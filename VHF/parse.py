
from datetime import datetime
from datetime import timedelta
from functools import cached_property
import logging
import numpy as np
import os
from _io import BufferedRandom


class VHFparser:
    # Written with reference to SVN r14
    """
    Class that parses binary, hexadecimal and ASCII output from VHF board,
    from file objects and data streams.

    Init arguments
    -----
    start_time: datetime.datetime compatible
    end_time: datetime.datetime compatible
        Trims VHFparser.data and derived objects at init-time to be bounded
        within start_time and end_time

    Init-time available properties
    -----
    filesize: Number of bytes
    header: Dictionary containing all paramaters used to call runVHF

    Available methods
    -----
    parse_header: Gets relevant front number of bytes and returns
    read_words_numpy: Converts binary words into (Q, I, M) arrays

    Run-time available properties
    -----
    reduced_phase: Phase/2pi array
    radii: I^2 + Q^2 array. Division by N points is not done.
    """

    def __init__(self, filename: str | os.PathLike | BufferedRandom, *,
                 start_time: datetime | None = None,
                 end_time: datetime | None = None):
        """Take a VHF output file and populates relevant properties."""
        self.__create_logger()
        self.logger.info('VHF parser has been run for file: %s', filename)

        # attributes:
        # filesize: bytes -
        #     length of data + header
        # _num_head_bytes: mutable, bytes -
        #     aims to show number of bytes of header read

        # init guard: parse time params
        if start_time is not None:
            assert isinstance(start_time, datetime)
        if end_time is not None:
            assert isinstance(end_time, datetime)

        # Create class specific modifiers
        self._num_head_bytes = 0
        self._fix_m_called = False

        # init: Begin Parsing logic, populating header
        if isinstance(filename, BufferedRandom):
            self.logger.debug("Obtained Buffered Random object")
            self.filesize = filename.tell()
            filename.seek(0)
            self._init_buffer(filename)
        elif isinstance(filename, str) or isinstance(filename, os.PathLike):
            self.logger.debug("Obtained File-like object")
            self.filesize = os.path.getsize(filename)  # number of bytes
            self._init_file(filename)
        else:
            self.logger.warning("No file-like/buffer object given to init.")
            print("No file-like/buffer object given to init.")
            return

        # init: Populate body "data" to within (start_time, end_time)
        # Available data in file in contrast to header['s']'s expected
        # file-length in the event that sampling
        max_data_size = int((self.filesize - self._num_head_bytes) / 8)
        if end_time is not None:
            # cast both to offset-aware if either times are
            if (self._datetime_aware(end_time)
                    ^ self._datetime_aware(self.header["Time start"])):
                end_time, self.header["Time start"] = self._coerce_dt_aware(
                    end_time, self.header["Time start"])
            target_diff = end_time - self.header["Time start"]
            # ensure to invoke only file is exceeds time argument bounds
            if (t_diff_s := target_diff.total_seconds()) > 0:
                max_data_size = int((
                    t_diff_s * self.header["sampling freq"])
                )

        # now read to right boundary of file
        self.data = np.memmap(
            filename,
            dtype=np.dtype(np.uint64).newbyteorder("<"),
            mode="r",
            offset=self._num_head_bytes,
            shape=(max_data_size,),
        )

        # shape has trimmed all data after end_time, _drop_left to trim before
        # start_time, which can only be done after obtaining (I, Q, M)
        if start_time is not None:
            if (self._datetime_aware(start_time)
                    ^ self._datetime_aware(self.header["Time start"])):
                start_time, self.header["Time start"] = self._coerce_dt_aware(
                    start_time, self.header["Time start"]
                )
            target_diff = start_time - self.header["Time start"]

            if (t_diff_s := target_diff.total_seconds()) > 0:
                self._drop_left(int(t_diff_s * self.header["sampling freq"]))

        # init: get (I, Q, M)
        self.read_words_numpy(self.data)

    def __create_logger(self):
        self.logger = logging.getLogger("vhfparser")

    def _init_file(self, filename):
        """For files. Opens file object, and parse by _init_buffer."""
        with open(filename, "rb") as file:  # binary read
            self._init_buffer(file)

    def _init_buffer(self, buffer):
        """For bufferedRandom."""
        header = buffer.read(8)
        actual = (0xFFFFFFFFFFFF0000 & int.from_bytes(header, "little"))
        expected = 0x123456ABCDEF0000
        if actual != expected:
            self.logger.error(
                "Buffer not found to conform to header expectations.")
            self.logger.debug("buffer.tell() = %s", buffer.tell())
            self.logger.debug(
                "Buffer read(1024): %s", header +
                buffer.read(min(1024, self.filesize-8))
            )
            raise ValueError(
                "Buffer does not conform to header expectations.")

        # in accordance with convert_bin_to_text.c
        header_count_b = (int.from_bytes(header, "little") & 0xFFFF) - 1
        # in teststream.c (SVN v17), 1 + i lines(8 bytes) were written (see line
        # 353), where i was (magic mask + 1). we note that due to improper null
        # termination of headerbuffer (in line 352) meant that teststream.c was
        # reading beyond buffer, which was not guranteed to be all 0s.
        # Nonetheless, consume it as header in accordance with masked magic
        # number and dump away. As such, stripping supposed 0s are ok.
        # -----
        # + 1 to read one more byte than spec (convert_bin_to_text.c)
        header_count = (header_count_b+1) * 8
        self._num_head_bytes += header_count  # this is the claimed headersize
        self.headerraw: bytes = buffer.read(header_count)
        self.headerraw = self.headerraw.rstrip(b"\x00")
        self.parse_header(self.headerraw)

    def _drop_left(self, val: int):
        """Mutates self.data and derivatives by dropping val number of points.

        Timing information is correspondingly updated.
        """
        try:
            val = int(val)
        except TypeError:
            raise ValueError(f"{val = } given could not be cast to int!")
        if val == 0:
            # Nothing to do
            return
        elif val < 0:
            raise ValueError(f"{val = } given is too small!")

        self.header["Time start"] += timedelta(
            seconds=(val / self.header["sampling freq"]))
        self.data = self.data[val:]

    def _datetime_aware(self, dt: datetime) -> bool:
        """Determine if a datetime object is aware, or otherwise (naive)."""
        # doc: https://docs.python.org/3/library/datetime.html#determining-if-an-object-is-aware-or-naive # noqa: E501
        # Resorts to False and undefined -> False
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None

    def _coerce_dt_aware(self, dt1: datetime,
                         dt2: datetime) -> tuple[datetime, datetime]:
        """Coerces one datetime object to be aware like the other object.

        Datetime objects are not orderable if exclusively one is 'aware'.
        The naive object is forced to inherit the aware datetime object tzinfo.
        """
        if not (self._datetime_aware(dt1) ^ self._datetime_aware(dt2)):
            # No coercion required
            return dt1, dt2

        # determine which to convert, and inherit timeone from the other
        if self._datetime_aware(dt1):
            dt2 = dt2.astimezone(dt1.tzinfo)
        elif self._datetime_aware(dt2):
            dt1 = dt1.astimezone(dt2.tzinfo)
        else:
            # Logical error
            raise NotImplementedError
        return dt1, dt2

    # def read_word(self, b: bytes, idx=int):
    #     """converts 8 bytes in (I, Q, M)."""
    #     m = struct.unpack_from('<h', b, 6)[0]
    #     i = struct.unpack_from('<i', b, 2)[0] >> 8
    #     # i = struct.unpack_from('<i', b, 3)[0] & 0xffffff
    #     q = struct.unpack_from('<i', b, 0)[0] & 0xffffff
    #
    #     sign_extend = -1 & ~0x7fffff
    #     if i & 0x800000:
    #         i |= sign_extend
    #     # i = i - (i >> 23) * 2**24 # branchless
    #     if q & 0x800000:
    #         q |= sign_extend
    #
    #     # substitute for non-local variable
    #     self.m_arr[idx] = m
    #     self.i_arr[idx] = i
    #     self.q_arr[idx] = q

    def read_words_numpy(self, data: np.ndarray, fix_m: bool = False):
        """Convert binary words into numpy arrays.

        Input
        -----
        data: np.ndarray
            This contains numpy binary data that contains 1 word of data
            per array element.
        fix_m: bool
            Accounts for 16-bit overflow of m during reading.
        """
        self.i_arr = np.bitwise_and(
            np.right_shift(data, 24), 0xFFFFFF, dtype=np.dtype(np.int32)
        )
        self.i_arr = self.i_arr - (self.i_arr >> 23) * 2**24
        self.q_arr = np.bitwise_and(data, 0xFFFFFF, dtype=np.dtype(np.int32))
        self.q_arr = self.q_arr - (self.q_arr >> 23) * 2**24
        self.m_arr = np.right_shift(data, 48, dtype=np.dtype(np.int64))
        # self.m_arr = self._m_arr_copy.astype(np.int16)

        # failed alternative to m_arr creation
        # self.m_arr.dtype = np.int16 # in-place data change, may be depreciated
        # ? setting the dtype seems to be as buggy as ndarray.view() method
        # this is in contrast to ndarray.astype() method, which returns a copy

        if fix_m:
            self._fix_overflow_m()

    def parse_header(self, header_raw: bytes):
        """Convert binary file header into a header property."""
        if header_raw is None:
            self.logger.error("parse_header invoked with empty argument: header_raw")
            raise ValueError("header_raw was not given.")

        # init
        self.header: dict = dict()
        header_raw: list[bytes] = header_raw.split(b"# ")[1:]
        self.logger.debug("parsed_header recieved header_raw = %s", header_raw)
        # populate self.header from command line
        func_cmd_line = lambda x: "command line: " in x.decode()
        for header_line in filter(func_cmd_line, header_raw):
            for x in header_line.split(b" -")[1:]:
                x = x.decode().strip(" ")
                self.header[x[0]] = x[1:].strip()

        # populate self.header["Time start"]
        func_record = lambda x: "recording start: " in x.decode()
        if any(map(func_record, header_raw)):
            filtered_header = filter(func_record, header_raw)
            entry = next(filtered_header)
            val = entry.split(b': ')[1]
            iso_str = val.split(b'\n')[0].decode().strip()
            self.header["Time start"] = datetime.fromisoformat(iso_str)
            try:
                next(filtered_header)
                self.logger.warn("More than one `recording start` found in header.")
            except StopIteration:
                pass
            except Exception as e:
                self.logger.warn("Error obtained trying to obtain next on filter obtaining `recording start`")
                self.logger.warn("Exception: %s", e)
        else:
            self.logger.warn("Command date and time could not be found within header. Attempting to obtain date and time from filename.")
            # TODO: obtain from filename.

        for k, v in self.header.items():
            try:
                self.header[k] = int(v)
            except ValueError:
                pass
            except TypeError:
                pass

        # generate explicit params
        if 'l' in self.header:
            self.header["base sampling freq"] = 10e6
        elif 'h' in self.header:
            self.header["base sampling freq"] = 20e6
        else:
            self.logger.warning(
                "Sampling frequency was not explictly given in header. "
                "Defaulting to 20 MHz.")
            self.header["base sampling freq"] = 20e6

        if 's' in self.header and self.header['s'] is not None:
            self.header["sampling freq"] = (self.header["base sampling freq"]
                                            / (1 + self.header["s"]))
        else:
            self.header["sampling freq"] = self.header["base sampling freq"]

    @cached_property
    def _potential_m_overflow(self) -> bool:
        """Determine if elements in self.m_arr close exceeds +-tol."""
        # Used to calibrate invokation of _m_fix()
        tol: int = 0x7F00
        if tol < 0:
            raise ValueError
        if (np.size(np.where(self.m_arr > tol))
                + np.size(np.where(self.m_arr < -tol))) > 0:
            return True
        return False

    def _fix_overflow_m(self) -> None:
        """Class method for fixing m value with 16-bit overflow."""
        if not self._fix_m_called:
            # get m_shift of +- n as according before multiplying by 2**16 to
            # account for overflow

            # 1. get location of +/-ve overflows
            # index 0 being +1 would mean that m[0] has flowed above the 16-bit
            # limit into a negative m[1], similarly, index 1 being -1 would
            # mean that m[1] has flowed below the 16-bit minimum into a large
            # positive m[2]
            deltas = np.diff(self.m_arr)  # subtraction between views
            # 2. now round towards +-1
            tol = 0xF000  # keep as int
            deltas = (np.greater(deltas, tol).astype(int)
                      - np.less(deltas, -tol).astype(int))
            # 3. finally fix
            self.m_arr[1:] -= 0x10000 * np.cumsum(deltas)
            self._fix_m_called = True
        else:
            print("overflow m_arr has been fixed before!")

    @cached_property
    def reduced_phase(self):
        """Obtain phase(time)/2pi for the given file.

        phase is obtained by atan(Q/I).
        """
        # Account for 16 bit overflow of m.
        if not self._fix_m_called and self._potential_m_overflow:
            self._fix_overflow_m()

        phase = -np.arctan2(self.i_arr, self.q_arr)
        phase /= 2*np.pi
        phase -= self.m_arr
        return phase

    @cached_property
    def radii(self):
        """Obtains radius(time) for the given file.

        Radius is obtained by sqrt(Q^2 + I^2).
        """
        # dtype starting from int32 should be a non-issue
        radii = np.hypot(self.q_arr, self.i_arr)
        return radii
