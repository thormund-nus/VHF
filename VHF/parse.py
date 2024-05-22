from datetime import datetime
from datetime import timedelta
from functools import cached_property
import logging
from typing import Optional
import numpy as np
import numpy.typing as npt
import os
from _io import BufferedRandom

__all__ = [
    "VHFparser",
]


class BinaryVHFTrace:
    """Collection of methods for parsing binary data out of VHF trace.

    Binary trace data is a contiguous binary array that is interpreted a word
    at a time. Each word is 8 bytes, intended to be unpacked as (I, Q, M).
    """

    bytes_per_word: int = 8

    @staticmethod
    def read_i_arr(trace: npt.NDArray[np.uint64]) -> npt.NDArray[np.int32]:
        """Gets the I portion of a word."""
        i_arr = np.bitwise_and(
            np.right_shift(trace, 24), 0xFFFFFF, dtype=np.dtype(np.int32)
        )
        i_arr = i_arr - (i_arr >> 23) * 2**24
        return i_arr

    @staticmethod
    def read_q_arr(trace: npt.NDArray[np.uint64]) -> npt.NDArray[np.int32]:
        """Gets the Q portion of a word."""
        q_arr = np.bitwise_and(trace, 0xFFFFFF, dtype=np.dtype(np.int32))
        q_arr = q_arr - (q_arr >> 23) * 2**24
        return q_arr

    @staticmethod
    def read_m_arr(trace: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
        """Gets the M portion of a word."""
        # is it safe to lower the space of this?
        return np.right_shift(trace, 48, dtype=np.dtype(np.int64))


class TraceTimer:
    """Timing and index management for VHF parser."""

    def __init__(
        self,
        trace_start: datetime,
        trace_freq: float,
        trace_size: int
    ):
        """Populate timing start/end of trace within VHF trace file.

        Init arguments
        -----
        trace_start: datetime.datetime
            This is the absolute time associated with trace[0].
            Unpacking trace[0] gives (I, Q, M).
        trace_freq: float
            This is the frequency in Hertz associated to trace[1] - trace[0].
        trace_size:
            By viewing the entire binary file as being a sequence of words,
            where a word is some specified number of bytes, the trace_size is
            the value of n for
                file = [header[0] ... header[j] trace[0] ... trace[n-1] ]
            which is the number of words in trace.
        """
        self.logger = logging.getLogger("vhfparser")
        self.trace_start = trace_start
        self.trace_duration = timedelta(seconds=trace_size/trace_freq)
        self.sample_interval = timedelta(seconds=1/trace_freq)
        self.trace_end = self.trace_start + self.trace_duration
        self.trace_freq = trace_freq
        self.plot_start = self.trace_start
        self.plot_end = self.trace_end

    def update_plot_timing(self, **kwargs) -> bool:
        """Update TraceTimer to select trace to specified interval.

        Output
        ------
        bool: True if plotting region has been changed.

        Parameters (**kwargs)
        ----------
        start: Optional[datetime | timedelta]
            If start is absolute, plot_start will be set to
            min(max(trace_start, start), trace_end).
            If start is relative, plot_start will be set to
            min(trace_start + start, trace_end).
            If start is not specified, prior decided start will be used.
        duration: Optional[timedelta]
            Duration specifies end relative to plot start. Will still be
            bounded to within trace_end. This makes the strong assumption that
            the user wants the trace to have the duration be duration relative
            to the specified start, prior to this method having coerced start
            to be bound to the trace's start and end.
        end: Optional[datetime]
            Specify plot end. Will still be bounded to trace end.
        """
        # Populate from kwargs
        start: Optional[datetime | timedelta] = kwargs.get(
            "start") if "start" in kwargs else None
        duration: Optional[timedelta] = kwargs.get(
            "duration") if "duration" in kwargs else None
        end: Optional[datetime] = kwargs.get(
            "end") if "end" in kwargs else None

        # Guard clause
        if duration is not None and end is not None:
            self.logger.warning(
                "Both duration and end were specified to update_plot_timing!")
            raise ValueError(
                "Both duration and end were specified to update_plot_timing!")
        if start is None and duration is None and end is None:
            self.logger.warning("Nothing passed into update_plot_timing.")
            return False

        start_changed: bool = False
        end_changed: bool = False

        # Time zone coercion
        if start is not None and isinstance(start, datetime):
            if self._datetime_aware(start) ^ self._datetime_aware(self.trace_start):
                start, _ = self._coerce_dt_aware(start, self.trace_start)
        if end is not None:
            if self._datetime_aware(end) ^ self._datetime_aware(self.trace_end):
                end, _ = self._coerce_dt_aware(end, self.trace_end)

        user_start = start

        # Change user's wanted start time to absolute time.
        if start is not None:
            if isinstance(start, timedelta):
                start: datetime = self.trace_start + start
            # Coerce start to be within trace
            if start > self.trace_end or start < self.trace_start:
                self.logger.warning(
                    "Specified plot start is out of bounds. Coercing...")
                start = min(max(self.trace_start, start), self.trace_end)
            start_changed = start != self.plot_start
            self.plot_start = start

        # If duration is used, we calculate the desired end time before seeing
        # if it needs to be coerced to within the bounds of the trace
        if duration is not None:
            # Get end to be absolute
            if isinstance(user_start, timedelta):
                end: datetime = self.trace_start + user_start + duration
            elif isinstance(user_start, datetime):
                end: datetime = user_start + duration
            else:
                raise ValueError
        elif end is not None:
            # By this point, variable end is datetime object.
            pass

        # Coerce end to be within trace
        if end is not None:
            if end > self.trace_end or end < self.trace_start:
                self.logger.warning(
                    "Specified plot end is out of bounds.")
                end = max(min(self.trace_end, end), self.trace_start)
            end_changed = end != self.plot_end
            self.plot_end = end

        return start_changed or end_changed

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

        # determine which to convert, and inherit timezone from the other
        if self._datetime_aware(dt1):
            dt2 = dt2.astimezone(dt1.tzinfo)
        elif self._datetime_aware(dt2):
            dt1 = dt1.astimezone(dt2.tzinfo)
        else:
            # Logical error
            raise NotImplementedError
        return dt1, dt2

    @property
    def start_idx(self) -> int:
        """Start index relative to trace[0] for plot_start."""
        delta = self.plot_start - self.trace_start
        return int(delta/self.sample_interval)

    @property
    def end_idx(self) -> int:
        """End index relative to file head for plot_start."""
        return self.start_idx + self.duration_idx

    @property
    def duration_idx(self) -> int:
        """Duration index specifying number of bytes to read."""
        delta = self.plot_end - self.plot_start
        return int(delta/self.sample_interval)


class ManifoldRollover:
    """Double-pass of trace's manifold rollover manager.

    For better memory management, we never intended on reading the entire file
    at once. However this meant that there is a need for a double pass on the
    entire trace data set, to then determine if there has been any
    "manifold"(m)-rollovers that might occur before the existing plot window
    that has to be accounted for.

    Run-time Methods
    -----
    update: Keeps track of overflows with successive blocks of trace's m_arr.
    lock: Blocks any more use of update, allows for use of populated fix_m_arr.
    """

    def __init__(self, trace_block_size: int):
        """Creates empty state of ManifoldRollover.

        The empty/base case state of ManfioldRollever is the state associated
        with Trace Data that has no manifold rollover.
        """
        self.__create_logger()
        self._lock: bool = False  # Toggles btwn receiving trc data and else

        # Variables needed in self._update
        self._ptl_overflow = BinaryVHFTrace.potential_m_overflow_tolerance
        self._trace_blk_id: int = -1
        self._prev_trc_last_m: Optional[np.int64] = None
        self._blk_size = trace_block_size

        # This stores the number of times to displace the plot upwards due to
        # packing into i16.
        self.sparse_m_delta: npt.NDArray[np.int16]
        self._list_m_delta: list[np.int16] = []
        # This is the associated index along the trace where m_delta occurs.
        # Ideally, we should be using usize (from Rust).
        self.sparse_m_delta_idx: npt.NDArray[np.uint64]
        self._list_m_delta_idx: list[np.uint64] = []
        self.logger.info("Initialization completed.")

    def __create_logger(self):
        self.logger = logging.getLogger("manifoldmanager")

    def lock(self):
        """Conclude firstpass of trace, locking in state of ManifoldRollover.

        This transits the behaviour of ManifoldRollover from consuming trace
        (in arbitrary sized chunks) to being able to give well-formed output.
        """
        if self._lock:
            self.logger.warning("Manifold Management has already been locked!")
            return

        self._lock = True
        self.sparse_m_delta = np.array(self._list_m_delta, dtype=np.int16)
        self.sparse_m_delta_idx = np.array(self._list_m_delta_idx, dtype=np.uint64)
        del self._list_m_delta
        del self._list_m_delta_idx
        self.logger.info("Locked and created sparse repr.")

    def _potential_overflow(self, m_block: npt.NDArray[np.int64]) -> bool:
        """Determines is a block of m_arr might require delta-obtaining."""
        if (np.size(np.where(m_block > self._ptl_overflow))
                + np.size(np.where(m_block < -self._ptl_overflow))) > 0:
            return True
        return False

    def _rollover_lemma(self, d_block: float | npt.NDArray[np.int64]):
        """Gets (idx, roll-over) for diff_block."""
        diff_overflow = BinaryVHFTrace.actual_m_overflow
        deltas = (np.greater(d_block, diff_overflow).astype(int)
                  - np.less(d_block, -diff_overflow).astype(int))
        idx = np.ravel(np.argwhere(deltas))  # assumes 1D
        rollover = deltas[idx]
        idx = idx + self._trace_blk_id * self._blk_size
        return idx, rollover

    def update(self, trace_block: npt.NDArray[np.uint64]):
        """Consume a chunk of trace.

        Takes in all chunks of entire trace file to then build up a
        representation of "manifold"(m)-rollover.
        """
        if trace_block.size < 1:
            self.logger.warning("Empty trace_block received in update!")
            return

        self._trace_blk_id += 1
        m_block: npt.NDArray[np.int64] = BinaryVHFTrace.read_m_arr(trace_block)
        last_m: np.int64 = m_block[-1]

        # We first perform the np.diff for the self._trace_blk_id > 1 case:
        x = None
        if self._prev_trc_last_m is not None:
            x = m_block[0] - self._prev_trc_last_m
        # Populate the np.diff for the given block
        if x is not None:
            diff_offset = 1
            diff = np.zeros_like(m_block, dtype=np.int64)
            diff[0] = x
        else:
            diff_offset = 0
            diff = np.zeros((m_block.size-1,), dtype=np.int64)
        if self._potential_overflow(m_block):  # Perform only if necessary
            diff[diff_offset:] = np.diff(m_block)
            # Next, get indices and rollover direction
            idx, deltas = self._rollover_lemma(diff)

            # Place into spare list
            self._list_m_delta.extend(deltas)
            self._list_m_delta_idx.extend(idx)

        # End the loop
        self._prev_trc_last_m = last_m


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
    header: Dictionary containing all parameters used to call runVHF

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

        max_data_size = int((self.filesize - self._num_head_bytes) / 8)

        # now read to right boundary of file
        self.data = np.memmap(
            filename,
            dtype=np.dtype(np.uint64).newbyteorder("<"),
            mode="r",
            offset=self._num_head_bytes,
            shape=(max_data_size,),
        )

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
        # in teststream.c (SVN v17), 1 + i lines(8 bytes) were written (see
        # line 353), where i was (magic mask + 1). we note that due to improper
        # null termination of headerbuffer (in line 352) meant that
        # teststream.c was reading beyond buffer, which was not guaranteed to
        # be all 0s. Nonetheless, consume it as header in accordance with
        # masked magic number and dump away. As such, stripping supposed 0s are
        # ok.
        # -----
        # + 1 to read one more byte than spec (convert_bin_to_text.c)
        header_count = (header_count_b+1) * 8
        self._num_head_bytes += header_count  # this is the claimed headersize
        self.headerraw: bytes = buffer.read(header_count)
        self.headerraw = self.headerraw.rstrip(b"\x00")
        self.parse_header(self.headerraw)

    def parse_header(self, header_raw: bytes):
        """Convert binary file header into a header property."""
        if header_raw is None:
            self.logger.error("parse_header invoked with empty argument: header_raw")
            raise ValueError("header_raw was not given.")

        # init
        self.header: dict = dict()
        header_raw: list[bytes] = header_raw.split(b"# ")[1:]
        self.logger.debug("parsed_header received header_raw = %s", header_raw)
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
                "Sampling frequency was not explicitly given in header. "
                "Defaulting to 20 MHz.")
            self.header["base sampling freq"] = 20e6

        if 's' in self.header and self.header['s'] is not None:
            self.header["sampling freq"] = (self.header["base sampling freq"]
                                            / (1 + self.header["s"]))
        else:
            self.header["sampling freq"] = self.header["base sampling freq"]

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
        self.i_arr: npt.NDArray[np.int32] = BinaryVHFTrace.read_i_arr(data)
        self.q_arr: npt.NDArray[np.int32] = BinaryVHFTrace.read_q_arr(data)
        self.m_arr: npt.NDArray[np.int64] = BinaryVHFTrace.read_m_arr(data)

        if fix_m:
            self._fix_overflow_m()

    @cached_property
    def _potential_m_overflow(self) -> bool:
        """Determine if elements in self.m_arr close exceeds +-tol."""
        # Used to calibrate invocation of _m_fix()
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
