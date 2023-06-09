from datetime import datetime
import logging
import numpy as np
import os
import struct


class VHFparser():
    # Written with reference to SVN r14

    def __init__(self, filename: str | os.PathLike):
        """Takes a VHF output file and populates relevant properties."""
        self.__createLogger()
        self._num_read_bytes = 0
        self.filesize = os.path.getsize(filename)  # number of bytes

        # populate header
        with open(filename, 'rb') as file:  # binary read
            header = file.read(8)
            if not 0xFFFFFFFFFFFF0000 & int.from_bytes(header, 'little') == 0x123456ABCDEF0000:
                self.logger.error("File: %s", filename)
                self.logger.error(
                    "not found to conform to header expectations.")
                raise ValueError(
                    'File does not conform to header expectations.')

            # 1 to account for first word used to determine size of header
            header_count_b = (int.from_bytes(header, 'little') & 0xFFFF) - 1
            header_count = header_count_b*8
            self._num_read_bytes += header_count
            self.headerraw: bytes = file.read((header_count)).rstrip(b"\x00")
            self.parse_header(self.headerraw)

        # populate body "data"
        max_data_size = int((self.filesize - self._num_read_bytes)/8)
        self.data = np.memmap(filename,
                              dtype=np.dtype(np.uint64).newbyteorder('<'),
                              mode='r', offset=self._num_read_bytes,
                              shape=(max_data_size,))
        # get (I, Q, M)
        self.read_words_numpy(self.data)

    def __createLogger(self):
        self.logger = logging.getLogger("vhfparser")

    # def read_word(self, b: bytes, idx=int):
    #     """converts 8 bytes in (I, Q, M)."""
    #     m = struct.unpack_from('<h', b, 6)[0]
    #     i = struct.unpack_from('<i', b, 2)[0] >> 8
    #     # i = struct.unpack_from('<i', b, 3)[0] & 0xffffff
    #     q = struct.unpack_from('<i', b, 0)[0] & 0xffffff

    #     sign_extend = -1 & ~0x7fffff
    #     if i & 0x800000:
    #         i |= sign_extend
    #     # i = i - (i >> 23) * 2**24 # branchless
    #     if q & 0x800000:
    #         q |= sign_extend

    #     # substitute for non-local variable
    #     self.m_arr[idx] = m
    #     self.i_arr[idx] = i
    #     self.q_arr[idx] = q

    def read_words_numpy(self, data: np.ndarray):
        self.i_arr = np.bitwise_and(np.right_shift(
            data, 24), 0xFFFFFF, dtype=np.dtype(np.int32))
        self.i_arr = self.i_arr - (self.i_arr >> 23) * 2**24
        self.q_arr = np.bitwise_and(data, 0XFFFFFF, dtype=np.dtype(np.int32))
        self.q_arr = self.q_arr - (self.q_arr >> 23) * 2**24
        self.m_arr = np.right_shift(data, 48, dtype=np.dtype(np.int64))
        # self.m_arr = self._m_arr_copy.astype(np.int16)

        # failed alternative to m_arr creation
        # self.m_arr.dtype = np.int16 # in-place data change, may be depreciated
        # ? setting the dtype seems to be as buggy as ndarray.view() method
        # this is in contrast to ndarray.astype() method, which returns a copy
        return

    # @property
    # def m_arr(self):
    #     return self._m_arr_copy.view(dtype=np.int16)

    def parse_header(self, header_raw: bytes):
        if header_raw is None:
            raise ValueError("header_raw was not given.")

        # init
        self.header: dict = dict()
        header_raw: list[bytes] = header_raw.split(b'# ')[1:]
        print(f"Debug: {header_raw = }")
        for x in header_raw[0].split(b' -')[1:]:  # command line record
            x = x.decode().strip(' ')
            # print(f"Debug: {x = }")
            self.header[x[0]] = x[1:].strip()
            # print(f"Debug: {self.header[x[0]] = }")

        # print(f"Debug: {header_raw[1].split(b': ')[1].decode().strip() = }")
        self.header['Time start'] = datetime.fromisoformat(header_raw[1].split(b': ')[1].decode().strip())
        for k, v in self.header.items():
            try:
                self.header[k] = int(v)
            except ValueError:
                pass
            except TypeError:
                pass
        
        # generate explicit params
        if 'l' in self.header.keys():
            self.header['base sampling freq'] = 10e6
        elif 'h' in self.header.keys():
            self.header['base sampling freq'] = 20e6
        else:
            self.header['base sampling freq'] = 20e6
        
        if self.header['s'] is not None:
            self.header['sampling freq'] = self.header['base sampling freq'] / (1+self.header['s'])


if __name__ == '__main__':
    # x = VHFparser(os.path.join(os.path.dirname(__file__), 'vhf_func_gen/Data/60.000_020MHz.txt'))
    # print(f'{x.header = }')
    # print(f"{x.m_arr[-12:] = }")

    print()
    y = VHFparser(os.path.join(os.path.dirname(__file__),
                               'Data/2023-05-16T15:20:51.480758_s4_q10000.bin'))
    # print(f'{y.header = }')
    # print(f'{y.i_arr[:12] = }\n{y.m_arr[:12] = }')
    # print(f"{y.i_arr[-5:] = }\n{y.m_arr[-5:] = }")

