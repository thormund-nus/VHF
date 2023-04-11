import numpy as np
import os
import struct

class VHFparser():
    # Written with reference to SVN r14

    def __init__(self, filename: str | os.PathLike):
        """Takes a VHF output file and populates relevant properties."""
        self._num_read_bytes = 0
        self.filesize = os.path.getsize(filename) # number of bytes

        # populate header
        with open(filename, 'rb') as file: # binary read
            header = file.read(8)
            if not 0xFFFFFFFFFFFF0000 & int.from_bytes(header, 'little') == 0x123456ABCDEF0000:
                raise ValueError('File does not conform to header expectations.')
            
            # 1 to account for first word used to determine size of header
            header_count_b = (int.from_bytes(header, 'little') & 0xFFFF) - 1 
            header_count = header_count_b*8
            self._num_read_bytes += header_count
            self.header = file.read((header_count)).rstrip(b"\x00")

        # populate body "data"
        max_data_size = int((self.filesize - self._num_read_bytes)/8)
        self.data = np.memmap(filename, 
                              dtype=np.dtype(np.uint64).newbyteorder('<'), 
                              mode='r', offset=self._num_read_bytes, 
                              shape=(max_data_size,))
        # get (I, Q, M)
        self.read_words_numpy(self.data)

    def read_word(self, b: bytes, idx = int):
        """Converts 8 bytes in (I,Q, M)."""
        m = struct.unpack_from('<h', b, 6)[0]
        i = struct.unpack_from('<I', b, 2)[0] >> 8
        # i = struct.unpack_from('<I', b, 3)[0] & 0xFFFFFF
        q = struct.unpack_from('<I', b, 0)[0] & 0xFFFFFF

        sign_extend = -1 & ~0x7fffff
        if i & 0x800000:
            i |= sign_extend
        # i = i - (i >> 23) * 2**24 # branchless
        if q  & 0x800000:
            q |= sign_extend

        # substitute for non-local variable
        self.m_arr[idx] = m
        self.i_arr[idx] = i
        self.q_arr[idx] = q
    
    def read_words_numpy(self, data: np.ndarray):
        self.i_arr = np.bitwise_and(np.right_shift(data, 24), 0xFFFFFF, dtype=np.dtype(np.int32))
        self.i_arr = self.i_arr - (self.i_arr >> 23) * 2**24
        self.q_arr = np.bitwise_and(data, 0xFFFFFF, dtype=np.dtype(np.int32))
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

    
if __name__ == '__main__':
    # x = VHFparser(os.path.join(os.path.dirname(__file__), 'vhf_func_gen/Data/60.000_020MHz.txt'))
    # print(f'{x.header = }')
    # print(f"{x.m_arr[-12:] = }")

    print()
    y = VHFparser(os.path.join(os.path.dirname(__file__), 'Data/2023-04-05T16:56:09.551460_s4_q10000000.txt'))
    # print(f'{y.header = }')
    # print(f'{y.i_arr[:12] = }\n{y.m_arr[:12] = }')
    # print(f"{y.i_arr[-5:] = }\n{y.m_arr[-5:] = }")