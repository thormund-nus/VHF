import array
import os
import struct

class VHFparser():
    # Written with reference to SVN r14

    def __init__(self, filename: str | os.PathLike):
        # self.tmp = 0
        self.num_read_bytes = 0
        self.filesize = os.path.getsize(filename) # number of bytes
        with open(filename, 'rb') as file: # binary read
            header = file.read(8)
            # self.num_read_bytes += 8
            if not 0xFFFFFFFFFFFF0000 & int.from_bytes(header, 'little')== 0x123456ABCDEF0000:
                raise ValueError('File does not conform to header expectations.')
            
            header_count_b = (int.from_bytes(header, 'little') & 0xffff) - 1 # 1 to account for first word used to determine size of header
            header_count = header_count_b*8
            self.num_read_bytes += header_count
            self.header = file.read((header_count)).rstrip(b"\x00")
            # print(f'{self.header = }\n{len(self.header) = }\n{header_count = }\n{self.num_read_bytes = }')

            max_size = int((self.filesize - self.num_read_bytes)/8)
            self.i_arr, self.q_arr, self.m_arr = array.array('h', [0]*max_size), array.array('l', [0]*max_size), array.array('l', [0]*max_size)  # not consistent with struct module
            idx, pop = 0, 0
            while (word := file.read(8)):
                # if self.tmp < 4:
                #     print(f"{word = }")
                self.num_read_bytes += 8
                if word == b'\x00\x00\x00\x00\x00\x00\x00\x00':
                    pop += 1
                    continue
                self.read_word(word, idx)
                del word
                idx += 1
                # self.tmp += 1
                # if self.tmp > 11:
                #     break

            # print(f"{pop = }")
            [(self.i_arr.pop(), self.q_arr.pop(), self.m_arr.pop()) for _ in range(pop)]

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
        # try:
        self.i_arr[idx] = i
        # except OverflowError as exc:
        #     print(f"{b = }")
        #     print(f"{struct.unpack_from('<I', b, 2)[0] >> 8 = }")
        #     print(f"{i = }")
        #     print(f"{struct.unpack_from('<I', b, 3)[0] & 0xFFFFFF = }")
        #     print(exc)
        #     os.exit(-1)

        self.q_arr[idx] = q

if __name__ == '__main__':
    # x = VHFparser(os.path.join(os.path.dirname(__file__), 'vhf_func_gen/Data/60.000_020MHz.txt'))
    # print(f'{x.header = }')
    # print(f"{x.m_arr[-12:] = }")

    print()
    y = VHFparser(os.path.join(os.path.dirname(__file__), 'Data/2023-04-05T16:56:09.551460_s4_q10000000.txt'))
    # print(f'{y.header = }')
    # print(f'{y.i_arr[:12] = }')
    # print(f"{y.i_arr[-5:] = }\n{y.m_arr[-5:] = }")