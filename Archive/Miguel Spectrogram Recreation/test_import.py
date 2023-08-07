# this file is functions that Miguel have written vs I would have 
# written, for the purpose of importing in testing

import numpy as np

def block_avg(my_arr: np.ndarray, N: int):
    """Returns a block average of 1D my_arr in blocks of N."""
    if N == 1:
        return my_arr
    return np.mean(my_arr.reshape(np.shape(my_arr)[0]//N, N), axis=1)

def block_avg_tail(my_arr: np.ndarray, N: int):
    """Block averages the main body, up till last block that might 
    contain less than N items. Assumes 1D."""
    if N == 1:
        return my_arr
    if np.size(my_arr) % N == 0:
        return block_avg(my_arr, N)
    else:
        result = np.zeros(np.size(my_arr)//N + 1)
        result[:-1] = block_avg(my_arr[:np.size(my_arr)//N * N], N)
        result[-1] = np.mean(my_arr[np.size(my_arr)//N * N:])
        return result

def miguel_average_tail(my_arr: np.ndarray, N: int):
    result = []
    len_arr = len(my_arr)
    n = 0

    while n < len_arr:
        if (n + N -1) < len_arr:
            t = 0
            for i in range(N):
                t += my_arr[n+i]
            result.append(t/N)
        else:
            t = 0
            c = 0
            while n < len_arr:
                c += 1
                t += my_arr[n]
                n += 1
            if c != 0:
                result.append(t/c)
        n += N

    result = np.array(result)
    return result

if __name__ == '__main__':
    print("Import test_import.py.")