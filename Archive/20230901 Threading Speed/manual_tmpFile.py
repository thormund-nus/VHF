from pathlib import Path


def get_fname():
    with open('/dev/urandom', 'rb') as f:
        x = f.read(16)
    return Path('/dev/shm').joinpath(x.hex())
