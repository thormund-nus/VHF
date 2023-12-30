# Subprocess will call this file to
# 1. Run the VHF into a temporary file
# 2. Signal to main that it has completed collection
#
# 3. Process temp data and write into targetted output
# 4. Wait to run step 1, unless called to terminate.

# from ..parseVHF import VHFparser # relative import
from multiprocessing.connection import Connection

def sample_and_analyse_thread(comm: Connection, cmd, ):
    return

if __name__ == '__main__':
    print('sample_and_analyse.py was called directly!')