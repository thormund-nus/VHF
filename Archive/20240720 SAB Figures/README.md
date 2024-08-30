# Plots

This is a folder that is a proof of concept for now that its possible to parse
data as a time series in as much of a file agnostic way as possible with as
minimal an overhead in memory as possible, whilst maximising analytical throughput.

As of the time of this commit, the misnamed `data_to_npy.py` file takes a
series of files as given in `SEARCH_FILES` to then get all the spectrogram
information possible for the timings specified between `TIME_START` and
`TIME_END`, in intervals of `INTERVAL` (15 minutes), giving the plot that is
used in the SAB poster.

## The plot as of this commit
![spec](https://github.com/user-attachments/assets/e7ec59d1-1bf7-4e37-a0a7-fd0220089949)
