from datetime import date, datetime, timedelta
from data_to_npy import aware_to_naive
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
from numpy import datetime64
from pathlib import Path

ARBITRARY_DAY = date(1970, 1, 1)
ARBITRARY_DATETIME = datetime(1970, 1, 1)
BASE_DIR = Path(__file__).parent
NPZ_FILE = BASE_DIR.joinpath("./SAB_Collated_JanData.npz")
FREQ_IDX = 600
FREQ_UNDER_PLOT = 15.715925744992672

def plot_data():
    """time, amplitude(linear, Δλ/√Hz), error for plot"""

    # 1. Pull from files
    with open(NPZ_FILE, "rb+") as file:
        npz = np.load(file, allow_pickle=True)
        times = npz["times"]
        freqs = npz["freqs"]
        ampls = npz["ampls"]
        times = times.flatten()

    # if times is None or freqs is None or ampls is None:
    #     raise ValueError

    # 1a. Ensure we pulled the correct data
    assert times.shape == (344,)
    assert freqs[0, FREQ_IDX] == FREQ_UNDER_PLOT

    # 2. Clean data up into matplotlib friendlier terms
    mean, std = ampls[:, 0, FREQ_IDX],  ampls[:, 1, FREQ_IDX]
    lower = np.power(10, (mean-std)/20)
    upper = np.power(10, (mean+std)/20)
    mean = np.power(10, mean/20)
    err = np.vstack((mean-lower, upper-mean))

    # 3. sort in time
    idx_sort = np.argsort(times)
    times = times[idx_sort]
    mean = mean[idx_sort]
    err = err[:, idx_sort]
    print(f"{type(times[0]) = }")
    return times, mean, err


def is_day(day: date):
    def func(x: list[datetime]):
        t = np.fromiter(map(aware_to_naive, x), dtype='datetime64[D]')
        t_year = t.astype('datetime64[Y]').astype(int) + 1970
        t_month =  t.astype('datetime64[M]').astype(int) % 12 + 1
        t_date = t - t.astype('datetime64[M]') + 1
        t_date = t_date.astype('int')

        return np.logical_and(
            np.logical_and(t_date == day.day, t_month == day.month),
            t_year == day.year
        )
    return func


def fig_size_etc(case: str):
    """Choose between SAB and QE and Indiv."""
    match case:
        case "SAB":
            return (5, 3), 12, 3, 2, 4
        case "QE":
            inches_per_mm = 0.0393
            scale_factor = 0.75
            return (scale_factor*160*inches_per_mm, 0.8*scale_factor*90*inches_per_mm), 10, 4, 2, 3
        case "Indiv":
            # This did not go into the SAB-Individual poster component in the end.
            scale_factor = 0.5
            return ((x_in:=0.5*(11.7-2*0.6-0.1)), scale_factor*x_in), 10, 4, 2, 3
        case _:
            raise ValueError


def extract_time_as_datetime_from_NDarr_of_npdatetime64_w_dtype_obj(x):
    """Matplotlib requires that the x-axis be datetime and not time objects.
    We are filtering down from what is an NDArray of datetime64.
    To ensure a cleaner plot we set all object to having the same date, so that
    matplotlib doesn't extend the x-axis for showing 4 days.
    """
    return list(map(lambda y: datetime.combine(ARBITRARY_DAY, y.time()), x))


def main(err_bars: bool = False):
    """Plot of average Sxx(@15.7Hz) of every 15 mins over 4 days."""
    # 1. Get data for plotting
    times, mean, err = plot_data()

    # 2. Get plot parameters
    for plot_type in ["SAB", "QE", "Indiv"]:
        fig_size, font_size, msize, elinew, csize = fig_size_etc(plot_type)

        # 3. Perform plotting; We have the spectrogram in a separate figure
        # 3a. Set figure up
        fig, ax = plt.subplots(figsize=fig_size, dpi=300)
        for day_offset in range(4):
            current_day = date(year=2024, month=1, day=19+day_offset)
            if not err_bars:  # There are no error bars
                target_y_lim = (10**-0.98, 10**-0.72)
                err = np.zeros_like(err)
            else:  # There are error bars
                target_y_lim = (10**-2.0, 10**-0.0)
                ax.set_yscale("log")
            ax.set_ylim(*target_y_lim)
            idx = np.where(is_day(current_day)(times))
            # 3b. Perform error plot
            f_err = err[:, idx]  # filtered err
            times_after_offset = times[idx]
            times_after_offset = extract_time_as_datetime_from_NDarr_of_npdatetime64_w_dtype_obj(times_after_offset)

            ys = mean[idx]
            if day_offset == 3:
                cutoff = 9
                times_after_offset = times_after_offset[:-cutoff]
                ys = ys[:-cutoff]
                f_err = f_err[:, :, :-cutoff]
            ax.plot(
                times_after_offset, ys,
                "-x",
                linewidth=0.5,
                markersize=msize,
            )
            if err_bars:
                ax.errorbar(
                    times_after_offset, ys,
                    yerr=f_err.reshape(-1, *f_err.shape[-1:]),
                    fmt=".",
                    markersize=msize,
                    elinewidth=elinew,
                    capsize=csize
                )
            # 3c. y-axis names
            ax.set_ylabel(
                r"Amplitude ($\Delta\lambda / \sqrt{\text{Hz}}$)",
                # r"Amplitude $\left[\Delta\lambda/\sqrt{\text{Hz}}\right]$",
                # usetex=True,
                fontsize=font_size
            )

        # a. set x_axis
        dfmt = mdates.DateFormatter("%-I%P")
        plt.xlim(ARBITRARY_DATETIME, ARBITRARY_DATETIME+timedelta(hours=24))
        plt.xticks([
            datetime(1970, 1, 1, 0),
            datetime(1970, 1, 1, 6),
            datetime(1970, 1, 1, 12),
            datetime(1970, 1, 1, 18),
            datetime(1970, 1, 2),
        ])
        ax.xaxis.set_major_formatter(dfmt)
        plt.legend(["Friday", "Saturday", "Sunday", "Monday"], loc="lower right", prop={"size":8}, ncol=2)
        plt.tight_layout()
        # b. Other titling
        # fig.suptitle(r"Spectrum observed over 4 days")

        if err_bars:
            err_name = "_wErr"
        else:
            err_name = ""
        # plt.show()
        fig.savefig(BASE_DIR.joinpath(f"{plot_type}_{FREQ_UNDER_PLOT:.5f}Hz_Jan{err_name}.png"), dpi=300)
        # 4. Close figure
        plt.close()

    return


def qe_plot_data():
    """For data point of 2024Jan22, get (mean,err) against frequencies."""
    # 1. Pull from files
    with open(NPZ_FILE, "rb+") as file:
        npz = np.load(file, allow_pickle=True)
        times = npz["times"]
        freqs = npz["freqs"]
        ampls = npz["ampls"]
        times = times.flatten()

    # if times is None or freqs is None or ampls is None:
    #     raise ValueError

    # 1a. Ensure we pulled the correct data
    assert times.shape == (344,)
    assert freqs[0, FREQ_IDX] == FREQ_UNDER_PLOT

    # 2. Clean data up into matplotlib friendlier terms
    # there is nothing
    
    # 3. sort in time
    idx_sort = np.argsort(times)
    times = times[idx_sort]

    time_idx = -39
    times = times[time_idx]
    freqs = freqs[time_idx]
    mean, std = ampls[time_idx, 0, :],  ampls[time_idx, 1, :]
    lower = np.power(10, (mean-std)/20)
    upper = np.power(10, (mean+std)/20)
    mean = np.power(10, mean/20)
    err = np.vstack((mean-lower, upper-mean))

    print(f"{times = }")
    return freqs, mean, err


def qe_plot_main():
    """Plot for a single time point (2024Jan22) across all frequencies."""
    freqs, mean, err = qe_plot_data()
    fig_size, font_size, msize, elinew, csize = fig_size_etc("Indiv")

    fig, p_ampx = plt.subplots(figsize=fig_size, dpi=300)
    p_ampx.set_yscale("log")
    p_ampx.plot(freqs, mean)
    p_ampx.fill_between(freqs, mean-err[0], mean+err[1],
                        alpha=0.5)
    p_ampx.set_title("22 Jan 2024 6.00-6.15am")
    p_ampx.set_ylabel(
        r"Amplitude ($\Delta\lambda / \sqrt{\text{Hz}}$)",
        fontsize=font_size
    )
    p_ampx.set_xlabel(
        r"Freq (Hz)",
        fontsize=font_size
    )
    fig.tight_layout(pad=1.01)
    fig.savefig(BASE_DIR.joinpath("Indiv_freq.png"), dpi=300)


if __name__ == "__main__":
    main(True)
    main()
    print("Sxx over freq plot!")
    qe_plot_main()
