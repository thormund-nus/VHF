from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.parse import TraceTimer  # noqa


def test_TraceTimer_regular():
    example_start = datetime(2024, 1, 19, 0, 34, 36,
                             tzinfo=timezone(timedelta(seconds=28800)))
    example_freq = 40000  # Hz
    example_size = example_freq * 2 * 60 * 60  # 2 hours
    timings = TraceTimer(0, example_start, example_freq, example_size)
    expected = timedelta(hours=2)
    assert timings.trace_end - example_start == expected

    rel_start = timedelta(seconds=70)
    change_statues = timings.update_plot_timing(start=rel_start)
    expected_duration = timedelta(hours=2) - rel_start
    assert change_statues
    assert timings.plot_end - timings.plot_start == expected_duration
    assert timings.plot_start == timings.trace_start + rel_start
    assert timings.plot_end == timings.trace_end

    new_end_rel_to_trc_start = timedelta(hours=1, seconds=12)
    new_end = example_start + new_end_rel_to_trc_start
    change_statues = timings.update_plot_timing(end=new_end)
    expected_duration = new_end_rel_to_trc_start - rel_start
    assert change_statues
    assert timings.plot_end - timings.plot_start == expected_duration
    assert timings.plot_start == timings.trace_start + rel_start
    assert timings.plot_end == timings.trace_start + new_end_rel_to_trc_start

    abs_start = datetime(2024, 1, 19, 1, 34, 36,
                         tzinfo=timezone(timedelta(seconds=28800)))
    dur = timedelta(seconds=5)
    change_statues = timings.update_plot_timing(start=abs_start, duration=dur)
    assert change_statues
    assert timings.plot_start == abs_start
    assert timings.plot_end - timings.plot_start == dur

    change_statues = timings.update_plot_timing(start=abs_start, duration=dur)
    assert not change_statues


def test_TraceTimer_OOB():
    example_start = datetime(2024, 1, 19, 0, 34, 36,
                             tzinfo=timezone(timedelta(seconds=28800)))
    example_freq = 40000  # Hz
    example_size = example_freq * 2 * 60 * 60  # 2 hours
    timings = TraceTimer(0, example_start, example_freq, example_size)
    expected = timedelta(hours=2)

    bad_start = example_start - timedelta(seconds=500)
    change_statues = timings.update_plot_timing(start=bad_start)
    assert not change_statues
    assert timings.plot_start == timings.trace_start

    bad_end = timings.trace_end + timedelta(seconds=500)
    change_statues = timings.update_plot_timing(end=bad_end)
    assert not change_statues
    assert timings.plot_end == timings.trace_end
