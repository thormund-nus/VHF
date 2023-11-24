import pytest
from datetime import datetime, timezone, timedelta

# test if Python's datetime lib can parse the following specific teststream
# generated datetime


def test_fromISOformat():
    """Check if the running Python has an expected .fromisoformat."""
    dt_str = "2023-11-23 17:03:08 +0800"
    actual = datetime.fromisoformat(dt_str)
    expected = datetime(2023, 11, 23, 17, 3, 8,
                        tzinfo=timezone(timedelta(hours=8)))
    assert expected == actual
