from functools import cache
from inspect import getmembers, isroutine
import logging
from pathlib import Path
import sys
module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
from VHF.multiprocess.signals import *  # noqa


@cache
def ChildSignals_attributes():
    attributes = getmembers(ChildSignals, lambda a: not (isroutine(a)))
    ms = [a for a in attributes if not (
        a[0].startswith('__') and a[0].endswith('__'))]
    logging.debug("ChildSignals's attributes = %s", ms)
    return ms


def test_ChildSignals_type():
    """ChildSignals (key, value) have values of type ChildSignals.type."""
    ms = ChildSignals_attributes()
    inner = [isinstance(ChildSignals().__getattribute__(x[0]),
                        ChildSignals().type) for x in ms if x[0] != 'type']
    result = all(inner)
    logging.info("inner = %s", inner)
    logging.info("isinstance(...) = %s", result)
    assert result


def test_ChildSignals_unique_values():
    """Require that the key: value pairs defined in ChildSignals be unique."""
    ms = ChildSignals_attributes()

    def num_element(idx):
        return len(set(map(lambda a: a[idx], ms)))

    assert num_element(0) == num_element(1)
