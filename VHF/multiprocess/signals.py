"""Shared constants in multiprocess classes.


"""

__all__ = [
    "cont",
    "is_cont",
    "HUP",
    "REQUEUE",
    "Signals",
    "ChildSignals",
]

HUP = (b'2',)
REQUEUE = b'REQUEUE'


def cont(*args) -> tuple:
    """Signals continue as expected.

    Inputs
    ------
    *args: Pickleable
        Message associated with continue is placed in args.
    """
    return (b'0', *args)


def is_cont(val: tuple):
    """Test if signal is continue.

    Construction is dependent on cont constructor.

    Inputs
    ------
    val: SIGNAL.cont
        True if val follows cont descriptor.
    """
    return type(val) == tuple and len(val) >= 1 and val[0] == b'0'


class Signals:
    """Dataclass for purpose of matching."""
    action_cont = cont()[0]
    action_hup = HUP[0]


class ChildSignals:
    """Dataclass for child processes. Use only send_bytes."""
    type = bytes  # Used for type hinting

    action_cont = cont()[0]
    action_request_requeue = REQUEUE
    action_hup = HUP[0]
    action_generic_error = b'1'
    too_many_attempts = b'3'
