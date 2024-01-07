"""Shared constants in multiprocess classes.


"""

__all__ = [
    "cont",
    "is_cont",
    "HUP",
]

HUP = (b'2',)


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
