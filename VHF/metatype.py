"""Extends ABC types."""
from typing import cast, Any, Callable, TypeVar

# sources: https://stackoverflow.com/a/50381071

__all__ = [
    "abstract_attribute",
]

R = TypeVar('R')


def abstract_attribute(obj: Callable[[Any], R] = None) -> R:
    """Provides decorator for ABCs with attribute.

    This is similar to abstract properties, but for attributes.
    """
    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(R, _obj)
