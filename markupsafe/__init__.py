"""Lightweight stub for the ``markupsafe`` package used by Jinja2.

Only the :class:`Markup` class is required for the test suite.  The real library
provides many utilities for HTML escaping; for our purposes a simple subclass of
``str`` that returns the original string is sufficient.
"""

from __future__ import annotations


def escape(s: object) -> str:
    """Simple escape function placeholder."""
    return str(s)


def soft_str(s: object) -> str:
    """Return a string representation of *s*.

    This mirrors the behaviour of ``markupsafe.soft_str`` which safely
    converts various objects to ``str`` without triggering HTML escaping.
    """
    return str(s)


def soft_unicode(s: object) -> str:  # pragma: no cover – compatibility shim
    return soft_str(s)


class Markup(str):
    """Minimal stand‑in for :class:`markupsafe.Markup`.

    Jinja2 treats ``Markup`` objects as safe HTML strings.  Subclassing ``str``
    preserves all string behaviour while allowing ``isinstance(value, Markup)``
    checks to succeed.
    """

    def __html__(self) -> str:  # pragma: no cover – compatibility shim
        return str(self)
