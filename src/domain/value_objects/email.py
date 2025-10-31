from __future__ import annotations
"""Module."""

    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class Email:
        value: str

