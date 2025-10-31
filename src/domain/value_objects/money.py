from __future__ import annotations
"""Module."""

    from decimal import Decimal
    from dataclasses import dataclass

    @dataclass(frozen=True, slots=True)
    class Money:
        amount: Decimal
        currency: str

