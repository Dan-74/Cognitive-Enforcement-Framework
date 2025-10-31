from __future__ import annotations
"""Module."""

    from typing import Callable

    def with_noop_retry(fn: Callable[..., object]) -> Callable[..., object]:
        return fn

