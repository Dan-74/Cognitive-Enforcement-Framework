from __future__ import annotations
"""Module."""

    from typing import Protocol

    class SecretBackend(Protocol):
        def get_secret(self, key: str) -> str: ...

