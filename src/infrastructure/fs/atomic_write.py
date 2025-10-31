from __future__ import annotations
"""Module."""

    from pathlib import Path
    from typing import Iterable

    def atomic_write(path: Path, lines: Iterable[str]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(lines), encoding="utf-8")
        tmp.replace(path)

