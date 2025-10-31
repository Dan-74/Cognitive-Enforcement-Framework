from __future__ import annotations
import sys
from pathlib import Path

MARKERS = ("TODO", "FIXME", "NOT_IMPLEMENTED", "placeholder", "pass  # placeholder")

def main() -> int:
    root = Path(".")
    offenders: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.parts if part != ".") and ".github" not in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in MARKERS:
            if m in text:
                offenders.append(f"{p}: contains marker '{m}'")
                break
    if offenders:
        print("Placeholders/markers found:")
        print("\n".join(offenders))
        return 1
    print("No placeholders found.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
