from __future__ import annotations
import hashlib, json, os, random, sys

def stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def main() -> int:
    random.seed(0)
    sample = [{"k": i, "v": i * i} for i in range(10)]
    payload = stable_json(sample)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(h)
    # Fixed expected hash for determinism
    return 0

if __name__ == "__main__":
    sys.exit(main())
