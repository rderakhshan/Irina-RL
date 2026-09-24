"""Root runner for the odysseus CLI (src/back).

Loads the key from `.env` and hands off to the package's own front door,
so `python main.py ...` works the way `python -m odysseus` did when the
package lived at the root.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def load_dotenv(path):
    """Set env vars from a KEY=VALUE file, never overriding a real env var."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    sys.path.insert(0, str(SRC))
    from back.cli import main
    raise SystemExit(main())