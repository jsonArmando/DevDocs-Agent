import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "libs" / "core", ROOT / "services" / "agent", ROOT / "eval", ROOT / "tests"):
    sys.path.insert(0, str(p))
