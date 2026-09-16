"""Make ``app.*`` and ``ml.*`` importable when tests run from the repository root."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
for entry in (BACKEND, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
