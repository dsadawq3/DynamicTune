"""Make the standalone project testable from its own root or workspace root."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
