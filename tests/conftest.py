"""Shared pytest config for the receptionist tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable from any test, so individual tests
# don't have to repeat the sys.path dance.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
