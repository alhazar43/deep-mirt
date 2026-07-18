"""Path setup for kt_mirt tests.

Adds src/ to sys.path so that ``kt_mirt`` is importable even without
the editable install, from any working directory.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
