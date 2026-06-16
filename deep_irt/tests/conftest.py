"""Path setup for deep_irt tests.

Adds the repo root to sys.path so that ``deep_irt.core`` is importable
when pytest is invoked from any working directory.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
