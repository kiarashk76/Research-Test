"""Interactive Programmatic Policy Lab.

Importing this package makes its own top-level modules (``core``,
``storage``, ``execution``, ``ui``, ``app``, ``cli``) and the parent repo's
top-level modules (``environments``, ``agents``, ``llm``, ``config``, ...)
both importable as bare names -- consistent with how every module in this
package already imports its neighbors and the parent repo. This runs once,
before any submodule of this package executes, whether the package is
launched via ``python -m programmatic_interactive_lab`` or imported by a
test collector.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _LAB_ROOT.parent

for _path in (str(_REPO_ROOT), str(_LAB_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
