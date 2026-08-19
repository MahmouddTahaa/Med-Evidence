"""Deprecated: cartesian StatPearls sweep removed.

Official freeze is sequential (store → embed → chunk → retrieval).
This entrypoint forwards to lock_winning_combo.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent


def main() -> None:
    print(
        "run_statpearls_sweep.py is deprecated: cartesian ingest×retrieval sweep removed.\n"
        "Forwarding to sequential freeze: scripts/lock_winning_combo.py\n"
        "(protocol: clinical_rag.eval.protocol)",
        flush=True,
    )
    sys.argv = [str(_SCRIPTS / "lock_winning_combo.py"), *sys.argv[1:]]
    runpy.run_path(str(_SCRIPTS / "lock_winning_combo.py"), run_name="__main__")


if __name__ == "__main__":
    main()
