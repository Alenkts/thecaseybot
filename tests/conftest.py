"""
Makes sure the project's root-level modules (llm_providers, llm_classifier,
llm_trim_evaluator, signal_classifier) import correctly no matter what
directory pytest is invoked from.

pytest's default import mode already does this on its own, because
tests/ has an __init__.py: it walks upward from this file until it
finds a directory *without* __init__.py — the repo root — and prepends
that to sys.path. This is a belt-and-suspenders guard that doesn't rely on
that mechanism, so `pytest tests/test_llm_providers.py` from some other
cwd, a different --import-mode, or a future pytest default change can't
silently break these imports.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
