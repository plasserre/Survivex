"""
Pytest configuration shared by the survivex test suite.

Two things are handled centrally here so individual test files stay untouched:

1. ``collect_ignore`` drops modules that cannot be collected in a clean
   checkout (e.g. a debug script that reads a data file which is not shipped).

2. ``pytest_collection_modifyitems`` auto-skips the tests that shell out to R
   (``Rscript`` / ``R``) when no R interpreter is on ``PATH``. This lets the
   fast CI tier run on a plain Python image (the R tests skip cleanly) while
   the nightly tier, which installs R + the gold-standard packages, runs them
   for real. See REPRODUCIBILITY.md for the two-tier layout.
"""

import shutil

import pytest

# Modules that invoke R at runtime via subprocess. They need an R interpreter
# plus the relevant CRAN packages (survival / cmprsk). Without R they are
# skipped rather than failed.
_R_RUNTIME_MODULES = {
    "test_competing_risk_r",
    "test_fine_gray",
}

# Modules that cannot be collected without artifacts absent from a fresh clone.
# test_pwp_tt.py is a manual debug script that reads a non-shipped
# ``cgd_data.csv`` at import time; it is not a real regression test.
collect_ignore = ["test_pwp_tt.py"]


def _have_r() -> bool:
    return shutil.which("Rscript") is not None or shutil.which("R") is not None


def pytest_collection_modifyitems(config, items):
    if _have_r():
        return
    skip_r = pytest.mark.skip(reason="R (Rscript) not installed; skipping R-backed test")
    for item in items:
        if item.module.__name__.split(".")[-1] in _R_RUNTIME_MODULES:
            item.add_marker(skip_r)
