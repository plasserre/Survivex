"""
One-command validation entry point for Table 2 of the survivex paper.

Runs every validation group that the paper's Table 2 reports and prints a
single pass/fail summary. This is the convenience aggregator; each group can
also be run on its own (see REPRODUCIBILITY.md).

Two kinds of checks:

  * R gold-standard scripts -- authoritative numerical checks that exit
    non-zero on disagreement (Cox PH vs ``survival``/``penalized`` to machine
    epsilon; gamma frailty vs ``frailtyEM`` EM; recurrent events vs
    ``survival``). These need R + the CRAN packages; they are reported as
    SKIP when no R interpreter is on PATH.

  * lifelines smoke/integration tests -- the per-model ``tests/test_*.py``
    files (KM / NA / parametric / competing risks / trees / GBM). They fit
    survivex alongside lifelines and pass if nothing raises. (The detailed
    ~1e-8 numerical agreement is printed by each script; the machine-checkable
    gold-standard assertions live in the R scripts above.)

Exit code is non-zero if any group that actually ran failed.

Run:  python tests/run_validation.py
      python tests/run_validation.py --no-r     # skip the R groups explicitly
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable

# group label -> (kind, command). kind is "r" (needs Rscript) or "py".
GROUPS = [
    ("Cox PH vs R survival + penalized",
     "r", [PY, os.path.join(HERE, "validate_coxph_vs_R.py")]),
    ("Gamma frailty vs frailtyEM (EM)",
     "r", [PY, os.path.join(HERE, "validate_frailty_vs_R.py")]),
    # test_recurrent.py is self-contained: it generates seeded AG / PWP-TT /
    # PWP-GT / frailty data and compares survivex against R survival::coxph
    # reference values baked into the script. (validate_recurrent_R.R is the
    # offline generator of those references; it reads a data CSV not shipped in
    # a fresh clone, so it is not part of the automated run.)
    ("Recurrent events (AG / PWP-TT / PWP-GT) vs R survival",
     "py", [PY, os.path.join(HERE, "test_recurrent.py")]),
    ("KM / NA / parametric / competing / trees / GBM vs lifelines",
     "py", [PY, "-m", "pytest",
            os.path.join(HERE, "test_model_kaplanmeier.py"),
            os.path.join(HERE, "test_nelsonaalen.py"),
            os.path.join(HERE, "test_parametric.py"),
            os.path.join(HERE, "test_competing_risk.py"),
            os.path.join(HERE, "test_survival_tree.py"),
            os.path.join(HERE, "test_random_survtree.py"),
            os.path.join(HERE, "test_gradient_boost.py"),
            "-q", "-p", "no:cacheprovider"]),
]


def have_r() -> bool:
    return shutil.which("Rscript") is not None or shutil.which("R") is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-r", action="store_true",
                    help="skip the R-backed gold-standard groups")
    args = ap.parse_args()

    r_ok = have_r() and not args.no_r
    if not r_ok:
        why = "--no-r given" if args.no_r else "Rscript not found on PATH"
        print(f"[note] R groups will be SKIPPED ({why}); "
              f"install R + survival/penalized/frailtyEM to run them.\n")

    results = []
    for label, kind, cmd in GROUPS:
        if kind == "r" and not r_ok:
            results.append((label, "SKIP"))
            continue
        print("=" * 80)
        print(f"RUN: {label}")
        print("=" * 80)
        proc = subprocess.run(cmd, cwd=REPO)
        results.append((label, "PASS" if proc.returncode == 0 else "FAIL"))

    print("\n" + "=" * 80)
    print("TABLE 2 VALIDATION SUMMARY")
    print("=" * 80)
    width = max(len(lbl) for lbl, _ in results)
    for label, status in results:
        print(f"  {label:<{width}}  {status}")
    print("=" * 80)

    n_fail = sum(1 for _, s in results if s == "FAIL")
    n_skip = sum(1 for _, s in results if s == "SKIP")
    if n_fail:
        print(f"{n_fail} group(s) FAILED.")
        return 1
    print(f"All run groups passed ({n_skip} skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
