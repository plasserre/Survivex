"""
Correctness validation of survivex gamma shared-frailty against R.

Why this exists
---------------
Reviewers flagged a ~4.88e-2 discrepancy between survivex's gamma frailty model
and R `survival::coxph(... + frailty(gamma))`. That gap is not a survivex
defect: the two fit *different estimators* of the same model.

  - survivex maximises the exact marginal likelihood via the EM algorithm.
  - R `survival::coxph` uses a penalized partial likelihood (PPL) approximation.

To prove this rather than assert it, we validate survivex against the canonical
EM implementation -- `frailtyEM::emfrail` -- which maximises the *same* objective
survivex does. When survivex(EM) matches frailtyEM(EM) closely while both differ
from coxph(PPL) by ~1e-2, the reviewer's discrepancy reduces to the well-known
EM-vs-PPL estimator difference, established independently.

What it validates (gamma frailty, two standard clustered datasets):
  rats   -- Surv(time, status) ~ rx,        cluster = litter
  kidney -- Surv(time, status) ~ age + sex, cluster = id

Data flow: tests/validate_frailty_R.R writes survival's `rats`/`kidney` to
tests/data/*.csv (so R and survivex fit byte-identical inputs), fits both
estimators, and writes tests/frailty_R_reference.csv. This script then fits
survivex on the same data and reports, per case, max|Delta| vs EM (the
correctness target) alongside the vs-PPL gap (the estimator difference).

Run:  .venv/bin/python tests/validate_frailty_vs_R.py
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    from survivex.models.frailty import FrailtyModel
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from survivex.models.frailty import FrailtyModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
R_SCRIPT = os.path.join(HERE, "validate_frailty_R.R")
R_REFERENCE = os.path.join(HERE, "frailty_R_reference.csv")

# Tight EM settings so survivex sits at the stationary point of the marginal
# likelihood rather than stopping early on a loose coefficient tolerance.
SX_KWARGS = dict(distribution="gamma", tie_method="breslow",
                 max_iter=1000, tol=1e-9)

# case -> (csv, covariates, time col, event col, cluster col)
CASES = {
    "rats":   ("rats.csv",   ["rx"],          "time", "status", "litter"),
    "kidney": ("kidney.csv", ["age", "sex"],  "time", "status", "id"),
}


def run_r():
    """Invoke the R reference generator. Raises on failure."""
    print("Running R reference generator (validate_frailty_R.R) ...")
    proc = subprocess.run(
        ["Rscript", R_SCRIPT],
        cwd=os.path.join(HERE, ".."),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("R stdout:\n", proc.stdout)
        print("R stderr:\n", proc.stderr)
        raise RuntimeError(f"Rscript failed (exit {proc.returncode})")
    print(proc.stdout)
    return pd.read_csv(R_REFERENCE)


def r_values(ref, case, method, variables):
    """Pull R coefficients (in order) and theta for one case/method."""
    sub = ref[(ref["case"] == case) & (ref["method"] == method)]
    sub = sub.set_index("variable")["value"]
    coef = np.array([sub.loc[v] for v in variables], dtype=np.float64)
    return coef, float(sub.loc["theta"])


def report(case, variables, beta_sx, theta_sx, ref):
    em_coef, em_theta = r_values(ref, case, "em", variables)
    ppl_coef, ppl_theta = r_values(ref, case, "ppl", variables)

    d_em = np.r_[np.abs(beta_sx - em_coef), abs(theta_sx - em_theta)]
    d_ppl = np.r_[np.abs(beta_sx - ppl_coef), abs(theta_sx - ppl_theta)]
    max_em, max_ppl = float(d_em.max()), float(d_ppl.max())

    print(f"\n  case: {case}")
    print(f"  {'param':<8} {'survivex':>16} {'frailtyEM':>16} "
          f"{'coxph(PPL)':>16} {'|Δ EM|':>10} {'|Δ PPL|':>10}")
    print("  " + "-" * 80)
    labels = variables + ["theta"]
    sx = list(beta_sx) + [theta_sx]
    em = list(em_coef) + [em_theta]
    ppl = list(ppl_coef) + [ppl_theta]
    for lbl, s, e, p in zip(labels, sx, em, ppl):
        print(f"  {lbl:<8} {s:>16.10f} {e:>16.10f} {p:>16.10f} "
              f"{abs(s - e):>10.2e} {abs(s - p):>10.2e}")
    print(f"  max|Δ vs EM| = {max_em:.3e}   max|Δ vs PPL| = {max_ppl:.3e}")
    return max_em, max_ppl


def main():
    ref = run_r()

    print("\n" + "=" * 82)
    print("survivex gamma frailty (EM) vs frailtyEM (EM, same estimator) "
          "and coxph (PPL)")
    print("=" * 82)

    results = []
    for case, (csv, cov, tcol, ecol, ccol) in CASES.items():
        df = pd.read_csv(os.path.join(DATA_DIR, csv))
        X = df[cov].values.astype(np.float64)
        T = df[tcol].values.astype(np.float64)
        E = df[ecol].values.astype(np.int64)
        g = df[ccol].values

        m = FrailtyModel(**SX_KWARGS)
        m.fit(X, T, E, g)
        max_em, max_ppl = report(case, cov, m.coefficients_,
                                 float(m.frailty_variance_), ref)
        results.append((case, max_em, max_ppl))

    print("\n" + "=" * 82)
    print("SUMMARY")
    print("=" * 82)
    print(f"{'case':<10} {'max|Δ vs EM|':>14} {'max|Δ vs PPL|':>14}   status")
    print("-" * 82)
    # Correctness target: survivex must track the EM reference (frailtyEM) far
    # more tightly than the PPL approximation it is sometimes compared against.
    # EM-vs-EM agreement is limited by the semiparametric baseline, not machine
    # epsilon, so the bar is 5e-3; the vs-PPL gap is ~1e-2 and is the estimator
    # difference we are documenting, not a failure.
    TOL_EM = 5e-3
    n_fail = 0
    for case, max_em, max_ppl in results:
        ok = (max_em < TOL_EM) and (max_em < max_ppl)
        n_fail += not ok
        print(f"{case:<10} {max_em:>14.2e} {max_ppl:>14.2e}   "
              f"{'OK' if ok else 'FAIL'}")
    print("-" * 82)
    if n_fail == 0:
        print("survivex MATCHES frailtyEM (EM); the R survival gap is EM-vs-PPL")
    else:
        print(f"{n_fail} case(s) failed -- inspect above")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
