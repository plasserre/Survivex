"""
Gold-standard correctness validation of survivex Cox PH against R's
`survival::coxph` (same algorithm: Newton-Raphson on the partial likelihood).

Why this exists
---------------
The TCGA pan-cancer benchmark (paper Table 4) shows a c-index gap between
survivex (Breslow ties) and lifelines (Efron ties) that widens with p
(Δc ≈ 0.009 at p=100 → 0.062 at p=5000). A reviewer could read that gap as
a survivex defect. It is not: it is the well-known Breslow-vs-Efron tie
approximation difference, amplified when p approaches the event count.

To prove this rather than assert it, we validate survivex against R's
`survival` package -- the canonical reference implementation -- using the
*same* tie method on each side. When survivex(Breslow) matches
coxph(ties="breslow") to machine precision, and survivex(Efron) matches
coxph(ties="efron") to machine precision, correctness is established
independently of lifelines. The Table 4 c-index gap then reduces to a
tie-method footnote, not a vulnerability.

What it validates (each vs R `survival`, same algorithm both sides):
  1. CoxPH, Breslow ties        -- Rossi, Lung
  2. CoxPH, Efron ties          -- Rossi, Lung
  3. Stratified CoxPH, Breslow  -- Rossi (strata=wexp), Lung (strata=sex)
  4. CoxPH + Ridge (L2)         -- Rossi, several penalties, vs penalized::penalized

Data flow: this script exports the datasets to tests/data/*.csv so R and
survivex fit byte-identical inputs, invokes tests/validate_coxph_R.R to
produce tests/coxph_R_reference.csv, then fits survivex on the same data and
reports max|Δβ| per case.

Run:  .venv/bin/python tests/validate_coxph_vs_R.py
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    from survivex.models.cox_ph import CoxPHModel, StratifiedCoxPHModel
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from survivex.models.cox_ph import CoxPHModel, StratifiedCoxPHModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
R_SCRIPT = os.path.join(HERE, "validate_coxph_R.R")
R_REFERENCE = os.path.join(HERE, "coxph_R_reference.csv")

# Penalty grid for the Ridge comparison (survivex/lifelines convention).
RIDGE_PENALTIES = [0.001, 0.01, 0.1, 1.0]

# Tight solver settings so both implementations sit at the true stationary
# point of a strictly-concave partial likelihood -- a fair machine-epsilon
# test rather than a tolerance-vs-tolerance comparison.
SX_KWARGS = dict(max_iter=200, tol=1e-12)


# --------------------------------------------------------------------------
# Dataset definitions
# --------------------------------------------------------------------------
def build_datasets():
    """Return tidy frames + metadata for Rossi and Lung."""
    from lifelines.datasets import load_lung, load_rossi

    rossi = load_rossi()
    rossi_df = pd.DataFrame(
        {
            "fin": rossi["fin"].astype(float),
            "age": rossi["age"].astype(float),
            "prio": rossi["prio"].astype(float),
            "wexp": rossi["wexp"].astype(int),  # stratum
            "time": rossi["week"].astype(float),
            "event": rossi["arrest"].astype(int),
        }
    )

    lung = load_lung()[["age", "sex", "ph.karno", "time", "status"]].dropna()
    lung_df = pd.DataFrame(
        {
            "age": lung["age"].astype(float),
            "sex": lung["sex"].astype(int),  # stratum
            "ph_karno": lung["ph.karno"].astype(float),
            "time": lung["time"].astype(float),
            # lifelines load_lung already recodes status to 0=censored, 1=dead
            "event": lung["status"].astype(int),
        }
    )

    return {
        "rossi": {
            "df": rossi_df,
            "covariates": ["fin", "age", "prio"],
            "stratum": "wexp",
            "strat_covariates": ["fin", "age", "prio"],
        },
        "lung": {
            "df": lung_df,
            "covariates": ["age", "sex", "ph_karno"],
            "stratum": "sex",
            "strat_covariates": ["age", "ph_karno"],
        },
    }


def export_data(datasets):
    """Write CSVs that both R and survivex consume."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, spec in datasets.items():
        spec["df"].to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)

    # Standardized Rossi for the Ridge comparison. Pre-standardizing with
    # ddof=1 (matching torch.std) means survivex's internal standardization
    # is the identity, so it penalizes in the *same* coordinate system R's
    # penalized(..., standardize=FALSE) uses. This removes any
    # standardization-convention mismatch from the Ridge test.
    rossi = datasets["rossi"]
    cov = rossi["covariates"]
    Xr = rossi["df"][cov].values.astype(np.float64)
    Xstd = (Xr - Xr.mean(axis=0)) / Xr.std(axis=0, ddof=1)
    std_df = pd.DataFrame(Xstd, columns=cov)
    std_df["time"] = rossi["df"]["time"].values
    std_df["event"] = rossi["df"]["event"].values
    std_df.to_csv(os.path.join(DATA_DIR, "rossi_std.csv"), index=False)

    # Ridge spec: penalized's L2 objective is loglik - (lambda2/2)*||b||^2
    # (gradient lambda2*b). survivex/lifelines use loglik - (n*penalty/2)*||b||^2
    # (gradient n*penalty*b). Equating the gradients => lambda2 = n * penalty.
    # (Verified empirically: penalized at lambda2=n*penalty reproduces survivex
    # penalty to 10 sig figs; lambda2=n*penalty/2 under-penalizes.)
    n = len(std_df)
    spec_df = pd.DataFrame(
        {
            "label": [f"rossi_ridge_{p}" for p in RIDGE_PENALTIES],
            "penalty": RIDGE_PENALTIES,
            "lambda2": [n * p for p in RIDGE_PENALTIES],
        }
    )
    spec_df.to_csv(os.path.join(DATA_DIR, "ridge_spec.csv"), index=False)

    return Xstd, std_df


def run_r():
    """Invoke the R reference generator. Raises on failure."""
    print("Running R reference generator (validate_coxph_R.R) ...")
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


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------
def r_coef(ref, case, variables):
    """Pull R coefficients for a case in the given variable order."""
    sub = ref[ref["case"] == case].set_index("variable")["coef"]
    missing = [v for v in variables if v not in sub.index]
    if missing:
        raise KeyError(f"R reference missing {missing} for case '{case}' "
                       f"(have {list(sub.index)})")
    return sub.loc[variables].values.astype(np.float64)


def report(case, variables, beta_sx, beta_r):
    diff = np.abs(beta_sx - beta_r)
    max_diff = float(diff.max())
    denom = max(float(np.max(np.abs(beta_r))), 1e-300)
    rel = max_diff / denom
    print(f"\n  case: {case}")
    print(f"  {'variable':<10} {'survivex':>18} {'R survival':>18} {'|Δ|':>12}")
    print("  " + "-" * 60)
    for v, bs, br, d in zip(variables, beta_sx, beta_r, diff):
        print(f"  {v:<10} {bs:>18.12f} {br:>18.12f} {d:>12.2e}")
    print(f"  max|Δβ| = {max_diff:.3e}   rel = {rel:.3e}")
    return max_diff, rel


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------
def case_coxph_unpenalized(datasets, ref, results):
    print("\n" + "=" * 70)
    print("1-2. CoxPH (Breslow & Efron) vs R survival::coxph")
    print("=" * 70)
    for name in ("rossi", "lung"):
        spec = datasets[name]
        df = spec["df"]
        cov = spec["covariates"]
        X = df[cov].values.astype(np.float64)
        T = df["time"].values.astype(np.float64)
        E = df["event"].values.astype(np.int64)
        for tie in ("breslow", "efron"):
            m = CoxPHModel(tie_method=tie, **SX_KWARGS)
            m.fit(X, T, E)
            case = f"{name}_{tie}"
            md, rel = report(case, cov, m.coefficients_, r_coef(ref, case, cov))
            results.append((case, md, rel))


def case_stratified(datasets, ref, results):
    print("\n" + "=" * 70)
    print("3. Stratified CoxPH (Breslow) vs R coxph(... + strata(g))")
    print("=" * 70)
    for name in ("rossi", "lung"):
        spec = datasets[name]
        df = spec["df"]
        cov = spec["strat_covariates"]
        X = df[cov].values.astype(np.float64)
        T = df["time"].values.astype(np.float64)
        E = df["event"].values.astype(np.int64)
        g = df[spec["stratum"]].values
        m = StratifiedCoxPHModel(tie_method="breslow", **SX_KWARGS)
        m.fit(X, T, E, strata=g)
        case = f"{name}_strat_{spec['stratum']}_breslow"
        md, rel = report(case, cov, m.coefficients_, r_coef(ref, case, cov))
        results.append((case, md, rel))


def case_ridge(Xstd, std_df, ref, results):
    print("\n" + "=" * 70)
    print("4. CoxPH + Ridge (Breslow) vs R penalized::penalized")
    print("=" * 70)
    print("    (data pre-standardized; both penalize identical coordinates)")
    cov = ["fin", "age", "prio"]
    T = std_df["time"].values.astype(np.float64)
    E = std_df["event"].values.astype(np.int64)
    for p in RIDGE_PENALTIES:
        m = CoxPHModel(tie_method="breslow", penalty=p, **SX_KWARGS)
        m.fit(Xstd, T, E)
        case = f"rossi_ridge_{p}"
        md, rel = report(case, cov, m.coefficients_, r_coef(ref, case, cov))
        results.append((case, md, rel))


# --------------------------------------------------------------------------
def main():
    datasets = build_datasets()
    Xstd, std_df = export_data(datasets)
    ref = run_r()

    results = []
    case_coxph_unpenalized(datasets, ref, results)
    case_stratified(datasets, ref, results)
    case_ridge(Xstd, std_df, ref, results)

    print("\n" + "=" * 70)
    print("SUMMARY  (survivex vs R survival, same algorithm both sides)")
    print("=" * 70)
    print(f"{'case':<32} {'max|Δβ|':>12} {'rel':>12}   status")
    print("-" * 70)
    # Unpenalized/stratified: expect machine epsilon. Ridge: a hair looser
    # because the two solvers stop on different convergence criteria, but
    # still far tighter than any tie-method effect.
    n_fail = 0
    for case, md, rel in results:
        tol = 1e-6 if "ridge" in case else 1e-8
        ok = md < tol
        n_fail += not ok
        print(f"{case:<32} {md:>12.2e} {rel:>12.2e}   {'OK' if ok else 'FAIL'}")
    print("-" * 70)
    if n_fail == 0:
        print("ALL CASES MATCH R survival TO TARGET PRECISION")
    else:
        print(f"{n_fail} case(s) exceeded tolerance -- inspect above")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
