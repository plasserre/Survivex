"""
CPU performance benchmark (paper Table 3).

Times survivex against lifelines on two small standard datasets:

  - gbsg2  -- Kaplan-Meier, Nelson-Aalen, Cox PH
  - rossi  -- Weibull / Log-Normal / Log-Logistic AFT

Each fit is timed as the median of ``--n-runs`` repetitions after one warm-up
call, and the lifelines/survivex speedup is reported. Timings are
hardware-dependent; the paper figures are from an Apple M2 Pro (16 GB), median
of 21 runs.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from lifelines.datasets import load_gbsg2, load_rossi
from lifelines import (
    KaplanMeierFitter,
    NelsonAalenFitter,
    CoxPHFitter,
    WeibullAFTFitter as LLWeibullAFT,
    LogNormalAFTFitter as LLLogNormalAFT,
    LogLogisticAFTFitter as LLLogLogisticAFT,
)

from survivex.models import (
    KaplanMeierEstimator,
    NelsonAalenEstimator,
    CoxPHModel,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    LogLogisticAFTFitter,
)


def _median_time(fit_once, n_runs, warmup):
    if warmup:
        fit_once()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fit_once()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _load_gbsg2():
    df = load_gbsg2()
    tcol = "time" if "time" in df.columns else next(c for c in df.columns if "time" in c)
    ecol = "cens" if "cens" in df.columns else next(c for c in df.columns if c in ("event", "status"))
    T = df[tcol].to_numpy(float)
    E = df[ecol].to_numpy(float)
    covariates = [c for c in df.columns if c not in (tcol, ecol)]
    X = df[covariates].copy()
    # Encode each categorical covariate as a single integer code (one column).
    for c in covariates:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    frame = X.copy()
    frame["T"], frame["E"] = T, E
    return X.to_numpy(float), T, E, frame


def _load_rossi():
    df = load_rossi()
    T = df["week"].to_numpy(float)
    E = df["arrest"].to_numpy(float)
    covariates = [c for c in df.columns if c not in ("week", "arrest")]
    frame = df.rename(columns={"week": "T", "arrest": "E"})
    return df[covariates].to_numpy(float), T, E, frame


def run(n_runs, warmup):
    gX, gT, gE, gframe = _load_gbsg2()
    rX, rT, rE, rframe = _load_rossi()
    rows = []

    def record(model, dataset, n, p, survivex_fit, lifelines_fit):
        sx = _median_time(survivex_fit, n_runs, warmup)
        ll = _median_time(lifelines_fit, n_runs, warmup)
        rows.append({
            "model": model, "dataset": dataset, "n": n, "p": p,
            "survivex_s": sx, "lifelines_s": ll, "speedup": ll / sx,
        })

    record("Kaplan-Meier", "gbsg2", len(gT), "",
           lambda: KaplanMeierEstimator(device="cpu").fit(gT, gE),
           lambda: KaplanMeierFitter().fit(gT, gE))
    record("Nelson-Aalen", "gbsg2", len(gT), "",
           lambda: NelsonAalenEstimator(device="cpu").fit(gT, gE),
           lambda: NelsonAalenFitter().fit(gT, gE))
    record("Cox PH", "gbsg2", len(gT), gX.shape[1],
           lambda: CoxPHModel(device="cpu", tie_method="efron").fit(gX, gT, gE),
           lambda: CoxPHFitter().fit(gframe, "T", "E"))
    for name, survivex_cls, lifelines_cls in [
        ("Weibull AFT", WeibullAFTFitter, LLWeibullAFT),
        ("Log-Normal AFT", LogNormalAFTFitter, LLLogNormalAFT),
        ("Log-Logistic AFT", LogLogisticAFTFitter, LLLogLogisticAFT),
    ]:
        record(name, "rossi", len(rT), rX.shape[1],
               lambda c=survivex_cls: c(device="cpu").fit(rX, rT, rE),
               lambda c=lifelines_cls: c().fit(rframe, "T", "E"))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="CPU benchmark vs lifelines (paper Table 3).")
    parser.add_argument("--n-runs", type=int, default=21, help="repetitions per fit (median reported)")
    parser.add_argument("--no-warmup", action="store_true", help="skip the warm-up fit")
    parser.add_argument("--output", default="results/cpu_standard_timings.csv")
    args = parser.parse_args()

    df = run(args.n_runs, not args.no_warmup)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    header = f"{'Model':<18}{'Dataset':<8}{'n':>6}{'p':>4}{'survivex(s)':>13}{'vs lifelines':>14}"
    print(header)
    for _, r in df.iterrows():
        print(f"{r['model']:<18}{r['dataset']:<8}{int(r['n']):>6}{str(r['p']):>4}"
              f"{r['survivex_s']:>13.4f}{r['speedup']:>12.1f}x")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
