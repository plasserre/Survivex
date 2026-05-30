"""
Synthetic GPU-scaling benchmark (Table 3 CPU rows + Table 4 synthetic scaling).

This is the CLI/seeded consolidation of the ``benchmark_gpu_scaling.ipynb``
notebook that produced the submitted SoftwareX numbers. It sweeps each model
family over the same (size, p) grid the notebook used, fits survivex on CPU and
GPU (and, where meaningful, lifelines on CPU as a baseline), and records median
wall-clock time, peak CPU memory (tracemalloc), peak GPU memory
(``torch.cuda.max_memory_allocated``), C-index, and CPU-vs-GPU coefficient
agreement.

Model families and their default sweeps (verbatim from the notebook):

    coxph     Cox PH (Breslow)         (10000,20) (50000,20) (100000,20)
                                       (200000,20) (50000,50) (50000,100)
                                       (50000,200)
    pwp-tt    PWP total-time           (5000,10) (10000,10) (10000,50)
    pwp-gt    PWP gap-time             (5000,10) (10000,10) (10000,50)
    frailty   Gamma shared frailty     (500,10,10) (1000,10,10) (2000,10,10)
                                       (1000,10,50)   [n_clusters, obs, p]
    weibull   Weibull AFT              (10000,10) (50000,10) (50000,50)
    gbm       Gradient boosting        (5000,10) (10000,10) (10000,50)
    rsf       Random survival forest   (5000,10) (10000,10) (10000,50)
    finegray  Fine-Gray (CPU only)     (500,5) (1000,5) (1000,10)

All randomness is seeded (default seed 42, the paper value); a given
(family, config, seed) yields byte-identical data via ``generators.py``.

Usage:
    python run_benchmark.py                       # all families, GPU if present
    python run_benchmark.py --families coxph weibull
    python run_benchmark.py --families coxph --backends survivex-cpu lifelines-cpu
    python run_benchmark.py --families frailty --configs 500:10:10 1000:10:10
    python run_benchmark.py --seeds 42 --n-runs 1     # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
import tracemalloc
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from generators import (
    generate_large_synthetic,
    generate_recurrent_data,
    generate_competing_risks_data,
    generate_clustered_data,
)


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkRow:
    family: str
    config: str
    n_rows: int
    p: int
    seed: int
    backend: str
    n_runs: int
    fit_seconds_median: float
    peak_cpu_mb: float
    peak_gpu_mb: float
    c_index: float
    coef_l2_norm: float
    coef_diff_vs_cpu: float  # max|Δβ| vs survivex-cpu (same family/config/seed)
    error: str
    notes: str


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# GPU helpers (mirror the notebook's sync/clear utilities)
# --------------------------------------------------------------------------- #
def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    try:
        import torch
    except ImportError:
        return False
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return False


def _sync(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def _clear_cache(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def _peak_gpu_mb(device: str) -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0  # MPS exposes no peak-allocation API


def _reset_gpu_peak(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# Timing core
# --------------------------------------------------------------------------- #
def _time_fits(
    fit_once: Callable[[], object], device: str, n_runs: int, warmup: bool,
) -> tuple[float, float, float, object, str]:
    """Run ``fit_once`` n_runs times; return (median_s, peak_cpu_mb,
    peak_gpu_mb, last_model, error)."""
    err = ""
    if warmup and device in ("cuda", "mps"):
        try:
            fit_once()
            _sync(device)
        except Exception:
            pass
        _clear_cache(device)

    gc.collect()
    _clear_cache(device)
    _reset_gpu_peak(device)

    tracemalloc.start()
    times: list[float] = []
    model = None
    try:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model = fit_once()
            _sync(device)
            times.append(time.perf_counter() - t0)
            _clear_cache(device)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    _, peak_cpu_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    median_s = float(np.median(times)) if times else float("nan")
    return median_s, peak_cpu_b / (1024**2), _peak_gpu_mb(device), model, err


# --------------------------------------------------------------------------- #
# Model-family registry
# --------------------------------------------------------------------------- #
@dataclass
class Family:
    name: str
    arity: int                                   # ints per config tuple
    default_configs: list[tuple]
    default_n_runs: int
    survivex_gpu: bool                           # whether GPU backends apply
    build: Callable[[tuple, int], dict]
    fit_survivex: Callable[[dict, str], object]  # (data, device) -> model
    n_p: Callable[[dict], tuple]                 # (n_rows, p) for reporting
    get_coef: Callable[[object], Optional[np.ndarray]] = lambda m: getattr(m, "coefficients_", None)
    get_cindex: Callable[[object], Optional[float]] = lambda m: getattr(m, "concordance_index_", None)
    fit_lifelines: Optional[Callable[[dict], object]] = None
    coef_lifelines: Optional[Callable[[object], Optional[np.ndarray]]] = None
    cindex_lifelines: Optional[Callable[[object], Optional[float]]] = None

    def config_label(self, cfg: tuple) -> str:
        return ":".join(str(c) for c in cfg)


def _coef(m) -> Optional[np.ndarray]:
    c = getattr(m, "coefficients_", None)
    return None if c is None else np.asarray(c)


def _cindex(m) -> float:
    v = getattr(m, "concordance_index_", None)
    try:
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


# ---- Cox PH ---------------------------------------------------------------- #
def _build_coxph(cfg, seed):
    return generate_large_synthetic(cfg[0], cfg[1], seed)


def _fit_coxph(data, device):
    from survivex.models import CoxPHModel
    m = CoxPHModel(device=device, tie_method="breslow")
    m.fit(data["X"], data["T"], data["E"])
    return m


def _ll_coxph(data):
    import pandas as pd
    from lifelines import CoxPHFitter
    df = pd.DataFrame(data["X"], columns=data["features"])
    df["T"], df["E"] = data["T"], data["E"]
    m = CoxPHFitter(penalizer=0.0)
    m.fit(df, duration_col="T", event_col="E")
    return m


# ---- PWP total-time / gap-time --------------------------------------------- #
def _build_recurrent(cfg, seed):
    return generate_recurrent_data(cfg[0], cfg[1], seed)


def _fit_pwptt(data, device):
    from survivex.models import PWPTTModel
    m = PWPTTModel(device=device, tie_method="breslow")
    m.fit(data["X"], data["start"], data["stop"], data["E"], data["id"], data["strata"])
    return m


def _fit_pwpgt(data, device):
    from survivex.models import PWPGTModel
    m = PWPGTModel(device=device, tie_method="breslow")
    m.fit(data["X"], data["gap_time"], data["E"], data["id"], data["strata"])
    return m


# ---- Gamma frailty --------------------------------------------------------- #
def _build_frailty(cfg, seed):
    return generate_clustered_data(cfg[0], cfg[1], cfg[2], seed)


def _fit_frailty(data, device):
    from survivex.models import FrailtyModel
    m = FrailtyModel(distribution="gamma", device=device, max_iter=100)
    m.fit(data["X"], data["T"], data["E"], data["cluster_id"])
    return m


# ---- Weibull AFT ----------------------------------------------------------- #
def _fit_weibull(data, device):
    from survivex.models import WeibullAFTFitter
    m = WeibullAFTFitter(device=device)
    m.fit(data["X"], data["T"], data["E"])
    return m


# ---- Gradient boosting / RSF ----------------------------------------------- #
def _fit_gbm(data, device):
    from survivex.models import GradientBoostingSurvivalAnalysis
    m = GradientBoostingSurvivalAnalysis(n_estimators=50, max_depth=3, device=device)
    m.fit(data["X"], data["T"], data["E"])
    return m


def _fit_rsf(data, device):
    from survivex.models import RandomSurvivalForest
    m = RandomSurvivalForest(n_estimators=50, max_depth=5, device=device, n_jobs=1)
    m.fit(data["X"], data["T"], data["E"])
    return m


# ---- Fine-Gray (CPU only) -------------------------------------------------- #
def _build_finegray(cfg, seed):
    return generate_competing_risks_data(cfg[0], cfg[1], seed)


def _fit_finegray(data, device):
    from survivex.models.competing_risk import FineGrayModel
    m = FineGrayModel(device=device, max_iter=50)
    m.fit(data["T"], data["event_type"], data["X"], event_of_interest=1)
    return m


FAMILIES: dict[str, Family] = {
    "coxph": Family(
        name="coxph", arity=2, default_n_runs=3, survivex_gpu=True,
        default_configs=[(10000, 20), (50000, 20), (100000, 20), (200000, 20),
                         (50000, 50), (50000, 100), (50000, 200)],
        build=_build_coxph, fit_survivex=_fit_coxph,
        n_p=lambda d: (len(d["T"]), d["X"].shape[1]),
        fit_lifelines=_ll_coxph,
        coef_lifelines=lambda m: np.asarray(m.params_.values),
        cindex_lifelines=lambda m: float(m.concordance_index_),
    ),
    "pwp-tt": Family(
        name="pwp-tt", arity=2, default_n_runs=3, survivex_gpu=True,
        default_configs=[(5000, 10), (10000, 10), (10000, 50)],
        build=_build_recurrent, fit_survivex=_fit_pwptt,
        n_p=lambda d: (len(d["E"]), d["X"].shape[1]),
    ),
    "pwp-gt": Family(
        name="pwp-gt", arity=2, default_n_runs=3, survivex_gpu=True,
        default_configs=[(5000, 10), (10000, 10), (10000, 50)],
        build=_build_recurrent, fit_survivex=_fit_pwpgt,
        n_p=lambda d: (len(d["E"]), d["X"].shape[1]),
    ),
    "frailty": Family(
        name="frailty", arity=3, default_n_runs=3, survivex_gpu=True,
        default_configs=[(500, 10, 10), (1000, 10, 10), (2000, 10, 10),
                         (1000, 10, 50)],
        build=_build_frailty, fit_survivex=_fit_frailty,
        n_p=lambda d: (len(d["T"]), d["X"].shape[1]),
    ),
    "weibull": Family(
        name="weibull", arity=2, default_n_runs=3, survivex_gpu=True,
        default_configs=[(10000, 10), (50000, 10), (50000, 50)],
        build=_build_coxph, fit_survivex=_fit_weibull,
        n_p=lambda d: (len(d["T"]), d["X"].shape[1]),
        get_cindex=lambda m: float("nan"),  # AFT exposes no concordance_index_
    ),
    "gbm": Family(
        name="gbm", arity=2, default_n_runs=3, survivex_gpu=True,
        default_configs=[(5000, 10), (10000, 10), (10000, 50)],
        build=_build_coxph, fit_survivex=_fit_gbm,
        n_p=lambda d: (len(d["T"]), d["X"].shape[1]),
        get_coef=lambda m: None,  # ensemble, no single coef vector
        get_cindex=lambda m: float("nan"),
    ),
    "rsf": Family(
        name="rsf", arity=2, default_n_runs=3, survivex_gpu=True,
        default_configs=[(5000, 10), (10000, 10), (10000, 50)],
        build=_build_coxph, fit_survivex=_fit_rsf,
        n_p=lambda d: (len(d["T"]), d["X"].shape[1]),
        get_coef=lambda m: None,
        get_cindex=lambda m: float("nan"),
    ),
    "finegray": Family(
        name="finegray", arity=2, default_n_runs=2, survivex_gpu=False,
        default_configs=[(500, 5), (1000, 5), (1000, 10)],
        build=_build_finegray, fit_survivex=_fit_finegray,
        n_p=lambda d: (len(d["T"]), d["X"].shape[1]),
        get_coef=_coef, get_cindex=lambda m: float("nan"),
    ),
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _parse_configs(fam: Family, raw: Optional[list[str]]) -> list[tuple]:
    if not raw:
        return fam.default_configs
    out = []
    for item in raw:
        parts = item.split(":")
        if len(parts) != fam.arity:
            raise SystemExit(
                f"--configs entry '{item}' has {len(parts)} fields; family "
                f"'{fam.name}' expects {fam.arity} (e.g. "
                f"'{fam.config_label(fam.default_configs[0])}')."
            )
        out.append(tuple(int(x) for x in parts))
    return out


def _empty_row(fam, cfg, n_rows, p, seed, backend, n_runs, err, notes=""):
    return BenchmarkRow(
        family=fam.name, config=fam.config_label(cfg), n_rows=n_rows, p=p,
        seed=seed, backend=backend, n_runs=n_runs,
        fit_seconds_median=float("nan"), peak_cpu_mb=0.0, peak_gpu_mb=0.0,
        c_index=float("nan"), coef_l2_norm=float("nan"),
        coef_diff_vs_cpu=float("nan"), error=err, notes=notes,
    )


def _summary_table(rows: list[BenchmarkRow]) -> str:
    from statistics import median
    grouped: dict[tuple, list[BenchmarkRow]] = {}
    for r in rows:
        grouped.setdefault((r.family, r.config, r.backend), []).append(r)
    header = (f"{'family':<9} {'config':<14} {'backend':<15} {'runs':>4} "
              f"{'median_s':>10} {'peak_cpu_mb':>11} {'peak_gpu_mb':>11} "
              f"{'coef_diff':>10} errs")
    lines = [header, "-" * len(header)]
    for (fam, cfg, backend), rs in sorted(grouped.items()):
        ok = [r for r in rs if not r.error]
        n = len(ok)
        med = median(r.fit_seconds_median for r in ok) if ok else float("nan")
        pcpu = median(r.peak_cpu_mb for r in ok) if ok else float("nan")
        pgpu = median(r.peak_gpu_mb for r in ok) if ok else float("nan")
        diffs = [r.coef_diff_vs_cpu for r in ok if not np.isnan(r.coef_diff_vs_cpu)]
        cd = max(diffs) if diffs else float("nan")
        errs = sum(1 for r in rs if r.error)
        lines.append(
            f"{fam:<9} {cfg:<14} {backend:<15} {n:>4} {med:>10.4f} "
            f"{pcpu:>11.1f} {pgpu:>11.1f} {cd:>10.2e} {errs}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Synthetic GPU-scaling benchmark for survivex (Table 3/4).")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES),
                    help="Model families to benchmark (default: all).")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Override the size grid as colon-joined ints, e.g. "
                         "'50000:20'. Arity must match the family (frailty "
                         "takes 3). Applies to every selected family.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds; each (family, config, backend) is timed once "
                         "per seed (default: 42, the paper value).")
    ap.add_argument("--backends", nargs="+",
                    default=["survivex-cpu", "survivex-cuda", "lifelines-cpu"],
                    choices=["survivex-cpu", "survivex-cuda", "survivex-mps",
                             "lifelines-cpu"],
                    help="Backends to time. survivex-cuda/mps are skipped for "
                         "CPU-only families (e.g. finegray) and when the "
                         "device is unavailable; lifelines-cpu only applies to "
                         "families with a lifelines comparator (coxph).")
    ap.add_argument("--n-runs", type=int, default=None,
                    help="Timed repetitions per cell (default: family-specific, "
                         "3 for most, 2 for finegray). Median is reported.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="Skip the discarded GPU warm-up fit (default: warm up).")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "results"
                    / "synthetic_scaling_timings.csv",
                    help="CSV output path.")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        cuda = torch.cuda.is_available()
        cuda_name = torch.cuda.get_device_name(0) if cuda else "n/a"
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        _log(f"torch {torch.__version__}; cuda={cuda} ({cuda_name}); mps={mps}")
    except ImportError:
        _log("torch not importable; GPU backends will be skipped")

    rows: list[BenchmarkRow] = []

    for fam_name in args.families:
        fam = FAMILIES[fam_name]
        n_runs = args.n_runs if args.n_runs is not None else fam.default_n_runs
        configs = _parse_configs(fam, args.configs)
        _log(f"== family {fam.name} | {len(configs)} configs | n_runs={n_runs} ==")

        for cfg in configs:
            for seed in args.seeds:
                data = fam.build(cfg, seed)
                n_rows, p = fam.n_p(data)
                cfg_lbl = fam.config_label(cfg)
                _log(f"  {fam.name} cfg={cfg_lbl} seed={seed} "
                     f"(n_rows={n_rows:,}, p={p})")

                cpu_coef: Optional[np.ndarray] = None

                for backend in args.backends:
                    # ---- survivex CPU/GPU ---------------------------------- #
                    if backend.startswith("survivex-"):
                        device = backend.split("-", 1)[1]
                        if device != "cpu" and not fam.survivex_gpu:
                            continue  # CPU-only family
                        if device != "cpu" and not _device_available(device):
                            rows.append(_empty_row(
                                fam, cfg, n_rows, p, seed, backend, n_runs,
                                err=f"{device}-unavailable"))
                            continue

                        med, pcpu, pgpu, model, err = _time_fits(
                            lambda d=device: fam.fit_survivex(data, d),
                            device, n_runs, warmup=not args.no_warmup)

                        coef = fam.get_coef(model) if (model is not None and not err) else None
                        coef = None if coef is None else np.asarray(coef)
                        if device == "cpu" and coef is not None:
                            cpu_coef = coef
                        diff = float("nan")
                        if coef is not None and cpu_coef is not None and coef.shape == cpu_coef.shape:
                            diff = float(np.max(np.abs(coef - cpu_coef)))
                        cidx = fam.get_cindex(model) if (model is not None and not err) else float("nan")
                        l2 = float(np.linalg.norm(coef)) if coef is not None else float("nan")

                        rows.append(BenchmarkRow(
                            family=fam.name, config=cfg_lbl, n_rows=n_rows, p=p,
                            seed=seed, backend=backend, n_runs=n_runs,
                            fit_seconds_median=med, peak_cpu_mb=pcpu,
                            peak_gpu_mb=pgpu, c_index=cidx, coef_l2_norm=l2,
                            coef_diff_vs_cpu=diff, error=err, notes=""))
                        _log(f"    {backend:<14}: {med:.4f}s  "
                             f"peak_cpu={pcpu:.0f}MB peak_gpu={pgpu:.0f}MB  "
                             f"coef_diff={diff:.2e}  err={err or '-'}")

                    # ---- lifelines CPU ------------------------------------- #
                    elif backend == "lifelines-cpu":
                        if fam.fit_lifelines is None:
                            continue
                        med, pcpu, _pg, model, err = _time_fits(
                            lambda: fam.fit_lifelines(data),
                            "cpu", n_runs, warmup=False)
                        diff = float("nan")
                        cidx = float("nan")
                        l2 = float("nan")
                        if model is not None and not err:
                            try:
                                ll_coef = (fam.coef_lifelines(model)
                                           if fam.coef_lifelines else None)
                                if ll_coef is not None:
                                    l2 = float(np.linalg.norm(ll_coef))
                                    if cpu_coef is not None and ll_coef.shape == cpu_coef.shape:
                                        diff = float(np.max(np.abs(ll_coef - cpu_coef)))
                            except Exception:
                                pass
                            try:
                                cidx = (fam.cindex_lifelines(model)
                                        if fam.cindex_lifelines else float("nan"))
                            except Exception:
                                pass
                        rows.append(BenchmarkRow(
                            family=fam.name, config=cfg_lbl, n_rows=n_rows, p=p,
                            seed=seed, backend=backend, n_runs=n_runs,
                            fit_seconds_median=med, peak_cpu_mb=pcpu,
                            peak_gpu_mb=0.0, c_index=cidx, coef_l2_norm=l2,
                            coef_diff_vs_cpu=diff, error=err, notes=""))
                        _log(f"    {backend:<14}: {med:.4f}s  "
                             f"peak_cpu={pcpu:.0f}MB  coef_diff={diff:.2e}  "
                             f"err={err or '-'}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchmarkRow.__annotations__))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    _log(f"Wrote {len(rows)} rows to {args.output}")

    print("\n" + _summary_table(rows))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
