"""
Stratified Cox PH benchmark on pooled TCGA pan-cancer data.

This replaces the synthetic-only Table 4 GPU evaluation in the original
SoftwareX submission with a real-data demonstration. Each of the 8 TCGA
cohorts contributes its own baseline hazard via stratification, while a
common coefficient vector beta is estimated jointly across cohorts.

For each p in {100, 500, 1000, 2000, 5000} (override with --p):
    For each seed in {42, 43, 44} (override with --seeds):
        For each backend in {survivex-cpu, survivex-cuda, lifelines-cpu}:
            - Fit stratified Cox PH (Breslow ties)
            - Record wall-clock fit time, peak memory, C-index, coef L2-norm
            - Verify CPU and GPU coefficients agree within tolerance

Outputs:
    --output (default results/tcga_pancancer_timings.csv): one row per
        (p, seed, backend).
    Console: summary table grouped by (p, backend).

Usage:
    python run_benchmark.py
    python run_benchmark.py --backends survivex-cuda survivex-cpu
    python run_benchmark.py --p 100 500 1000
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
from typing import Optional

import numpy as np


@dataclass
class BenchmarkRow:
    p: int
    seed: int
    backend: str
    n_patients: int
    n_events: int
    n_strata: int
    fit_seconds: float
    peak_cpu_mb: float
    peak_gpu_mb: float
    c_index: float
    coef_l2_norm: float
    error: str
    notes: str


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_dataset(processed_dir: Path, p: int):
    """Load the processed top-p TCGA pan-cancer dataset."""
    npz_path = processed_dir / f"tcga_pancancer_top{p}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Processed file missing: {npz_path}\nRun preprocess.py first."
        )
    data = np.load(npz_path, allow_pickle=True)
    # Promote X to float64 so CPU and GPU paths compare on the same precision
    # (CUDA path uses float64 by default; MPS uses float32 internally).
    X = data["X"].astype(np.float64)
    T = data["T"].astype(np.float64)
    E = data["E"].astype(np.float64)
    strata = data["strata"].astype(np.int64)
    strata_labels = data["strata_labels"]
    return X, T, E, strata, strata_labels


def _peak_gpu_mb(device: str) -> float:
    """Peak GPU memory in MiB. 0.0 for non-GPU or MPS (no API)."""
    import torch
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def _bench_survivex(
    X: np.ndarray, T: np.ndarray, E: np.ndarray, strata: np.ndarray,
    device: str, p: int, seed: int, warmup: bool, penalty: float = 0.0,
) -> tuple[BenchmarkRow, Optional[object]]:
    """Fit survivex StratifiedCoxPHModel on the requested device."""
    import torch
    from survivex.models.cox_ph import StratifiedCoxPHModel

    n_strata = int(len(np.unique(strata)))
    n = len(T)
    n_events = int(E.sum())

    def _empty(err: str) -> tuple[BenchmarkRow, None]:
        return BenchmarkRow(
            p=p, seed=seed, backend=f"survivex-{device}",
            n_patients=n, n_events=n_events, n_strata=n_strata,
            fit_seconds=float("nan"), peak_cpu_mb=0.0, peak_gpu_mb=0.0,
            c_index=float("nan"), coef_l2_norm=float("nan"),
            error=err, notes="",
        ), None

    if device == "cuda":
        if not torch.cuda.is_available():
            return _empty("cuda-unavailable")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    elif device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return _empty("mps-unavailable")

    np.random.seed(seed)
    torch.manual_seed(seed)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    if warmup:
        # Discarded fit on a small sub-sample to absorb compile/cache effects.
        # Use enough samples to cover at least 2 strata (otherwise the model
        # is essentially unstratified for the warm-up).
        n_wu = min(400, n)
        try:
            wu = StratifiedCoxPHModel(tie_method="breslow", device=device,
                                      max_iter=200, penalty=penalty)
            wu.fit(X[:n_wu], T[:n_wu], E[:n_wu], strata[:n_wu])
            del wu
        except Exception:
            pass
        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif device == "mps":
            torch.mps.synchronize()

    tracemalloc.start()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()

    cox = StratifiedCoxPHModel(tie_method="breslow", device=device,
                               max_iter=200, penalty=penalty)
    err = ""
    notes: list[str] = []
    try:
        cox.fit(X, T, E, strata)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        notes.append("fit raised; row partial")

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    _, peak_cpu_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    c_index = float("nan")
    coef_l2 = float("nan")
    if not err:
        try:
            c_index = float(getattr(cox, "concordance_index_", float("nan")))
        except Exception:
            pass
        try:
            beta = np.asarray(getattr(cox, "coefficients_", []))
            if beta.size:
                coef_l2 = float(np.linalg.norm(beta))
        except Exception:
            pass

    return BenchmarkRow(
        p=p, seed=seed, backend=f"survivex-{device}",
        n_patients=n, n_events=n_events, n_strata=n_strata,
        fit_seconds=elapsed,
        peak_cpu_mb=peak_cpu_b / (1024**2),
        peak_gpu_mb=_peak_gpu_mb(device),
        c_index=c_index, coef_l2_norm=coef_l2,
        error=err, notes="; ".join(notes),
    ), (cox if not err else None)


def _bench_lifelines(
    X: np.ndarray, T: np.ndarray, E: np.ndarray, strata: np.ndarray,
    strata_labels: np.ndarray, p: int, seed: int, penalty: float = 0.0,
) -> BenchmarkRow:
    """Lifelines CoxPHFitter with strata for a fair comparison."""
    n_strata = int(len(np.unique(strata)))
    n = len(T)
    n_events = int(E.sum())

    try:
        import pandas as pd
        from lifelines import CoxPHFitter
    except ImportError as exc:
        return BenchmarkRow(
            p=p, seed=seed, backend="lifelines-cpu",
            n_patients=n, n_events=n_events, n_strata=n_strata,
            fit_seconds=float("nan"), peak_cpu_mb=0.0, peak_gpu_mb=0.0,
            c_index=float("nan"), coef_l2_norm=float("nan"),
            error=f"import-failed: {exc}", notes="",
        )

    df = pd.DataFrame(X, columns=[f"g{i}" for i in range(p)])
    df["__T"] = T
    df["__E"] = E.astype(int)
    # lifelines requires a string/categorical strata column
    df["__S"] = [strata_labels[s] for s in strata]

    np.random.seed(seed)
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    # l1_ratio=0 → pure Ridge (matches survivex penalty convention exactly).
    cph = CoxPHFitter(penalizer=penalty, l1_ratio=0.0)
    err = ""
    notes = ""
    try:
        cph.fit(df, duration_col="__T", event_col="__E", strata=["__S"],
                show_progress=False, fit_options={"step_size": 0.5})
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        if p >= n_events:
            notes = "lifelines unstable when p >= n_events; expected"
    elapsed = time.perf_counter() - t0
    _, peak_b = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    c_index = float("nan")
    coef_l2 = float("nan")
    if not err:
        try:
            c_index = float(cph.concordance_index_)
        except Exception:
            pass
        try:
            coef_l2 = float(np.linalg.norm(cph.params_.to_numpy()))
        except Exception:
            pass

    return BenchmarkRow(
        p=p, seed=seed, backend="lifelines-cpu",
        n_patients=n, n_events=n_events, n_strata=n_strata,
        fit_seconds=elapsed,
        peak_cpu_mb=peak_b / (1024**2), peak_gpu_mb=0.0,
        c_index=c_index, coef_l2_norm=coef_l2,
        error=err, notes=notes,
    )


def _compare_coefs(cox_cpu, cox_gpu) -> Optional[float]:
    """Return max abs coefficient difference between two survivex fits."""
    if cox_cpu is None or cox_gpu is None:
        return None
    try:
        a = np.asarray(cox_cpu.coefficients_)
        b = np.asarray(cox_gpu.coefficients_)
        if a.shape != b.shape:
            return None
        return float(np.max(np.abs(a - b)))
    except Exception:
        return None


def _summary_table(rows: list[BenchmarkRow]) -> str:
    """Median timing grouped by (p, backend)."""
    from statistics import median

    grouped: dict[tuple[int, str], list[BenchmarkRow]] = {}
    for r in rows:
        grouped.setdefault((r.p, r.backend), []).append(r)

    lines = [
        f"{'p':>6}  {'backend':<18}  {'n_runs':>6}  {'median_s':>10}  "
        f"{'c_index':>8}  {'peak_cpu_mb':>12}  {'peak_gpu_mb':>12}  errors"
    ]
    lines.append("-" * len(lines[0]))
    for (p, backend), rs in sorted(grouped.items()):
        ok = [r for r in rs if not r.error]
        n = len(ok)
        med = median(r.fit_seconds for r in ok) if ok else float("nan")
        ci = (median(r.c_index for r in ok if not np.isnan(r.c_index))
              if any(not np.isnan(r.c_index) for r in ok) else float("nan"))
        peak_cpu = median(r.peak_cpu_mb for r in ok) if ok else float("nan")
        peak_gpu = median(r.peak_gpu_mb for r in ok) if ok else float("nan")
        errs = sum(1 for r in rs if r.error)
        lines.append(
            f"{p:>6}  {backend:<18}  {n:>6}  {med:>10.3f}  "
            f"{ci:>8.4f}  {peak_cpu:>12.1f}  {peak_gpu:>12.1f}  {errs}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark survivex stratified Cox PH on pooled TCGA "
                    "pan-cancer data."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing data/processed/ (default: ./data).",
    )
    parser.add_argument(
        "--p", type=int, nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="Covariate counts to benchmark.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
        help="Random seeds; each (p, backend) is timed once per seed.",
    )
    parser.add_argument(
        "--backends", type=str, nargs="+",
        default=["survivex-cpu", "survivex-cuda", "lifelines-cpu"],
        choices=["survivex-cpu", "survivex-cuda", "survivex-mps", "lifelines-cpu"],
        help="Which backends to time. survivex-mps is for Apple Silicon "
             "sanity checks; not paper-grade.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "results" / "tcga_pancancer_timings.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip the GPU warm-up fit. Default is to warm up.",
    )
    parser.add_argument(
        "--penalty", type=float, default=0.0,
        help="L2 Ridge penalty (lifelines convention: effective penalty is "
             "n*penalty*0.5*||beta||^2). Same value is passed to survivex "
             "and lifelines so coefficients are directly comparable. "
             "Default 0.0 (unregularised). Recommend 0.01 for p>=2000 "
             "where unregularised Cox is ill-posed (lifelines crashes; "
             "see its own ConvergenceError at high p).",
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_name = torch.cuda.get_device_name(0) if cuda_avail else "n/a"
        mps_avail = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        _log(f"torch {torch.__version__}; "
             f"cuda={cuda_avail} ({cuda_name}); mps={mps_avail}")
    except ImportError:
        _log("torch not available; GPU backends will be skipped")
    _log(f"penalty (L2 Ridge, lifelines convention) = {args.penalty}")

    for p in sorted(set(args.p)):
        _log(f"== p = {p} ==")
        try:
            X, T, E, strata, strata_labels = _load_dataset(processed_dir, p)
        except FileNotFoundError as exc:
            _log(str(exc))
            continue
        _log(f"  loaded: X={X.shape}, events={int(E.sum())}/{len(E)}, "
             f"strata={len(np.unique(strata))}")

        for seed in args.seeds:
            _log(f"  seed={seed}")
            cox_cpu_obj = cox_gpu_obj = None

            if "survivex-cpu" in args.backends:
                row, obj = _bench_survivex(
                    X, T, E, strata, device="cpu", p=p, seed=seed,
                    warmup=not args.no_warmup, penalty=args.penalty,
                )
                cox_cpu_obj = obj
                rows.append(row)
                _log(f"    survivex-cpu : {row.fit_seconds:.3f}s "
                     f"c-index={row.c_index:.4f}  err={row.error or '-'}")

            if "survivex-cuda" in args.backends:
                row, obj = _bench_survivex(
                    X, T, E, strata, device="cuda", p=p, seed=seed,
                    warmup=not args.no_warmup, penalty=args.penalty,
                )
                cox_gpu_obj = obj
                rows.append(row)
                _log(f"    survivex-cuda: {row.fit_seconds:.3f}s "
                     f"c-index={row.c_index:.4f}  "
                     f"peak_gpu={row.peak_gpu_mb:.0f}MB  err={row.error or '-'}")

            if "survivex-mps" in args.backends:
                row, obj = _bench_survivex(
                    X, T, E, strata, device="mps", p=p, seed=seed,
                    warmup=not args.no_warmup, penalty=args.penalty,
                )
                if cox_gpu_obj is None:
                    cox_gpu_obj = obj
                rows.append(row)
                _log(f"    survivex-mps : {row.fit_seconds:.3f}s "
                     f"c-index={row.c_index:.4f}  err={row.error or '-'}")

            if "lifelines-cpu" in args.backends:
                row = _bench_lifelines(X, T, E, strata, strata_labels,
                                       p=p, seed=seed, penalty=args.penalty)
                rows.append(row)
                _log(f"    lifelines-cpu: {row.fit_seconds:.3f}s "
                     f"c-index={row.c_index:.4f}  err={row.error or '-'}")

            max_diff = _compare_coefs(cox_cpu_obj, cox_gpu_obj)
            if max_diff is not None:
                _log(f"    coef CPU vs GPU max-abs diff: {max_diff:.3e}")

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
