# Synthetic GPU-scaling benchmark (Table 3 CPU rows + Table 4 synthetic scaling)

This directory is the seeded, CLI-driven consolidation of the
`benchmark_gpu_scaling.ipynb` notebook that produced the synthetic timing and
scaling numbers in the submitted SoftwareX paper. The notebook was the version
of record; the two files here reproduce it without Jupyter:

| File | Role |
|---|---|
| `generators.py` | The four synthetic data generators, ported verbatim from notebook cell 3. Each takes an explicit `seed` (default `42`, the paper value). |
| `run_benchmark.py` | Sweeps each model family over the notebook's size grid, fits survivex on CPU/GPU (and lifelines on CPU where applicable), and records median wall-clock, peak CPU/GPU memory, C-index, and CPU-vs-GPU coefficient agreement to a CSV. |

`results/` is git-ignored (machine-specific); commit only the canonical paper
run if you want it tracked.

## Quick start

```bash
# From the repo root, with the project installed (pip install -e ".[all]")
cd benchmarks/synthetic

# Full sweep, all families, GPU used automatically if present:
python run_benchmark.py

# A single family with the seeds/runs used in the paper:
python run_benchmark.py --families coxph --seeds 42 --n-runs 3

# Fast smoke run (tiny grid, 1 rep):
python run_benchmark.py --families coxph --configs 2000:5 --n-runs 1 \
    --backends survivex-cpu lifelines-cpu
```

Output CSV (default `results/synthetic_scaling_timings.csv`) has one row per
`(family, config, seed, backend)` with columns: `family, config, n_rows, p,
seed, backend, n_runs, fit_seconds_median, peak_cpu_mb, peak_gpu_mb, c_index,
coef_l2_norm, coef_diff_vs_cpu, error, notes`. A median-grouped summary table is
also printed to the console.

## Model families and default size grids

These grids are verbatim from the notebook (cells 12, 18, 20, 22, 24, 26).
Config tuples are passed to `--configs` as colon-joined ints; the arity must
match the family.

| Family | Model | Config `(...)` | Default grid |
|---|---|---|---|
| `coxph` | Cox PH (Breslow) | `(n, p)` | (10000,20) (50000,20) (100000,20) (200000,20) (50000,50) (50000,100) (50000,200) |
| `pwp-tt` | PWP total-time | `(n_subjects, p)` | (5000,10) (10000,10) (10000,50) |
| `pwp-gt` | PWP gap-time | `(n_subjects, p)` | (5000,10) (10000,10) (10000,50) |
| `frailty` | Gamma shared frailty | `(n_clusters, obs_per_cluster, p)` | (500,10,10) (1000,10,10) (2000,10,10) (1000,10,50) |
| `weibull` | Weibull AFT | `(n, p)` | (10000,10) (50000,10) (50000,50) |
| `gbm` | Gradient boosting | `(n, p)` | (5000,10) (10000,10) (10000,50) |
| `rsf` | Random survival forest | `(n, p)` | (5000,10) (10000,10) (10000,50) |
| `finegray` | Fine-Gray (CPU only) | `(n, p)` | (500,5) (1000,5) (1000,10) |

`n_runs` defaults to 3 for every family except `finegray` (2), matching the
notebook. The reported time is the median over `n_runs`.

## Backends

`--backends` selects which to time (default: `survivex-cpu survivex-cuda
lifelines-cpu`):

- `survivex-cpu` / `survivex-cuda` / `survivex-mps` — the same survivex model on
  the named device. GPU backends are **skipped** for CPU-only families
  (`finegray`, whose IPCW weighting is Python-loop bound — GPU adds overhead,
  not speedup, exactly as the notebook documents) and when the requested device
  is unavailable (an `error` row records the skip rather than silently running
  on CPU).
- `lifelines-cpu` — a CPU baseline for families that have a lifelines
  comparator (currently `coxph`). For other families the backend is silently
  skipped. This supplies the Table 3 "vs lifelines" CPU rows.

## Notes on the recorded columns

- **Peak GPU memory** uses `torch.cuda.max_memory_allocated()`; MPS exposes no
  peak-allocation API and reports `0.0`.
- **Peak CPU memory** uses `tracemalloc` (Python allocations only).
- **`coef_diff_vs_cpu`** is `max|Δβ|` against the `survivex-cpu` fit for the
  same `(family, config, seed)`. For `survivex-cuda`/`mps` it is the CPU-vs-GPU
  agreement (≈1e-15 on CUDA float64, ≈1e-7 on MPS float32); for `lifelines-cpu`
  on `coxph` it is the survivex-vs-lifelines coefficient gap. Tree/ensemble
  families (`gbm`, `rsf`) and the AFT/`weibull` family expose no single
  comparable coefficient vector, so the column is `nan` there.
- **Determinism:** all data is seeded; survivex Cox/frailty/AFT fits are
  deterministic (Newton-Raphson / EM, no stochastic steps). `gbm` and `rsf` use
  seeded RNG internally. Timings vary run-to-run; the reported figure is the
  median.

## Hardware in the submitted paper

Table 3 CPU rows: Apple M2 Pro (16 GB). Table 4 synthetic scaling: NVIDIA RTX
A6000 (48 GB), `n_runs=3`, `seed=42`. See the top-level `REPRODUCIBILITY.md` for
the full environment and the real-data (TCGA pan-cancer) GPU benchmark.
