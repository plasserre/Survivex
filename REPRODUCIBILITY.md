# Reproducibility

This document maps every table in the survivex paper to the exact code,
command, random seed, and hardware needed to reproduce it. It is the entry
point for reviewers and users who want to re-run the validation and benchmarks.

| Paper artifact | What it reports | How to reproduce | Hardware |
|---|---|---|---|
| Table 1 | Library feature comparison | Static claims table — see `README.md` feature list (no computation) | — |
| Table 2 | Validation accuracy vs reference implementations | [Validation suite](#table-2--validation-accuracy) | CPU (any) |
| Table 3 | CPU performance vs lifelines | [CPU benchmark](#table-3--cpu-benchmark) | Apple M2 Pro, 16 GB |
| Table 4 | GPU scaling + real-data benchmark | [GPU benchmark](#table-4--gpu-benchmark) | NVIDIA RTX A6000, 48 GB |
| Ablation | Analytic vs. autodiff derivatives | [Ablation](#analytic-vs-autodiff-ablation) | CPU + NVIDIA RTX A6000 |

All randomized procedures use fixed seeds (documented per section). Numerical
agreement targets are stated alongside each command.

---

## Environment

### CPU / validation machine (Mac)

```bash
git clone https://github.com/TaninZeraati/survivex.git
cd survivex
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[all]"
.venv/bin/pip install tqdm scikit-learn          # benchmark/test helpers
```

### GPU machine

Same as above on a CUDA host; install the CUDA build of PyTorch first
(`pip install torch --index-url https://download.pytorch.org/whl/cu130` or the
version matching your driver). survivex selects the device from the `device=`
argument and **falls back to CPU with a warning** if the requested accelerator
is unavailable (`survivex/models/cox_ph.py`); there is no automatic n/p-based
device switch.

### R (only needed for the gold-standard validations in Table 2)

```r
install.packages(c("survival", "penalized", "frailtyEM", "coxme"))
```

- `survival` — Cox PH / stratified / recurrent (Newton-Raphson, same algorithm as survivex)
- `penalized` — Cox Ridge (L2) gold standard
- `frailtyEM` — gamma shared-frailty EM (same estimator as survivex)
- `coxme` — log-normal frailty marginal likelihood (cross-check)

### Pinned package versions (validation reference)

The Table 2 validation figures were produced with the versions below. All
validation runs use **double precision (float64)** on CPU and CUDA (MPS is
float32-only and is not used for the reported tables); solver tolerances are
pinned tight on both sides (survivex `tol=1e-12`, up to 200 iterations; R
`coxph.control(eps=1e-12)`); and all randomized procedures use **seed 42**. The
CI matrix below installs current releases, which reproduce the same agreement.

| Component | Version | Component | Version |
|---|---|---|---|
| Python | 3.12 | lifelines | 0.30.3 |
| PyTorch | 2.12.0 | R `survival` | 3.8.3 |
| NumPy | 2.4.6 | R `penalized` | 0.9.53 |
| SciPy | 1.17.1 | R `frailtyEM` | 1.0.1 |
| pandas | 2.3.3 | R `coxme` | 2.2.22 |
| numba | 0.65.1 | scikit-learn | 1.8.0 |

### Continuous integration

Two GitHub Actions workflows in `.github/workflows/`:

- `ci.yml` — fast tier, on every push: **ubuntu-latest × Python 3.10 / 3.11 /
  3.12**, `pip install -e ".[all]"`, then `pytest tests/`. R-backed tests
  auto-skip when `Rscript` is absent.
- `validation.yml` — nightly tier (cron) + manual dispatch: installs R
  (`survival`, `penalized`, `frailtyEM`, `coxme`) and runs the full R
  gold-standard validators plus the test suite.

---

## Table 2 — Validation accuracy

Each model is checked against a reference implementation. The R-backed scripts
export byte-identical data to `tests/data/*.csv`, invoke the matching `.R`
reference generator, then compare survivex to the reference and print
`max|Δ|` per case.

**One command for everything** (runs all groups, skips the R-backed ones with a
note if R is not installed; exit code is non-zero on any failure):

```bash
.venv/bin/python tests/run_validation.py          # full Table 2
.venv/bin/python tests/run_validation.py --no-r   # lifelines groups only
```

Or run each group on its own:

| Group | Reference | Command | Target |
|---|---|---|---|
| Cox PH (Breslow/Efron, stratified, Ridge) | R `survival` + `penalized` | `.venv/bin/python tests/validate_coxph_vs_R.py` | ~1e-15 |
| Gamma frailty | R `frailtyEM` (EM) | `.venv/bin/python tests/validate_frailty_vs_R.py` | ~1e-4 (rats) to ~3.6e-3 (kidney) vs EM |
| Recurrent events (AG, PWP-TT, PWP-GT) | R `survival` (references baked into the test) | `.venv/bin/python tests/test_recurrent.py` | machine ε |
| KM / NA / parametric / competing risks / trees / GBM | lifelines | per-model `tests/test_*.py` | ~1e-8 |

Notes:
- The Cox and frailty drivers regenerate their input CSVs and R references on
  every run, so no committed artifacts are required; the `.R` scripts and
  reference CSVs are tracked for offline inspection.
- The recurrent check (`tests/test_recurrent.py`) is self-contained: it
  generates seeded AG / PWP-TT / PWP-GT / frailty datasets and compares
  survivex against R `survival::coxph` values baked into the script.
  `tests/validate_recurrent_R.R` is the offline generator of those reference
  numbers (it reads a `tests/recurrent_event_data.csv` not shipped in a fresh
  clone) and is tracked only for inspection.
- The frailty driver also reports the survivex-vs-`coxph` (penalized partial
  likelihood) gap to document that the historical ~4.88e-2 difference is an
  EM-vs-PPL estimator difference, not a defect.
- Solver tolerances are pinned tight on both sides (survivex `tol=1e-12`,
  R `coxph.control(eps=1e-12)`) so the comparison is solver-vs-solver, not
  tolerance-vs-tolerance.

---

## Table 3 — CPU benchmark

Median of 21 runs on an **Apple M2 Pro (16 GB)**, survivex vs lifelines on the
small standard datasets (gbsg2, rossi). Fully scripted in
`benchmarks/cpu_standard/`:

```bash
cd benchmarks/cpu_standard
python run_cpu_benchmark.py            # writes results/cpu_standard_timings.csv
```

The script loads gbsg2 (Kaplan-Meier, Nelson-Aalen, Cox PH) and rossi
(Weibull / Log-Normal / Log-Logistic AFT) from lifelines, times each model
against its lifelines counterpart as the median over `--n-runs` repetitions
(default 21) after one warm-up call, and reports the lifelines/survivex
speedup. See `benchmarks/cpu_standard/README.md` for details. Timings are
hardware-dependent; the table figures are from the Apple M2 Pro above.

---

## Table 4 — GPU benchmark

Two parts:

### 4a. Synthetic scaling (submitted paper)

NVIDIA **RTX A6000 (48 GB)**. Synthetic datasets across a covariate sweep,
`n_runs=3`, all generators seeded with `seed=42`. Fully scripted in
`benchmarks/synthetic/`:

```bash
cd benchmarks/synthetic
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --backends survivex-cpu survivex-cuda \
    --seeds 42 --n-runs 3 \
    --output results/synthetic_scaling_timings.csv
# or restrict to one family, e.g.  --families coxph
```

The generators in `benchmarks/synthetic/generators.py` (all default `seed=42`):

- `generate_large_synthetic(n, p, seed=42)` — Cox PH, Weibull AFT, GBM, RSF rows; `β ~ U(−0.5, 0.5)`, ~20% admin censoring
- `generate_recurrent_data(n_subjects, p, seed=42)` — AG / PWP-TT / PWP-GT; `β ~ U(−0.3, 0.3)`, ~20% censoring
- `generate_competing_risks_data(n, p, seed=42)` — Fine-Gray (two cause-specific event types); `β ~ U(−0.3, 0.3)`
- `generate_clustered_data(n_clusters, obs_per_cluster, p, seed=42)` — gamma frailty; `β ~ U(−0.3, 0.3)`, ~25% censoring

The per-family size grids (verbatim from the original notebook) and the output
CSV schema are documented in `benchmarks/synthetic/README.md`. GPU backends are
skipped for the CPU-only Fine-Gray family and when the requested device is
unavailable.

### 4b. Real-data: TCGA pan-cancer (revision)

A public, high-dimensional clinical benchmark added in revision. Pools 8 TCGA
cohorts (4,112 patients, 1,382 events) and fits a `StratifiedCoxPHModel` across
a covariate sweep on three backends (`survivex-cpu`, `survivex-cuda`,
`lifelines-cpu`). Fully scripted in `benchmarks/tcga_pancancer/`:

```bash
cd benchmarks/tcga_pancancer
python download.py        # ~700 MB from cBioPortal; logs SHA-256 per file
python preprocess.py      # gene intersection + per-cohort z-score + pool + top-K
mkdir -p results
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py \
    --p 100 500 1000 2000 5000 --seeds 42 43 44 \
    --backends survivex-cpu survivex-cuda lifelines-cpu \
    --penalty 0.01 \
    --output results/tcga_pancancer_timings.csv
```

See `benchmarks/tcga_pancancer/README.md` for the math pipeline. Output CSV
columns: backend, p, seed, fit time, peak GPU memory, c-index, coefficient
agreement. cBioPortal data is mutable — the gene universe can shift between
downloads — so cite the download date; `download.py` logs per-file SHA-256 for
provenance.

---

## Analytic-vs-autodiff ablation

Quantifies the speed contributed by survivex's closed-form Cox gradient and
Hessian versus automatic differentiation (paper Section 2.3.3). Fully scripted
in `benchmarks/ablation_autodiff/`:

```bash
cd benchmarks/ablation_autodiff
python run_ablation.py --p 20 100 500 1000 --n 2000 --seed 42 \
    --device cpu  --output results/ablation_cpu.csv
CUDA_VISIBLE_DEVICES=0 python run_ablation.py --p 20 100 500 1000 --n 2000 \
    --seed 42 --device cuda --output results/ablation_cuda.csv
```

The same Breslow partial likelihood is fit with a Newton-Raphson loop using
either closed-form or autograd-computed derivatives, plus a gradient-only
L-BFGS variant. The script validates the analytic gradient and Hessian against
autograd before timing (so the comparison is on the same objective), and both
Newton paths converge in the same number of iterations. See
`benchmarks/ablation_autodiff/README.md` for details.

---

## Determinism notes

- survivex Cox PH / stratified / frailty fits are deterministic given the data
  and solver settings (Newton-Raphson / EM, no stochastic steps).
- GBM and tree models use seeded RNG; pass the documented `seed=42`.
- Benchmark *timings* vary run-to-run; the reported figures are medians
  (Table 3: 21 runs; Table 4: 3 runs).
- At `p > n_events`, unpenalized Cox coefficients are non-identifiable (the
  Hessian is near-singular) and differ across hardware along the null space
  even though the c-index and predictions agree; use `--penalty` to regularize.
