# Analytic vs. autodiff ablation (Cox partial likelihood)

This directory isolates the speed contributed by survivex's use of closed-form
gradients and a closed-form Hessian, as opposed to automatic differentiation,
for Cox-type models.

The same Breslow partial likelihood is fit three ways on identical data and the
same device:

| Method | Optimizer | Derivatives |
|---|---|---|
| `newton-analytic` | Newton-Raphson | closed-form gradient + Hessian (the survivex path) |
| `newton-autodiff` | Newton-Raphson | gradient + Hessian from `torch.autograd` |
| `lbfgs-autodiff` | L-BFGS | gradient from `torch.autograd`, no Hessian |

`newton-analytic` and `newton-autodiff` run the *identical* loop and converge
in the same number of iterations, so the wall-clock gap between them is the pure
cost of automatic differentiation. `lbfgs-autodiff` is the gradient-only
alternative one would reach for if building a Cox fitter without analytic
derivatives; it avoids the Hessian but converges to looser precision and yields
no information matrix for standard errors.

Before timing, the analytic gradient and Hessian are checked against autograd
(`grad_chk`, `hess_chk` columns) so the comparison is provably on the same
objective. Data uses continuous, tie-free event times, so Breslow is exact.

## Quick start

```bash
# From the repo root, with the project installed (pip install -e ".[all]")
cd benchmarks/ablation_autodiff

# CPU (float64):
python run_ablation.py --p 20 100 500 1000 --n 2000 --seed 42 \
    --device cpu --output results/ablation_cpu.csv

# NVIDIA GPU (float64):
CUDA_VISIBLE_DEVICES=0 python run_ablation.py --p 20 100 500 1000 --n 2000 \
    --seed 42 --device cuda --output results/ablation_cuda.csv
```

`--device mps` also runs on Apple Silicon, but MPS is float32-only, so the
Newton tolerance is relaxed accordingly and the agreement columns are at
float32 precision.

## Output

One CSV row per (p, method) with median wall-clock over `--repeats` runs
(after one warm-up), iteration count, `beta_err` (distance to the analytic
solution), and the `grad_chk` / `hess_chk` analytic-vs-autograd residuals.
