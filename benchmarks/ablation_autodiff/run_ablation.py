"""
Ablation: analytical gradient/Hessian vs automatic differentiation for the
Cox partial likelihood.

The Cox models in survivex use closed-form gradients and a closed-form Hessian
inside a Newton-Raphson loop. This script quantifies the speed contributed by
that choice by fitting the *same* Breslow partial likelihood three ways on the
same data and device:

  1. newton-analytic : Newton-Raphson with closed-form gradient and Hessian
                       (the survivex algorithm).
  2. newton-autodiff : the identical Newton-Raphson loop, but the gradient and
                       Hessian come from torch.autograd. Same iterations, same
                       convergence path -> the timing gap is the pure
                       autodiff overhead.
  3. lbfgs-autodiff  : L-BFGS on the same loss with autograd gradients (no
                       Hessian). This is how a gradient-only autodiff Cox would
                       realistically be built, and shows the additional cost of
                       giving up second-order information.

Before timing, the analytic gradient and Hessian are checked against autograd
so the comparison is provably fair (same objective, same optimum).

Synthetic data uses continuous event times (no ties), so Breslow is exact.

Example:
    python run_ablation.py --p 20 100 500 --n 2000 --device cpu \
        --output results/ablation_cpu.csv
    CUDA_VISIBLE_DEVICES=0 python run_ablation.py --p 20 100 500 1000 \
        --n 2000 --device cuda --output results/ablation_cuda.csv
"""

import argparse
import csv
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch


def make_data(n, p, seed, device, dtype):
    """Cox data with continuous (tie-free) event times and ~20% censoring."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = rng.uniform(-0.5, 0.5, size=p)
    risk = np.exp(X @ beta_true)
    event_time = rng.exponential(1.0 / risk)
    censor_time = rng.exponential(event_time.mean() / 0.25)  # ~20% censored
    time_obs = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)

    # Sort once by descending time so risk sets are prefix sums.
    order = np.argsort(-time_obs)
    Xt = torch.as_tensor(X[order], device=device, dtype=dtype)
    et = torch.as_tensor(event[order], device=device, dtype=dtype)
    return Xt, et, int(event.sum())


def neg_log_partial_likelihood(beta, X, event):
    """Breslow negative log partial likelihood, no ties. Rows sorted by
    descending time, so logcumsumexp gives log of each risk-set sum."""
    eta = X @ beta
    log_risk = torch.logcumsumexp(eta, dim=0)
    return -torch.sum(event * (eta - log_risk))


def analytic_grad_hess(beta, X, event):
    """Closed-form gradient and observed information (negative Hessian) of the
    Breslow log partial likelihood, without materialising an (n, p, p) tensor."""
    eta = X @ beta
    r = torch.exp(eta)
    s0 = torch.cumsum(r, dim=0)                       # risk-set weight sums
    s1 = torch.cumsum(r[:, None] * X, dim=0)          # (n, p)
    mean_risk = s1 / s0[:, None]                      # weighted risk-set means

    grad = torch.sum(event[:, None] * (X - mean_risk), dim=0)

    # w_j = sum over events i in risk set of j of 1/s0_i  (reverse prefix sum)
    ev_over_s0 = event / s0
    w = torch.flip(torch.cumsum(torch.flip(ev_over_s0, [0]), dim=0), [0])
    A = (X * (r * w)[:, None]).T @ X                  # X' diag(r w) X
    M = mean_risk[event.bool()]                       # (n_events, p)
    B = M.T @ M
    info = A - B                                      # observed information
    return grad, info


def newton(grad_hess_fn, X, event, p, device, dtype, tol=1e-8, max_iter=100):
    """Newton-Raphson driven by a (gradient, information) callback."""
    beta = torch.zeros(p, device=device, dtype=dtype)
    iters = 0
    for iters in range(1, max_iter + 1):
        grad, info = grad_hess_fn(beta)
        step = torch.linalg.solve(info, grad)
        beta = beta + step
        if torch.linalg.vector_norm(step) < tol:
            break
    return beta, iters


def autodiff_grad_hess(beta, X, event):
    """Score and observed information via torch.autograd, in the same sign
    convention as analytic_grad_hess: the loss is the negative log partial
    likelihood, so its gradient is the negated score and its Hessian is the
    observed information."""
    b = beta.detach().clone().requires_grad_(True)
    loss = neg_log_partial_likelihood(b, X, event)
    g = torch.autograd.grad(loss, b, create_graph=True)[0]
    rows = [torch.autograd.grad(g[k], b, retain_graph=True)[0] for k in range(g.numel())]
    hess = torch.stack(rows)
    return -g.detach(), hess.detach()


def fit_lbfgs(X, event, p, device, dtype, max_iter=100, tol=1e-8):
    """L-BFGS on the loss with autograd gradients (no explicit Hessian)."""
    beta = torch.zeros(p, device=device, dtype=dtype, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=max_iter, tolerance_grad=tol,
                            tolerance_change=tol * 1e-2, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = neg_log_partial_likelihood(beta, X, event)
        loss.backward()
        return loss

    opt.step(closure)
    return beta.detach(), opt.state[beta]["n_iter"]


def timed(fn, device, repeats):
    """Median wall-clock over `repeats` runs after one warm-up."""
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return median(times), out


def run_case(n, p, seed, device, dtype, repeats):
    X, event, n_events = make_data(n, p, seed, device, dtype)
    tol = 1e-8 if dtype == torch.float64 else 1e-4   # float32 can't reach 1e-8

    # Correctness: analytic derivatives must match autograd before we trust timing.
    b0 = torch.zeros(p, device=device, dtype=dtype)
    ga, ha = analytic_grad_hess(b0, X, event)
    gd, hd = autodiff_grad_hess(b0, X, event)
    grad_err = torch.linalg.vector_norm(ga - gd).item()
    hess_err = torch.linalg.matrix_norm(ha - hd).item()

    t_an, (beta_an, it_an) = timed(
        lambda: newton(lambda b: analytic_grad_hess(b, X, event), X, event, p, device, dtype, tol=tol),
        device, repeats)
    t_ad, (beta_ad, it_ad) = timed(
        lambda: newton(lambda b: autodiff_grad_hess(b, X, event), X, event, p, device, dtype, tol=tol),
        device, repeats)
    t_lb, (beta_lb, it_lb) = timed(
        lambda: fit_lbfgs(X, event, p, device, dtype, tol=tol), device, repeats)

    ref = beta_an
    return [
        dict(n=n, p=p, n_events=n_events, method="newton-analytic",
             seconds=t_an, iters=it_an, beta_err=0.0,
             grad_check=grad_err, hess_check=hess_err),
        dict(n=n, p=p, n_events=n_events, method="newton-autodiff",
             seconds=t_ad, iters=it_ad,
             beta_err=torch.linalg.vector_norm(beta_ad - ref).item(),
             grad_check=grad_err, hess_check=hess_err),
        dict(n=n, p=p, n_events=n_events, method="lbfgs-autodiff",
             seconds=t_lb, iters=it_lb,
             beta_err=torch.linalg.vector_norm(beta_lb - ref).item(),
             grad_check=grad_err, hess_check=hess_err),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, nargs="+", default=[20, 100, 500])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "mps" else torch.float64

    print(f"device={device}  dtype={dtype}  n={args.n}  repeats={args.repeats}")
    header = f"{'p':>6} {'method':>16} {'sec':>10} {'iters':>6} {'speedup':>8} " \
             f"{'beta_err':>10} {'grad_chk':>10} {'hess_chk':>10}"
    print(header)

    rows = []
    for p in args.p:
        case = run_case(args.n, p, args.seed, device, dtype, args.repeats)
        base = case[0]["seconds"]
        for r in case:
            speed = r["seconds"] / base  # how many times slower than analytic
            print(f"{r['p']:>6} {r['method']:>16} {r['seconds']:>10.4f} "
                  f"{r['iters']:>6} {speed:>7.1f}x {r['beta_err']:>10.2e} "
                  f"{r['grad_check']:>10.2e} {r['hess_check']:>10.2e}")
            r["autodiff_over_analytic"] = speed
            rows.append(r)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
