# CPU benchmark (paper Table 3)

Reproduces Table 3: survivex vs lifelines fit times on the small standard
datasets. Kaplan-Meier, Nelson-Aalen and Cox PH are run on gbsg2; the Weibull,
Log-Normal and Log-Logistic AFT models are run on rossi (both loaded from
lifelines).

```bash
cd benchmarks/cpu_standard
python run_cpu_benchmark.py             # median of 21 runs, writes results/cpu_standard_timings.csv
python run_cpu_benchmark.py --n-runs 5  # quicker, noisier
```

Each fit is timed as the median over `--n-runs` repetitions after one warm-up
call. Timings are hardware-dependent; the paper figures are from an Apple M2
Pro (16 GB). `results/` is git-ignored.
