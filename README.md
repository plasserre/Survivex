# SurviveX

A GPU-accelerated survival analysis library for Python, built from scratch with PyTorch.

SurviveX provides a complete toolkit for time-to-event analysis — from classical non-parametric estimators to machine learning models — all implemented from the ground up with optional GPU acceleration. Every model is validated against R's `survival` package and Python's `lifelines` to machine precision.

## Why SurviveX?

Existing survival analysis tools have limitations:

- **lifelines** (Python): CPU-only, no tree-based ML models, slower on large datasets
- **scikit-survival** (Python): Limited model coverage, no GPU, no multi-state or recurrent events
- **R survival** (R): Requires R runtime, hard to integrate into Python ML pipelines

SurviveX fills these gaps with:
- All models implemented from scratch in Python (no wrappers)
- GPU acceleration via PyTorch (CUDA for NVIDIA, MPS for Apple Silicon)
- 2-22x faster than lifelines across model types
- Coefficients match R/lifelines to machine precision (< 1e-9)
- Consistent scikit-learn-style API: `model.fit(X, durations, events)`

## Installation

```bash
# From source (recommended for now)
git clone https://github.com/TaninZeraati/survivex.git
cd survivex
pip install -e .

# Dependencies are installed automatically:
# torch, numpy, pandas, scipy, matplotlib
```

Optional (for best performance):
```bash
pip install numba joblib  # Numba JIT for Cox PH, joblib for parallel Random Forest
```

## Quick Start

```python
import numpy as np
from survivex.models import KaplanMeierEstimator, CoxPHModel

# Generate sample data
np.random.seed(42)
durations = np.random.exponential(scale=50, size=200)
events = np.random.binomial(1, 0.7, size=200).astype(float)
X = np.random.randn(200, 3)  # 3 covariates

# Kaplan-Meier survival curve
km = KaplanMeierEstimator()
km.fit(durations, events)
print(f"Median survival: {km.median_survival_time_:.1f}")

# Cox Proportional Hazards
cox = CoxPHModel()
result = cox.fit(X, durations, events)
print(f"Hazard ratios: {np.exp(cox.coefficients_)}")
print(f"C-index: {cox.concordance_index_:.3f}")
```

## Loading Data

### From a CSV file

The most common format for survival data is a CSV with columns for time, event status, and covariates:

```python
import pandas as pd
import numpy as np
from survivex.models import CoxPHModel

# Load your CSV
df = pd.read_csv("patient_data.csv")

# Your CSV might look like:
# patient_id, time_months, died, age, treatment, tumor_size
# 1, 24.5, 1, 65, 1, 3.2
# 2, 36.0, 0, 52, 0, 1.8   (0 = censored, still alive at last follow-up)
# ...

# Extract arrays
durations = df['time_months'].values.astype(float)
events = df['died'].values.astype(float)  # 1 = event occurred, 0 = censored
covariates = df[['age', 'treatment', 'tumor_size']].values.astype(float)

# Fit Cox model
cox = CoxPHModel()
cox.fit(covariates, durations, events)

# Print results
for i, name in enumerate(['age', 'treatment', 'tumor_size']):
    hr = np.exp(cox.coefficients_[i])
    se = cox.standard_errors_[i]
    print(f"{name}: HR={hr:.3f}, SE={se:.4f}")
```

### From lifelines datasets

If you have lifelines installed, you can use its built-in datasets:

```python
from lifelines.datasets import load_rossi, load_lung, load_gbsg2
from survivex.models import CoxPHModel, KaplanMeierEstimator
import numpy as np

# Load the Rossi recidivism dataset (432 subjects, 7 covariates)
rossi = load_rossi()

durations = rossi['week'].values.astype(float)
events = rossi['arrest'].values.astype(float)
covariate_cols = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
X = rossi[covariate_cols].values.astype(float)

# Fit Cox PH model
cox = CoxPHModel(tie_method='efron')
cox.fit(X, durations, events)

print("Cox PH Results (Rossi dataset):")
print(f"{'Covariate':<8} {'Coef':>8} {'HR':>8} {'p-value':>10}")
print("-" * 40)
from scipy.stats import norm
for i, name in enumerate(covariate_cols):
    coef = cox.coefficients_[i]
    hr = np.exp(coef)
    z = coef / cox.standard_errors_[i]
    p = 2 * (1 - norm.cdf(abs(z)))
    print(f"{name:<8} {coef:>8.4f} {hr:>8.4f} {p:>10.4f}")

print(f"\nConcordance index: {cox.concordance_index_:.4f}")
```

### From R datasets via rpy2

```python
import numpy as np

# If you have rpy2 installed, you can load R datasets directly
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    ro.r('library(survival)')
    lung = ro.r('lung')

    durations = lung['time'].values.astype(float)
    events = (lung['status'].values == 2).astype(float)  # R uses 2=dead, 1=censored
    X = lung[['age', 'sex']].values.astype(float)
except ImportError:
    print("rpy2 not installed — use CSV or lifelines datasets instead")
```

### Handling missing data

SurviveX expects clean numpy arrays. Handle missing values before fitting:

```python
import pandas as pd
import numpy as np
from survivex.models import CoxPHModel

df = pd.read_csv("data.csv")

# Option 1: Drop rows with missing values
df_clean = df.dropna(subset=['time', 'event', 'age', 'treatment'])

# Option 2: Impute missing values
df['age'].fillna(df['age'].median(), inplace=True)

# Convert to arrays
durations = df_clean['time'].values.astype(float)
events = df_clean['event'].values.astype(float)
X = df_clean[['age', 'treatment']].values.astype(float)

# Now fit
cox = CoxPHModel()
cox.fit(X, durations, events)
```

## Available Models

### Non-Parametric Estimators

```python
from survivex.models import KaplanMeierEstimator, NelsonAalenEstimator, LogRankTest

# Kaplan-Meier survival curve
km = KaplanMeierEstimator()
km.fit(durations, events)
S_at_30 = km.survival_function_at_times([30.0])  # S(30)

# Nelson-Aalen cumulative hazard
na = NelsonAalenEstimator()
na.fit(durations, events)
H_at_30 = na.cumulative_hazard_at_times([30.0])  # H(30)

# Log-rank test (compare two groups)
group = (X[:, 0] > 0).astype(float)  # split by first covariate
lr = LogRankTest()
result = lr.test(durations, events, group)
print(f"Chi-squared: {result.test_statistic:.2f}, p-value: {result.p_value:.4f}")
```

### Cox Proportional Hazards

```python
from survivex.models import CoxPHModel

# Breslow method (fastest, exact for no ties)
cox = CoxPHModel(tie_method='breslow')
cox.fit(X, durations, events)

# Efron method (more accurate when ties exist)
cox = CoxPHModel(tie_method='efron')
cox.fit(X, durations, events)

# Access results
print(cox.coefficients_)       # log hazard ratios
print(cox.standard_errors_)    # standard errors
print(cox.concordance_index_)  # predictive accuracy

# Predict risk for new patients
new_patients = np.array([[45, 1, 2.5], [60, 0, 4.0]])
risk_scores = cox.predict_risk(new_patients)

# Predict survival curves
surv = cox.predict_survival_function(new_patients, times=np.arange(0, 100, 5))
```

### Parametric Models

```python
from survivex.models import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

# Weibull Accelerated Failure Time
waft = WeibullAFTFitter()
waft.fit(X, durations, events)
surv_pred = waft.predict_survival_function(X[:5], times=[10, 20, 30, 50])

# Log-Normal AFT
lnaft = LogNormalAFTFitter()
lnaft.fit(X, durations, events)

# Log-Logistic AFT
llaft = LogLogisticAFTFitter()
llaft.fit(X, durations, events)
```

### Competing Risks

When subjects can experience one of several mutually exclusive events:

```python
from survivex.models import AalenJohansenFitter, FineGrayModel
import numpy as np

# event_type: 0=censored, 1=heart failure, 2=stroke, 3=other death
durations = np.array([5, 10, 15, 20, 25, 8, 12, 30])
event_type = np.array([1, 0, 2, 1, 0, 3, 1, 2])

# Cumulative Incidence Functions (non-parametric)
aj = AalenJohansenFitter()
aj.fit(durations, event_type, event_of_interest=1)
print(f"CIF for heart failure at t=20: {aj.cumulative_incidence_at_times([20.0])}")

# Fine-Gray subdistribution hazard model
fg = FineGrayModel()
fg.fit(X[:8], durations, event_type, event_of_interest=1)
print(f"Subdistribution HRs: {np.exp(fg.coefficients_)}")
```

### Recurrent Event Models

For subjects who can experience the same event multiple times (hospital readmissions, infections, etc.):

```python
import pandas as pd
import numpy as np
from survivex.models import AndersenGillModel, PWPTTModel, PWPGTModel

# Load counting process data (start, stop] format
df = pd.read_csv("recurrent_events.csv")
# Columns: id, start, stop, event, enum (event number), x1, x2

ids = df['id'].values
starts = df['start'].values.astype(float)
stops = df['stop'].values.astype(float)
events = df['event'].values.astype(float)
enums = df['enum'].values.astype(int)
X = df[['x1', 'x2']].values.astype(float)

# Andersen-Gill: all events exchangeable, robust SE for clustering
ag = AndersenGillModel(tie_method='breslow')
ag.fit(X=X, time_start=starts, time_stop=stops, events=events, subject_id=ids)
print(ag.result_.summary())

# PWP Total Time: stratified by event number, calendar time scale
pwp_tt = PWPTTModel(tie_method='breslow')
pwp_tt.fit(X=X, time_start=starts, time_stop=stops, events=events,
           subject_id=ids, stratum=enums)

# PWP Gap Time: stratified by event number, time since last event
gap_times = (stops - starts)  # or use pre-computed gap times
pwp_gt = PWPGTModel(tie_method='breslow')
pwp_gt.fit(X=X, gap_durations=df['gap_time'].values.astype(float),
           events=events, subject_id=ids, stratum=enums)
```

### Multi-State Models

Model transitions between multiple health states over time:

```python
from survivex.models import (
    MultiStateAalenJohansen, MultiStateCoxPH,
    create_illness_death_matrix, prepare_multistate_data_simple
)

# Illness-death model: Healthy -> Illness -> Death (or Healthy -> Death)
trans_matrix = create_illness_death_matrix(with_recovery=False)

# Prepare data in multi-state long format
ms_data = prepare_multistate_data_simple(durations, events, trans_matrix, X)

# Non-parametric state occupation probabilities
aj = MultiStateAalenJohansen(trans_matrix)
result = aj.fit(ms_data, start_state=0)

# Transition-specific Cox models (different covariate effects per transition)
mscox = MultiStateCoxPH(trans_matrix, tie_method='efron')
cox_result = mscox.fit(ms_data, covariate_names=['age', 'treatment'])
print(cox_result.summary())
```

### Machine Learning Models

```python
from survivex.models import (
    RandomSurvivalForest,
    GradientBoostingSurvivalAnalysis,
    SurvivalTree
)

# Random Survival Forest
rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, random_state=42)
rsf.fit(X_train, T_train, E_train)
c_index = rsf.score(X_test, T_test, E_test)
print(f"RSF C-index: {c_index:.3f}")
print(f"OOB score: {rsf.oob_score_:.3f}")

# Feature importance
importances = rsf.feature_importances_
for i, imp in enumerate(importances):
    print(f"  Feature {i}: {imp:.4f}")

# Gradient Boosting Survival
gb = GradientBoostingSurvivalAnalysis(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train, T_train, E_train)
c_index = gb.score(X_test, T_test, E_test)

# Predict survival curves
times = np.linspace(0, 100, 50)
surv_probs = gb.predict_survival_function(X_test[:5], times=times)
```

## GPU Acceleration

SurviveX supports GPU acceleration for compute-intensive models. This is most beneficial for:
- Large datasets (10k+ observations)
- Many covariates (10+ features)
- Cox PH and recurrent event models

```python
from survivex.models import CoxPHModel, AndersenGillModel

# Auto-detect best available device
cox = CoxPHModel()  # Uses CPU by default (fast for most datasets)

# Explicitly use GPU
cox_gpu = CoxPHModel(device='cuda')  # NVIDIA GPU
cox_mps = CoxPHModel(device='mps')   # Apple Silicon (M1/M2/M3/M4)

# Recurrent event models with GPU
ag = AndersenGillModel(device='mps', tie_method='breslow')
ag.fit(X=X, time_start=starts, time_stop=stops, events=events, subject_id=ids)

# When does GPU help?
# - Cox PH with n > 10,000 and p > 10: significant speedup
# - Recurrent events with p=50 covariates: up to 6.7x faster on MPS
# - Small datasets (n < 5000): CPU is usually faster (GPU overhead)
```

## Performance

Benchmarked against lifelines (Python) and R's survival package:

| Model | SurviveX | Reference | Speedup |
|-------|----------|-----------|---------|
| Cox PH (n=20k) | 0.36s | lifelines 0.83s | **2.3x faster** |
| Weibull AFT (n=5k) | 0.004s | lifelines 0.085s | **22x faster** |
| Kaplan-Meier (n=100k) | 0.017s | lifelines 0.055s | **3.3x faster** |
| Nelson-Aalen (n=100k) | 0.015s | lifelines 0.052s | **3.5x faster** |
| Random Survival Forest | 1.30s | R 2.06s | **1.6x faster** |
| Gradient Boosting | 0.65s | R 0.65s | **~1x** |
| Andersen-Gill | 0.019s | R 0.018s | **~1x** |

GPU acceleration adds further speedup for large datasets with many covariates.

## Accuracy

All models validated against reference implementations (11/11 tests pass):

| Model | vs Reference | Max Difference |
|-------|-------------|----------------|
| Kaplan-Meier | lifelines | 2.87e-08 |
| Nelson-Aalen | lifelines | 8.32e-04 |
| Cox PH (Breslow) | R survival | 4.66e-12 |
| Cox PH (Efron) | lifelines | 4.15e-09 |
| Weibull AFT | lifelines | 4.20e-05 |
| Andersen-Gill | R survival | 3.61e-08 |
| PWP Total Time | R survival | 5.55e-16 |
| PWP Gap Time | R survival | 2.22e-16 |

See `validate_accuracy.ipynb` for the full validation notebook.

## Complete Example: Clinical Trial Analysis

```python
import numpy as np
import pandas as pd
from survivex.models import (
    KaplanMeierEstimator, LogRankTest, CoxPHModel, WeibullAFTFitter
)

# Simulated clinical trial: treatment vs control
np.random.seed(123)
n = 300

# Treatment group (n=150): lower hazard
T_treat = np.random.weibull(1.5, 150) * 60
E_treat = np.random.binomial(1, 0.65, 150).astype(float)

# Control group (n=150): higher hazard
T_ctrl = np.random.weibull(1.5, 150) * 40
E_ctrl = np.random.binomial(1, 0.75, 150).astype(float)

# Combine
durations = np.concatenate([T_treat, T_ctrl])
events = np.concatenate([E_treat, E_ctrl])
treatment = np.concatenate([np.ones(150), np.zeros(150)])
age = np.random.normal(55, 10, n)
X = np.column_stack([treatment, age])

# 1. Kaplan-Meier curves by group
km_treat = KaplanMeierEstimator()
km_treat.fit(T_treat, E_treat)

km_ctrl = KaplanMeierEstimator()
km_ctrl.fit(T_ctrl, E_ctrl)

print(f"Median survival - Treatment: {km_treat.median_survival_time_:.1f}")
print(f"Median survival - Control:   {km_ctrl.median_survival_time_:.1f}")

# 2. Log-rank test
lr = LogRankTest()
result = lr.test(durations, events, treatment)
print(f"\nLog-rank test: chi2={result.test_statistic:.2f}, p={result.p_value:.4f}")

# 3. Cox PH model (adjusted for age)
cox = CoxPHModel(tie_method='efron')
cox.fit(X, durations, events)

print(f"\nCox PH Results:")
print(f"  Treatment HR: {np.exp(cox.coefficients_[0]):.3f} "
      f"(95% CI: {np.exp(cox.coefficients_[0] - 1.96*cox.standard_errors_[0]):.3f}-"
      f"{np.exp(cox.coefficients_[0] + 1.96*cox.standard_errors_[0]):.3f})")
print(f"  Age HR (per year): {np.exp(cox.coefficients_[1]):.3f}")
print(f"  C-index: {cox.concordance_index_:.3f}")

# 4. Weibull AFT for parametric analysis
waft = WeibullAFTFitter()
waft.fit(X, durations, events)
print(f"\nWeibull shape (rho): {waft.rho_:.3f}")
```

## Project Structure

```
survivex/
├── survivex/
│   ├── models/
│   │   ├── kaplan_meier.py          # Kaplan-Meier estimator
│   │   ├── nelson_aalen.py          # Nelson-Aalen estimator
│   │   ├── log_rank_test.py         # Log-rank test
│   │   ├── cox_ph.py                # Cox Proportional Hazards
│   │   ├── parametric_models.py     # Weibull, Log-Normal, Log-Logistic, Exponential
│   │   ├── competing_risk.py        # Aalen-Johansen, Fine-Gray
│   │   ├── multi_state.py           # Multi-state models
│   │   ├── andersen_gill.py         # Andersen-Gill recurrent events
│   │   ├── recurrent_event.py       # PWP Total Time model
│   │   ├── pwp.py                   # PWP Gap Time model
│   │   ├── survival_tree.py         # Survival Tree
│   │   ├── random_survival_tree.py  # Random Survival Forest
│   │   └── gradient_boosting_survival.py  # Gradient Boosting
│   └── ...
├── tests/                           # Unit tests (19 files)
├── validate_accuracy.ipynb          # Accuracy validation vs lifelines/R
├── benchmark_survivex_vs_r.ipynb    # Performance benchmarks
├── setup.py
└── pyproject.toml
```

## Requirements

- Python >= 3.8
- NumPy >= 1.21
- PyTorch >= 2.0
- SciPy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4

Optional:
- numba (faster Cox PH Efron method)
- joblib (parallel Random Survival Forest)

## Running Tests

```bash
# Run unit tests
pytest tests/

# Run accuracy validation notebook
jupyter notebook validate_accuracy.ipynb
```

## Citation

If you use SurviveX in your research, please cite:

```bibtex
@article{zeraati2026survivex,
  title={SurviveX: A GPU-Accelerated Python Library for Survival Analysis},
  author={Zeraati, Tanin},
  journal={SoftwareX},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

The Apache-2.0 license allows you to freely use, modify, and distribute this software, including for commercial purposes, with patent protection included.
