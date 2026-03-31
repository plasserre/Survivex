# SurviveX

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/plasserre/Survivex/main?labpath=examples%2Fgetting_started.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
git clone https://github.com/plasserre/Survivex.git
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
print(f"Median survival: {km.median_survival_time():.1f}")

# Cox Proportional Hazards
cox = CoxPHModel()
result = cox.fit(X, durations, events)
print(f"Hazard ratios: {np.exp(cox.coefficients_)}")
print(f"C-index: {cox.concordance_index_:.3f}")
```

## Loading Data

SurviveX provides built-in tools for loading, validating, and preparing survival data from any source. You can either use the **universal loader** (recommended) or work with numpy arrays directly.

### Universal Loader (recommended)

The `load_survival_dataset` function handles everything: loading, validation, missing values, categorical encoding, and format conversion:

```python
from survivex.datasets.loaders import load_survival_dataset

# From a CSV file — auto-detects time and event columns
data = load_survival_dataset("patient_data.csv", verbose=True)

# Explicitly specify columns
data = load_survival_dataset(
    "patient_data.csv",
    time_col="time_months",
    event_col="died",
    feature_cols=["age", "treatment", "tumor_size"],
    handle_missing="drop",   # or 'impute', 'warn', 'raise'
    validate=True,
    verbose=True
)

# The result is a SurvivalData object — use it directly with models
from survivex.models import CoxPHModel
cox = CoxPHModel()
cox.fit(data.X.numpy(), data.time.numpy(), data.event.numpy().astype(float))
```

The loader automatically:
- Validates data (checks for non-positive times, missing values, event rates)
- Handles missing values (drop, impute with median/mean, or raise)
- Encodes categorical features (label encoding, one-hot, or drop)
- Converts event columns (handles 0/1, 1/2, True/False, Dead/Alive, Yes/No formats)
- Supports date ranges (`time_col=["start_date", "end_date"]` computes duration in days)

### From a CSV file (manual)

If you prefer working with numpy arrays directly:

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

# Extract arrays
durations = df['time_months'].values.astype(float)
events = df['died'].values.astype(float)  # 1 = event occurred, 0 = censored
covariates = df[['age', 'treatment', 'tumor_size']].values.astype(float)

# Fit Cox model
cox = CoxPHModel()
cox.fit(covariates, durations, events)
```

### From lifelines datasets

```python
from survivex.datasets.loaders import load_survival_dataset

# Load any lifelines dataset by name (auto-detected)
data = load_survival_dataset(
    "rossi",                    # Dataset name — auto-loads from lifelines
    time_col="week",
    event_col="arrest",
    feature_cols=["fin", "age", "race", "wexp", "prio"],
    verbose=True
)

# Available datasets: rossi, lung, waltons, kidney, stanford, dd, regression
```

Or load manually with lifelines:

```python
from lifelines.datasets import load_rossi
from survivex.models import CoxPHModel
import numpy as np

rossi = load_rossi()
durations = rossi['week'].values.astype(float)
events = rossi['arrest'].values.astype(float)
X = rossi[['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']].values.astype(float)

cox = CoxPHModel(tie_method='efron')
cox.fit(X, durations, events)
print(f"C-index: {cox.concordance_index_:.4f}")
```

### Using SurvivalData objects

The `SurvivalData` class provides a validated container for survival data with automatic type conversion:

```python
from survivex.core.data import SurvivalData

# From arrays
data = SurvivalData(
    time=[5, 10, 15, 20, 25],
    event=[1, 0, 1, 1, 0],
    X=[[65, 1], [70, 0], [55, 1], [60, 0], [75, 1]],
    feature_names=['age', 'sex']
)
print(data)  # SurvivalData(n_obs=5, n_events=3, event_rate=60.0%, n_features=2)

# From a pandas DataFrame (auto-detects numeric features)
import pandas as pd
df = pd.DataFrame({
    'time': [12, 24, 36, 48, 60],
    'event': [1, 0, 1, 1, 0],
    'age': [55.0, 60.0, 45.0, 70.0, 50.0],
    'bmi': [22.0, 28.0, 25.0, 30.0, 24.0],
})
data = SurvivalData.from_pandas(df, time_col='time', event_col='event')
print(data.feature_names)  # ['age', 'bmi'] — auto-detected

# Convert back to pandas
df_out = data.to_pandas()
```

`SurvivalData` validates on creation — it raises errors for:
- Non-positive survival times
- Event values not in {0, 1}
- Mismatched array lengths

### Working with date columns

If your data has start/end dates instead of durations:

```python
from survivex.datasets.loaders import load_survival_dataset

data = load_survival_dataset(
    "clinical_trial.csv",
    time_col=["enrollment_date", "last_followup_date"],  # Computes days between dates
    event_col="died",
    feature_cols=["age", "treatment_arm"],
    verbose=True
)
# Duration is automatically computed in days
```

### Data validation

Validate your data before fitting models to catch common issues:

```python
from survivex.datasets.validators import validate_survival_data
import pandas as pd

df = pd.read_csv("my_data.csv")

result = validate_survival_data(
    df,
    time_col='survival_months',
    event_col='status',
    feature_cols=['age', 'stage', 'treatment'],
    verbose=True
)
# Prints a validation report:
#   - Checks for non-positive times
#   - Checks for missing values
#   - Warns about low event rates
#   - Warns about outliers
#   - Reports data summary statistics

if result['valid']:
    print("Data is ready for analysis")
else:
    print(f"Issues found: {result['issues']}")
```

### Converting recurrent event data

Transform repeated events data into counting process (start, stop] format for Andersen-Gill or PWP models:

```python
from survivex.datasets.converters import convert_recurrent_events
from survivex.models import AndersenGillModel
import pandas as pd

# Your data: one row per event per patient
df = pd.DataFrame({
    'patient_id': [1, 1, 1, 2, 2, 3],
    'time_days':  [10, 25, 40, 15, 35, 20],
    'readmitted': [1, 1, 0, 1, 0, 0],  # 1=readmitted, 0=censored
    'age':        [65, 65, 65, 50, 50, 70],
    'severity':   [2, 3, 3, 1, 2, 1]
})

# Convert to counting process format
cp_data = convert_recurrent_events(
    df, subject_col='patient_id', time_col='time_days', event_col='readmitted'
)
# Result has: patient_id, start, stop, event, age, severity

# Fit Andersen-Gill model
ag = AndersenGillModel(tie_method='breslow')
ag.fit(
    X=cp_data[['age', 'severity']].values.astype(float),
    time_start=cp_data['start'].values.astype(float),
    time_stop=cp_data['stop'].values.astype(float),
    events=cp_data['event'].values.astype(float),
    subject_id=cp_data['patient_id'].values
)
print(ag.result_.summary())
```

### Handling missing data

The universal loader handles missing values automatically, or you can do it manually:

```python
from survivex.datasets.loaders import load_survival_dataset

# Option 1: Drop rows with missing values
data = load_survival_dataset("data.csv", handle_missing="drop")

# Option 2: Impute with median (numeric) / mode (categorical)
data = load_survival_dataset("data.csv", handle_missing="impute",
                             impute_strategy="median")

# Option 3: Raise error if any missing values found
data = load_survival_dataset("data.csv", handle_missing="raise")
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

### Frailty Models

Account for unobserved heterogeneity (random effects) in clustered or recurrent event data:

```python
from survivex.models import FrailtyModel
import numpy as np

# Clustered data: multiple observations per subject
# e.g., kidney dialysis data with 2 observations per patient

# Prepare data
durations = np.array([8, 16, 23, 13, 22, 28, ...])  # Event times
events = np.array([1, 1, 1, 0, 1, 1, ...])           # 1=event, 0=censored
cluster_id = np.array([1, 1, 2, 2, 3, 3, ...])       # Subject/cluster ID
X = np.array([[28, 1, 0, 0, 0],                      # Covariates
              [28, 1, 0, 0, 0],
              [48, 2, 1, 0, 0], ...])

# Gamma frailty (most common choice)
# Model: h_i(t) = z_g * h_0(t) * exp(beta' X_i)
# where z_g ~ Gamma(1/theta, 1/theta) with E[z]=1, Var[z]=theta
frailty_gamma = FrailtyModel(distribution='gamma', tie_method='breslow')
frailty_gamma.fit(X, durations, events, cluster_id)
print(frailty_gamma.result_.summary())

# Access results
print(f"Coefficients: {frailty_gamma.coefficients_}")
print(f"Frailty variance (theta): {frailty_gamma.frailty_variance_:.4f}")
print(f"Cluster frailties: {frailty_gamma.frailty_values_}")

# Gaussian (log-normal) frailty
# log(z_g) ~ N(0, sigma^2)
frailty_gauss = FrailtyModel(distribution='gaussian', tie_method='breslow')
frailty_gauss.fit(X, durations, events, cluster_id)
```

The frailty model uses an EM algorithm to estimate both the regression coefficients
and the frailty variance. Gamma frailty is recommended for most applications.

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

All models validated against reference implementations (12/12 tests pass):

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
| Frailty (Gamma) | R survival | 3.59e-02* |

*Frailty model uses EM algorithm which may converge to slightly different optimum than R's penalized likelihood.

See `validate_accuracy.ipynb` for the full validation notebook.


## Project Structure

```
survivex/
├── survivex/
│   ├── core/
│   │   └── data.py                  # SurvivalData class (validated container)
│   ├── datasets/
│   │   ├── loaders.py               # Universal dataset loader
│   │   ├── validators.py            # Data validation utilities
│   │   └── converters.py            # Format converters (lifelines, sksurv, pycox)
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
│   │   ├── frailty.py               # Frailty models (gamma, gaussian)
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

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

The Apache-2.0 license allows you to freely use, modify, and distribute this software, including for commercial purposes, with patent protection included.
