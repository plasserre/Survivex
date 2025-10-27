"""
Test Parametric Models against lifelines
"""

import numpy as np
import pandas as pd
from survivex.models.parametric_models import WeibullPHFitter, WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter
from lifelines import WeibullFitter, WeibullAFTFitter as LifelinesWeibullAFTFitter
from lifelines.datasets import load_rossi

from lifelines import LogLogisticAFTFitter as LifelinesLogLogisticAFTFitter

from lifelines import LogNormalAFTFitter as LifelinesLogNormalAFTFitter

print("=" * 80)
print("TEST 1: WEIBULL PH - NO COVARIATES")
print("=" * 80)

rossi = load_rossi()
T = rossi['week'].values
E = rossi['arrest'].values

wph = WeibullPHFitter()
wph.fit(T, E, X=None)

wf_ll = WeibullFitter()
wf_ll.fit(T, E)

print(f"\nOur Model:")
print(f"  rho: {wph.rho_:.6f}")
print(f"  lambda_effective: {wph.lambda_ * np.exp(-wph.coef_[0] / wph.rho_):.6f}")
print(f"  log-likelihood: {wph.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  rho: {wf_ll.rho_:.6f}")
print(f"  lambda: {wf_ll.lambda_:.6f}")

# Test survival predictions
times = np.array([10, 20, 30, 50])
S_ours = wph.predict_survival_function(X=None, times=times)
S_ll = wf_ll.survival_function_at_times(times).values

print(f"\nSurvival predictions at t={times}:")
print(f"  Ours:     {S_ours}")
print(f"  Lifelines: {S_ll}")
print(f"  Max diff: {np.max(np.abs(S_ours - S_ll)):.8f}")

print("\n" + "=" * 80)
print("TEST 2: WEIBULL AFT - NO COVARIATES")
print("=" * 80)

waft = WeibullAFTFitter()
waft.fit(T, E, X=None)

waft_ll = LifelinesWeibullAFTFitter()
waft_ll.fit(rossi[['week', 'arrest']], duration_col='week', event_col='arrest')

print(f"\nOur Model:")
print(f"  Intercept: {waft.lambda_params_[0]:.6f}")
print(f"  rho: {waft.rho_:.6f}")
print(f"  log-likelihood: {waft.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  Intercept: {waft_ll.params_['lambda_', 'Intercept']:.6f}")
print(f"  rho: {np.exp(waft_ll.params_['rho_', 'Intercept']):.6f}")

S_ours = waft.predict_survival_function(X=None, times=times)
S_ll_df = waft_ll.predict_survival_function(rossi.iloc[0:1][[]])
S_ll = np.array([S_ll_df.loc[t, :].values[0] for t in times])

print(f"\nSurvival predictions at t={times}:")
print(f"  Ours:      {S_ours}")
print(f"  Lifelines: {S_ll}")
print(f"  Max diff: {np.max(np.abs(S_ours - S_ll)):.8f}")

print("\n" + "=" * 80)
print("TEST 3: WEIBULL AFT - WITH COVARIATES")
print("=" * 80)

covariates = ['age', 'prio']
X = rossi[covariates].values

waft_cov = WeibullAFTFitter()
waft_cov.fit(T, E, X)

waft_ll_cov = LifelinesWeibullAFTFitter()
waft_ll_cov.fit(rossi[['week', 'arrest'] + covariates], duration_col='week', event_col='arrest')

print(f"\nOur Model:")
print(f"  Intercept: {waft_cov.lambda_params_[0]:.6f}")
print(f"  age coef:  {waft_cov.lambda_params_[1]:.6f}")
print(f"  prio coef: {waft_cov.lambda_params_[2]:.6f}")
print(f"  rho:       {waft_cov.rho_:.6f}")
print(f"  log-likelihood: {waft_cov.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  Intercept: {waft_ll_cov.params_['lambda_', 'Intercept']:.6f}")
print(f"  age coef:  {waft_ll_cov.params_['lambda_', 'age']:.6f}")
print(f"  prio coef: {waft_ll_cov.params_['lambda_', 'prio']:.6f}")
print(f"  rho:       {np.exp(waft_ll_cov.params_['rho_', 'Intercept']):.6f}")

# Test predictions for first 3 individuals
for i in range(3):
    sample_X = X[i:i+1, :]
    sample_df = rossi.iloc[i:i+1][covariates]
    
    median_ours = waft_cov.predict_median(sample_X)
    median_ll = waft_ll_cov.predict_median(sample_df).values[0]
    
    print(f"\nIndividual {i+1} (age={X[i,0]:.0f}, prio={X[i,1]:.0f}):")
    print(f"  Median - Ours: {median_ours:.2f}, Lifelines: {median_ll:.2f}, Diff: {abs(median_ours - median_ll):.6f}")

print("\n" + "=" * 80)
print("TEST 4: SYNTHETIC DATA WITH KNOWN PARAMETERS")
print("=" * 80)

np.random.seed(42)
n = 500
true_rho = 1.5
true_lambda = 50.0

U = np.random.uniform(0, 1, n)
T_synth = true_lambda * (-np.log(U)) ** (1/true_rho)
C = np.random.exponential(1/0.015, n)
T_obs = np.minimum(T_synth, C)
E_obs = (T_synth <= C).astype(int)

print(f"True parameters: rho={true_rho}, lambda={true_lambda}")
print(f"Censoring rate: {(1-E_obs.mean())*100:.1f}%")

wph_synth = WeibullPHFitter()
wph_synth.fit(T_obs, E_obs, X=None)

wf_synth = WeibullFitter()
wf_synth.fit(T_obs, E_obs)

# Our effective lambda
lambda_eff = wph_synth.lambda_ * np.exp(-wph_synth.coef_[0] / wph_synth.rho_)

print(f"\nOur estimates:")
print(f"  rho: {wph_synth.rho_:.4f} (error: {abs(wph_synth.rho_ - true_rho):.4f})")
print(f"  lambda_effective: {lambda_eff:.4f} (error: {abs(lambda_eff - true_lambda):.4f})")

print(f"\nLifelines estimates:")
print(f"  rho: {wf_synth.rho_:.4f} (error: {abs(wf_synth.rho_ - true_rho):.4f})")
print(f"  lambda: {wf_synth.lambda_:.4f} (error: {abs(wf_synth.lambda_ - true_lambda):.4f})")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE")
print("=" * 80)

print("\n" + "=" * 80)
print("TEST 5: LOG-NORMAL AFT - NO COVARIATES")
print("=" * 80)



lnaft = LogNormalAFTFitter()
lnaft.fit(T, E, X=None)

lnaft_ll = LifelinesLogNormalAFTFitter()
lnaft_ll.fit(rossi[['week', 'arrest']], duration_col='week', event_col='arrest')

print(f"\nOur Model:")
print(f"  Intercept: {lnaft.lambda_params_[0]:.6f}")
print(f"  sigma: {lnaft.sigma_:.6f}")
print(f"  log-likelihood: {lnaft.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  Intercept: {lnaft_ll.params_['mu_', 'Intercept']:.6f}")
print(f"  sigma: {np.exp(lnaft_ll.params_['sigma_', 'Intercept']):.6f}")

times = np.array([10, 20, 30, 50])
S_ours = lnaft.predict_survival_function(X=None, times=times)
S_ll_df = lnaft_ll.predict_survival_function(rossi.iloc[0:1][[]])
S_ll = np.array([S_ll_df.loc[t, :].values[0] for t in times])

print(f"\nSurvival predictions at t={times}:")
print(f"  Ours:      {S_ours}")
print(f"  Lifelines: {S_ll}")
print(f"  Max diff: {np.max(np.abs(S_ours - S_ll)):.8f}")

print("\n" + "=" * 80)
print("TEST 6: LOG-NORMAL AFT - WITH COVARIATES")
print("=" * 80)

lnaft_cov = LogNormalAFTFitter()
lnaft_cov.fit(T, E, X)

lnaft_ll_cov = LifelinesLogNormalAFTFitter()
lnaft_ll_cov.fit(rossi[['week', 'arrest'] + covariates], duration_col='week', event_col='arrest')

print(f"\nOur Model:")
print(f"  Intercept: {lnaft_cov.lambda_params_[0]:.6f}")
print(f"  age coef:  {lnaft_cov.lambda_params_[1]:.6f}")
print(f"  prio coef: {lnaft_cov.lambda_params_[2]:.6f}")
print(f"  sigma:     {lnaft_cov.sigma_:.6f}")
print(f"  log-likelihood: {lnaft_cov.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  Intercept: {lnaft_ll_cov.params_['mu_', 'Intercept']:.6f}")
print(f"  age coef:  {lnaft_ll_cov.params_['mu_', 'age']:.6f}")
print(f"  prio coef: {lnaft_ll_cov.params_['mu_', 'prio']:.6f}")
print(f"  sigma:     {np.exp(lnaft_ll_cov.params_['sigma_', 'Intercept']):.6f}")

for i in range(3):
    sample_X = X[i:i+1, :]
    sample_df = rossi.iloc[i:i+1][covariates]
    
    median_ours = lnaft_cov.predict_median(sample_X)
    median_ll = lnaft_ll_cov.predict_median(sample_df).values[0]
    
    print(f"\nIndividual {i+1} (age={X[i,0]:.0f}, prio={X[i,1]:.0f}):")
    print(f"  Median - Ours: {median_ours:.2f}, Lifelines: {median_ll:.2f}, Diff: {abs(median_ours - median_ll):.6f}")

print("\n" + "=" * 80)
print("TEST 7: LOG-LOGISTIC AFT - NO COVARIATES")
print("=" * 80)



llaft = LogLogisticAFTFitter()
llaft.fit(T, E, X=None)

llaft_ll = LifelinesLogLogisticAFTFitter()
llaft_ll.fit(rossi[['week', 'arrest']], duration_col='week', event_col='arrest')

print(f"\nOur Model:")
print(f"  Intercept: {llaft.lambda_params_[0]:.6f}")
print(f"  alpha (1/sigma): {llaft.alpha_:.6f}")
print(f"  log-likelihood: {llaft.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  Intercept: {llaft_ll.params_['alpha_', 'Intercept']:.6f}")
print(f"  alpha: {np.exp(llaft_ll.params_['beta_', 'Intercept']):.6f}")

S_ours = llaft.predict_survival_function(X=None, times=times)
S_ll_df = llaft_ll.predict_survival_function(rossi.iloc[0:1][[]])
S_ll = np.array([S_ll_df.loc[t, :].values[0] for t in times])

print(f"\nSurvival predictions at t={times}:")
print(f"  Ours:      {S_ours}")
print(f"  Lifelines: {S_ll}")
print(f"  Max diff: {np.max(np.abs(S_ours - S_ll)):.8f}")

print("\n" + "=" * 80)
print("TEST 8: LOG-LOGISTIC AFT - WITH COVARIATES")
print("=" * 80)

llaft_cov = LogLogisticAFTFitter()
llaft_cov.fit(T, E, X)

llaft_ll_cov = LifelinesLogLogisticAFTFitter()
llaft_ll_cov.fit(rossi[['week', 'arrest'] + covariates], duration_col='week', event_col='arrest')

print(f"\nOur Model:")
print(f"  Intercept: {llaft_cov.lambda_params_[0]:.6f}")
print(f"  age coef:  {llaft_cov.lambda_params_[1]:.6f}")
print(f"  prio coef: {llaft_cov.lambda_params_[2]:.6f}")
print(f"  alpha:     {llaft_cov.alpha_:.6f}")
print(f"  log-likelihood: {llaft_cov.log_likelihood_:.4f}")

print(f"\nLifelines:")
print(f"  Intercept: {llaft_ll_cov.params_['alpha_', 'Intercept']:.6f}")
print(f"  age coef:  {llaft_ll_cov.params_['alpha_', 'age']:.6f}")
print(f"  prio coef: {llaft_ll_cov.params_['alpha_', 'prio']:.6f}")
print(f"  alpha:     {np.exp(llaft_ll_cov.params_['beta_', 'Intercept']):.6f}")

for i in range(3):
    sample_X = X[i:i+1, :]
    sample_df = rossi.iloc[i:i+1][covariates]
    
    median_ours = llaft_cov.predict_median(sample_X)
    median_ll = llaft_ll_cov.predict_median(sample_df).values[0]
    
    print(f"\nIndividual {i+1} (age={X[i,0]:.0f}, prio={X[i,1]:.0f}):")
    print(f"  Median - Ours: {median_ours:.2f}, Lifelines: {median_ll:.2f}, Diff: {abs(median_ours - median_ll):.6f}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETE - 8 TESTS")
print("=" * 80)