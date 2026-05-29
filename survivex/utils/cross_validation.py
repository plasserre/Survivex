"""
Cross-validation utilities for survival models.

Provides K-fold cross-validation with held-out c-index evaluation,
the standard approach for honest predictive evaluation of Cox PH and
related survival models. Without held-out CV, in-sample c-index is
upward-biased — especially at p > n_events where the model can fit
training data perfectly while generalising poorly.

Use this when you care about *predictive performance*. For benchmarking
backend equivalence (CPU vs GPU), in-sample c-index is sufficient and
this utility is not needed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CVResult:
    """Held-out c-index across K folds."""

    fold_c_indices: list[float]
    mean: float
    std: float
    n_folds: int


def _fit_clone(model, X_train, T_train, E_train, strata_train=None):
    """Fit a fresh copy of `model` on training data."""
    m = deepcopy(model)
    if strata_train is not None:
        m.fit(X_train, T_train, E_train, strata_train)
    else:
        m.fit(X_train, T_train, E_train)
    return m


def _predict_risk(model, X_test):
    """Return risk scores (higher = higher hazard) for c-index."""
    # Cox PH-family: risk score is exp(β'X). The c-index is invariant
    # to monotone transforms, so β'X (linear predictor) is equivalent
    # and avoids overflow at very large β. We use β'X directly.
    if hasattr(model, "coefficients_") and model.coefficients_ is not None:
        return np.asarray(X_test) @ np.asarray(model.coefficients_)
    if hasattr(model, "predict_partial_hazard"):
        return np.asarray(model.predict_partial_hazard(X_test)).ravel()
    if hasattr(model, "predict_risk"):
        return np.asarray(model.predict_risk(X_test)).ravel()
    raise ValueError(
        f"Cannot extract risk scores from {type(model).__name__}: "
        "expected `coefficients_`, `predict_partial_hazard`, or `predict_risk`."
    )


def _harrell_c_index(T, E, risk_scores):
    """
    Harrell's concordance index for right-censored survival data.

    Iterates over all comparable pairs (i, j): i had an event at time
    T_i, and j was either uncensored with T_j > T_i, or censored with
    T_j >= T_i. Concordant if risk_i > risk_j; tied risks contribute 0.5.
    Returns NaN if there are no comparable pairs.
    """
    T = np.asarray(T, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64).astype(bool)
    r = np.asarray(risk_scores, dtype=np.float64)

    n = len(T)
    if n < 2:
        return float("nan")

    concordant = 0.0
    total = 0.0
    # O(n^2) — fine for K-fold test sets (typically <1000 patients).
    for i in range(n):
        if not E[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if T[j] > T[i] or (T[j] == T[i] and not E[j]):
                total += 1.0
                if r[i] > r[j]:
                    concordant += 1.0
                elif r[i] == r[j]:
                    concordant += 0.5

    if total == 0:
        return float("nan")
    return concordant / total


def cross_validate_cindex(
    model,
    X,
    T,
    E,
    strata=None,
    k: int = 5,
    seed: int = 42,
    shuffle: bool = True,
) -> CVResult:
    """
    K-fold cross-validation with held-out Harrell c-index.

    Parameters
    ----------
    model : survivex.models.CoxPHModel | StratifiedCoxPHModel | similar
        An UNFITTED model instance configured with all hyperparameters
        (tie_method, device, penalty, etc.). A fresh copy is fit per fold.
    X : array, shape (n, p)
        Covariate matrix.
    T : array, shape (n,)
        Event or censoring times.
    E : array, shape (n,)
        Event indicators (1 = event, 0 = censored).
    strata : array, shape (n,), optional
        Stratification labels. If provided, the model is assumed to
        accept a `strata` argument in `fit()` (e.g. StratifiedCoxPHModel).
    k : int, default 5
        Number of folds.
    seed : int, default 42
        Random seed for the fold split (if shuffle=True).
    shuffle : bool, default True
        Whether to shuffle indices before splitting.

    Returns
    -------
    CVResult
        With `fold_c_indices`, `mean`, `std`, and `n_folds` populated.

    Notes
    -----
    Use the held-out c-index for predictive-performance claims. For
    backend-equivalence checks (CPU vs GPU agreement), in-sample
    c-index from a single fit is sufficient and faster.

    Strata are NOT used for stratified sampling — folds are simple
    random splits of the rows. If you need stratified splits (e.g. to
    keep cancer-type proportions balanced), pre-compute fold indices
    externally and call the underlying fit/predict yourself.
    """
    X = np.asarray(X)
    T = np.asarray(T, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    n = len(T)
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n]; got k={k}, n={n}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    fold_size = n // k
    extras = n % k

    fold_starts = []
    cursor = 0
    for fold in range(k):
        size = fold_size + (1 if fold < extras else 0)
        fold_starts.append((cursor, cursor + size))
        cursor += size

    fold_cs: list[float] = []
    for start, stop in fold_starts:
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        E_train, E_test = E[train_idx], E[test_idx]
        strata_train = strata[train_idx] if strata is not None else None

        fitted = _fit_clone(model, X_train, T_train, E_train, strata_train)
        risk_test = _predict_risk(fitted, X_test)
        c = _harrell_c_index(T_test, E_test, risk_test)
        fold_cs.append(float(c))

    finite = [c for c in fold_cs if np.isfinite(c)]
    mean = float(np.mean(finite)) if finite else float("nan")
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    return CVResult(fold_c_indices=fold_cs, mean=mean, std=std, n_folds=k)
