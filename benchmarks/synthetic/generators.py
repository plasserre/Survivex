"""
Synthetic data generators for the Table 3 / Table 4 benchmarks.

These are the *version of record* generators used to produce the timing and
scaling numbers in the submitted SoftwareX paper. They were originally defined
inline in the ``benchmark_gpu_scaling.ipynb`` notebook (cell 3) on the GPU
workstation; this module ports them verbatim so the same numbers are
reproducible from a CLI script (see ``run_benchmark.py``).

Every generator takes an explicit ``seed`` (default ``42``, the value used for
the paper) and seeds NumPy's global RNG at the top, so a given (size, seed)
always yields byte-identical data. The return dicts intentionally mirror the
notebook's schema (``X``/``T``/``E``/``features``/... keys) so the benchmark
harness can stay generic across model families.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "generate_large_synthetic",
    "generate_recurrent_data",
    "generate_competing_risks_data",
    "generate_clustered_data",
]


def generate_large_synthetic(n, p, seed=42):
    """Cox PH / Weibull AFT / GBM / RSF scaling data.

    beta ~ U(-0.5, 0.5); exponential survival times with ~20% administrative
    censoring at the 80th percentile of T.
    """
    np.random.seed(seed)
    X = np.random.randn(n, p).astype(np.float64)
    beta = np.random.uniform(-0.5, 0.5, p)
    scale = np.exp(-X @ beta) / 0.01
    T = np.random.exponential(scale).astype(np.float64)
    censor = np.percentile(T, 80)
    E = (T <= censor).astype(np.int32)
    T = np.minimum(T, censor)
    features = [f"X{i}" for i in range(p)]
    return {
        "name": f"Synthetic_{n // 1000}k_p{p}",
        "X": X, "T": T, "E": E,
        "features": features,
        "description": f"Synthetic data (n={n:,}, p={p})",
    }


def generate_recurrent_data(n_subjects, p, seed=42):
    """Recurrent-event data for AG / PWP-TT / PWP-GT with per-event strata.

    beta ~ U(-0.3, 0.3); per-subject gap times are exponential with rate
    proportional to the linear predictor, capped by an exponential censoring
    time (mean 50). ``strata`` is the event order (1, 2, ...) for PWP.
    """
    np.random.seed(seed)
    X_list, starts, stops, events, ids, strata, gap_times = [], [], [], [], [], [], []
    betas = np.random.uniform(-0.3, 0.3, p)

    for i in range(n_subjects):
        x = np.random.randn(p)
        risk = np.exp(x @ betas)
        t = 0.0
        event_num = 0
        max_events = np.random.randint(1, 8)
        censor_time = np.random.exponential(50)

        for _ in range(max_events):
            gap = np.random.exponential(10.0 / risk)
            new_t = t + gap
            event_num += 1
            if new_t > censor_time:
                starts.append(t)
                stops.append(censor_time)
                gap_times.append(censor_time - t)
                events.append(0)
                ids.append(i)
                strata.append(event_num)
                X_list.append(x)
                break
            else:
                starts.append(t)
                stops.append(new_t)
                gap_times.append(new_t - t)
                events.append(1)
                ids.append(i)
                strata.append(event_num)
                X_list.append(x)
                t = new_t

    return {
        "name": f"Recurrent_{n_subjects}subj_p{p}",
        "X": np.array(X_list, dtype=np.float64),
        "start": np.array(starts, dtype=np.float64),
        "stop": np.array(stops, dtype=np.float64),
        "gap_time": np.array(gap_times, dtype=np.float64),
        "E": np.array(events, dtype=np.int32),
        "id": np.array(ids, dtype=np.int32),
        "strata": np.array(strata, dtype=np.int32),
        "features": [f"X{i}" for i in range(p)],
        "description": f"Recurrent events ({n_subjects:,} subjects)",
    }


def generate_competing_risks_data(n, p, seed=42):
    """Competing-risks data with two cause-specific event types (Fine-Gray).

    beta1, beta2 ~ U(-0.3, 0.3); two exponential cause-specific times plus an
    exponential censoring time; the earliest determines T and the event type
    (0 = censored, 1 = cause 1, 2 = cause 2).
    """
    np.random.seed(seed)
    X = np.random.randn(n, p).astype(np.float64)
    beta1 = np.random.uniform(-0.3, 0.3, p)
    beta2 = np.random.uniform(-0.3, 0.3, p)

    scale1 = np.exp(-X @ beta1) * 50
    scale2 = np.exp(-X @ beta2) * 80
    censor_scale = 100

    T1 = np.random.exponential(scale1)
    T2 = np.random.exponential(scale2)
    C = np.random.exponential(censor_scale, n)

    T = np.minimum(np.minimum(T1, T2), C)
    event_type = np.zeros(n, dtype=np.int32)
    event_type[(T1 < T2) & (T1 < C)] = 1
    event_type[(T2 <= T1) & (T2 < C)] = 2

    return {
        "name": f"CompetingRisks_{n // 1000}k_p{p}",
        "X": X,
        "T": T.astype(np.float64),
        "event_type": event_type,
        "features": [f"X{i}" for i in range(p)],
        "description": f"Competing risks (n={n:,}, p={p})",
    }


def generate_clustered_data(n_clusters, obs_per_cluster, p, seed=42):
    """Clustered data for the gamma shared-frailty model.

    beta ~ U(-0.3, 0.3); cluster-specific Gamma(2, 0.5) frailties multiply the
    hazard; ~25% administrative censoring at the 75th percentile of T.
    """
    np.random.seed(seed)
    n = n_clusters * obs_per_cluster
    X = np.random.randn(n, p).astype(np.float64)
    cluster_ids = np.repeat(np.arange(n_clusters), obs_per_cluster)

    frailties = np.random.gamma(2, 0.5, n_clusters)
    obs_frailties = frailties[cluster_ids]

    beta = np.random.uniform(-0.3, 0.3, p)
    scale = np.exp(-X @ beta) * obs_frailties * 50
    T = np.random.exponential(scale).astype(np.float64)
    censor = np.percentile(T, 75)
    E = (T <= censor).astype(np.int32)
    T = np.minimum(T, censor)

    return {
        "name": f"Clustered_{n_clusters}cl_p{p}",
        "X": X,
        "T": T,
        "E": E,
        "cluster_id": cluster_ids,
        "features": [f"X{i}" for i in range(p)],
        "description": f"Clustered data ({n_clusters} clusters, {obs_per_cluster} obs each)",
    }
