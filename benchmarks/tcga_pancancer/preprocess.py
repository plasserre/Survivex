"""
Preprocess TCGA PanCancer Atlas cohorts for the survivex stratified Cox PH
benchmark.

Mathematical pipeline (executed once over all cohorts):

  Per cohort c:
    1. Load clinical data; drop patients with missing or non-positive OS.
    2. Load RNA-seq RSEM expression matrix (genes x samples).
    3. Collapse duplicate gene symbols by mean (different Entrez IDs).
    4. Keep one expression sample per patient: primary solid tumor (TCGA
       sample-type code 01); average if multiple primary samples exist.
    5. Inner-join clinical and expression on patient ID.

  Cross-cohort:
    6. Compute the intersection of gene symbols across all cohorts.
       Only genes present in every cohort are kept; this guarantees a
       homogeneous covariate space for the pooled model.

  Per cohort c (after restricting to common genes):
    7. Apply log2(x + 1) transform to RSEM values (variance stabilization).
    8. Z-score each gene WITHIN the cohort: (x - mean_c) / std_c. This is
       the standard correction for inter-cohort batch effects (sequencing
       depth, library prep, lab). After this step every gene has mean=0
       and std=1 in every cohort.

  Pooling:
    9. Vertically concatenate the z-scored expression matrices and the
       (T, E) outcome vectors. Assign integer stratum labels (one per cohort).

 10. For each requested covariate count p, select the top-p genes by
     variance across the pooled z-scored data and save:
         X (n x p, float32)       z-scored top-p gene expression
         T (n,)    float64         OS time in months
         E (n,)    int32           event indicator
         strata (n,) int32         cohort label (0..n_cohorts-1)
         strata_labels (n_cohorts,) string array mapping strata -> name
         gene_names (p,)           Hugo gene symbols of the selected genes
         patient_ids (n,)          TCGA patient barcodes (for traceability)

After z-scoring within cohort, the pooled per-gene variance decomposes
(by the law of total variance) into (i) the variance of the cohort-specific
means (which is zero by construction) and (ii) the average within-cohort
variance (which is one by construction). In practice rounding and missing
data make the pooled variance very close to 1; the "top-p by pooled
variance" ranking is then driven by tiny deviations and ties. To pick
biologically meaningful genes, we therefore rank by per-cohort variance
of the pre-z-score log2 expression and select the genes with the largest
AVERAGE variance across cohorts. See `_score_genes` below for the exact
definition.

Outputs to <data-dir>/processed/tcga_pancancer_top{p}.npz, one per p.

Usage:
    python preprocess.py [--data-dir ./data] [--p 100 500 1000 2000 5000]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cohorts import COHORTS, Cohort

DEFAULT_P_VALUES = (100, 500, 1000, 2000, 5000)

# TCGA sample barcode example: TCGA-A1-A0SB-01A-11R-A084-07
# Positions 1-3 are the patient barcode TCGA-A1-A0SB; position 4 is the
# sample type (01..09 = tumor, 10..19 = normal, etc.).
SAMPLE_BARCODE_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})[A-Z]?")
PRIMARY_TUMOR_TYPE = 1  # TCGA sample-type code for primary solid tumor


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sample_to_patient_and_type(sample_id: str) -> tuple[str | None, int | None]:
    m = SAMPLE_BARCODE_RE.match(sample_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_clinical(cohort_dir: Path) -> pd.DataFrame:
    """Return DataFrame indexed by PATIENT_ID with OS_MONTHS, OS_STATUS_BINARY."""
    path = cohort_dir / "data_clinical_patient.txt"
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)

    needed = {"PATIENT_ID", "OS_MONTHS", "OS_STATUS"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")

    status_map = {"0:LIVING": 0, "1:DECEASED": 1, "LIVING": 0, "DECEASED": 1}
    df["OS_STATUS_BINARY"] = df["OS_STATUS"].map(status_map)
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")

    df = df.dropna(subset=["OS_MONTHS", "OS_STATUS_BINARY"])
    df = df[df["OS_MONTHS"] > 0]

    return df.set_index("PATIENT_ID")[["OS_MONTHS", "OS_STATUS_BINARY"]]


def load_expression(cohort_dir: Path) -> pd.DataFrame:
    """Return DataFrame (patient x gene), primary tumor samples only."""
    path = cohort_dir / "data_mrna_seq_v2_rsem.txt"
    df = pd.read_csv(path, sep="\t", low_memory=False)

    if "Hugo_Symbol" not in df.columns:
        raise RuntimeError(f"{path}: missing Hugo_Symbol column")

    if "Entrez_Gene_Id" in df.columns:
        df = df.drop(columns=["Entrez_Gene_Id"])

    # Collapse duplicate Hugo symbols (different Entrez IDs map to same name).
    if df["Hugo_Symbol"].duplicated().any():
        df = df.groupby("Hugo_Symbol", as_index=True).mean(numeric_only=True)
    else:
        df = df.set_index("Hugo_Symbol")

    # Restrict to primary tumor sample columns and relabel to patient IDs.
    keep_cols: list[str] = []
    sample_to_patient: dict[str, str] = {}
    for s in df.columns:
        pid, stype = _sample_to_patient_and_type(s)
        if pid is None or stype != PRIMARY_TUMOR_TYPE:
            continue
        keep_cols.append(s)
        sample_to_patient[s] = pid
    df = df[keep_cols]
    df.columns = [sample_to_patient[c] for c in df.columns]

    # Transpose to patient x gene, collapse multiple primary samples per patient.
    expr_T = df.T
    if expr_T.index.duplicated().any():
        expr_T = expr_T.groupby(level=0).mean()
    return expr_T


def merge_cohort(cohort_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load one cohort, return (expression_df, OS_MONTHS series, OS_STATUS series),
    all aligned by patient ID and restricted to patients with both clinical and
    expression data.
    """
    clinical = load_clinical(cohort_dir)
    expression = load_expression(cohort_dir)

    common = clinical.index.intersection(expression.index)
    if len(common) < 50:
        raise RuntimeError(
            f"Only {len(common)} patients matched between clinical and "
            f"expression for {cohort_dir.name}; expected many more."
        )

    clinical = clinical.loc[common]
    expression = expression.loc[common]
    return expression, clinical["OS_MONTHS"], clinical["OS_STATUS_BINARY"]


def _score_genes(
    per_cohort_log2: list[pd.DataFrame], common_genes: list[str]
) -> np.ndarray:
    """
    Score each common gene by its AVERAGE within-cohort variance computed on
    the log2-transformed expression (before z-scoring). Higher score = the
    gene varies a lot within typical cohorts -> more informative for
    within-cohort Cox PH.

    Returns an array of length len(common_genes) of scores aligned with
    common_genes order.
    """
    score = np.zeros(len(common_genes), dtype=np.float64)
    for cohort_df in per_cohort_log2:
        # Variance with ddof=1 to be unbiased; clip negative numerical drift.
        v = cohort_df[common_genes].var(axis=0, ddof=1).to_numpy()
        v = np.where(np.isfinite(v), v, 0.0)
        score += v
    score /= len(per_cohort_log2)
    return score


def preprocess_all(
    data_dir: Path, p_values: tuple[int, ...]
) -> None:
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-cohort load & log2 transform ---
    per_cohort_log2: list[pd.DataFrame] = []
    per_cohort_T:    list[np.ndarray] = []
    per_cohort_E:    list[np.ndarray] = []
    per_cohort_pid:  list[list[str]]  = []
    cohort_labels:   list[str]        = []
    cohort_metrics:  list[tuple[str, int, int]] = []  # (label, n, events)

    for cohort in COHORTS:
        _log(f"Loading {cohort.label} ({cohort.cbioportal_id}) ...")
        cohort_dir = data_dir / cohort.cbioportal_id
        if not cohort_dir.is_dir():
            raise RuntimeError(
                f"Missing cohort directory {cohort_dir}; run download.py."
            )

        expr, T_series, E_series = merge_cohort(cohort_dir)

        # log2(x + 1) — variance stabilization for RSEM. Replace NaN with 0
        # BEFORE log so log(0+1)=0 (consistent with "no expression observed").
        expr_log2 = np.log2(expr.fillna(0.0).clip(lower=0.0) + 1.0)
        expr_log2 = expr_log2.astype(np.float32)

        n_total = len(T_series)
        n_events = int(E_series.sum())
        _log(f"  {cohort.label}: n={n_total}, events={n_events} "
             f"({100.0 * n_events / max(n_total, 1):.1f}%)")
        cohort_metrics.append((cohort.label, n_total, n_events))

        per_cohort_log2.append(expr_log2)
        per_cohort_T.append(T_series.to_numpy(dtype=np.float64))
        per_cohort_E.append(E_series.to_numpy(dtype=np.int32))
        per_cohort_pid.append(list(T_series.index))
        cohort_labels.append(cohort.label)

    # --- Gene intersection ---
    common: set[str] = set(per_cohort_log2[0].columns)
    for df in per_cohort_log2[1:]:
        common &= set(df.columns)
    common_genes = sorted(common)
    _log(f"Common genes across {len(COHORTS)} cohorts: {len(common_genes)}")

    if len(common_genes) < max(p_values):
        _log(f"WARNING: only {len(common_genes)} common genes; some p values"
             f" exceed this and will be skipped.")

    # --- Score genes by mean within-cohort variance (pre z-score) ---
    _log("Scoring genes by mean within-cohort variance ...")
    scores = _score_genes(per_cohort_log2, common_genes)

    # --- Z-score within cohort and stack ---
    _log("Z-scoring per cohort and pooling ...")
    z_scored_chunks: list[np.ndarray] = []
    for cohort_df in per_cohort_log2:
        block = cohort_df[common_genes].to_numpy(dtype=np.float32)
        mu = block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, ddof=0, keepdims=True)
        # Guard against constant genes within a cohort (sd=0): leave as zero
        # after centering, since (x - mu) = 0 anyway when sd = 0.
        sd[sd < 1e-8] = 1.0
        z_scored_chunks.append((block - mu) / sd)

    X_pooled = np.vstack(z_scored_chunks).astype(np.float32)
    T_pooled = np.concatenate(per_cohort_T)
    E_pooled = np.concatenate(per_cohort_E).astype(np.int32)
    strata = np.concatenate([
        np.full(len(per_cohort_T[i]), i, dtype=np.int32)
        for i in range(len(per_cohort_T))
    ])
    patient_ids = np.array(
        [pid for chunk in per_cohort_pid for pid in chunk]
    )
    strata_labels = np.array(cohort_labels)

    n_pooled = X_pooled.shape[0]
    n_events_pooled = int(E_pooled.sum())
    _log(f"Pooled cohort: n={n_pooled}, events={n_events_pooled} "
         f"({100.0 * n_events_pooled / n_pooled:.1f}%), genes={X_pooled.shape[1]}")

    # --- Rank genes by score (highest = most informative) ---
    gene_order = np.argsort(-scores)
    genes_sorted = np.array(common_genes)[gene_order]
    X_sorted = X_pooled[:, gene_order]

    # --- Save per-p subsets ---
    for p in sorted(set(p_values)):
        if p > X_sorted.shape[1]:
            _log(f"Skipping p={p} (only {X_sorted.shape[1]} common genes)")
            continue
        X_p = np.ascontiguousarray(X_sorted[:, :p], dtype=np.float32)
        out = out_dir / f"tcga_pancancer_top{p}.npz"
        np.savez(
            out,
            X=X_p,
            T=T_pooled,
            E=E_pooled,
            strata=strata,
            strata_labels=strata_labels,
            gene_names=genes_sorted[:p],
            patient_ids=patient_ids,
        )
        _log(f"  wrote {out.name}  shape=({X_p.shape[0]}, {X_p.shape[1]})  "
             f"size={out.stat().st_size / 1e6:.1f} MB")

    # --- Final summary ---
    _log("Per-cohort summary:")
    for label, n, events in cohort_metrics:
        _log(f"  {label:6s}  n={n:5d}  events={events:5d}  "
             f"rate={100.0*events/max(n,1):4.1f}%")
    _log(f"Total: n={n_pooled}, events={n_events_pooled}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess pooled TCGA pan-cancer data for survivex."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing the cohort subdirectories "
             "(default: ./data next to this script).",
    )
    parser.add_argument(
        "--p", type=int, nargs="+",
        default=list(DEFAULT_P_VALUES),
        help="Target covariate counts to save (default: 100 500 1000 2000 5000).",
    )
    args = parser.parse_args()

    try:
        preprocess_all(args.data_dir.resolve(), tuple(args.p))
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
