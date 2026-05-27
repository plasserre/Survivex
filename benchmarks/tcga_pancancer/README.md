# TCGA pan-cancer GPU benchmark for survivex

Real-world, high-dimensional **stratified** Cox PH benchmark added during the
SoftwareX revision to replace the synthetic-only Table 4 with a public
clinical dataset.

## Why pan-cancer pooling

Single-cohort TCGA studies (e.g., BRCA, OV alone) have at most ~200 events.
That caps the events-per-covariate ratio so low that Cox PH is statistically
under-determined at the p ≥ 500 covariate counts where GPU acceleration is
most informative. Pooling 8 TCGA PanCancer Atlas cohorts under a **stratified
Cox model** (one baseline hazard per cancer type, common β across cohorts)
yields ~4,000 patients and ~1,400 events — enough to make high-p Cox
well-determined while exercising survivex's `StratifiedCoxPHModel`.

This is the standard PanCancer Atlas survival-analysis methodology
(Liu et al., *Cell* 2018; "An Integrated TCGA Pan-Cancer Clinical Data Resource").

## Cohorts

Defined in [cohorts.py](./cohorts.py). Eight studies, each from cBioPortal's
GitHub LFS mirror:

| ID | Label | Cancer |
|---|---|---|
| `brca_tcga_pan_can_atlas_2018` | BRCA | Breast invasive carcinoma |
| `ov_tcga_pan_can_atlas_2018`   | OV   | Ovarian serous cystadenocarcinoma |
| `luad_tcga_pan_can_atlas_2018` | LUAD | Lung adenocarcinoma |
| `lusc_tcga_pan_can_atlas_2018` | LUSC | Lung squamous cell carcinoma |
| `kirc_tcga_pan_can_atlas_2018` | KIRC | Kidney renal clear cell carcinoma |
| `lihc_tcga_pan_can_atlas_2018` | LIHC | Liver hepatocellular carcinoma |
| `stad_tcga_pan_can_atlas_2018` | STAD | Stomach adenocarcinoma |
| `hnsc_tcga_pan_can_atlas_2018` | HNSC | Head and neck squamous cell carcinoma |

Total download is ~700 MB. **License**: ODC Open Database License (cBioPortal).

## One-time setup

```bash
cd benchmarks/tcga_pancancer

# 1. Download all 8 cohorts (~700 MB, ~5 min on a fast connection)
python download.py

# 2. Preprocess: gene intersection + per-cohort z-score + pool
python preprocess.py

# 3. Run the benchmark (uses StratifiedCoxPHModel)
python run_benchmark.py
```

## Mathematical pipeline (preprocess.py)

For statistical correctness when pooling distinct populations:

1. **Per cohort**: load clinical (drop missing OS), load RNA-seq RSEM
   expression, collapse duplicate Hugo symbols by mean, keep primary solid
   tumor samples only (TCGA sample-type code `01`), inner-join on
   patient ID.
2. **Cross-cohort**: take the **intersection** of gene symbols. Only genes
   measured in every cohort are kept (typically ~18,000-19,000).
3. **Per cohort**: apply `log2(x + 1)` for variance stabilization, then
   **z-score each gene within the cohort**: `(x − μ_c) / σ_c`. This is
   the standard correction for inter-cohort batch effects
   (sequencing depth, library prep, lab).
4. **Pool**: vertically concatenate the z-scored expression matrices.
   Each patient retains an integer `strata` label (0 .. n_cohorts−1)
   indicating its cancer of origin.
5. **Gene ranking**: score each common gene by its **mean within-cohort
   variance computed on the pre-z-score log2 expression**. This picks
   genes that vary within typical cohorts (most informative for the
   coefficient β under stratification) rather than between cohorts
   (already absorbed by the per-stratum baseline).
6. **Output**: for each `p ∈ {100, 500, 1000, 2000, 5000}` save an `.npz`
   with `X (n×p, float32), T, E, strata, strata_labels, gene_names,
   patient_ids`.

## Mathematical pipeline (run_benchmark.py)

`StratifiedCoxPHModel` fits the partial likelihood

```
  ℓ(β) = Σ_c Σ_{i ∈ D_c} [ β^T x_i − log Σ_{j ∈ R_c(t_i)} exp(β^T x_j) ]
```

where the inner sum is computed *within* cohort `c` only — each cancer
type contributes its own risk sets `R_c(t)` and event set `D_c`, but
all share the same coefficient vector `β`. Optimization is the same
Newton-Raphson with analytical gradient and Hessian used in
`CoxPHModel`.

The lifelines comparison passes `strata=["__S"]` so its `CoxPHFitter`
fits the same model.

## What gets benchmarked

For each `p ∈ {100, 500, 1000, 2000, 5000}` and each random seed:

| Backend | What it does |
|---|---|
| `survivex-cpu` | `StratifiedCoxPHModel(...).fit(X, T, E, strata)` on CPU |
| `survivex-cuda` | Same with `device="cuda"` (skipped if no GPU) |
| `lifelines-cpu` | `CoxPHFitter(penalizer=0.0).fit(df, ..., strata=["__S"])` |

Recorded per cell: wall-clock fit time (after warm-up), peak CPU memory
(`tracemalloc`), peak GPU memory (`torch.cuda.max_memory_allocated`),
concordance index, coefficient L2-norm. Per (p, seed) the script also
logs the max absolute coefficient difference between
`survivex-cpu` and `survivex-cuda` (expected `< 1e-10`).

## Reproducibility

- Random seeds fixed at `42, 43, 44` (override with `--seeds`).
- `numpy` and `torch` seeds set per benchmark cell.
- The processed `.npz` files are deterministic given the cBioPortal raw
  files; each download is SHA-256 logged at fetch time.

## Customization

```bash
# Subset of p values
python run_benchmark.py --p 100 500 1000

# Only survivex backends (skip lifelines)
python run_benchmark.py --backends survivex-cpu survivex-cuda

# Different output location
python run_benchmark.py --output ../../paper_data/tcga_table.csv

# Download just one cohort to debug
python download.py --only luad_tcga_pan_can_atlas_2018
```

## Mac sanity check (Apple Silicon, MPS)

Use this to verify the pipeline end-to-end on a Mac before transferring to
the CUDA machine. MPS numbers are not paper-quality (different hardware
than the rest of Table 4) and Apple's MPS backend does not expose a
peak-memory counter, so `peak_gpu_mb` will be 0.

```bash
python run_benchmark.py \
    --p 100 500 \
    --seeds 42 \
    --backends survivex-cpu survivex-mps lifelines-cpu \
    --output results/tcga_pancancer_mac_sanity.csv
```

After this passes, run the full benchmark on the CUDA machine; do **not**
commit the Mac sanity CSV.

## Hardware target

Designed to be run on the NVIDIA RTX A6000 (48 GB) that produced the
existing synthetic-data Table 4 numbers. The peak GPU memory column
makes it clear whether the chosen `p` fits on a given device.
