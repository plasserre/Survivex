"""
TCGA PanCancer Atlas cohort registry for the survivex pan-cancer benchmark.

The list below was chosen to balance two competing goals:
  1. Sample size — pooling gives us enough events per covariate for high-p
     Cox PH to be statistically well-determined.
  2. Diversity — including cancers with different baseline hazards motivates
     the use of stratified Cox PH (each cohort = one stratum), which the
     survivex library implements as `StratifiedCoxPHModel`.

Each cohort is fetched from cBioPortal's GitHub LFS mirror at
    https://github.com/cBioPortal/datahub/tree/master/public/<cohort_id>

Selection notes:
  - COAD (colon adenocarcinoma) excluded: cBioPortal LFS file size 0 indicates
    the RNA-seq matrix is not currently available for the PanCancer Atlas
    version. Easy to add back if/when restored.
  - GBM (glioblastoma) excluded: tiny RNA-seq subset (~160 patients), would
    add noise for little signal.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cohort:
    cbioportal_id: str  # study folder in cBioPortal datahub
    label: str          # short human-readable name used in plots / Table 4
    description: str    # one-line description for the README


COHORTS: tuple[Cohort, ...] = (
    Cohort("brca_tcga_pan_can_atlas_2018", "BRCA",
           "Breast invasive carcinoma"),
    Cohort("ov_tcga_pan_can_atlas_2018",   "OV",
           "Ovarian serous cystadenocarcinoma"),
    Cohort("luad_tcga_pan_can_atlas_2018", "LUAD",
           "Lung adenocarcinoma"),
    Cohort("lusc_tcga_pan_can_atlas_2018", "LUSC",
           "Lung squamous cell carcinoma"),
    Cohort("kirc_tcga_pan_can_atlas_2018", "KIRC",
           "Kidney renal clear cell carcinoma"),
    Cohort("lihc_tcga_pan_can_atlas_2018", "LIHC",
           "Liver hepatocellular carcinoma"),
    Cohort("stad_tcga_pan_can_atlas_2018", "STAD",
           "Stomach adenocarcinoma"),
    Cohort("hnsc_tcga_pan_can_atlas_2018", "HNSC",
           "Head and neck squamous cell carcinoma"),
)

DATAHUB_BASE = (
    "https://media.githubusercontent.com/media/"
    "cBioPortal/datahub/master/public"
)
