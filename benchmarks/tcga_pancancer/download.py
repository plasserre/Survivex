"""
Download TCGA PanCancer Atlas cohorts from cBioPortal for the survivex
pan-cancer Cox PH benchmark.

For each cohort listed in `cohorts.py`, fetches two files from cBioPortal's
GitHub LFS mirror:
    - data_clinical_patient.txt   (overall-survival labels)
    - data_mrna_seq_v2_rsem.txt   (RNA-seq RSEM expression)

The total download is roughly 700 MB for the eight cohorts.

License: cBioPortal data is freely available for academic use under the ODC
Open Database License. See https://www.cbioportal.org/faq for details.

Usage:
    python download.py [--data-dir ./data] [--force]
    python download.py --only brca_tcga_pan_can_atlas_2018  # one cohort
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

from cohorts import COHORTS, DATAHUB_BASE, Cohort

REQUIRED_FILES = (
    "data_clinical_patient.txt",
    "data_mrna_seq_v2_rsem.txt",
)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest`, printing single-line progress."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    msg = (
                        f"  {_human_bytes(downloaded)} / {_human_bytes(total)} "
                        f"({pct:5.1f}%)"
                    )
                else:
                    msg = f"  {_human_bytes(downloaded)} downloaded"
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
    sys.stdout.write("\n")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_cohort(cohort: Cohort, data_dir: Path, force: bool) -> Path:
    """Download the required files for one cohort. Returns the cohort directory."""
    cohort_dir = data_dir / cohort.cbioportal_id
    cohort_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {cohort.label}  ({cohort.cbioportal_id}) ===")
    for fname in REQUIRED_FILES:
        target = cohort_dir / fname
        url = f"{DATAHUB_BASE}/{cohort.cbioportal_id}/{fname}"

        if target.is_file() and target.stat().st_size > 0 and not force:
            print(f"Already present: {fname} "
                  f"({_human_bytes(target.stat().st_size)})")
            continue

        _download_with_progress(url, target)
        digest = _sha256_of_file(target)
        print(f"  SHA-256: {digest}")

    for fname in REQUIRED_FILES:
        fpath = cohort_dir / fname
        if not fpath.is_file() or fpath.stat().st_size == 0:
            raise RuntimeError(f"Empty or missing after download: {fpath}")
        print(f"  ok  {fname:35s} {_human_bytes(fpath.stat().st_size):>10s}")

    return cohort_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download TCGA PanCancer Atlas cohorts from cBioPortal."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent / "data",
        help="Directory to store cohort subdirectories "
             "(default: ./data next to this script).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if already present.",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Download only one cohort by cbioportal_id (debugging).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cohorts = COHORTS
    if args.only:
        cohorts = tuple(c for c in COHORTS if c.cbioportal_id == args.only)
        if not cohorts:
            print(f"Unknown cohort id: {args.only}", file=sys.stderr)
            return 1

    try:
        for cohort in cohorts:
            download_cohort(cohort, data_dir, force=args.force)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2

    print("\nAll cohorts ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
