"""Download the open-access (CC BY 4.0) arXiv papers used for tests and evaluation.

The PDFs are *not* committed to the repository; run this script instead:

    python evaluation/download_papers.py

Files are saved to ``sample_data/papers/`` with stable names that match
``evaluation/evaluation_questions.csv``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = ROOT / "sample_data" / "papers"

# file name -> arXiv identifier (pinned version so page numbers stay stable)
PAPERS: dict[str, str] = {
    "immunotherapy_generalisation.pdf": "2604.05478v5",
    "immunotherapy_cohort_effects.pdf": "2610.33010v1",
    "gut_microbiome_circadian.pdf": "2608.00662v1",
    "rnaseq_gene_clusters.pdf": "2511.09590v1",
}


def download(name: str, arxiv_id: str, force: bool = False) -> Path:
    target = PAPERS_DIR / name
    if target.exists() and not force:
        print(f"[skip] {name} already exists")
        return target
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    print(f"[get ] {name} <- {url}")
    response = requests.get(url, timeout=60, headers={"User-Agent": "researchmind-ai/0.1"})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF")
    target.write_bytes(response.content)
    return target


def main() -> int:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv
    for name, arxiv_id in PAPERS.items():
        download(name, arxiv_id, force=force)
        time.sleep(3)  # be polite to arXiv
    print(f"Done. Papers are in {PAPERS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
