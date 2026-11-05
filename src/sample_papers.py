"""The four open-access (CC BY 4.0) arXiv papers used for the demo and the evaluation.

The PDFs are not committed to the repository. ``download`` fetches a pinned
version from arXiv into ``sample_data/papers/``; ``ensure_downloaded`` fetches
only the missing ones (used by the app's "Load sample papers" button and by
``evaluation/download_papers.py``).
"""

from __future__ import annotations

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
        return target
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = requests.get(url, timeout=60, headers={"User-Agent": "researchmind-ai/0.1"})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF")
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def ensure_downloaded(force: bool = False, pause_s: float = 3.0, on_progress=None) -> list[Path]:
    """Return the paths of all sample papers, downloading the missing ones.
    ``on_progress(name, downloaded: bool)`` is called once per paper."""
    paths = []
    for name, arxiv_id in PAPERS.items():
        needed = force or not (PAPERS_DIR / name).exists()
        paths.append(download(name, arxiv_id, force=force))
        if on_progress is not None:
            on_progress(name, needed)
        if needed:
            time.sleep(pause_s)  # be polite to arXiv
    return paths
