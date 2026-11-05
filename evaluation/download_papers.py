"""Download the open-access (CC BY 4.0) arXiv papers used for tests and evaluation.

The PDFs are *not* committed to the repository; run this script instead:

    python evaluation/download_papers.py [--force]

Files are saved to ``sample_data/papers/`` with stable names that match
``evaluation/evaluation_questions.csv``. The paper list lives in
``src/sample_papers.py`` so the app can load the same papers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sample_papers import PAPERS_DIR, ensure_downloaded  # noqa: E402


def main() -> int:
    def report(name: str, downloaded: bool) -> None:
        print(f"[{'get ' if downloaded else 'skip'}] {name}")

    ensure_downloaded(force="--force" in sys.argv, on_progress=report)
    print(f"Done. Papers are in {PAPERS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
