# Sample data

The evaluation and the examples in this repository use four open-access papers
from arXiv, each released under the **Creative Commons Attribution 4.0 (CC BY 4.0)**
license. The PDFs are **not** committed to the repository; download them with:

```bash
python evaluation/download_papers.py
```

They are saved to `sample_data/papers/` (gitignored). Versions are pinned so that
page numbers in `evaluation/evaluation_questions.csv` stay valid.

| Local file name | Paper | Authors | arXiv | License |
|---|---|---|---|---|
| `immunotherapy_generalisation.pdf` | Transcriptomic Models for Immunotherapy Response Prediction Show Limited Cross-cohort Generalisability | Y. Liang et al. | [2604.05478v5](https://arxiv.org/abs/2604.05478v5) | CC BY 4.0 |
| `immunotherapy_cohort_effects.pdf` | Cohort Effects, Not Treatment Response, Dominate Public Immunotherapy Cohorts: Evidence from Microbiome and Tumor Genomic Data | A. Garg | [2610.33010v1](https://arxiv.org/abs/2610.33010v1) | CC BY 4.0 |
| `gut_microbiome_circadian.pdf` | No persistent circadian oscillator at genome resolution: pseudo-coherence in gut microbiome dynamics | V. Troude et al. | [2608.00662v1](https://arxiv.org/abs/2608.00662v1) | CC BY 4.0 |
| `rnaseq_gene_clusters.pdf` | A Graphical Method for Identifying Gene Clusters from RNA Sequencing Data | J. R. Patock, R. Ratnapriya, A. Barman | [2511.09590v1](https://arxiv.org/abs/2511.09590v1) | CC BY 4.0 |

Licenses were checked in the arXiv metadata (OAI-PMH `license` field) when the
papers were selected. The papers were chosen because they are openly licensed and
span related topics (immunotherapy, transcriptomics, microbiome), which makes
cross-document questions meaningful. Their inclusion does not imply endorsement
of this project by the authors.

You can use ResearchMind AI with any text-based PDFs of your own; nothing you
upload through the app is written to disk.
