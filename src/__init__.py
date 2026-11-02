"""ResearchMind AI: a retrieval-augmented research assistant for scientific PDFs."""

import os
import sys

# On macOS, both faiss-cpu and PyTorch wheels bundle their own copy of the
# OpenMP runtime (libomp.dylib). When both initialise, libomp aborts the process
# ("OMP: Error #15"). Allowing the duplicate runtime is the widely used
# workaround for this pairing; tests/test_retrieval.py checks that FAISS
# results still match a NumPy brute-force reference with the real model loaded.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
