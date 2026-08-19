"""
Tunable constants for the retrieval pipeline.

Everything the pipeline can be tuned by lives here so the Streamlit layer can
surface a subset as widgets without hardcoding values in two places.
"""

import os

# -----------------------------
# Ollama connection
# -----------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Model listing is instant; embedding a batch can take a while on cold load.
TAGS_TIMEOUT = 5
EMBED_TIMEOUT = 300

# -----------------------------
# Embeddings
# -----------------------------
DEFAULT_EMBED_MODEL = "nomic-embed-text"

# Ollama's /api/embed accepts a list. Batching keeps HTTP overhead down without
# building a request so large it stalls on limited VRAM.
EMBED_BATCH_SIZE = 32

# -----------------------------
# Chunking
# -----------------------------
# Child chunks are what get embedded and matched: small enough to be precise.
CHILD_CHUNK_CHARS = 1600      # ~400 tokens
CHILD_CHUNK_OVERLAP = 200

# Parent windows are what the LLM actually reads: large enough to be coherent.
PARENT_MAX_CHARS = 6000

# A fragment shorter than this carries no retrievable signal (page numbers,
# stray headers) and is dropped rather than indexed.
MIN_CHUNK_CHARS = 80

# -----------------------------
# Retrieval
# -----------------------------
DENSE_TOP_K = 20          # candidates from vector search
SPARSE_TOP_K = 20         # candidates from BM25
FUSED_TOP_K = 12          # survivors of RRF, handed to the reranker
FINAL_TOP_K = 5           # what reaches the prompt

# Reciprocal Rank Fusion damping. 60 is the value from the original RRF paper
# and needs no tuning; it mainly controls how much rank 1 outweighs rank 10.
RRF_K = 60

# 0.5 = balanced. Raise toward 1.0 to favor semantic, lower to favor keyword.
DENSE_WEIGHT = 0.5

# -----------------------------
# Context assembly
# -----------------------------
# Ollama tokenizers vary; 4 chars/token is a reasonable cross-model heuristic.
# Deliberately conservative -- undershooting the budget costs a little recall,
# overshooting it silently truncates the prompt, which is the bug we are fixing.
CHARS_PER_TOKEN = 4.0

# Share of num_ctx reserved for retrieved context. The remainder holds the
# system prompt, chat history, the question, and the model's own answer.
CONTEXT_BUDGET_RATIO = 0.6

# -----------------------------
# Chroma
# -----------------------------
COLLECTION_NAME = "rag_chunks"   # must be 3-512 chars of [a-zA-Z0-9._-]
DISTANCE_METRIC = "cosine"
