"""Central configuration for DocRAG."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Language of the interface, prompts, messages and heading patterns: "en" or "id".
LANGUAGE = os.getenv("DOCRAG_LANG", "en").strip().lower()

# Documents, the index, sessions and logs live here. Override to keep several indexes side by side.
DATA_DIR = Path(os.getenv("DOCRAG_DATA_DIR", str(BASE_DIR / "data")))
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_DIR = DATA_DIR / "sqlite"
SQLITE_PATH = SQLITE_DIR / "docrag.db"
SESSION_LOGS_DIR = DATA_DIR / "session_logs"
CHUNK_LOGS_DIR = DATA_DIR / "chunk_logs"

for _d in (DATA_DIR, CORPUS_DIR, CHROMA_DIR, SQLITE_DIR, SESSION_LOGS_DIR, CHUNK_LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Multilingual embedding model, works for both English and Indonesian.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Local language model served by Ollama. Full-document prompts are long, so they get a longer timeout.
OLLAMA_MODEL = os.getenv("DOCRAG_OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_HOST = os.getenv("DOCRAG_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_SECONDS = 180
OLLAMA_FULL_DOC_TIMEOUT_SECONDS = 420

# Optional vision model that describes pages that are mostly diagrams.
VLM_ENABLED = os.getenv("DOCRAG_VLM", "1") != "0"  # DOCRAG_VLM=0 skips the diagram descriptions
VLM_MODEL = "qwen2.5vl:7b"
VLM_TIMEOUT_SECONDS = 300  # vision inference is slower than text-only generation
VLM_CONTEXT_TOKENS = 8192
VLM_MIN_CURVES = 20  # a page counts as a diagram with at least this many large vector curves
VLM_MIN_CURVE_SIZE_PT = 15  # ... where "large" means bigger than this many points

# Minimum Tesseract orientation confidence before a scanned page is rotated.
OCR_ORIENTATION_MIN_CONFIDENCE = 2.0

# Passages sent to the language model, and the chunk size used at ingestion.
TOP_K = 8
CHUNK_SIZE_CHARS = int(os.getenv("DOCRAG_CHUNK_SIZE", "1000"))  # the benchmark overrides it to compare sizes
CHUNK_OVERLAP_CHARS = round(CHUNK_SIZE_CHARS * 0.15)

# Cross-encoder reranker: reorders RERANK_CANDIDATE_K candidates and keeps TOP_K.
RERANK_ENABLED = True
RERANK_CANDIDATE_K = 100
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # about 470 MB, multilingual

# Relevance gate: a question is answered only if the best raw cosine score reaches the first value
# OR the best rerank score reaches the second. Otherwise the system refuses without calling the model.
RELEVANCE_MIN_SCORE = 0.62
RELEVANCE_MIN_RERANK_SCORE = 0.0
# A question with a word found nowhere in the documents is answered only if the best rerank score reaches this.
UNKNOWN_TERM_RERANK_MIN = 3.0

# Hybrid search: BM25 keyword candidates are merged with vector candidates by reciprocal rank fusion.
HYBRID_SEARCH_ENABLED = True
BM25_CANDIDATE_K = 100
RRF_K = 60  # standard RRF damping constant (Cormack et al.)

# Documents whose best rerank score is within this gap of the top one count as competing.
MULTI_SOURCE_RERANK_GAP = 3.0

# Context window used for prompts and for the full-document size check.
OLLAMA_CONTEXT_TOKENS = 16384
FULL_DOC_PROMPT_MARGIN_TOKENS = 1200  # system prompt + question + generation headroom
CHARS_PER_TOKEN_ESTIMATE = 4  # rough heuristic, no tokenizer call needed for a fast fit-check

# A single small named document up to this size can be sent whole instead of as chunks.
FULL_DOC_FALLBACK_MAX_CHARS = 25000

OCR_RENDER_TIMEOUT_SECONDS = 60  # convert_from_path (Poppler pdftoppm)
OCR_RECOGNIZE_TIMEOUT_SECONDS = 60  # pytesseract.image_to_data
OCR_OSD_TIMEOUT_SECONDS = 15  # pytesseract.image_to_osd (a cheaper pre-pass)

# Tesseract page segmentation mode tried when the default mode reads a page poorly.
OCR_FALLBACK_PSM = 6

# When no document is named: candidate chunks retrieved, and alternatives offered next to a best-match answer.
MULTI_SOURCE_CANDIDATE_K = 15
MULTI_SOURCE_CANDIDATE_LIST_MAX = 8

# Heading detection and merging.
HEADING_MAX_MERGE_LINES = 5  # a heading may wrap over up to this many lines
HEADING_MERGE_EXCLUDE_MIN_WORDS = 4
HEADING_MIN_CATEGORY_HEADINGS = 2  # a broad-category word needs at least 2 distinct headings to qualify

FILENAME_CACHE_TTL_SECONDS = 10

# Background sync of the corpus folder.
SYNC_INTERVAL_MINUTES = 15

# Parallel ingestion workers: bounded by free RAM and CPU cores. Override with INGEST_WORKERS_OVERRIDE.
try:
    import psutil as _psutil

    _available_gb = _psutil.virtual_memory().available / (1024**3)
    _ram_worker_cap = max(1, int((_available_gb - 3) // 1.5))
except Exception:
    _ram_worker_cap = 3  # psutil unavailable: fall back to a conservative fixed cap

_override = os.getenv("INGEST_WORKERS_OVERRIDE")
if _override:
    INGEST_PARALLEL_WORKERS = max(1, int(_override))
else:
    INGEST_PARALLEL_WORKERS = max(1, min((os.cpu_count() or 4) - 1, _ram_worker_cap))

# The API has no sign-in, so it listens on this machine only unless DOCRAG_API_HOST says otherwise.
API_HOST = os.getenv("DOCRAG_API_HOST", "127.0.0.1")
API_PORT = 8000

# Chroma reads are paged in batches to stay under its SQL variable limit.
CHROMA_PAGE_SIZE = 500
