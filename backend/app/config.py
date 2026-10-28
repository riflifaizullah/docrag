"""Central configuration for DocRAG.

Profile as of the migration to the new machine (i7-240H, RTX 5050 8GB VRAM)
-- raised from the original temporary-laptop profile (8GB RAM, MX330 2GB
VRAM) that every value here was tuned down for. sentence-transformers'
SentenceTransformer/CrossEncoder auto-detect CUDA with zero code changes;
they just need a CUDA-enabled torch build installed (the default pip
install pulls CPU-only; see the README) to actually use the GPU.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_DIR = DATA_DIR / "sqlite"
SQLITE_PATH = SQLITE_DIR / "docrag.db"
SESSION_LOGS_DIR = DATA_DIR / "session_logs"
CHUNK_LOGS_DIR = DATA_DIR / "chunk_logs"

for _d in (DATA_DIR, CORPUS_DIR, CHROMA_DIR, SQLITE_DIR, SESSION_LOGS_DIR, CHUNK_LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Embeddings
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Local LLM via Ollama. Raised from qwen2.5:3b-instruct-q4_K_M (the
# temporary-laptop profile) now that there's a real GPU to run on.
# qwen3.5:9b (the OFFICIAL Ollama library tag -- not a third-party
# namespace like frob/ or FieldMouse-AI/, and definitely not the
# "abliterated" uncensored community variant that also shows up in search
# results under a similar name) -- newer generation than qwen2.5, a 256K
# architecture context window vs qwen2.5's ~32K, explicitly expanded to
# 201 languages, and confirmed to fit comfortably in 8GB VRAM (~6.6GB).
# aisingapore/Llama-SEA-LION-v2-8B-IT remains the original spec's intended
# model and is worth trying as a follow-up experiment once migration
# itself is stable, not gambled on now.
OLLAMA_MODEL = "qwen3.5:9b"
# Literal 127.0.0.1, not "localhost" -- on this machine "localhost" resolves
# IPv6 (::1) first, Ollama isn't reachable there, and every request pays a
# ~2 second connect-timeout-then-fallback-to-IPv4 tax before actually
# starting (confirmed live: 2040ms vs 4ms for the identical request).
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_SECONDS = 180
# Full-document-fallback prompts (the whole document injected, not just
# retrieved chunks) are far bigger than a normal retrieval prompt -- 180s
# was confirmed live to be too tight for one (~4200 tokens) right after the
# VLM had been swapped into VRAM, raising a Timeout instead of completing.
OLLAMA_FULL_DOC_TIMEOUT_SECONDS = 420

# Vision-language model for diagram/image-heavy pages (see
# ingestion._page_is_graphical) -- confirmed real via Ollama's official
# library (qwen2.5vl:7b, 6.0GB, requires Ollama 0.7.0+; this machine runs
# 0.34.3). Only ever invoked for a page structurally flagged as graphical,
# never for ordinary text/OCR pages, since it can't stay loaded in VRAM at
# the same time as OLLAMA_MODEL -- 6.0GB + ~6.6GB is over this GPU's 8GB
# budget. Ollama swaps models on demand, so this only works because
# ingestion and the live API server are never run at the same time (same
# rule as never touching Chroma from both at once).
VLM_ENABLED = True
VLM_MODEL = "qwen2.5vl:7b"
VLM_TIMEOUT_SECONDS = 300  # vision inference is slower than text-only generation
# Confirmed live: Ollama's default num_ctx (4096) for this model left almost
# no generation room -- a single rendered page image alone consumed 4054 of
# those 4096 tokens, cutting the response off after one sentence. 8192 was
# confirmed sufficient for a full multi-paragraph description to finish
# naturally (done_reason "stop") on the real diagram this was tested against.
VLM_CONTEXT_TOKENS = 8192
# Provisional -- calibrated live against exactly one confirmed real diagram
# page (69 vector curves, from pipe/valve icons drawn as actual vector
# paths) versus an ordinary text/table page (6 curves, its many rects are
# table-cell borders not diagram content) and two flat scanned pages (0
# curves -- a scan is one raster image with no vector drawing at all, so
# image presence/area alone would wrongly flag every scanned page too).
# Needs revalidating against more real diagram pages as they turn up.
VLM_MIN_CURVES = 20
# Confirmed live this threshold needs to apply to curve SIZE, not just raw
# count: a signed/export-optimized PDF renders its text as outlined vector
# paths (one tiny curve per letter stroke) instead of embedded font glyphs
# -- an ordinary text-and-table page came back with 3148 curves this way
# and was misflagged as a diagram on all 53 pages of that document, each
# paying a real VLM call for nothing. A genuine diagram's curves (pipe/
# valve icon shapes) are drawn large enough to be visually legible on the
# page; a confirmed real flowchart page had 40 curves over 15pt in either
# dimension, while the false-positive text page had zero that large out of
# 3148. See ingestion._page_is_graphical.
VLM_MIN_CURVE_SIZE_PT = 15

# Tesseract OSD's own confidence for a detected rotation -- confirmed live
# that trusting OSD unconditionally can flip an already-upright page into
# gibberish when OSD's guess is a near coin-flip (0.53 confidence ruined a
# real cover page), versus a genuinely correct rotation detection (4.71
# confidence, the real rotated diagram). A wide, clean gap between the two
# real examples calibrated this threshold; only revisit if more real
# examples land closer to it than these two did.
OCR_ORIENTATION_MIN_CONFIDENCE = 2.0

# Retrieval
# Raised again for the new hardware (was 4 on the temporary laptop, then 6
# after confirming live that correct-but-lower-scoring chunks sometimes
# missed the cutoff). A real GPU + more RAM can afford considering more
# candidates without the latency cost this mattered for before.
TOP_K = 8
CHUNK_SIZE_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150

# Reranking: retrieve a wider embedding candidate set, then rerank with a
# cross-encoder that scores (question, chunk) jointly -- helps specifically
# when several near-identical boilerplate clauses compete for the same slot.
RERANK_ENABLED = True
RERANK_CANDIDATE_K = 100  # 200-question A/B: 35 -> 100 lifted doc hit 81.5->87.0%, page 72.5->79.0%; most page misses never reached the reranker
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # ~470MB, multilingual incl. Indonesian, now GPU-capable

# Relevance gate: below this top-hit score, skip the LLM call entirely and
# refuse -- confirmed live there was NO such gate at all, so any retrieval
# (however weak) reached the LLM and refusal was left entirely to the
# model's own judgment. A 100-question grounded eval found this let genuinely
# off-topic questions (guitar, cooking, geography -- nothing this corpus has)
# get answered from the model's general knowledge more than half the time
# (44% correct-refusal rate).
#
# First calibration (0.55) used k=8; the actual no-named-document path
# (generate_answer's multi-source discovery branch) retrieves at
# MULTI_SOURCE_CANDIDATE_K=15, and a wider candidate pool raises the noise
# floor -- a wider net always turns up SOME marginally-higher-scoring
# irrelevant chunk purely by having more chances to. Confirmed live at the
# real k=15: an off-topic "rendang recipe" question topped out at 0.574
# (an unrelated HSSE risk-management document, not a real match) and still
# slipped past 0.55. Recalibrated at the actual k=15 used in production:
# off-topic top scores 0.36-0.57.
#
# This alone isn't a clean separator, though: a broader real-question sample
# found genuine, in-corpus questions as low as 0.51 (short/colloquial
# phrasing embeds weakly) -- BELOW the off-topic ceiling. See
# RELEVANCE_MIN_RERANK_SCORE below, which is what actually catches those.
RELEVANCE_MIN_SCORE = 0.62

# Second relevance signal, checked with OR against RELEVANCE_MIN_SCORE
# (generation._passes_relevance_gate) -- confirmed live the raw score alone
# misclassifies real questions with weak/colloquial phrasing (e.g. "cara
# mengajukan cuti" scored 0.514, below the off-topic ceiling above). The
# cross-encoder rerank score (already computed for free inside
# retrieval.retrieve() when RERANK_ENABLED) separates the same two groups
# far more cleanly: off-topic ceiling -0.63, genuine floor ~2.1 for
# questions the corpus can actually answer well. 0.0 sits in that gap with
# large margin on both sides.
#
# Known gap this does NOT fix: a question whose real answer document uses
# different vocabulary than the question ("cuti" vs. the document's formal
# "Istirahat Tahunan") can score low on BOTH signals -- that's a retrieval-
# ranking/vocabulary problem, not a gating one, and needs query/synonym
# expansion, not a threshold change.
RELEVANCE_MIN_RERANK_SCORE = 0.0

# Hybrid search: pure embedding similarity can rank a chunk containing the
# exact WRONG Pasal number as more similar than the chunk with the right
# one, on short structurally-repetitive legal text where every clause uses
# near-identical wording. A BM25 (sparse/keyword) pass catches the literal
# identifier match embeddings can miss; both rank lists are combined via
# reciprocal rank fusion before reranking.
HYBRID_SEARCH_ENABLED = True
BM25_CANDIDATE_K = 100  # widened together with RERANK_CANDIDATE_K, see above
RRF_K = 60  # standard RRF damping constant (Cormack et al.)

# Multi-source discovery's "how many distinct documents compete for this
# question" decision (see generation.generate_answer) must use the
# reranked score, not the raw cosine score -- confirmed live: for a
# question genuinely about one document (a policy document), the raw cosine
# score alone put two unrelated documents' boilerplate-matching chunks at
# 0.28-0.41 similarity, enough to count as "competing" sources, while the
# cross-encoder rerank score correctly scored the one truly relevant chunk
# at +2.74 and every other chunk from ALL documents (including the correct
# one's other chunks) between -1.7 and -6.2 -- a clean, wide gap the raw
# cosine score doesn't show at all. A source only really "competes" if its
# best reranked score comes within this gap of the single best score
# across all candidates; anything further behind is noise the reranker
# already flagged, not real ambiguity. Provisional -- calibrated against
# one real case, revisit if more real examples land closer to the gap.
# retrieval.detect_ambiguity() (bare-identifier hard-block ambiguity) used
# to reuse this same gap, but was switched to a structural heading-count
# check instead -- score gap conflates "one document has richer retrievable
# content" with "the identifier is unambiguous" (real eval case, 1k-corpus
# scale: "Apa isi Lampiran 12?", 15+ documents share that heading but only
# one had enough body text to score far ahead of the rest).
MULTI_SOURCE_RERANK_GAP = 3.0

# Full-document fallback: for "enumerate" or small-document "broad"
# questions, skip retrieval entirely and inject the whole document when it
# clearly fits the model's context window, guaranteeing completeness
# regardless of chunk-level retrieval quality.
# Verified live on this machine (confirmed via `ollama ps` showing
# CONTEXT 16384 while call_ollama's explicit num_ctx passthrough is
# active) -- this is now the model's real running context, not an
# assumption. qwen3.5:9b's architecture supports up to 256K; 16384 is a
# deliberately smaller, confirmed-working value for this GPU's VRAM
# budget, not a guess.
OLLAMA_CONTEXT_TOKENS = 16384
FULL_DOC_PROMPT_MARGIN_TOKENS = 1200  # system prompt + question + generation headroom
CHARS_PER_TOKEN_ESTIMATE = 4  # rough heuristic, no tokenizer call needed for a fast fit-check

# "Fits in context" alone isn't a good enough bar for actually injecting a
# whole document -- confirmed live: a 60-page/135,704-char document
# answered a specific date question wrong (confused the real effective
# date with a different date elsewhere in the same document), while every
# document under ~15,342 chars in the same test answered correctly via
# this same full-document path. A small local model can technically fit
# a large document in its context window without being able to reliably
# find one specific fact inside it -- normal chunk retrieval (which finds
# the single most relevant passage directly) is more reliable for large
# documents even though full-injection would technically fit. This cap
# applies to any PDF, not a specific document -- above it, generate_answer
# falls through to normal retrieval instead of full-document injection.
# Calibrated with a wide, real margin on both sides (15,342 worked,
# 135,704 didn't); revisit if a real case lands closer to the gap.
FULL_DOC_FALLBACK_MAX_CHARS = 25000

# Poppler/Tesseract subprocess timeouts (seconds) -- confirmed live these
# were needed: with none at all, a real full-corpus ingest ran 15 parallel
# workers each rendering at a higher DPI, hit some kind of resource
# contention, and hung completely for 7+ hours with zero files completed
# and no error, because nothing was ever bounding how long a single
# subprocess call could take. ingestion.py's existing broad try/except
# around each of these calls already degrades a single page gracefully
# (empty OCR text, page treated as unreadable) -- these timeouts just make
# sure that except block is actually reachable instead of blocking forever.
OCR_RENDER_TIMEOUT_SECONDS = 60  # convert_from_path (Poppler pdftoppm)
OCR_RECOGNIZE_TIMEOUT_SECONDS = 60  # pytesseract.image_to_data
OCR_OSD_TIMEOUT_SECONDS = 15  # pytesseract.image_to_osd (a cheaper pre-pass)

# Tesseract's default page-segmentation mode (PSM 3, fully-automatic) can
# drop/merge characters on dense title-block tables -- confirmed live on a
# real page: PSM 3 read "EFFECTIVE DATE : 2 August 2025" (digit dropped),
# while PSM 6 ("assume a single uniform block of text") read the same
# image correctly as "21 August 2025". But PSM 6 is NOT a safe blanket
# replacement -- confirmed live on two decorative cover pages (logo/graphic-
# heavy, sparse scattered text) it produced total garbage where PSM 3 read
# fine. The two modes' own average word confidence reliably tells them
# apart on all pages tested (both fix cases and both regression cases), so
# _ocr_page runs both and keeps whichever result has the higher average
# confidence, rather than trusting either PSM unconditionally.
OCR_FALLBACK_PSM = 6

# Multi-source discovery: when no document is named in the question, look
# at how many DISTINCT documents actually compete for the top retrieval
# slots -- not an absolute relevance-score cutoff (the cross-encoder's raw
# scores aren't calibrated to a known "this counts as relevant" threshold),
# just how many different documents show up among the best matches.
# One distinct source -> just answer it. More than one -> answer from the
# highest-ranked source only, note how many other documents are also
# relevant, and list up to MULTI_SOURCE_CANDIDATE_LIST_MAX of them as
# candidates the UI can offer as a "narrow it down" choice.
MULTI_SOURCE_CANDIDATE_K = 15
MULTI_SOURCE_CANDIDATE_LIST_MAX = 8

# Heading detection / merging
HEADING_MAX_MERGE_LINES = 5  # was 3; that silently dropped legitimate 4-line headings
HEADING_MERGE_EXCLUDE_MIN_WORDS = 4  # bold line with >=4 words ending in .!? is a sentence, not a heading fragment
HEADING_MIN_CATEGORY_HEADINGS = 2  # a broad-category word needs >=2 distinct headings to qualify

# Caching
FILENAME_CACHE_TTL_SECONDS = 10

# Sync
SYNC_INTERVAL_MINUTES = 15
# Per-file PDF parsing/OCR (the CPU-bound half of ingestion, see
# ingestion.extract_ingest_data) runs in a process pool during sync -- this
# machine's multi-core CPU can OCR several files at once instead of one at a
# time. The actual Chroma/embedder/sqlite writes stay single-threaded
# regardless (see ingestion.write_ingest_data and retrieval._chroma_thread),
# so raising this only speeds up extraction, not the write phase.
#
# Confirmed live this needs a RAM ceiling, not just a CPU-count one: on this
# machine's 16GB total RAM, a plain cpu_count()-1 sizing gave 15 workers,
# and the real full-corpus ingest (1149 files, a much larger and more varied
# set than any small-batch test covered) repeatedly died with MemoryError
# under that many concurrent OCR/VLM-heavy workers -- three fix attempts
# chasing an unrelated, real-but-minor pdfminer edge case (a malformed
# font's CMap parsing) never actually resolved it, because the true
# bottleneck was system-wide memory exhaustion from over-parallelization,
# not that code path specifically; the crash kept surfacing there only
# because parsing an embedded CID font is common early per-page work that
# happens to need a fresh allocation right when the system is already
# critically low, not because that one file was uniquely broken. Budgets
# RAM per worker against currently-available RAM (via psutil, reserving
# headroom for the main process's embedding model, Chroma/sqlite, and
# Ollama) and takes whichever of that or the CPU-count bound is lower, so
# a future higher-RAM machine still scales up by CPU count while a tight
# one like this backs off automatically instead of repeating this
# incident.
#
# First attempt at this budget (1GB/worker, 2GB reserved) still resolved
# to 6 workers on a re-run once some RAM had freed up between attempts,
# and confirmed live via Get-Counter '\Memory\Available MBytes' that left
# only ~1.4GB truly available with those 6 workers just barely started (no
# heavy pages yet) -- real observed per-worker footprint is already
# ~650-800MB before doing any substantial OCR/VLM work, well above the 1GB
# budget's assumed margin for that plus in-flight spikes. Raised to 1.5GB/
# worker and 3GB reserved to leave real headroom for large-page spikes
# instead of running at the edge of the observed danger zone.
try:
    import psutil as _psutil

    _available_gb = _psutil.virtual_memory().available / (1024**3)
    _ram_worker_cap = max(1, int((_available_gb - 3) // 1.5))
except Exception:
    _ram_worker_cap = 3  # psutil unavailable -- fall back to a conservative fixed cap

# Manual escape hatch: the RAM formula above is deliberately conservative
# (3GB reserved + 1.5GB/worker), so it can land on a lower count than a
# human watching live available RAM would judge safe -- confirmed live it
# computed 1 worker at 4.3GB available, just under the 4.5GB it wants for a
# 2nd. Setting INGEST_WORKERS_OVERRIDE skips the formula entirely and uses
# that number as-is -- this is a deliberate risk trade a person made with
# eyes open (e.g. via Get-Counter), not something to reach for by default;
# going too high reproduces the exact MemoryError crashes documented above.
_override = os.getenv("INGEST_WORKERS_OVERRIDE")
if _override:
    INGEST_PARALLEL_WORKERS = max(1, int(_override))
else:
    INGEST_PARALLEL_WORKERS = max(1, min((os.cpu_count() or 4) - 1, _ram_worker_cap))

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Chroma "too many SQL variables" style limits: page reads in batches
CHROMA_PAGE_SIZE = 500
