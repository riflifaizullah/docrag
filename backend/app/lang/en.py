"""English: prompts, messages, refusal phrases and heading patterns."""
import re

CODE = "en"

SYSTEM_PROMPT = """You are a question answering assistant for a collection of documents.
Answer ONLY from the context given below. Do not invent information.
If the answer is not in the context, say clearly that the information was not found in the documents.
If several sources give different values for what looks like the same fact, state the value from each source
separately and name the document, instead of merging them into one contradictory answer.
Do not copy context labels such as "[Source: ...]" into the answer, and do not start every paragraph with the
same phrase. Refer to pages briefly at the end of a sentence, for example (p. 24).

Choose the answer format to fit the content, do not force one shape on every case:
- If the answer is one simple fact, write one short flowing paragraph, as if explaining it directly to a colleague.
- If the content really has several distinct parts (for example several stages, several steps, or several
  sub-items with their own values), write one short line per part with a bold label (**Label** -- short
  explanation) instead of forcing it into one long paragraph.
Answer in the same language as the question, even if the documents are in another language.
Do not copy context sentences word for word or repeat the same template phrase for every clause or
sub-section. State numbers, dates and reference codes exactly as written (never change their values)."""

SHORT_DISCLAIMER = "Generated automatically -- verify anything important."

PAGE_WORD = "Page"
SOURCE_WORD = "Source"
DOCUMENT_LABEL = "DOCUMENT"
FULL_DOCUMENT_LABEL = "FULL DOCUMENT"
CONTEXT_LABEL = "CONTEXT"
QUESTION_LABEL = "QUESTION"
ANSWER_LABEL = "ANSWER"
NO_CONTEXT = "(no relevant context)"

NOT_FOUND = "Sorry, that information was not found in the available documents."
DEMO_ANSWER_INTRO = "Demo mode: no language model is used here. The best matching passages are:"
CLARIFY = (
    "Your question refers to an identifier found in several documents: {candidates}. "
    "Please say which document you mean."
)
MULTI_SOURCE_NOTE = (
    "\n\nNote: this question also matches {count} other document(s) that may be relevant. "
    "The answer above comes from the best matching document ({top}); "
    "pick one below to focus on another document."
)
BROAD_LISTING = "Found {count} '{category}' sections:\n{lines}"
DOC_TYPE_LISTING = "Found {count} documents of type '{doc_type}':\n{lines}"
FULL_DOC_NOTE = "Generated from the whole document, not from text excerpts."
UNVERIFIED_NOTE = "Note: {items} could not be verified against the source document."
LOW_OCR_NOTE = (
    "Note: part of the context comes from a scan (OCR) with low confidence (~{confidence:.0f}/100)."
)
PROCESSING_ERROR = "[An error occurred while processing this question. Please try again.]"

RESUME_TEMPLATE = "{question} for document {document}"
RESUME_RE = re.compile(r"^(.*?)\s+for document\s+(\S.*)$", re.IGNORECASE | re.DOTALL)

REFUSAL_PHRASES = [
    "not found in the document", "not found in the available", "was not found",
    "cannot find", "could not find", "not available in the document",
    "no information", "i cannot answer", "not mentioned in the document",
    "out of scope", "i don't have that information", "sorry, that information",
]
HEDGING_PHRASES = [
    "possibly", "it seems", "may be", "might be", "not certain", "unclear", "perhaps",
    "needs to be verified",
]
REFUSAL_TEMPLATES = [
    "That information was not found in the available documents.",
    "Information about this was not found in the documents provided.",
]

VALUE_UNIT_RE = re.compile(
    r"^\s*(million|thousand|billion|percent|%|dollars?|usd|days?|weeks?|months?|years?|minutes?|hours?|times)\b",
    re.IGNORECASE,
)
MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

META_REQUEST_PATTERNS = [
    re.compile(
        r"\bplease\s+(?:be\s+brief|be\s+detailed|answer\s+briefly|explain\s+in\s+detail|in\s+detail)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:be\s+brief|be\s+detailed|answer\s+briefly|explain\s+in\s+detail)\b", re.IGNORECASE),
]
FOLLOWUP_PATTERNS = [re.compile(r"^\s*why\?\s*$", re.IGNORECASE)]
AND_WORDS = "and"
AND_JOINER = "and"
# A later clause that points back at the earlier one ("... must it be changed?") cannot stand alone.
REFERENCE_RE = re.compile(r"\b(?:it|its|they|them|their|this|that|these|those)\b", re.IGNORECASE)
QUESTION_WORDS = r"(?:what|how|when|where|why|who|which|whose|is|are|does|do|can|may)"

ENUMERATE_TRIGGERS = (
    "list all", "list the", "name all", "what are the", "how many", "all the", "which ones",
)
CONTRACT_WORDS = "contract"
CONTRACT_FILENAME_WORD = "contract"
NUMBER_WORDS = r"no\.?|number"

IDENTIFIER_WORDS = ("article", "section", "appendix", "chapter")
HEADING_PATTERNS = [
    re.compile(r"^(ARTICLE|Article)\s+\d+", re.IGNORECASE),
    re.compile(r"^(SECTION|Section)\s+\d+", re.IGNORECASE),
    re.compile(r"^(APPENDIX|Appendix)\s+[A-Za-z0-9]+", re.IGNORECASE),
    re.compile(r"^(CHAPTER|Chapter)\s+[IVXLC0-9]+", re.IGNORECASE),
]

VLM_LABEL = "[Description of the diagram or image on this page]"

# Exported session log.
LOG_TITLE = "# Session {session_id}"
LOG_CREATED = "Created: {created}"
LOG_UNKNOWN = "unknown"
LOG_TOTAL = "Total tokens: {total} (prompt: {prompt}, response: {response})"
LOG_TURN = "## {created} -- {tokens} tokens"

UNKNOWN_TYPE = "Unknown"
EMPTY_SESSION = "(empty session)"
