"""Indonesian: prompts, messages, refusal phrases and heading patterns."""
import re

CODE = "id"

SYSTEM_PROMPT = """Anda adalah asisten tanya jawab untuk kumpulan dokumen.
Jawab HANYA berdasarkan konteks yang diberikan di bawah ini. Jangan mengarang informasi.
Jika jawabannya tidak ada dalam konteks, katakan dengan jelas bahwa informasi tersebut tidak ditemukan dalam dokumen.
Jika beberapa sumber memberikan nilai yang berbeda untuk fakta yang tampaknya sama, sebutkan nilai dari masing-masing
sumber secara terpisah dengan menyebut nama dokumennya, jangan menggabungkannya menjadi satu jawaban yang kontradiktif.
Jangan menyalin label konteks seperti "[Sumber: ...]" ke dalam jawaban, dan jangan mengawali setiap paragraf dengan
frasa yang sama. Cukup rujuk halaman secara singkat di akhir kalimat, misalnya (hlm. 24).

Pilih format jawaban sesuai bentuk isinya, jangan dipaksakan satu bentuk untuk semua kasus:
- Jika jawabannya satu fakta sederhana, tulis sebagai satu paragraf singkat yang mengalir, seperti menjelaskan
  langsung ke rekan kerja.
- Jika isinya memang terdiri dari beberapa bagian yang berbeda (misalnya beberapa tahap/stage, beberapa
  langkah, atau beberapa sub-item dengan nilai masing-masing), tulis satu baris singkat per bagian dengan
  label tebal (**Label** -- penjelasan singkat), bukan dipaksakan jadi satu paragraf panjang.
Jawab dalam bahasa yang sama dengan pertanyaan, meskipun dokumennya berbahasa lain.
Jangan menyalin ulang kalimat konteks kata per kata atau mengulang frasa templat yang sama untuk setiap
ayat/sub-bagian. Sebutkan angka, tanggal, dan kode referensi persis apa adanya (jangan diubah nilainya)."""

SHORT_DISCLAIMER = "Dihasilkan otomatis -- verifikasi ulang untuk hal penting."

# Prompt labels. PAGE_WORD is also how a page citation is recognised in an answer.
PAGE_WORD = "Halaman"
SOURCE_WORD = "Sumber"
DOCUMENT_LABEL = "DOKUMEN"
FULL_DOCUMENT_LABEL = "DOKUMEN LENGKAP"
CONTEXT_LABEL = "KONTEKS"
QUESTION_LABEL = "PERTANYAAN"
ANSWER_LABEL = "JAWABAN"
NO_CONTEXT = "(tidak ada konteks yang relevan)"

# Messages shown to the user.
NOT_FOUND = "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
CLARIFY = (
    "Pertanyaan Anda merujuk pada identifikasi yang ditemukan di beberapa dokumen: {candidates}. "
    "Mohon sebutkan dokumen yang dimaksud."
)
MULTI_SOURCE_NOTE = (
    "\n\nCatatan: pertanyaan ini juga cocok dengan {count} dokumen lain yang "
    "mungkin relevan. Jawaban di atas diambil dari dokumen dengan kecocokan tertinggi "
    "({top}); pilih salah satu di bawah untuk fokus ke dokumen lain."
)
BROAD_LISTING = "Ditemukan {count} bagian '{category}':\n{lines}"
DOC_TYPE_LISTING = "Ditemukan {count} dokumen bertipe '{doc_type}':\n{lines}"
FULL_DOC_NOTE = "Dihasilkan dari keseluruhan isi dokumen, bukan potongan teks."
UNVERIFIED_NOTE = "Catatan: {items} belum terverifikasi terhadap dokumen sumber."
LOW_OCR_NOTE = (
    "Catatan: sebagian konteks berasal dari hasil pemindaian (OCR) dengan keyakinan rendah "
    "(~{confidence:.0f}/100)."
)
PROCESSING_ERROR = "[Terjadi kesalahan saat memproses pertanyaan ini. Silakan coba lagi.]"

# The website sends "<question> untuk dokumen <file>" when the user picks a document.
RESUME_TEMPLATE = "{question} untuk dokumen {document}"
RESUME_RE = re.compile(r"^(.*?)\s+untuk dokumen\s+(\S.*)$", re.IGNORECASE | re.DOTALL)

# Refusal detection also knows the English phrases, so a model that slips into English is still caught.
REFUSAL_PHRASES = [
    "tidak ditemukan", "tidak tersedia", "tidak dapat menemukan",
    "maaf, saya tidak", "tidak ada informasi", "di luar cakupan",
    "tidak disebutkan dalam dokumen",
    "cannot find", "not available in the document", "no information",
    "i cannot answer", "not mentioned in the document", "out of scope",
    "i don't have that information",
]
HEDGING_PHRASES = [
    "mungkin", "kemungkinan", "sepertinya", "perlu diverifikasi",
    "possibly", "it seems", "may be", "not certain", "unclear",
]
REFUSAL_TEMPLATES = [
    "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia.",
    "Informasi mengenai hal tersebut tidak ditemukan dalam dokumen yang diberikan.",
]

# Identifier grounding.
VALUE_UNIT_RE = re.compile(
    r"^\s*(juta|ribu|miliar|persen|%|rupiah|hari|bulan|tahun|menit|jam|kali)\b", re.IGNORECASE
)
MONTH_NAMES = {
    "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
    "september", "oktober", "november", "desember",
    "january", "february", "march", "june", "july", "august", "october", "december",
}

# Question splitting.
META_REQUEST_PATTERNS = [
    re.compile(
        r"\b(?:tolong|please)\s+(?:jawab\s+singkat|be\s+brief|be\s+detailed|detail(?:kan)?|jelaskan\s+secara\s+detail)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:jawab\s+singkat|be\s+brief|be\s+detailed|jelaskan\s+secara\s+detail)\b", re.IGNORECASE),
]
FOLLOWUP_PATTERNS = [re.compile(r"^\s*(why\?|kenapa\?|mengapa\?)\s*$", re.IGNORECASE)]
AND_WORDS = "dan|and"
AND_JOINER = "dan"
# A later clause that points back at the earlier one ("... diganti?") cannot stand alone.
REFERENCE_RE = re.compile(r"\b(?:\w+nya|tersebut|itu)\b", re.IGNORECASE)
QUESTION_WORDS = (
    r"(?:apa|berapa|siapa|kapan|dimana|di\s+mana|bagaimana|mengapa|kenapa|"
    r"what|how|when|where|why|who|which)"
)

# Retrieval.
ENUMERATE_TRIGGERS = (
    "list all", "daftar", "sebutkan semua", "apa saja", "berapa banyak", "semua", "ada berapa",
)
CONTRACT_WORDS = "contract|kontrak"
CONTRACT_FILENAME_WORD = "kontrak"
NUMBER_WORDS = r"no\.?|nomor|number"

# Headings. The first word of a heading names its category (for example "Pasal 5").
IDENTIFIER_WORDS = ("pasal", "lampiran", "bab")
HEADING_PATTERNS = [
    re.compile(r"^(PASAL|Pasal)\s+\d+", re.IGNORECASE),
    re.compile(r"^(LAMPIRAN|Lampiran)\s+[A-Za-z0-9]+", re.IGNORECASE),
    re.compile(r"^(BAB|Bab)\s+[IVXLC0-9]+", re.IGNORECASE),
    re.compile(r"^(AYAT|Ayat)\s+\(?\d+\)?", re.IGNORECASE),
]

# Text added to a page that a vision model described.
VLM_LABEL = "[Deskripsi diagram/gambar dari halaman ini]"

# Exported session log.
LOG_TITLE = "# Sesi {session_id}"
LOG_CREATED = "Dibuat: {created}"
LOG_UNKNOWN = "tidak diketahui"
LOG_TOTAL = "Total token: {total} (prompt: {prompt}, response: {response})"
LOG_TURN = "## {created} -- {tokens} token"

UNKNOWN_TYPE = "Tidak diketahui"
EMPTY_SESSION = "(sesi kosong)"
