"""PDF parsing (with OCR fallback for scanned pages), heading detection, chunking, embedding,
and writing into ChromaDB.
"""
from __future__ import annotations

import base64
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pdfplumber
import requests

from app import config, database, lang

try:
    import cv2
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image

    _OCR_AVAILABLE = True
except ImportError:  # pragma: no cover - OCR deps are optional
    _OCR_AVAILABLE = False


def _patch_pdfminer_cmap_range_bomb() -> None:
    """pdfminer.six's CMapParser expands a font's embedded begincidrange/ beginbfrange
    declarations with a bare `for i in range(end - start + 1)` and no sanity bound on the
    range size.
    """
    from pdfminer import cmapdb

    _MAX_CMAP_TOTAL = 200_000  # total entries per font, not per declaration
    _CHECK_EVERY = 4096
    _original_do_keyword = cmapdb.CMapParser.do_keyword

    def _map_size(cmap) -> int:
        d = getattr(cmap, "cid2unichr", None)
        if d is not None:
            return len(d)
        d = getattr(cmap, "code2cid", None)
        return len(d) if d is not None else 0

    _MAX_OPERAND_BYTES = 32

    def _patched_do_keyword(self, pos, token):
        if token is self.KEYWORD_ENDCIDRANGE:
            objs = [obj for (__, obj) in self.popall()]
            for (start_byte, end_byte, cid) in cmapdb.choplist(3, objs):
                if _map_size(self.cmap) >= _MAX_CMAP_TOTAL:
                    break
                if not isinstance(start_byte, bytes) or not isinstance(end_byte, bytes):
                    continue
                if not isinstance(cid, int):
                    continue
                if len(start_byte) != len(end_byte) or len(start_byte) > _MAX_OPERAND_BYTES:
                    continue
                if start_byte[:-4] != end_byte[:-4]:
                    continue
                svar, evar = start_byte[-4:], end_byte[-4:]
                start, end = cmapdb.nunpack(svar), cmapdb.nunpack(evar)
                if end < start:
                    continue
                vlen = len(svar)
                for i in range(end - start + 1):
                    if i % _CHECK_EVERY == 0 and _map_size(self.cmap) >= _MAX_CMAP_TOTAL:
                        break
                    x = start_byte[:-4] + cmapdb.struct.pack(">L", start + i)[-vlen:]
                    self.cmap.add_cid2unichr(cid + i, x)
            return

        if token is self.KEYWORD_ENDBFRANGE:
            objs = [obj for (__, obj) in self.popall()]
            for (start_byte, end_byte, code) in cmapdb.choplist(3, objs):
                if _map_size(self.cmap) >= _MAX_CMAP_TOTAL:
                    break
                if not isinstance(start_byte, bytes) or not isinstance(end_byte, bytes):
                    continue
                if len(start_byte) != len(end_byte) or len(start_byte) > _MAX_OPERAND_BYTES:
                    continue
                start, end = cmapdb.nunpack(start_byte), cmapdb.nunpack(end_byte)
                if end < start:
                    continue
                if isinstance(code, list):
                    for cid, unicode_value in zip(range(start, end + 1), code):
                        if _map_size(self.cmap) >= _MAX_CMAP_TOTAL:
                            break
                        self.cmap.add_cid2unichr(cid, unicode_value)
                elif isinstance(code, bytes):
                    if len(code) > _MAX_OPERAND_BYTES:
                        continue
                    var = code[-4:]
                    base = cmapdb.nunpack(var)
                    vlen = len(var)
                    for i in range(end - start + 1):
                        if i % _CHECK_EVERY == 0 and _map_size(self.cmap) >= _MAX_CMAP_TOTAL:
                            break
                        x = code[:-4] + cmapdb.struct.pack(">L", base + i)[-vlen:]
                        self.cmap.add_cid2unichr(start + i, x)
            return

        return _original_do_keyword(self, pos, token)

    cmapdb.CMapParser.do_keyword = _patched_do_keyword


_patch_pdfminer_cmap_range_bomb()


@dataclass
class Line:
    text: str
    bold: bool
    is_first_line_of_page: bool
    line_no: int


@dataclass
class Page:
    page_num: int  # 1-indexed
    lines: list[Line] = field(default_factory=list)
    full_text: str = ""
    had_text_layer: bool = True
    ocr_confidence: Optional[float] = None  # None for native-text pages


def _line_is_bold(chars: list[dict]) -> bool:
    if not chars:
        return False
    bold_chars = sum(1 for c in chars if "bold" in c.get("fontname", "").lower())
    return bold_chars >= max(1, len(chars) // 2)


def _extract_page_lines(page: "pdfplumber.page.Page") -> list[Line]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []
    chars = page.chars
    # group words into lines by rounded 'top' position
    lines_map: dict[int, list[dict]] = {}
    for w in words:
        key = round(w["top"])
        lines_map.setdefault(key, []).append(w)

    ordered_tops = sorted(lines_map.keys())
    lines: list[Line] = []
    for idx, top in enumerate(ordered_tops):
        line_words = sorted(lines_map[top], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in line_words).strip()
        if not text:
            continue
        # gather chars roughly on this line to check boldness
        line_chars = [
            c
            for c in chars
            if abs(round(c["top"]) - top) <= 2
        ]
        bold = _line_is_bold(line_chars)
        lines.append(
            Line(text=text, bold=bold, is_first_line_of_page=(idx == 0), line_no=idx)
        )
    return lines


_OCR_LANG = "ind+eng"  # Indonesian first, then English
_OCR_DPI = 200


def _preprocess_for_ocr(image: "Image.Image") -> "Image.Image":
    """Grayscale, Otsu binarization and deskew before OCR."""
    gray = np.array(image.convert("L"))
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binarized < 255))
    if len(coords) > 20:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.1:
            h, w = binarized.shape
            matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binarized = cv2.warpAffine(
                binarized, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
    return Image.fromarray(binarized)


def _reconstruct_text_from_ocr_data(data: dict) -> str:
    """Rebuilds line text from Tesseract's own word-level bounding boxes instead of trusting
    image_to_string()'s line merging, which happily joins two side-by-side table columns
    into one interleaved line.
    """
    n = len(data.get("text", []))
    groups: dict[tuple, list[int]] = {}
    for i in range(n):
        if data["level"][i] != 5 or not data["text"][i].strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append(i)

    def _group_top(key: tuple) -> int:
        return min(data["top"][i] for i in groups[key])

    output_lines = []
    for key in sorted(groups.keys(), key=_group_top):
        idxs = sorted(groups[key], key=lambda i: data["left"][i])
        words = [(data["left"][i], data["width"][i], data["text"][i]) for i in idxs]
        gaps = [words[k + 1][0] - (words[k][0] + words[k][1]) for k in range(len(words) - 1)]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
        gap_threshold = max(40, median_gap * 4)

        current = [words[0][2]]
        for k in range(1, len(words)):
            gap = words[k][0] - (words[k - 1][0] + words[k - 1][1])
            if gap > gap_threshold:
                output_lines.append(" ".join(current))
                current = [words[k][2]]
            else:
                current.append(words[k][2])
        output_lines.append(" ".join(current))

    return "\n".join(output_lines)


def _correct_orientation(image: "Image.Image") -> "Image.Image":
    """The main OCR pass assumes upright, left-to-right text."""
    try:
        osd = pytesseract.image_to_osd(
            image, output_type=pytesseract.Output.DICT, timeout=config.OCR_OSD_TIMEOUT_SECONDS
        )
        rotation = osd.get("rotate", 0)
        confidence = osd.get("orientation_conf", 0)
    except Exception:
        return image
    if not rotation or confidence < config.OCR_ORIENTATION_MIN_CONFIDENCE:
        return image
    return image.rotate(-rotation, expand=True)


def _run_tesseract(processed: "Image.Image", psm: Optional[int] = None) -> tuple[str, Optional[float]]:
    kwargs = {}
    if psm is not None:
        kwargs["config"] = f"--psm {psm}"
    data = pytesseract.image_to_data(
        processed, lang=_OCR_LANG, output_type=pytesseract.Output.DICT,
        timeout=config.OCR_RECOGNIZE_TIMEOUT_SECONDS, **kwargs,
    )
    text = _reconstruct_text_from_ocr_data(data)
    confidences = [float(c) for c in data.get("conf", []) if c not in ("-1", -1) and float(c) >= 0]
    avg_confidence = round(sum(confidences) / len(confidences), 1) if confidences else None
    return text, avg_confidence


def _ocr_page(pdf_path: Path, page_num: int) -> tuple[str, Optional[float]]:
    """Returns (text, average_word_confidence)."""
    if not _OCR_AVAILABLE:
        return "", None
    try:
        images = convert_from_path(
            str(pdf_path), first_page=page_num, last_page=page_num, dpi=_OCR_DPI,
            timeout=config.OCR_RENDER_TIMEOUT_SECONDS,
        )
        if not images:
            return "", None
        oriented = _correct_orientation(images[0])
        processed = _preprocess_for_ocr(oriented)

        text, avg_confidence = _run_tesseract(processed)
        if config.OCR_FALLBACK_PSM:
            alt_text, alt_confidence = _run_tesseract(processed, psm=config.OCR_FALLBACK_PSM)
            if alt_confidence is not None and (avg_confidence is None or alt_confidence > avg_confidence):
                text, avg_confidence = alt_text, alt_confidence

        return text, avg_confidence
    except Exception as exc:
        print(f"WARNING: OCR failed on {pdf_path.name} page {page_num}: {exc!r}")
        return "", None


def _page_is_graphical(page: "pdfplumber.page.Page") -> bool:
    """A page whose real content is a diagram reads very differently at the structural level
    than either a flat scan or an ordinary text/table page.
    """
    large_curves = sum(
        1 for c in page.curves
        if max(c.get("width", 0), c.get("height", 0)) > config.VLM_MIN_CURVE_SIZE_PT
    )
    return large_curves >= config.VLM_MIN_CURVES


def _describe_page_image(pdf_path: Path, page_num: int) -> Optional[str]:
    """Renders the page and asks the vision-language model to describe it -- only called for
    pages _page_is_graphical() flagged, never for ordinary pages.
    """
    if not _OCR_AVAILABLE or not config.VLM_ENABLED:
        return None
    try:
        images = convert_from_path(
            str(pdf_path), first_page=page_num, last_page=page_num, dpi=_OCR_DPI,
            timeout=config.OCR_RENDER_TIMEOUT_SECONDS,
        )
        if not images:
            return None
        oriented = _correct_orientation(images[0])
    except Exception as exc:
        print(f"WARNING: rendering {pdf_path.name} page {page_num} for VLM failed: {exc!r}")
        return None
    return _describe_image_with_vlm(oriented)


_VLM_PROMPT = (
    "Describe this diagram or image in detail. List every "
    "visible label or text exactly as written, then describe "
    "the shapes and how they connect (arrows, lines, flow "
    "direction, groupings). Respond in the same language as "
    "the labels in the image."
)


def _image_content_hash(png_bytes: bytes, cache_context: str = "") -> str:
    return hashlib.sha256(png_bytes + cache_context.encode("utf-8")).hexdigest()


def _describe_image_with_vlm(image: "Image.Image", _retried: bool = False) -> Optional[str]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    image_hash = _image_content_hash(png_bytes, f"{_VLM_PROMPT}|{config.VLM_MODEL}")
    cached = database.get_vlm_description(image_hash)
    if cached is not None:
        return cached

    b64 = base64.b64encode(png_bytes).decode("ascii")
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.VLM_MODEL,
                "prompt": _VLM_PROMPT,
                "images": [b64],
                "stream": False,
                "options": {"num_ctx": config.VLM_CONTEXT_TOKENS, "temperature": 0},
            },
            timeout=config.VLM_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        if _retried:
            return None
        return _describe_image_with_vlm(image, _retried=True)
    except Exception as exc:
        print(f"WARNING: VLM description failed: {exc!r}")
        return None
    description = resp.json().get("response", "").strip() or None
    if description:
        database.set_vlm_description(image_hash, description)
    return description


_CID_ARTIFACT_RE = re.compile(r"\(cid:\d+\)")


def _looks_like_real_text(text: str) -> bool:
    """A PDF's own text layer can be present but garbage."""
    stripped = text.strip()
    if not stripped:
        return False
    printable = sum(1 for ch in stripped if ch.isprintable() or ch in "\n\t")
    if printable / len(stripped) <= 0.9:
        return False
    return not _CID_ARTIFACT_RE.search(stripped)


def extract_pdf_pages(pdf_path: Path) -> list[Page]:
    pages: list[Page] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            lines = _extract_page_lines(page)
            text = "\n".join(l.text for l in lines)
            had_text_layer = _looks_like_real_text(text)
            ocr_confidence = None
            if not had_text_layer:
                ocr_text, ocr_confidence = _ocr_page(pdf_path, i)
                text = ocr_text
                lines = [
                    Line(text=t, bold=False, is_first_line_of_page=(j == 0), line_no=j)
                    for j, t in enumerate(t for t in ocr_text.splitlines() if t.strip())
                ]

            if _page_is_graphical(page):
                vlm_description = _describe_page_image(pdf_path, i)
                if vlm_description:
                    label = lang.get().VLM_LABEL
                    text = f"{text}\n\n{label}\n{vlm_description}" if text.strip() else f"{label}\n{vlm_description}"
                    lines = lines + [
                        Line(text=t, bold=False, is_first_line_of_page=False, line_no=len(lines) + j)
                        for j, t in enumerate(
                            l for l in (label + "\n" + vlm_description).splitlines() if l.strip()
                        )
                    ]

            pages.append(
                Page(
                    page_num=i, lines=lines, full_text=text,
                    had_text_layer=had_text_layer, ocr_confidence=ocr_confidence,
                )
            )
    return pages


_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+(?=\S*[A-Za-z])\S+")


def _heading_patterns() -> list[re.Pattern]:
    return [p for mod in lang.all_modules() for p in mod.HEADING_PATTERNS] + [_NUMBERED_HEADING]

_SENTENCE_END = re.compile(r"[.!?]$")

_MIDSENTENCE_CONTINUATION_RE = re.compile(r"^[,;:]?\s*[a-z]")

_LOWERCASE_START_RE = re.compile(r"^[a-z]")

_PAGINATION_LINE_RE = re.compile(r"\d+\s*(?:dari|of|/)\s*\d+\s*$", re.IGNORECASE)


def _is_midsentence_reference(line_text: str, match_end: int, prev_line: Optional[Line]) -> bool:
    remainder = line_text[match_end:]
    if remainder.strip() and _MIDSENTENCE_CONTINUATION_RE.match(remainder):
        return True
    if prev_line is not None:
        prev_text = prev_line.text.strip()
        prev_words = prev_text.split()
        if prev_words and not _SENTENCE_END.search(prev_text) and prev_words[-1][0].islower():
            return True
    return False


def _looks_like_heading_candidate(line: Line, prev_line: Optional[Line] = None) -> bool:
    """A line is a heading candidate if it's bold OR the first line of the page OR matches a
    known structural pattern.
    """
    pattern_match = next((p.match(line.text) for p in _heading_patterns() if p.match(line.text)), None)
    if pattern_match:
        return not _is_midsentence_reference(line.text, pattern_match.end(), prev_line)
    if _LOWERCASE_START_RE.match(line.text.strip()):
        return False
    if _PAGINATION_LINE_RE.search(line.text.strip()):
        return False
    if line.bold and len(line.text.split()) <= 12:
        return True
    if line.is_first_line_of_page and len(line.text.split()) <= 12:
        return True
    return False


def _is_ordinary_sentence(line: Line) -> bool:
    """Bug #3: a bold ordinary-prose sentence must not merge into (and
    thereby be able to silently delete) an adjacent heading."""
    words = line.text.split()
    return len(words) >= config.HEADING_MERGE_EXCLUDE_MIN_WORDS and bool(
        _SENTENCE_END.search(line.text.strip())
    )


def _filter_repeating_headers(headings: list[dict], total_pages: int) -> list[dict]:
    """A genuine heading occurs once; a running header/footer template (a page-number line, a
    revision-row, a document title repeated on every page) occurs.
    """
    if total_pages < 4 or not headings:
        return headings

    def _normalize(text: str) -> str:
        return re.sub(r"\d+", "#", text.strip().lower())

    counts: dict[str, int] = {}
    for h in headings:
        norm = _normalize(h["text"])
        counts[norm] = counts.get(norm, 0) + 1

    repeat_threshold = max(3, total_pages // 3)
    repeating_norms = {norm for norm, c in counts.items() if c >= repeat_threshold}

    dropped_lines_by_page: dict[int, set[int]] = {}
    kept: list[dict] = []
    for h in headings:
        if _normalize(h["text"]) in repeating_norms:
            dropped_lines_by_page.setdefault(h["page"], set()).add(h["line_no"])
        else:
            kept.append(h)

    final: list[dict] = []
    for h in kept:
        page_drops = dropped_lines_by_page.get(h["page"], set())
        glued_to_dropped_header = h["line_no"] == 0 and any(ln <= 2 for ln in page_drops)
        if not glued_to_dropped_header:
            final.append(h)
    return final


def detect_headings(pages: list[Page]) -> list[dict]:
    """Returns a list of {page, text, line_no} heading dicts for one document."""
    headings: list[dict] = []

    for page_idx, page in enumerate(pages):
        i = 0
        n = len(page.lines)
        while i < n:
            line = page.lines[i]
            if i > 0:
                prev_line = page.lines[i - 1]
            else:
                prev_line = None
                for back in range(page_idx - 1, -1, -1):
                    if pages[back].lines:
                        prev_line = pages[back].lines[-1]
                        break
            if not _looks_like_heading_candidate(line, prev_line):
                i += 1
                continue

            run_lines = [line]
            j = i + 1
            while (
                j < n
                and len(run_lines) < config.HEADING_MAX_MERGE_LINES
                and _looks_like_heading_candidate(page.lines[j], page.lines[j - 1])
                and not _is_ordinary_sentence(page.lines[j])
            ):
                run_lines.append(page.lines[j])
                j += 1

            # the line immediately after the merged run: cross into the
            # next page if the run ran off the end of this one.
            if j < n:
                next_line = page.lines[j]
            else:
                next_line = None
                for fwd in range(page_idx + 1, len(pages)):
                    if pages[fwd].lines:
                        next_line = pages[fwd].lines[0]
                        break
            next_text = next_line.text.strip() if next_line is not None else ""
            prose_continuation = bool(next_text) and (
                _LOWERCASE_START_RE.match(next_text) or _PAGINATION_LINE_RE.search(next_text)
            )

            if len(run_lines) <= config.HEADING_MAX_MERGE_LINES and not prose_continuation:
                merged_text = " ".join(l.text for l in run_lines).strip()
                if merged_text:
                    headings.append(
                        {"page": page.page_num, "text": merged_text, "line_no": line.line_no}
                    )
            i = j if j > i else i + 1

    return _filter_repeating_headers(headings, len(pages))


_CLAUSE_MARKER_RE = re.compile(r"^\d+\.\d+$")
_STEP_MARKER_RE = re.compile(r"^\d+\)\s")


def _split_on_markers(lines: list[str], is_marker) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and is_marker(line.strip()):
            segments.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        segments.append("\n".join(current).strip())
    return [s for s in segments if s]


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(text):
            break
        start = end - overlap
    return pieces


def _find_heading_positions(text: str, heading_texts: list[str]) -> list[int]:
    """detect_headings() merges lines with a single space, while the source
    text keeps real newlines between them -- match whitespace-tolerantly to
    find each heading's actual start offset in the page text."""
    positions = []
    for heading in heading_texts:
        pattern = re.escape(heading).replace(r"\ ", r"\s+")
        m = re.search(pattern, text)
        if m:
            positions.append(m.start())
    return sorted(set(positions))


def _split_by_heading_positions(text: str, positions: list[int]) -> list[str]:
    if not positions:
        return [text] if text.strip() else []
    bounds = [0] + positions + [len(text)]
    bounds = sorted(set(bounds))
    segments = [text[bounds[i]:bounds[i + 1]].strip() for i in range(len(bounds) - 1)]
    return [s for s in segments if s]


def chunk_page_text(text: str, page_num: int, page_headings: Optional[list[str]] = None) -> list[dict]:
    """Structure-aware: a chunk boundary aligns with a detected heading (one PASAL-with-its-
    sub-clauses, or one LAMPIRAN/SOP-section, becomes one top-level segment) instead of an
    arbitrary character count.
    """
    text = text.strip()
    if not text:
        return []
    size = config.CHUNK_SIZE_CHARS
    overlap = config.CHUNK_OVERLAP_CHARS

    top_sections = None
    if page_headings:
        positions = _find_heading_positions(text, page_headings)
        if positions:
            top_sections = _split_by_heading_positions(text, positions)

    if top_sections is None:
        top_sections = _split_on_markers(
            text.split("\n"), lambda l: any(p.match(l) for p in _heading_patterns())
        )

    final_texts: list[str] = []
    for section in top_sections:
        if len(section) <= size:
            final_texts.append(section)
            continue
        sub_segments = _split_on_markers(
            section.split("\n"),
            lambda l: bool(_CLAUSE_MARKER_RE.match(l) or _STEP_MARKER_RE.match(l)),
        )
        for sub in sub_segments:
            if len(sub) <= size:
                final_texts.append(sub)
            else:
                final_texts.extend(_sliding_window(sub, size, overlap))

    return [
        {"text": t, "page": page_num, "chunk_index": i}
        for i, t in enumerate(final_texts)
    ]


def content_hash(pdf_path: Path) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(pdf_path: Path) -> str:
    """Cheap mtime+size fingerprint -- a single stat() call, no file read.
    See database.get_document_fingerprint() for why this matters at scale."""
    stat = pdf_path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _normalize_line_for_repeat_check(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.strip().lower())
    return re.sub(r"\d+", "#", text).strip()


def _strip_repeating_page_lines(pages: list[Page]) -> None:
    """Mutates pages in place, removing lines that repeat (punctuation/ digit-normalized)
    across a large fraction of the document's pages.
    """
    if len(pages) < 4:
        return
    counts: dict[str, int] = {}
    for page in pages:
        for line in page.lines:
            counts[_normalize_line_for_repeat_check(line.text)] = (
                counts.get(_normalize_line_for_repeat_check(line.text), 0) + 1
            )
    threshold = max(3, len(pages) // 2)
    repeating = {norm for norm, c in counts.items() if norm and c >= threshold}

    kept_once: set[str] = set()
    for page in pages:
        kept = []
        changed = False
        for line in page.lines:
            norm = _normalize_line_for_repeat_check(line.text)
            if norm in repeating:
                if norm in kept_once:
                    changed = True
                    continue
                kept_once.add(norm)
            kept.append(line)
        if changed:
            page.lines = kept
            page.full_text = "\n".join(l.text for l in kept)


def extract_ingest_data(pdf_path: Path) -> dict:
    """CPU-bound half of ingestion: PDF parsing, OCR, heading detection, and chunking -- no
    Chroma, no embedder, no sqlite.
    """
    source = pdf_path.name
    pages = extract_pdf_pages(pdf_path)
    _strip_repeating_page_lines(pages)

    headings = detect_headings(pages)

    headings_by_page: dict[int, list[str]] = {}
    for h in headings:
        headings_by_page.setdefault(h["page"], []).append(h["text"])

    all_chunks: list[dict] = []
    ocr_confidence_by_page: dict[int, Optional[float]] = {}
    for page in pages:
        chunks = chunk_page_text(page.full_text, page.page_num, headings_by_page.get(page.page_num))
        all_chunks.extend(chunks)
        ocr_confidence_by_page[page.page_num] = page.ocr_confidence

    return {
        "source": source,
        "headings": headings,
        "all_chunks": all_chunks,
        "ocr_confidence_by_page": ocr_confidence_by_page,
        "page_count": len(pages),
    }


def _chunk_log_records(all_chunks: list[dict], ocr_confidence_by_page: dict[int, Optional[float]]) -> list[dict]:
    return [
        {
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "ocr_confidence": ocr_confidence_by_page.get(c["page"]),
            "text": c["text"],
        }
        for c in sorted(all_chunks, key=lambda c: (c["page"], c["chunk_index"]))
    ]


def write_chunk_log_md(source: str, records: list[dict]) -> Path:
    """Human-readable per-document Markdown log under data/chunk_logs/, rewritten in full every
    time this source is (re-)ingested.
    """
    config.CHUNK_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CHUNK_LOGS_DIR / f"{source}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {source}\n\n{len(records)} chunks\n\n")
        for r in records:
            conf = r["ocr_confidence"]
            conf_note = "not OCR'd" if conf is None or conf < 0 else f"OCR confidence {conf:.1f}"
            f.write(f"## Page {r['page']}, chunk {r['chunk_index']} ({conf_note})\n\n")
            f.write("```\n" + r["text"].strip() + "\n```\n\n")
    return out_path


def remove_chunk_log(source: str) -> None:
    (config.CHUNK_LOGS_DIR / f"{source}.md").unlink(missing_ok=True)


def write_ingest_data(pdf_path: Path, data: dict, collection, embedder) -> int:
    """Chroma/sqlite-writing half of ingestion."""
    source = data["source"]
    headings = data["headings"]
    all_chunks = data["all_chunks"]
    ocr_confidence_by_page = data["ocr_confidence_by_page"]

    database.replace_headings(source, headings)

    # remove any previously-indexed chunks for this source before re-adding
    try:
        collection.delete(where={"source": source})
    except Exception as exc:
        print(f"WARNING: collection.delete failed for source={source!r}: {exc!r}")

    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        ids = [f"{source}::{c['page']}::{c['chunk_index']}" for c in all_chunks]
        metadatas = [
            {
                "source": source,
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "ocr_confidence": ocr_confidence_by_page.get(c["page"]) if ocr_confidence_by_page.get(c["page"]) is not None else -1.0,
            }
            for c in all_chunks
        ]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        write_chunk_log_md(source, _chunk_log_records(all_chunks, ocr_confidence_by_page))
    else:
        remove_chunk_log(source)

    size_kb = round(pdf_path.stat().st_size / 1024, 1)
    database.upsert_document(
        source, content_hash(pdf_path), file_fingerprint(pdf_path), data["page_count"], size_kb
    )
    return len(all_chunks)


def ingest_file(pdf_path: Path, collection, embedder) -> int:
    data = extract_ingest_data(pdf_path)
    return write_ingest_data(pdf_path, data, collection, embedder)


def remove_file(source: str, collection) -> None:
    try:
        collection.delete(where={"source": source})
    except Exception as exc:
        print(f"WARNING: collection.delete failed for source={source!r}: {exc!r}")
    remove_chunk_log(source)
    database.delete_document(source)
