"""Language selection. Every user-facing string and language-specific pattern lives in lang/<code>.py.

The default language comes from config.LANGUAGE. A request can answer in another language with
`use(code)`, and `detect(text)` picks the language a question is written in.
"""
import importlib
import re
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Iterator, Optional

from app import config

SUPPORTED = ("en", "id")

_default: Optional[ModuleType] = None
_override: ContextVar[Optional[ModuleType]] = ContextVar("docrag_language", default=None)


def _load(code: str) -> ModuleType:
    code = code.strip().lower()
    if code not in SUPPORTED:
        raise ValueError(f"Unsupported language {code!r}; use one of {SUPPORTED}")
    return importlib.import_module(f"app.lang.{code}")


def set_language(code: str) -> ModuleType:
    """Set the default language (tests and startup)."""
    global _default
    _default = _load(code)
    return _default


def get() -> ModuleType:
    override = _override.get()
    if override is not None:
        return override
    return _default if _default is not None else set_language(config.LANGUAGE)


def all_modules() -> list[ModuleType]:
    return [_load(code) for code in SUPPORTED]


@contextmanager
def use(code: str) -> Iterator[ModuleType]:
    """Run a block in another language without touching the default."""
    token = _override.set(_load(code))
    try:
        yield _override.get()
    finally:
        _override.reset(token)


_WORD_RE = re.compile(r"[a-z]+")

# Common words that mostly belong to one language. Short and frequent, so even a short question scores.
_ID_WORDS = frozenset(
    "apa yang dan di berapa untuk dokumen dengan adalah bagaimana siapa kapan jika atau dari pada ini itu "
    "tidak saya harus bisa dalam ke sebutkan apakah ada akan oleh sudah boleh sebesar setiap".split()
)
_EN_WORDS = frozenset(
    "the what is are of how who when does do can to for in and a an this that with or from any many much "
    "which list all say must should it be have has by on at as".split()
)


def detect(text: str) -> str:
    """The language a question is written in, or the default language when it is unclear."""
    words = _WORD_RE.findall(text.lower())
    id_hits = sum(w in _ID_WORDS for w in words)
    en_hits = sum(w in _EN_WORDS for w in words)
    if id_hits > en_hits:
        return "id"
    if en_hits > id_hits:
        return "en"
    return get().CODE
