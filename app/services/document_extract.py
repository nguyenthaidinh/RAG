from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Final
import re


class UnsupportedFileType(Exception):
    """Raised when input format is not supported or required dependency missing."""


class FileTooLarge(Exception):
    """Raised when input file exceeds allowed size."""


class ExtractedTextTooLarge(Exception):
    """Raised when extracted text exceeds allowed size."""


@dataclass(frozen=True)
class ExtractLimits:
    max_bytes: int = 10 * 1024 * 1024        # input file bytes
    max_chars: int = 1_000_000               # rough guard
    max_text_bytes: int = 5 * 1024 * 1024    # OUTPUT utf-8 bytes (MUST match schema)


_ZERO_WIDTH_RE: Final = re.compile(r"[\u200B-\u200D\uFEFF]")
_NULL_RE: Final = re.compile(r"\x00+")
_WS_RE: Final = re.compile(r"[ \t]+\n")


def _normalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig", errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _cleanup_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _NULL_RE.sub("", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WS_RE.sub("\n", text)
    return text.strip()


def _enforce_char_limit(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        raise ExtractedTextTooLarge(f"Text extracted too large: {len(text)} chars > {max_chars}")
    return text


def _clamp_utf8_bytes(text: str, max_bytes: int) -> str:
    b = text.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return text

    cut = b[:max_bytes]
    # ensure valid UTF-8 boundary
    while True:
        try:
            return cut.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            if not cut:
                return ""
            cut = cut[:-1]


def _enforce_text_bytes(text: str, max_text_bytes: int) -> str:
    # clamp (không raise) để router/service không dính 422 schema
    return _clamp_utf8_bytes(text, max_text_bytes)


def extract_text(
    filename: str | None,
    content_type: str | None,
    data: bytes,
    *,
    limits: ExtractLimits = ExtractLimits(),
) -> str:
    if len(data) > limits.max_bytes:
        raise FileTooLarge(f"File too large: {len(data)} bytes > {limits.max_bytes}")

    name = (filename or "").lower().strip()
    ctype = _normalize_content_type(content_type)

    # TXT / MD
    if name.endswith(".txt") or name.endswith(".md") or (ctype in {"text/plain", "text/markdown"}):
        text = _cleanup_text(_decode_text(data))
        text = _enforce_char_limit(text, limits.max_chars)
        return _enforce_text_bytes(text, limits.max_text_bytes)

    # PDF
    if name.endswith(".pdf") or (ctype == "application/pdf"):
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise UnsupportedFileType("PDF chưa hỗ trợ (thiếu pymupdf). Cài: pip install pymupdf") from e

        parts: list[str] = []
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for page in doc:
                parts.append(page.get_text("text") or "")
        finally:
            doc.close()

        text = _cleanup_text("\n".join(parts))
        text = _enforce_char_limit(text, limits.max_chars)
        return _enforce_text_bytes(text, limits.max_text_bytes)

    # DOCX
    if name.endswith(".docx") or (ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        try:
            from docx import Document
        except Exception as e:
            raise UnsupportedFileType("DOCX chưa hỗ trợ (thiếu python-docx). Cài: pip install python-docx") from e

        d = Document(BytesIO(data))
        text = "\n".join((p.text or "") for p in d.paragraphs)
        text = _cleanup_text(text)
        text = _enforce_char_limit(text, limits.max_chars)
        return _enforce_text_bytes(text, limits.max_text_bytes)

    # DOC legacy
    if name.endswith(".doc") or (ctype == "application/msword"):
        raise UnsupportedFileType("File .doc (Word cũ) chưa hỗ trợ. Hãy chuyển sang .docx hoặc bật convert bằng LibreOffice.")

    raise UnsupportedFileType(
        f"Định dạng không hỗ trợ. filename={filename!r}, content_type={content_type!r}. Hỗ trợ: txt, md, pdf, docx."
    )