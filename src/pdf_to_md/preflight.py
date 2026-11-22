"""PDF validation and per-page text-layer inspection."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf

logger = logging.getLogger(__name__)


class PdfInputError(Exception):
    """Raised when the input PDF is missing, unreadable, encrypted, or invalid."""


PageClassification = Literal["digital", "scanned_or_image", "empty"]
DocumentKind = Literal["digital", "scanned", "mixed", "empty"]


@dataclass(slots=True)
class PagePreflightInfo:
    """Diagnostics for a single PDF page prior to Docling conversion."""

    page_no: int
    width_pt: float
    height_pt: float
    text_char_count: int
    word_count: int
    image_count: int
    drawing_count: int
    has_embedded_text: bool
    classification: PageClassification
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PdfPreflightReport:
    """Full pre-flight report for an input PDF document."""

    file_path: Path
    file_size_bytes: int
    sha256: str
    page_count: int
    pdf_version: str | None
    is_encrypted: bool
    metadata: dict[str, str]
    has_embedded_text_layer: bool
    document_kind: DocumentKind
    digital_pages: list[int]
    scanned_pages: list[int]
    empty_pages: list[int]
    pages: list[PagePreflightInfo]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize pre-flight diagnostics into a JSON-compatible dict."""
        return {
            "file_path": str(self.file_path),
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "pdf_version": self.pdf_version,
            "is_encrypted": self.is_encrypted,
            "metadata": self.metadata,
            "has_embedded_text_layer": self.has_embedded_text_layer,
            "document_kind": self.document_kind,
            "digital_pages": self.digital_pages,
            "scanned_pages": self.scanned_pages,
            "empty_pages": self.empty_pages,
            "warnings": self.warnings,
            "pages": [
                {
                    "page_no": p.page_no,
                    "width_pt": round(p.width_pt, 2),
                    "height_pt": round(p.height_pt, 2),
                    "text_char_count": p.text_char_count,
                    "word_count": p.word_count,
                    "image_count": p.image_count,
                    "drawing_count": p.drawing_count,
                    "has_embedded_text": p.has_embedded_text,
                    "classification": p.classification,
                    "warnings": p.warnings,
                }
                for p in self.pages
            ],
        }


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_pdf(pdf_path: Path, min_page_chars: int = 50) -> PdfPreflightReport:
    """Validate an input PDF (read-only) and inspect its embedded text layer."""
    if not pdf_path.exists():
        raise PdfInputError(
            f"Input PDF not found at '{pdf_path}'. Place the academic PDF at '{pdf_path}' or pass --input."
        )
    if not pdf_path.is_file():
        raise PdfInputError(f"Input path '{pdf_path}' is not a regular file.")

    file_size = pdf_path.stat().st_size
    if file_size == 0:
        raise PdfInputError(f"Input PDF '{pdf_path}' is empty (0 bytes).")

    with pdf_path.open("rb") as handle:
        header = handle.read(8)
    if not header.startswith(b"%PDF-"):
        raise PdfInputError(
            f"File '{pdf_path}' does not have a valid PDF header (got {header!r})."
        )

    sha256 = _compute_sha256(pdf_path)

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise PdfInputError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    try:
        if doc.needs_pass:
            raise PdfInputError(
                f"Input PDF '{pdf_path}' is password-protected/encrypted and cannot be read without a password."
            )
        if doc.page_count == 0:
            raise PdfInputError(f"Input PDF '{pdf_path}' contains 0 pages.")

        raw_meta = doc.metadata or {}
        metadata = {
            str(k): str(v).strip()
            for k, v in raw_meta.items()
            if v is not None and str(v).strip()
        }
        pdf_version = metadata.get("format")

        pages: list[PagePreflightInfo] = []
        digital_pages: list[int] = []
        scanned_pages: list[int] = []
        empty_pages: list[int] = []
        doc_warnings: list[str] = []

        for idx in range(doc.page_count):
            page_no = idx + 1
            page = doc.load_page(idx)
            rect = page.rect
            text = page.get_text("text") or ""
            stripped = text.strip()
            char_count = len(stripped)
            word_count = len(stripped.split()) if stripped else 0
            images = page.get_images(full=True)
            drawings = page.get_drawings()
            image_count = len(images)
            drawing_count = len(drawings)

            page_warnings: list[str] = []
            has_text = char_count >= 10

            if char_count == 0 and image_count == 0 and drawing_count == 0:
                classification: PageClassification = "empty"
                empty_pages.append(page_no)
                page_warnings.append(f"Page {page_no} appears completely blank (no text, images, or vector drawings).")
            elif not has_text and (image_count > 0 or drawing_count > 0):
                classification = "scanned_or_image"
                scanned_pages.append(page_no)
                page_warnings.append(
                    f"Page {page_no} has little/no embedded text ({char_count} chars) and {image_count} image(s); OCR required."
                )
            else:
                classification = "digital"
                digital_pages.append(page_no)
                if char_count < min_page_chars:
                    page_warnings.append(
                        f"Page {page_no} has unusually short embedded text ({char_count} chars)."
                    )

            if rect.width <= 0 or rect.height <= 0:
                page_warnings.append(f"Page {page_no} has invalid dimensions ({rect.width}x{rect.height}).")

            pages.append(
                PagePreflightInfo(
                    page_no=page_no,
                    width_pt=float(rect.width),
                    height_pt=float(rect.height),
                    text_char_count=char_count,
                    word_count=word_count,
                    image_count=image_count,
                    drawing_count=drawing_count,
                    has_embedded_text=has_text,
                    classification=classification,
                    warnings=page_warnings,
                )
            )

        if len(empty_pages) == doc.page_count:
            document_kind: DocumentKind = "empty"
            doc_warnings.append("All pages in the PDF appear empty.")
        elif len(digital_pages) == doc.page_count:
            document_kind = "digital"
        elif len(digital_pages) == 0 and len(scanned_pages) > 0:
            document_kind = "scanned"
            doc_warnings.append("No embedded text layer detected; treating document as scanned PDF requiring OCR.")
        else:
            document_kind = "mixed"
            if scanned_pages:
                doc_warnings.append(
                    f"Mixed PDF detected: scanned/image-only pages {scanned_pages}, digital pages {digital_pages}."
                )
            if empty_pages:
                doc_warnings.append(f"Empty pages detected: {empty_pages}.")

        has_embedded_text_layer = len(digital_pages) > 0
        logger.info(
            "Pre-flight check complete: %d pages, kind=%s, digital=%d, scanned=%d, empty=%d",
            doc.page_count,
            document_kind,
            len(digital_pages),
            len(scanned_pages),
            len(empty_pages),
        )

        return PdfPreflightReport(
            file_path=pdf_path,
            file_size_bytes=file_size,
            sha256=sha256,
            page_count=doc.page_count,
            pdf_version=pdf_version,
            is_encrypted=bool(doc.is_encrypted),
            metadata=metadata,
            has_embedded_text_layer=has_embedded_text_layer,
            document_kind=document_kind,
            digital_pages=digital_pages,
            scanned_pages=scanned_pages,
            empty_pages=empty_pages,
            pages=pages,
            warnings=doc_warnings,
        )
    finally:
        doc.close()
