"""Unit tests for pre-flight PDF validation and text-layer detection."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from pdf_to_md.preflight import PdfInputError, analyze_pdf


def test_missing_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(PdfInputError, match="not found"):
        analyze_pdf(tmp_path / "missing.pdf")


def test_invalid_header_raises(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_pdf.pdf"
    bad.write_text("hello world", encoding="utf-8")
    with pytest.raises(PdfInputError, match="valid PDF header"):
        analyze_pdf(bad)


def test_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(PdfInputError, match="empty"):
        analyze_pdf(empty)


def test_digital_scanned_and_empty_pages_classification(tmp_path: Path) -> None:
    pdf_path = tmp_path / "mixed.pdf"
    img_path = tmp_path / "scan.png"
    Image.new("RGB", (400, 400), color=(240, 240, 240)).save(img_path)

    doc = pymupdf.open()
    # Page 1: digital text
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Academic Paper Title\nThis page has plenty of embedded digital text for testing.")
    # Page 2: scanned image only (no text)
    p2 = doc.new_page()
    p2.insert_image(pymupdf.Rect(50, 50, 350, 350), filename=str(img_path))
    # Page 3: completely blank
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    report = analyze_pdf(pdf_path)
    assert report.page_count == 3
    assert report.document_kind == "mixed"
    assert report.has_embedded_text_layer is True
    assert report.digital_pages == [1]
    assert report.scanned_pages == [2]
    assert report.empty_pages == [3]
