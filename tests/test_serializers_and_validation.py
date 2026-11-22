"""Unit tests for math serialization, table complexity and validation checks."""

from __future__ import annotations

from pathlib import Path

from docling_core.types.doc import DoclingDocument, DocItemLabel, TableCell, TableData

from pdf_to_md.assets import AssetExtractionResult
from pdf_to_md.config import PipelineConfig
from pdf_to_md.converter import DoclingConversionOutput
from pdf_to_md.preflight import PagePreflightInfo, PdfPreflightReport
from pdf_to_md.serializers import (
    _escape_underscores_preserving_math,
    analyze_table_complexity,
    assess_formula_quality,
    export_academic_markdown,
    extract_equation_number,
    normalize_inline_math,
)
from pdf_to_md.validation import validate_extraction


def test_normalize_inline_math_preserves_currency() -> None:
    text = "The loss is $L = x_i^2 + y_j$ while the dataset cost $100 or $5.5M."
    out = normalize_inline_math(text)
    assert r"\(L = x_i^2 + y_j\)" in out
    assert "$100" in out
    assert "$5.5M" in out


def test_escape_underscores_preserves_math_and_links() -> None:
    text = r"Given \(x_1 + x_2\) and $$a_i$$ and ![alt_text](assets/figures/fig_01.png) and foo_bar."
    out = _escape_underscores_preserving_math(text)
    assert r"\(x_1 + x_2\)" in out
    assert "$$a_i$$" in out
    assert "![alt_text](assets/figures/fig_01.png)" in out
    assert r"foo\_bar" in out


def test_assess_formula_quality_flags_corrupted_and_undecoded() -> None:
    status_ok, reasons_ok = assess_formula_quality(r"\frac{a}{b} = c", None)
    assert status_ok.startswith("Extracted")
    assert reasons_ok == []

    status_empty, _ = assess_formula_quality("", "raw")
    assert status_empty.startswith("Uncertain")

    status_unbal, reasons_unbal = assess_formula_quality(r"\frac{a}{b", None)
    assert status_unbal.startswith("Uncertain")
    assert any("Unbalanced curly braces" in r for r in reasons_unbal)

    status_dblsup, reasons_dblsup = assess_formula_quality(r"\mu ^ { ^ { - } }", None)
    assert status_dblsup.startswith("Uncertain")
    assert any("superscript" in r for r in reasons_dblsup)


def test_extract_equation_number() -> None:
    assert extract_equation_number(r"E = mc^2 \tag{1}") == "1"
    assert extract_equation_number(r"x + y = z (2.3)") == "2.3"
    assert extract_equation_number(r"a^2 + b^2 = c^2") is None


def test_complex_table_renders_html_while_simple_renders_markdown() -> None:
    doc = DoclingDocument(name="test")
    # Simple table
    simple_data = TableData(
        num_rows=2,
        num_cols=2,
        table_cells=[
            TableCell(start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, text="Model", column_header=True),
            TableCell(start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2, text="Score", column_header=True),
            TableCell(start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1, text="Baseline"),
            TableCell(start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2, text="0.91"),
        ],
    )
    simple_tbl = doc.add_table(data=simple_data)
    assert analyze_table_complexity(simple_tbl, doc)[0] is False

    # Complex table with colspan=2
    complex_data = TableData(
        num_rows=2,
        num_cols=2,
        table_cells=[
            TableCell(start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=2, text="Combined Header", col_span=2, column_header=True),
            TableCell(start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1, text="A"),
            TableCell(start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2, text="B"),
        ],
    )
    complex_tbl = doc.add_table(data=complex_data)
    is_complex, reasons = analyze_table_complexity(complex_tbl, doc)
    assert is_complex is True
    assert "merged_cells" in reasons

    md = export_academic_markdown(doc)
    assert "| Model" in md
    assert "<table" in md and 'colspan="2"' in md


def test_validation_detects_broken_asset_link_and_missing_eq_number(tmp_path: Path) -> None:
    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.TEXT, text="As shown in Eq. (2), the bound holds.")
    preflight = PdfPreflightReport(
        file_path=tmp_path / "paper.pdf",
        file_size_bytes=100,
        sha256="abc",
        page_count=2,
        pdf_version="PDF 1.7",
        is_encrypted=False,
        metadata={},
        has_embedded_text_layer=True,
        document_kind="digital",
        digital_pages=[1, 2],
        scanned_pages=[],
        empty_pages=[],
        pages=[
            PagePreflightInfo(1, 612, 792, 120, 20, 0, 0, True, "digital"),
            PagePreflightInfo(2, 612, 792, 120, 20, 0, 0, True, "digital"),
        ],
    )
    conversion = DoclingConversionOutput(
        document=doc,
        status="success",
        ocr_enabled=True,
        force_full_page_ocr=False,
        formula_enrichment_enabled=True,
        code_enrichment_enabled=True,
    )
    paper_json = {
        "blocks": [
            {"block_id": "blk_0001", "page_no": 1, "text": "See Eq. (1) and Eq. (3)."},
        ],
        "equations": [
            {
                "equation_id": "eq_001",
                "block_id": "blk_0002",
                "page_no": 1,
                "section_path": ["Intro"],
                "equation_number": "1",
                "latex": "a = b",
                "verification_status": "Extracted — not independently verified",
                "uncertainty_reasons": [],
            },
            {
                "equation_id": "eq_002",
                "block_id": "blk_0003",
                "page_no": 1,
                "section_path": ["Intro"],
                "equation_number": "3",
                "latex": r"\frac{1}{2",
                "verification_status": "Uncertain — requires manual inspection",
                "uncertainty_reasons": ["Unbalanced curly braces"],
            },
        ],
        "tables": [],
        "code_blocks": [],
        "figures": [],
    }
    md_text = "See ![fig](assets/figures/nonexistent.png)\n"
    summary = validate_extraction(
        preflight=preflight,
        conversion=conversion,
        doc=doc,
        paper_json=paper_json,
        markdown_text=md_text,
        assets=AssetExtractionResult(),
        output_dir=tmp_path,
        config=PipelineConfig(),
    )
    cats = {i.category for i in summary.issues}
    assert "broken_asset_reference" in cats
    assert "missing_equation_numbering" in cats
    assert "uncertain_equation" in cats
    assert "missing_page_content" in cats
    assert (tmp_path / "validation" / "report.json").exists()
    assert (tmp_path / "validation" / "equations.md").exists()
    assert (tmp_path / "validation" / "issues.md").exists()
