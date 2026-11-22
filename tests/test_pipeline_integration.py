"""Integration test for the full ingestion pipeline using a synthetic academic PDF."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from pdf_to_md.config import PipelineConfig
from pdf_to_md.pipeline import run_pipeline


def _build_synthetic_academic_pdf(pdf_path: Path, tmp_dir: Path) -> None:
    """Create a realistic 2-page synthetic academic PDF with text, table, code and figure."""
    fig_path = tmp_dir / "chart.png"
    img = Image.new("RGB", (480, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 450, 210], outline=(20, 60, 140), width=3)
    draw.line([(60, 180), (160, 120), (280, 140), (410, 60)], fill=(200, 40, 40), width=4)
    img.save(fig_path)

    doc = pymupdf.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(
        (72, 72),
        "A Study on Local Document Ingestion Pipelines",
        fontsize=18,
    )
    p1.insert_text(
        (72, 102),
        "Jane Doe, John Smith",
        fontsize=11,
    )
    p1.insert_text(
        (72, 136),
        "Abstract",
        fontsize=13,
    )
    p1.insert_text(
        (72, 156),
        "We present a reproducible local-first pipeline for converting academic PDFs into\n"
        "structured Markdown and JSON representations while preserving mathematical formulas\n"
        "such as f(x) = x^2 + 1 and tabular benchmarks.",
        fontsize=10.5,
    )
    p1.insert_text((72, 220), "1. Introduction and Algorithm", fontsize=13)
    p1.insert_text(
        (72, 244),
        "The core scoring function is implemented below in Python:",
        fontsize=10.5,
    )
    p1.insert_text(
        (84, 268),
        "def score_page(chars: int, images: int) -> float:\n"
        "    return chars / max(1, images + 1)",
        fontsize=9.5,
        fontname="cour",
    )

    # Simple table drawn with lines and text on page 1
    y0 = 330
    headers = ["Method", "Precision", "Recall"]
    rows = [["Baseline", "0.82", "0.79"], ["Docling", "0.95", "0.93"]]
    all_rows = [headers] + rows
    col_x = [72, 220, 350, 480]
    for r_idx, row in enumerate(all_rows):
        y = y0 + r_idx * 24
        p1.draw_line((col_x[0], y), (col_x[-1], y))
        for c_idx, val in enumerate(row):
            p1.insert_text((col_x[c_idx] + 8, y + 16), val, fontsize=10)
    p1.draw_line((col_x[0], y0 + len(all_rows) * 24), (col_x[-1], y0 + len(all_rows) * 24))
    for cx in col_x:
        p1.draw_line((cx, y0), (cx, y0 + len(all_rows) * 24))

    p1.insert_text((72, 430), "Table 1: Benchmark precision and recall across extractors.", fontsize=9.5)

    # Page 2: Figure + caption + references
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 72), "2. Experimental Results", fontsize=13)
    p2.insert_image(pymupdf.Rect(72, 100, 480, 300), filename=str(fig_path))
    p2.insert_text((72, 320), "Figure 1: Accuracy curve across document complexity levels.", fontsize=9.5)
    p2.insert_text((72, 360), "References", fontsize=13)
    p2.insert_text((72, 384), "[1] Auer et al. Docling Technical Report. 2024.", fontsize=10)

    doc.save(str(pdf_path))
    doc.close()


@pytest.mark.slow
def test_end_to_end_pipeline_with_synthetic_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    _build_synthetic_academic_pdf(pdf_path, tmp_path)

    out_dir = tmp_path / "output"
    config = PipelineConfig(
        input_path=pdf_path,
        output_dir=out_dir,
        ocr_mode="off",
        do_formula_enrichment=False,
        do_code_enrichment=False,
        enable_remote_services=False,
    )
    result = run_pipeline(config)

    assert result.validation.original_page_count == 2
    assert result.validation.processed_page_count == 2
    assert (out_dir / "paper.md").exists()
    assert (out_dir / "paper.json").exists()
    assert (out_dir / "AGENTS.md").exists()
    assert (out_dir / "README.md").exists()
    assert (out_dir / "validation" / "report.json").exists()
    assert (out_dir / "validation" / "report.md").exists()
    assert (out_dir / "validation" / "equations.md").exists()
    assert (out_dir / "validation" / "issues.md").exists()

    parsed = json.loads((out_dir / "paper.json").read_text(encoding="utf-8"))
    assert parsed["source"]["page_count"] == 2
    assert len(parsed["blocks"]) > 0
    md_text = (out_dir / "paper.md").read_text(encoding="utf-8")
    assert "Local Document Ingestion Pipelines" in md_text
