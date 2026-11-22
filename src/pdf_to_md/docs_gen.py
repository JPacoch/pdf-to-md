from __future__ import annotations

from pathlib import Path

from pdf_to_md.preflight import PdfPreflightReport
from pdf_to_md.validation import ValidationSummary


def write_output_docs(
    *,
    output_dir: Path,
    preflight: PdfPreflightReport,
    paper_json: dict[str, object],
    validation: ValidationSummary,
) -> None:
    """Write output/AGENTS.md and output/README.md."""
    agents_md = """# Agent Consumption Rules (`output/`)

1. `paper.md` is the primary human-readable representation of the paper.
2. `paper.json` contains structured blocks, section hierarchy, bounding boxes, page numbers, tables, code blocks, figures and equation metadata.
3. Mathematical expressions use LaTeX (`\\( ... \\)` inline, `$$ ... $$` display).
4. The original PDF (`input/paper.pdf`) is the authoritative source of truth.
5. Cite original PDF page numbers (from `paper.json`) when referencing claims, tables, figures, or equations.
6. Do not treat expressions marked `Uncertain` or `Extracted — not independently verified` in `validation/equations.md` (or flagged with `<!-- UNCERTAIN_FORMULA ... -->` in `paper.md`) as authoritative without checking the source PDF.
7. Extracted figures, table snapshots and standalone images live in `assets/figures/`, `assets/tables/` and `assets/images/`.
8. Inspect the original PDF whenever extracted text, math, tables, or code appear incomplete or ambiguous and check `validation/issues.md` and `validation/report.md`.
"""
    (output_dir / "AGENTS.md").write_text(agents_md, encoding="utf-8")

    counts = validation.counts
    title = (paper_json.get("metadata") or {}).get("title") or preflight.file_path.name  # type: ignore[union-attr]
    readme_md = f"""# Extracted Paper Output

- Source: `{preflight.file_path.name}`
- Title: {title}
- Kind: `{preflight.document_kind}` (embedded text layer: `{preflight.has_embedded_text_layer}`)
- Pages: {validation.processed_page_count} / {validation.original_page_count}
- Status: `{validation.overall_status}`

## Files

- `paper.md`: Markdown export of the paper (complex tables rendered as HTML).
- `paper.json`: Structured JSON export (`schema_version`: `{paper_json.get("schema_version")}`).
- `AGENTS.md`: Consumption rules for automated agents.
- `assets/figures/`: Extracted figures ({counts.get("figures", 0)}).
- `assets/tables/`: Cropped table snapshots ({counts.get("tables", 0)}).
- `assets/images/`: Standalone embedded images ({len(paper_json.get("standalone_images", []))}).
- `validation/report.json`, `validation/report.md`: Validation counts, fallbacks and flagged pages.
- `validation/equations.md`: Display equations and verification status.
- `validation/issues.md`: Warnings and errors from conversion.
"""
    (output_dir / "README.md").write_text(readme_md, encoding="utf-8")
