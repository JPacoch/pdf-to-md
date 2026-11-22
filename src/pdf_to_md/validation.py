"""Post-conversion validation, quality assurance and report generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from docling_core.types.doc import DoclingDocument

from pdf_to_md.assets import AssetExtractionResult
from pdf_to_md.config import PipelineConfig
from pdf_to_md.converter import DoclingConversionOutput
from pdf_to_md.preflight import PdfPreflightReport

Severity = Literal["error", "warning", "info"]

_MD_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_EQ_REF_IN_TEXT_RE = re.compile(r"\b(?:Eqs?\.?|Equations?)\s*\(?(\d+(?:\.\d+)*)\)?", re.IGNORECASE)
_IMAGE_PLACEHOLDER_RE = re.compile(r"(?:<!--\s*image\s*-->|🖼️❌)")
_GARBLED_TEXT_RE = re.compile(r"(?:\ufffd|cid:\d+)", re.IGNORECASE)


@dataclass(slots=True)
class ValidationIssue:
    """Single validation error or warning tied to a category and optional page."""

    severity: Severity
    category: str
    message: str
    page_no: int | None = None
    block_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "page_no": self.page_no,
            "block_id": self.block_id,
            "message": self.message,
        }


@dataclass(slots=True)
class ValidationSummary:
    """Complete validation outcome and artifact writer inputs."""

    overall_status: str
    original_page_count: int
    processed_page_count: int
    counts: dict[str, int]
    issues: list[ValidationIssue] = field(default_factory=list)
    pages_requiring_inspection: list[int] = field(default_factory=list)
    fallbacks_used: list[str] = field(default_factory=list)
    nougat_pages_run: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        errors = [i.to_dict() for i in self.issues if i.severity == "error"]
        warnings = [i.to_dict() for i in self.issues if i.severity == "warning"]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_status": self.overall_status,
            "original_page_count": self.original_page_count,
            "processed_page_count": self.processed_page_count,
            "counts": self.counts,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "pages_requiring_manual_inspection": self.pages_requiring_inspection,
            "fallbacks_used": self.fallbacks_used,
            "nougat_pages_run": self.nougat_pages_run,
            "errors": errors,
            "warnings": warnings,
        }


def validate_extraction(
    *,
    preflight: PdfPreflightReport,
    conversion: DoclingConversionOutput,
    doc: DoclingDocument,
    paper_json: dict[str, Any],
    markdown_text: str,
    assets: AssetExtractionResult,
    output_dir: Path,
    config: PipelineConfig,
    nougat_pages_run: list[int] | None = None,
    nougat_warnings: list[str] | None = None,
) -> ValidationSummary:
    """Validate conversion completeness and write reports under validation/."""
    issues: list[ValidationIssue] = []
    inspect_pages: set[int] = set()

    for w in preflight.warnings:
        issues.append(ValidationIssue(severity="warning", category="preflight", message=w))
    for w in conversion.warnings:
        issues.append(ValidationIssue(severity="warning", category="conversion", message=w))
    for w in assets.warnings:
        issues.append(ValidationIssue(severity="warning", category="assets", message=w))
    for w in nougat_warnings or []:
        issues.append(ValidationIssue(severity="warning", category="nougat", message=w))

    for p in preflight.empty_pages:
        inspect_pages.add(p)
        issues.append(
            ValidationIssue(
                severity="warning",
                category="empty_page",
                page_no=p,
                message=f"Page {p} is empty in the source PDF.",
            )
        )
    for p in preflight.scanned_pages:
        inspect_pages.add(p)
        issues.append(
            ValidationIssue(
                severity="warning",
                category="scanned_page",
                page_no=p,
                message=f"Page {p} lacks an embedded text layer and relied on OCR.",
            )
        )

    doc_pages_count = len(doc.pages) if getattr(doc, "pages", None) else 0
    pages_with_blocks = {
        blk["page_no"] for blk in paper_json.get("blocks", []) if isinstance(blk.get("page_no"), int)
    }
    processed_page_count = doc_pages_count or len(pages_with_blocks)
    if doc_pages_count and doc_pages_count != preflight.page_count:
        issues.append(
            ValidationIssue(
                severity="error",
                category="page_count_mismatch",
                message=(
                    f"Original PDF has {preflight.page_count} page(s), "
                    f"but Docling processed {doc_pages_count} page(s)."
                ),
            )
        )

    total_extracted_chars = 0
    page_text_chars: dict[int, int] = {i: 0 for i in range(1, preflight.page_count + 1)}
    for blk in paper_json.get("blocks", []):
        p = blk.get("page_no")
        txt = (blk.get("text") or "").strip()
        if isinstance(p, int) and p in page_text_chars:
            page_text_chars[p] += len(txt)
        total_extracted_chars += len(txt)

    for page_no in range(1, preflight.page_count + 1):
        if page_no not in pages_with_blocks and page_no not in preflight.empty_pages:
            inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="missing_page_content",
                    page_no=page_no,
                    message=f"No content blocks were extracted for non-empty page {page_no}.",
                )
            )
        elif (
            page_no not in preflight.empty_pages
            and page_text_chars.get(page_no, 0) < config.min_page_chars_warning
            and not any(a.page_no == page_no for a in assets.assets)
        ):
            inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="short_page_text",
                    page_no=page_no,
                    message=(
                        f"Page {page_no} yielded unusually short extracted text "
                        f"({page_text_chars.get(page_no, 0)} chars)."
                    ),
                )
            )

    if total_extracted_chars < config.short_document_chars_warning and preflight.document_kind != "empty":
        issues.append(
            ValidationIssue(
                severity="warning",
                category="short_document_text",
                message=(
                    f"Total extracted text across document is unusually short "
                    f"({total_extracted_chars} chars)."
                ),
            )
        )

    equations = paper_json.get("equations", [])
    detected_eq_numbers: list[tuple[int | None, str]] = []
    uncertain_eq_count = 0
    verified_eq_count = 0

    for eq in equations:
        page_no = eq.get("page_no")
        blk_id = eq.get("block_id")
        status = eq.get("verification_status", "")
        reasons = eq.get("uncertainty_reasons") or []
        if status.startswith("Uncertain"):
            uncertain_eq_count += 1
            if isinstance(page_no, int):
                inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="uncertain_equation",
                    page_no=page_no,
                    block_id=blk_id,
                    message=f"{eq.get('equation_id')}: {'; '.join(reasons) or status}",
                )
            )
        elif status.startswith("Verified"):
            verified_eq_count += 1

        eq_num = eq.get("equation_number")
        if eq_num:
            detected_eq_numbers.append((page_no, eq_num))

    int_eq_nums: list[int] = []
    for _, num in detected_eq_numbers:
        if num.isdigit():
            int_eq_nums.append(int(num))
    if len(int_eq_nums) >= 2:
        expected = set(range(1, max(int_eq_nums) + 1))
        missing_nums = sorted(expected - set(int_eq_nums))
        if missing_nums:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="missing_equation_numbering",
                    message=f"Possible missing equation numbers in sequence: {missing_nums} (detected {sorted(set(int_eq_nums))}).",
                )
            )

    known_num_set = {num for _, num in detected_eq_numbers}
    referenced_nums: set[str] = set()
    for blk in paper_json.get("blocks", []):
        txt = blk.get("text") or ""
        for m in _EQ_REF_IN_TEXT_RE.finditer(txt):
            referenced_nums.add(m.group(1))
    if referenced_nums and equations and not known_num_set:
        issues.append(
            ValidationIssue(
                severity="warning",
                category="missing_equation_numbering",
                message=(
                    f"Document references equation number(s) {sorted(referenced_nums)}, "
                    "but extracted display equations have no detected equation numbers."
                ),
            )
        )

    for blk in paper_json.get("blocks", []):
        txt = blk.get("text") or ""
        if _GARBLED_TEXT_RE.search(txt):
            page_no = blk.get("page_no")
            if isinstance(page_no, int):
                inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="garbled_text",
                    page_no=page_no,
                    block_id=blk.get("block_id"),
                    message=f"Block {blk.get('block_id')} contains replacement/CID glyph artifacts.",
                )
            )

    tables = paper_json.get("tables", [])
    for tbl in tables:
        page_no = tbl.get("page_no")
        blk_id = tbl.get("block_id")
        num_rows = tbl.get("num_rows", 0)
        num_cols = tbl.get("num_cols", 0)
        rows = tbl.get("rows") or []
        if num_rows == 0 or num_cols == 0 or not rows:
            if isinstance(page_no, int):
                inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="broken_table",
                    page_no=page_no,
                    block_id=blk_id,
                    message=f"{tbl.get('table_id')} has empty dimensions ({num_rows}x{num_cols}).",
                )
            )
            continue
        row_lengths = {len(r) for r in rows}
        if len(row_lengths) > 1:
            if isinstance(page_no, int):
                inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="broken_table",
                    page_no=page_no,
                    block_id=blk_id,
                    message=f"{tbl.get('table_id')} has ragged row widths {sorted(row_lengths)}.",
                )
            )
        total_cells = sum(len(r) for r in rows)
        empty_cells = sum(1 for r in rows for c in r if not str(c).strip())
        if total_cells > 0 and empty_cells / total_cells > 0.75:
            if isinstance(page_no, int):
                inspect_pages.add(page_no)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    category="sparse_table",
                    page_no=page_no,
                    block_id=blk_id,
                    message=(
                        f"{tbl.get('table_id')} has {empty_cells}/{total_cells} empty cells; "
                        "verify structure against table snapshot."
                    ),
                )
            )

    if _IMAGE_PLACEHOLDER_RE.search(markdown_text):
        issues.append(
            ValidationIssue(
                severity="warning",
                category="unresolved_image_placeholder",
                message="paper.md still contains unresolved image placeholder comments.",
            )
        )

    for m in _MD_IMAGE_LINK_RE.finditer(markdown_text):
        raw_target = unquote(m.group(1).strip())
        if raw_target.startswith(("http://", "https://", "data:")):
            continue
        target_path = (output_dir / raw_target).resolve()
        if not target_path.exists() or target_path.stat().st_size == 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="broken_asset_reference",
                    message=f"Markdown references missing or empty asset '{raw_target}'.",
                )
            )

    for fig in paper_json.get("figures", []):
        asset_rel = fig.get("asset_path")
        if not asset_rel:
            p = fig.get("page_no")
            if isinstance(p, int):
                inspect_pages.add(p)
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="missing_figure_asset",
                    page_no=p,
                    block_id=fig.get("block_id"),
                    message=f"Figure {fig.get('figure_id')} has no extracted image file.",
                )
            )

    has_errors = any(i.severity == "error" for i in issues)
    has_warnings = any(i.severity == "warning" for i in issues)
    if has_errors:
        overall_status = "completed_with_errors"
    elif has_warnings:
        overall_status = "completed_with_warnings"
    else:
        overall_status = "success"

    counts = {
        "equations_total": len(equations),
        "equations_verified": verified_eq_count,
        "equations_unverified": len(equations) - verified_eq_count - uncertain_eq_count,
        "equations_uncertain": uncertain_eq_count,
        "tables": len(tables),
        "code_blocks": len(paper_json.get("code_blocks", [])),
        "figures": len(paper_json.get("figures", [])),
        "assets_extracted": len(assets.assets),
    }

    summary = ValidationSummary(
        overall_status=overall_status,
        original_page_count=preflight.page_count,
        processed_page_count=processed_page_count,
        counts=counts,
        issues=issues,
        pages_requiring_inspection=sorted(inspect_pages),
        fallbacks_used=conversion.fallbacks_used,
        nougat_pages_run=sorted(nougat_pages_run or []),
    )

    _write_validation_files(
        summary=summary,
        equations=equations,
        output_dir=output_dir,
        preflight=preflight,
        conversion=conversion,
    )
    return summary


def _write_validation_files(
    *,
    summary: ValidationSummary,
    equations: list[dict[str, Any]],
    output_dir: Path,
    preflight: PdfPreflightReport,
    conversion: DoclingConversionOutput,
) -> None:
    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = val_dir / "report.json"
    report_dict = summary.to_dict()
    report_dict["preflight"] = preflight.to_dict()
    report_dict["conversion"] = {
        "status": conversion.status,
        "ocr_enabled": conversion.ocr_enabled,
        "force_full_page_ocr": conversion.force_full_page_ocr,
        "formula_enrichment_enabled": conversion.formula_enrichment_enabled,
        "code_enrichment_enabled": conversion.code_enrichment_enabled,
        "elapsed_seconds": round(conversion.elapsed_seconds, 2),
    }
    report_json_path.write_text(
        json.dumps(report_dict, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    errors = [i for i in summary.issues if i.severity == "error"]
    warnings = [i for i in summary.issues if i.severity == "warning"]
    lines = [
        "# Validation Report",
        "",
        f"- **Overall Status:** `{summary.overall_status}`",
        f"- **Source PDF:** `{preflight.file_path}` ({preflight.document_kind}, embedded text layer: `{preflight.has_embedded_text_layer}`)",
        f"- **Original Pages:** {summary.original_page_count}",
        f"- **Processed Pages:** {summary.processed_page_count}",
        f"- **Conversion Elapsed:** {conversion.elapsed_seconds:.2f}s",
        "",
        "## Extraction Counts",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
    ]
    for k, v in summary.counts.items():
        lines.append(f"| `{k}` | {v} |")

    lines.extend(["", "## Fallback Processing", ""])
    if summary.fallbacks_used:
        for fb in summary.fallbacks_used:
            lines.append(f"- {fb}")
    else:
        lines.append("- None required.")

    if summary.nougat_pages_run:
        lines.append(f"- Nougat secondary verification executed on page(s): {summary.nougat_pages_run}")

    lines.extend(["", "## Pages Requiring Manual Inspection", ""])
    if summary.pages_requiring_inspection:
        lines.append(
            "- Pages: " + ", ".join(str(p) for p in summary.pages_requiring_inspection)
        )
    else:
        lines.append("- None flagged automatically (spot-check math and tables against the source PDF).")

    lines.extend(["", f"## Errors ({len(errors)})", ""])
    if errors:
        for err in errors:
            loc = f" (page {err.page_no})" if err.page_no else ""
            lines.append(f"- **[{err.category}]**{loc}: {err.message}")
    else:
        lines.append("- None.")

    lines.extend(["", f"## Warnings ({len(warnings)})", ""])
    if warnings:
        for warn in warnings:
            loc = f" (page {warn.page_no})" if warn.page_no else ""
            lines.append(f"- **[{warn.category}]**{loc}: {warn.message}")
    else:
        lines.append("- None.")

    (val_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    eq_lines = [
        "# Extracted Equations & Mathematical Verification Log",
        "",
        "Status definitions:",
        "- `Extracted — not independently verified`: Syntactically valid LaTeX extracted by Docling; not independently verified against the source PDF.",
        "- `Verified against source` / `Verified against secondary extractor (Nougat)`: Cross-checked.",
        "- `Uncertain — requires manual inspection`: Undecoded, syntactically suspect, or conflicting with secondary extraction.",
        "",
    ]
    if not equations:
        eq_lines.append("_No display `FormulaItem` blocks were detected by Docling._")
    else:
        for eq in equations:
            sec = " > ".join(eq.get("section_path") or []) or "Document Body"
            page_no = eq.get("page_no") or "?"
            eq_num = eq.get("equation_number")
            num_badge = f" (Eq. {eq_num})" if eq_num else ""
            eq_lines.extend(
                [
                    f"## {eq['equation_id']}{num_badge} — Page {page_no}",
                    "",
                    f"- **Section:** {sec}",
                    f"- **Block ID:** `{eq['block_id']}`",
                    f"- **Verification Status:** `{eq['verification_status']}`",
                ]
            )
            if eq.get("uncertainty_reasons"):
                eq_lines.append(
                    "- **Notes:** " + "; ".join(eq["uncertainty_reasons"])
                )
            latex = eq.get("latex") or ""
            eq_lines.append("")
            if latex:
                eq_lines.append("```latex")
                eq_lines.append(latex)
                eq_lines.append("```")
            else:
                eq_lines.append("_Primary LaTeX extraction empty._")
                if eq.get("raw_orig"):
                    eq_lines.append(f"Raw text fallback: `{eq['raw_orig']}`")
            if eq.get("nougat_latex"):
                eq_lines.extend(
                    [
                        "",
                        f"**Nougat Secondary Extraction (`{eq.get('nougat_comparison')}`):**",
                        "",
                        "```latex",
                        eq["nougat_latex"],
                        "```",
                    ]
                )
            eq_lines.append("")

    (val_dir / "equations.md").write_text("\n".join(eq_lines) + "\n", encoding="utf-8")

    issue_lines = [
        "# Extraction Issues & Manual Review Checklist",
        "",
        f"Total errors: {len(errors)} | Total warnings: {len(warnings)}",
        "",
    ]
    if not summary.issues:
        issue_lines.append("- No automated errors or warnings were raised during conversion.")
    else:
        for idx, issue in enumerate(summary.issues, start=1):
            loc_parts = []
            if issue.page_no is not None:
                loc_parts.append(f"page {issue.page_no}")
            if issue.block_id:
                loc_parts.append(f"block `{issue.block_id}`")
            loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
            issue_lines.append(
                f"{idx}. **[{issue.severity.upper()}] `{issue.category}`**{loc}: {issue.message}"
            )
    (val_dir / "issues.md").write_text("\n".join(issue_lines) + "\n", encoding="utf-8")
