"""Pipeline runner coordinating preflight, conversion, asset extraction and validation."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from pdf_to_md.assets import AssetExtractionResult, extract_assets
from pdf_to_md.config import PipelineConfig
from pdf_to_md.converter import DoclingConversionOutput, convert_pdf_with_docling
from pdf_to_md.docs_gen import write_output_docs
from pdf_to_md.nougat import verify_pages_with_nougat
from pdf_to_md.preflight import PdfPreflightReport, analyze_pdf
from pdf_to_md.serializers import build_structured_json, export_academic_markdown
from pdf_to_md.validation import ValidationSummary, validate_extraction

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineRunResult:
    """Outputs and telemetry produced by a full pipeline execution."""

    output_dir: Path
    preflight: PdfPreflightReport
    conversion: DoclingConversionOutput
    assets: AssetExtractionResult
    validation: ValidationSummary


def _prepare_output_dir(output_dir: Path, clean: bool) -> None:
    if clean and output_dir.exists():
        for sub in ("assets", "validation"):
            sub_path = output_dir / sub
            if sub_path.exists():
                shutil.rmtree(sub_path)
        for fname in ("paper.md", "paper.json", "README.md", "AGENTS.md"):
            f_path = output_dir / fname
            if f_path.exists():
                f_path.unlink()
    for rel in ("assets/figures", "assets/tables", "assets/images", "validation"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def run_pipeline(config: PipelineConfig) -> PipelineRunResult:
    """Run the PDF-to-Markdown pipeline end-to-end."""
    input_path, output_dir = config.resolve_paths()
    logger.info("Starting ingestion pipeline: input=%s, output=%s", input_path, output_dir)

    preflight = analyze_pdf(input_path, min_page_chars=config.min_page_chars_warning)
    _prepare_output_dir(output_dir, clean=config.clean_output)

    conversion = convert_pdf_with_docling(input_path, preflight, config)
    doc = conversion.document

    assets = extract_assets(
        doc=doc,
        pdf_path=input_path,
        output_dir=output_dir,
        images_scale=config.images_scale,
    )

    markdown_text = export_academic_markdown(doc)
    paper_md_path = output_dir / "paper.md"
    paper_md_path.write_text(markdown_text.rstrip() + "\n", encoding="utf-8")

    paper_json = build_structured_json(doc=doc, preflight=preflight, assets=assets)

    nougat_pages_run: list[int] = []
    nougat_warnings: list[str] = []
    if config.verify_nougat:
        target_pages = config.nougat_pages
        if not target_pages:
            uncertain_pages = sorted(
                {
                    int(eq["page_no"])
                    for eq in paper_json["equations"]
                    if eq.get("page_no") and str(eq.get("verification_status", "")).startswith("Uncertain")
                }
            )
            eq_pages = sorted(
                {int(eq["page_no"]) for eq in paper_json["equations"] if eq.get("page_no")}
            )
            target_pages = (uncertain_pages or eq_pages)[: config.nougat_max_pages]

        if target_pages:
            nougat_results, nougat_warnings = verify_pages_with_nougat(
                pdf_path=input_path,
                pages=target_pages,
                equations=paper_json["equations"],
                model_id=config.nougat_model_id,
                local_files_only=not config.enable_remote_services,
            )
            nougat_pages_run = sorted(nougat_results.keys())
            if nougat_pages_run:
                conversion.fallbacks_used.append(
                    f"Nougat secondary mathematical verification ran on page(s): {nougat_pages_run}."
                )

    paper_json_path = output_dir / "paper.json"
    paper_json_path.write_text(
        json.dumps(paper_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    validation = validate_extraction(
        preflight=preflight,
        conversion=conversion,
        doc=doc,
        paper_json=paper_json,
        markdown_text=markdown_text,
        assets=assets,
        output_dir=output_dir,
        config=config,
        nougat_pages_run=nougat_pages_run,
        nougat_warnings=nougat_warnings,
    )

    write_output_docs(
        output_dir=output_dir,
        preflight=preflight,
        paper_json=paper_json,
        validation=validation,
    )

    logger.info(
        "Pipeline complete: status=%s, pages=%d/%d, equations=%d, tables=%d, code=%d, figures=%d",
        validation.overall_status,
        validation.processed_page_count,
        validation.original_page_count,
        validation.counts["equations_total"],
        validation.counts["tables"],
        validation.counts["code_blocks"],
        validation.counts["figures"],
    )

    return PipelineRunResult(
        output_dir=output_dir,
        preflight=preflight,
        conversion=conversion,
        assets=assets,
        validation=validation,
    )
