"""Command-line interface for the academic PDF-to-Markdown ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from pdf_to_md.config import PipelineConfig
from pdf_to_md.converter import PdfConversionError
from pdf_to_md.pipeline import run_pipeline
from pdf_to_md.preflight import PdfInputError


def _parse_pages(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    pages: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            pages.extend(range(min(start, end), max(start, end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages)) or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-to-md",
        description="Convert an academic PDF into structured Markdown, JSON, assets and validation reports.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/paper.pdf"),
        help="Path to the input PDF (default: input/paper.pdf).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output).",
    )
    parser.add_argument(
        "--ocr",
        choices=["auto", "always", "off"],
        default="auto",
        help="OCR strategy: auto (default), always (force full-page OCR), or off.",
    )
    parser.add_argument(
        "--ocr-lang",
        nargs="+",
        default=["en"],
        help="OCR language codes for RapidOCR (default: en).",
    )
    parser.add_argument(
        "--table-mode",
        choices=["accurate", "fast"],
        default="accurate",
        help="TableFormer table structure mode (default: accurate).",
    )
    parser.add_argument(
        "--no-formula-enrichment",
        action="store_true",
        help="Disable Docling formula-to-LaTeX enrichment.",
    )
    parser.add_argument(
        "--no-code-enrichment",
        action="store_true",
        help="Disable Docling code block enrichment.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=2.0,
        help="Rasterization scale for extracted figures/tables (default: 2.0).",
    )
    parser.add_argument(
        "--verify-nougat",
        action="store_true",
        help="Run local Nougat secondary verification on uncertain/math pages.",
    )
    parser.add_argument(
        "--nougat-pages",
        type=str,
        default=None,
        help="Comma-separated page numbers or ranges for Nougat verification (e.g. 2,4-6).",
    )
    parser.add_argument(
        "--allow-remote-downloads",
        action="store_true",
        help="Allow Hugging Face / model downloads during conversion if local cache is incomplete.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean existing files inside output/ before running.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        nougat_pages = _parse_pages(args.nougat_pages)
    except ValueError as exc:
        parser.error(f"Invalid --nougat-pages value '{args.nougat_pages}': {exc}")
        return 2

    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output,
        ocr_mode=args.ocr,
        ocr_lang=args.ocr_lang,
        table_mode=args.table_mode,
        do_formula_enrichment=not args.no_formula_enrichment,
        do_code_enrichment=not args.no_code_enrichment,
        images_scale=args.image_scale,
        verify_nougat=args.verify_nougat,
        nougat_pages=nougat_pages,
        enable_remote_services=args.allow_remote_downloads,
        clean_output=not args.no_clean,
        log_level=args.log_level,
    )

    try:
        result = run_pipeline(config)
    except PdfInputError as exc:
        logging.error("Input validation error: %s", exc)
        return 1
    except PdfConversionError as exc:
        logging.error("PDF conversion error: %s", exc)
        return 1
    except Exception as exc:
        logging.exception("Unexpected pipeline failure: %s", exc)
        return 1

    print(
        f"Done ({result.validation.overall_status}): "
        f"{result.validation.processed_page_count}/{result.validation.original_page_count} pages -> {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
