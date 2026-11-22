"""Docling PDF conversion, OCR configuration and digital code-block recovery."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import re
import textwrap
from typing import Any

import pymupdf
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import CodeItem, DoclingDocument

from pdf_to_md.config import PipelineConfig
from pdf_to_md.preflight import PdfPreflightReport

logger = logging.getLogger(__name__)


class PdfConversionError(Exception):
    """Raised when Docling fails to convert the PDF document."""


@dataclass(slots=True)
class DoclingConversionOutput:
    """Container for Docling conversion results and fallback telemetry."""

    document: DoclingDocument
    status: str
    ocr_enabled: bool
    force_full_page_ocr: bool
    formula_enrichment_enabled: bool
    code_enrichment_enabled: bool
    fallbacks_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _resolve_bundled_rapidocr_paths() -> tuple[str | None, str | None, str | None]:
    """Locate bundled ONNX models shipped inside the rapidocr package for offline OCR."""
    try:
        import rapidocr

        models_dir = Path(rapidocr.__file__).parent / "models"
        det = models_dir / "PP-OCRv6_det_small.onnx"
        cls = models_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
        rec = models_dir / "PP-OCRv6_rec_small.onnx"
        if det.is_file() and cls.is_file() and rec.is_file():
            return str(det), str(cls), str(rec)
    except Exception:
        pass
    return None, None, None


def _resolve_local_artifacts_path() -> Path | None:
    """Return Docling's local artifacts cache directory if models are pre-downloaded."""
    candidate = Path(settings.cache_dir) / "models"
    if (candidate / "docling-project--docling-layout-heron").is_dir():
        return candidate
    return None


def _build_pipeline_options(
    config: PipelineConfig,
    *,
    do_ocr: bool,
    force_full_page_ocr: bool,
    do_formula_enrichment: bool,
    do_code_enrichment: bool,
) -> PdfPipelineOptions:
    table_mode = (
        TableFormerMode.ACCURATE
        if config.table_mode == "accurate"
        else TableFormerMode.FAST
    )
    artifacts_path = _resolve_local_artifacts_path()
    det_path, cls_path, rec_path = _resolve_bundled_rapidocr_paths()
    ocr_opts = RapidOcrOptions(
        lang=config.ocr_lang,
        backend="onnxruntime",
        mode=OcrMode.FULL_PAGE if force_full_page_ocr else OcrMode.DEFAULT,
        det_model_path=det_path,
        cls_model_path=cls_path,
        rec_model_path=rec_path,
    )
    opts = PdfPipelineOptions(
        artifacts_path=artifacts_path,
        enable_remote_services=config.enable_remote_services,
        document_timeout=config.document_timeout,
        do_ocr=do_ocr,
        ocr_options=ocr_opts,
        do_table_structure=config.do_table_structure,
        table_structure_options=TableStructureOptions(
            mode=table_mode,
            do_cell_matching=True,
        ),
        do_formula_enrichment=do_formula_enrichment,
        do_code_enrichment=do_code_enrichment,
        generate_picture_images=config.generate_picture_images,
        generate_page_images=config.generate_page_images,
        images_scale=config.images_scale,
    )
    return opts


def _should_use_ocr(config: PipelineConfig, preflight: PdfPreflightReport) -> tuple[bool, bool]:
    """Return (do_ocr, force_full_page_ocr) for the document."""
    if config.ocr_mode == "off":
        return False, False
    if config.ocr_mode == "always":
        return True, True
    if preflight.document_kind == "scanned":
        return True, True
    if preflight.scanned_pages:
        return True, False
    # Keep RapidOCR enabled on digital PDFs so embedded bitmap figures can still be OCRed.
    return True, False


def _run_docling(pdf_path: Path, pipeline_options: PdfPipelineOptions) -> Any:
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter.convert(str(pdf_path), raises_on_error=False)


def convert_pdf_with_docling(
    pdf_path: Path,
    preflight: PdfPreflightReport,
    config: PipelineConfig,
) -> DoclingConversionOutput:
    """Convert a PDF into a DoclingDocument with automatic fallbacks and diagnostics."""
    start = time.monotonic()
    do_ocr, force_full_page_ocr = _should_use_ocr(config, preflight)
    do_formula = config.do_formula_enrichment
    do_code = config.do_code_enrichment
    fallbacks_used: list[str] = []
    warnings: list[str] = []

    if force_full_page_ocr and preflight.document_kind == "scanned":
        fallbacks_used.append(
            "Enabled full-page RapidOCR because no embedded text layer was detected."
        )
    elif preflight.scanned_pages and do_ocr:
        fallbacks_used.append(
            f"Enabled RapidOCR for scanned/image-only pages: {preflight.scanned_pages}."
        )

    logger.info(
        "Running Docling conversion (ocr=%s, force_full_page_ocr=%s, formula=%s, code=%s, table_mode=%s)",
        do_ocr,
        force_full_page_ocr,
        do_formula,
        do_code,
        config.table_mode,
    )

    conv_res = None
    try:
        opts = _build_pipeline_options(
            config,
            do_ocr=do_ocr,
            force_full_page_ocr=force_full_page_ocr,
            do_formula_enrichment=do_formula,
            do_code_enrichment=do_code,
        )
        conv_res = _run_docling(pdf_path, opts)
    except Exception as exc:
        logger.warning("Primary Docling pipeline failed (%s); attempting fallback configuration.", exc)
        warnings.append(f"Primary Docling conversion raised exception: {exc}")

        # Retry without VLM enrichments first if model loading fails offline
        if do_formula or do_code:
            try:
                do_formula = False
                do_code = False
                fallbacks_used.append(
                    f"Disabled formula/code VLM enrichment after primary conversion error: {exc}"
                )
                opts = _build_pipeline_options(
                    config,
                    do_ocr=do_ocr,
                    force_full_page_ocr=force_full_page_ocr,
                    do_formula_enrichment=False,
                    do_code_enrichment=False,
                )
                conv_res = _run_docling(pdf_path, opts)
            except Exception as exc2:
                warnings.append(f"Fallback conversion without enrichment also failed: {exc2}")
                conv_res = None

        # Retry with OCR disabled if RapidOCR fails to initialize
        if conv_res is None and do_ocr:
            try:
                do_ocr = False
                force_full_page_ocr = False
                fallbacks_used.append("Disabled RapidOCR after OCR initialization failure.")
                opts = _build_pipeline_options(
                    config,
                    do_ocr=False,
                    force_full_page_ocr=False,
                    do_formula_enrichment=False,
                    do_code_enrichment=False,
                )
                conv_res = _run_docling(pdf_path, opts)
            except Exception as exc3:
                raise PdfConversionError(
                    f"Docling failed to convert '{pdf_path}' even after fallbacks: {exc3}"
                ) from exc3

        if conv_res is None:
            raise PdfConversionError(f"Docling failed to convert '{pdf_path}': {exc}") from exc

    if conv_res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        err_msgs = [str(getattr(e, "error_message", e)) for e in getattr(conv_res, "errors", [])]
        raise PdfConversionError(
            f"Docling conversion failed with status={conv_res.status}: {'; '.join(err_msgs) or 'unknown error'}"
        )

    if conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
        warnings.append("Docling reported PARTIAL_SUCCESS; some pages or items may have errors.")

    for err in getattr(conv_res, "errors", []) or []:
        warnings.append(f"Docling error: {getattr(err, 'error_message', err)}")

    doc: DoclingDocument = conv_res.document

    extracted_text_len = sum(len((t.text or "").strip()) for t in doc.texts)
    if (
        extracted_text_len < 30
        and preflight.document_kind != "empty"
        and config.ocr_mode != "off"
        and not force_full_page_ocr
    ):
        logger.warning(
            "Extracted text is very short (%d chars); retrying with force_full_page_ocr=True.",
            extracted_text_len,
        )
        try:
            opts_retry = _build_pipeline_options(
                config,
                do_ocr=True,
                force_full_page_ocr=True,
                do_formula_enrichment=do_formula,
                do_code_enrichment=do_code,
            )
            retry_res = _run_docling(pdf_path, opts_retry)
            if retry_res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                retry_len = sum(len((t.text or "").strip()) for t in retry_res.document.texts)
                if retry_len > extracted_text_len:
                    conv_res = retry_res
                    doc = retry_res.document
                    force_full_page_ocr = True
                    do_ocr = True
                    fallbacks_used.append(
                        f"Retried with force_full_page_ocr=True (recovered {retry_len} chars vs {extracted_text_len})."
                    )
        except Exception as exc:
            warnings.append(f"Full-page OCR retry failed: {exc}")

    elapsed = time.monotonic() - start
    refined_code_count = refine_digital_code_blocks(doc=doc, pdf_path=pdf_path, preflight=preflight)
    if refined_code_count:
        fallbacks_used.append(
            f"Recovered exact embedded PDF text for {refined_code_count} code block(s) on digital pages to avoid VLM OCR artifacts."
        )

    logger.info(
        "Docling conversion finished in %.2fs (status=%s, texts=%d, tables=%d, pictures=%d)",
        elapsed,
        conv_res.status.value if hasattr(conv_res.status, "value") else conv_res.status,
        len(doc.texts),
        len(doc.tables),
        len(doc.pictures),
    )

    return DoclingConversionOutput(
        document=doc,
        status=str(conv_res.status.value if hasattr(conv_res.status, "value") else conv_res.status),
        ocr_enabled=do_ocr,
        force_full_page_ocr=force_full_page_ocr,
        formula_enrichment_enabled=do_formula,
        code_enrichment_enabled=do_code,
        fallbacks_used=fallbacks_used,
        warnings=warnings,
        elapsed_seconds=elapsed,
    )


def _infer_code_language(code_text: str, current_lang: str | None) -> str | None:
    """Fix obvious language misclassifications from VLM code enrichment."""
    stripped = code_text.strip()
    if re.search(r"(?m)^\[(?:package|dependencies|dev-dependencies|workspace|build-dependencies)\]\s*$", stripped):
        return "toml"
    if re.search(r"(?m)^(?:#\[(?:repr|derive)|pub\s+(?:struct|enum|fn|trait|mod)|impl\b)", stripped):
        return "rust"
    return current_lang


def refine_digital_code_blocks(
    *,
    doc: DoclingDocument,
    pdf_path: Path,
    preflight: PdfPreflightReport,
) -> int:
    """Replace VLM OCR code text with exact PDF text-layer content on digital pages."""
    digital_set = set(preflight.digital_pages)
    if not digital_set:
        return 0

    pdf_doc = pymupdf.open(str(pdf_path))
    refined = 0
    try:
        for item in doc.texts:
            if not isinstance(item, CodeItem) or not item.prov:
                continue
            prov = item.prov[0]
            if prov.page_no not in digital_set or prov.page_no > pdf_doc.page_count:
                continue
            page = pdf_doc.load_page(prov.page_no - 1)
            tl_bbox = prov.bbox.to_top_left_origin(page_height=float(page.rect.height))
            clip = pymupdf.Rect(tl_bbox.l - 1.0, tl_bbox.t - 1.0, tl_bbox.r + 1.0, tl_bbox.b + 1.0)
            extracted = page.get_text("text", clip=clip)
            lines = [ln.rstrip() for ln in extracted.splitlines()]
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            clean_code = textwrap.dedent("\n".join(lines))
            if not clean_code or "\ufffd" in clean_code:
                continue
            if clean_code != (item.text or "").strip():
                item.orig = item.text or item.orig
                item.text = clean_code
                refined += 1
            raw_lang = getattr(item, "code_language", None)
            lang_str = str(raw_lang.value if hasattr(raw_lang, "value") else raw_lang or "").lower() or None
            inferred = _infer_code_language(clean_code, lang_str)
            if inferred and inferred != lang_str:
                try:
                    from docling_core.types.doc.labels import CodeLanguageLabel

                    item.code_language = CodeLanguageLabel(inferred)
                except Exception:
                    pass
    finally:
        pdf_doc.close()

    return refined
