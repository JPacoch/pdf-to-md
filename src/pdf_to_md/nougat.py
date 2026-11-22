"""Optional local Nougat secondary extraction for mathematical verification."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from pdf_to_md.serializers import strip_outer_math_delimiters

logger = logging.getLogger(__name__)

_NOUGAT_EQ_RE = re.compile(r"(?:\$\$(.*?)\$\$|\\\[(.*?)\\\])", re.DOTALL)


@dataclass(slots=True)
class NougatPageResult:
    """Raw and parsed equation output from Nougat for a single PDF page."""

    page_no: int
    raw_markdown: str
    extracted_equations: list[str] = field(default_factory=list)
    error: str | None = None


def _normalize_for_compare(latex: str) -> str:
    return re.sub(r"\s+", "", strip_outer_math_delimiters(latex))


def verify_pages_with_nougat(
    pdf_path: Path,
    pages: list[int],
    equations: list[dict[str, Any]],
    *,
    model_id: str = "facebook/nougat-base",
    local_files_only: bool = False,
) -> tuple[dict[int, NougatPageResult], list[str]]:
    """Run Nougat on selected pages and attach non-destructive comparisons to equations.

    Conflicting equations are never automatically merged; instead both Docling's
    primary extraction and Nougat's secondary extraction are recorded for review.
    """
    warnings: list[str] = []
    results: dict[int, NougatPageResult] = {}

    if not pages:
        return results, warnings

    try:
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel
    except Exception as exc:
        warnings.append(f"Nougat verification unavailable (transformers/torch import failed): {exc}")
        return results, warnings

    try:
        logger.info("Loading Nougat model '%s' for secondary math verification...", model_id)
        processor = NougatProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        model = VisionEncoderDecoderModel.from_pretrained(model_id, local_files_only=local_files_only)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    except Exception as exc:
        warnings.append(f"Failed to load Nougat model '{model_id}': {exc}")
        return results, warnings

    pdf_doc = pymupdf.open(str(pdf_path))
    try:
        for page_no in sorted(set(pages)):
            if page_no < 1 or page_no > pdf_doc.page_count:
                warnings.append(f"Requested Nougat page {page_no} is out of bounds (1..{pdf_doc.page_count}).")
                continue
            try:
                page = pdf_doc.load_page(page_no - 1)
                pix = page.get_pixmap(dpi=150, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        pixel_values,
                        min_length=1,
                        max_length=3584,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    )
                decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                if hasattr(processor, "post_process_generation"):
                    decoded = processor.post_process_generation(decoded, fix_markdown=False)
                eqs = [
                    (m.group(1) or m.group(2) or "").strip()
                    for m in _NOUGAT_EQ_RE.finditer(decoded)
                    if (m.group(1) or m.group(2) or "").strip()
                ]
                results[page_no] = NougatPageResult(
                    page_no=page_no,
                    raw_markdown=decoded,
                    extracted_equations=eqs,
                )
            except Exception as exc:
                results[page_no] = NougatPageResult(
                    page_no=page_no,
                    raw_markdown="",
                    error=str(exc),
                )
                warnings.append(f"Nougat failed on page {page_no}: {exc}")
    finally:
        pdf_doc.close()

    by_page: dict[int, list[dict[str, Any]]] = {}
    for eq in equations:
        p = eq.get("page_no")
        if isinstance(p, int) and p in results:
            by_page.setdefault(p, []).append(eq)

    for page_no, page_eqs in by_page.items():
        nougat_eqs = results[page_no].extracted_equations
        if not nougat_eqs:
            continue
        for idx, eq in enumerate(page_eqs):
            primary_norm = _normalize_for_compare(eq.get("latex", ""))
            best_match = nougat_eqs[idx] if idx < len(nougat_eqs) else nougat_eqs[0]
            best_norm = _normalize_for_compare(best_match)
            exact_hit = next(
                (nq for nq in nougat_eqs if _normalize_for_compare(nq) == primary_norm and primary_norm),
                None,
            )
            if exact_hit is not None:
                eq["nougat_latex"] = exact_hit
                eq["nougat_comparison"] = "matches_primary"
                if not eq.get("uncertainty_reasons"):
                    eq["verification_status"] = "Verified against secondary extractor (Nougat) — source PDF still authoritative"
            else:
                eq["nougat_latex"] = best_match
                eq["nougat_comparison"] = "differs_from_primary"
                if eq.get("verification_status", "").startswith("Extracted"):
                    eq["verification_status"] = "Uncertain — primary and Nougat extractions differ; requires manual inspection"
                    eq.setdefault("uncertainty_reasons", []).append(
                        "Primary Docling LaTeX differs from Nougat secondary extraction."
                    )

    return results, warnings
