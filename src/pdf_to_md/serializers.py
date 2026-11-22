"""Custom Markdown serialization and structured JSON export for academic PDFs."""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
    MarkdownTextSerializer,
    _caption_goes_after,
)
from docling_core.types.doc import (
    CodeItem,
    DocItemLabel,
    DoclingDocument,
    FormulaItem,
    ImageRefMode,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.document import FloatingItem, ImageRef, RichTableCell

from pdf_to_md.assets import AssetExtractionResult
from pdf_to_md.preflight import PdfPreflightReport

_FORMULA_NOT_DECODED = "<!-- formula-not-decoded -->"
_MATH_TOKEN_RE = re.compile(
    r"(\\[a-zA-Z]+|\\[()\[\]]|\$\$.*?\$\$|\\\(.*?\\\)|\\\[.*?\\\]|"
    r"[\u0370-\u03FF\u2200-\u22FF\u2190-\u21FF]|[_^]\{|[a-zA-Z0-9]\s*[=<>≤≥≈≠±×÷]\s*[a-zA-Z0-9])",
    re.DOTALL,
)
_INLINE_DOLLAR_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^\n$]+?)(?<!\\)\$(?!\$)")
_CURRENCY_LIKE_RE = re.compile(r"^\s*\d[\d,]*(?:\.\d+)?\s*(?:[KMBTkmbt]|USD|EUR|GBP)?\s*$")
_TRAILING_EQ_NUM_RE = re.compile(r"(?:\\tag\{([^}]+)\}|\((\d+(?:\.\d+)*)\))\s*$")
_MATH_SEGMENT_RE = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$[^\n$]+?(?<!\\)\$|`[^`]+`|!\[.*?\]\(.*?\)|\[[^\]]+\]\([^)]+\))",
    re.DOTALL,
)


def strip_outer_math_delimiters(latex: str) -> str:
    """Remove redundant outer math delimiters from an extracted LaTeX formula."""
    s = latex.strip()
    for left, right in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if s.startswith(left) and s.endswith(right) and len(s) > len(left) + len(right):
            inner = s[len(left) : len(s) - len(right)].strip()
            if inner:
                return inner
    return s


def extract_equation_number(latex: str, orig: str | None = None) -> str | None:
    """Detect an equation number such as '(1)', '(2.3)', or '\\tag{1}' if present."""
    for candidate in (latex or "", orig or ""):
        m = _TRAILING_EQ_NUM_RE.search(candidate.strip())
        if m:
            return (m.group(1) or m.group(2) or "").strip() or None
    return None


def assess_formula_quality(text: str, orig: str | None) -> tuple[str, list[str]]:
    """Assess extracted LaTeX formula quality and return (status, uncertainty_reasons)."""
    reasons: list[str] = []
    cleaned = (text or "").strip()

    if not cleaned or cleaned == _FORMULA_NOT_DECODED:
        reasons.append("Formula text is empty or marked undecoded by primary extractor.")
        return "Uncertain — requires manual inspection", reasons

    if "\ufffd" in cleaned or "cid:" in cleaned.lower():
        reasons.append("Contains replacement character or unresolved PDF CID glyph reference.")

    if cleaned.count("{") != cleaned.count("}"):
        reasons.append(
            f"Unbalanced curly braces ({{={cleaned.count('{')}, }}={cleaned.count('}')})."
        )

    left_count = len(re.findall(r"\\left\b", cleaned))
    right_count = len(re.findall(r"\\right\b", cleaned))
    if left_count != right_count:
        reasons.append(f"Unbalanced \\left/\\right delimiters ({left_count} vs {right_count}).")

    begins = re.findall(r"\\begin\{([^}]+)\}", cleaned)
    ends = re.findall(r"\\end\{([^}]+)\}", cleaned)
    if begins != ends:
        reasons.append(f"Mismatched LaTeX environments (begin={begins}, end={ends}).")

    if re.search(r"(\\(?:quad|qquad|;|,)\s*){6,}", cleaned):
        reasons.append("Contains suspicious repeated spacing commands (possible decoder loop).")

    if re.search(r"\^\s*\{\s*\^", cleaned):
        reasons.append("Contains nested empty-base superscript (`^ { ^`), likely a misread diacritic/overbar.")

    if reasons:
        return "Uncertain — requires manual inspection", reasons
    return "Extracted — not independently verified", []


def normalize_inline_math(text: str) -> str:
    """Convert clear inline $...$ LaTeX expressions to \\( ... \\) while preserving currency."""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if _CURRENCY_LIKE_RE.match(inner):
            return match.group(0)
        # Skip if two currency values appear on the same line, e.g. "$5 to $10"
        if re.match(r"^\d", inner.strip()) and not re.search(r"[\\=<>^_{}+\-*/]", inner):
            return match.group(0)
        return rf"\({inner.strip()}\)"

    return _INLINE_DOLLAR_RE.sub(_replace, text)


def analyze_table_complexity(item: TableItem, doc: DoclingDocument) -> tuple[bool, list[str]]:
    """Determine whether a TableItem requires HTML instead of GFM Markdown."""
    reasons: list[str] = []
    header_rows: set[int] = set()
    has_merged = False
    has_math = False
    has_multiline = False

    for cell in item.data.table_cells:
        if cell.row_span > 1 or cell.col_span > 1:
            has_merged = True
        if cell.column_header:
            for r in range(cell.start_row_offset_idx, cell.end_row_offset_idx):
                header_rows.add(r)
        cell_text = cell.text or ""
        if "\n" in cell_text.strip():
            has_multiline = True
        if _MATH_TOKEN_RE.search(cell_text):
            has_math = True

    if not has_merged:
        for row in item.data.grid:
            for cell in row:
                if cell.row_span > 1 or cell.col_span > 1:
                    has_merged = True
                if isinstance(cell, RichTableCell):
                    has_multiline = True

    if has_merged:
        reasons.append("merged_cells")
    if len(header_rows) > 1:
        reasons.append("nested_or_multi_row_headers")
    if has_math:
        reasons.append("mathematical_notation_in_cells")
    if has_multiline:
        reasons.append("multiline_or_rich_cell_content")

    return bool(reasons), reasons


def _escape_underscores_preserving_math(text: str) -> str:
    """Escape bare Markdown underscores while leaving math, code and links untouched."""
    parts: list[str] = []
    last_end = 0
    for match in _MATH_SEGMENT_RE.finditer(text):
        before = text[last_end : match.start()]
        parts.append(re.sub(r"(?<!\\)_", r"\_", before))
        parts.append(match.group(0))
        last_end = match.end()
    if last_end < len(text):
        parts.append(re.sub(r"(?<!\\)_", r"\_", text[last_end:]))
    return "".join(parts)


class AcademicMarkdownDocSerializer(MarkdownDocSerializer):
    """Markdown serializer that preserves underscores inside LaTeX math segments."""

    @classmethod
    def _escape_underscores(cls, text: str) -> str:
        return _escape_underscores_preserving_math(text)


_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")


def _infer_heading_level(item: TitleItem | SectionHeaderItem) -> int:
    """Infer academic heading depth from TitleItem or dotted section numbering."""
    if isinstance(item, TitleItem):
        return 0
    txt = (item.text or "").strip()
    m = _NUMBERED_HEADING_RE.match(txt)
    if m:
        depth = len(m.group(1).split("."))
        return min(depth, 5)
    return max(1, int(getattr(item, "level", 1) or 1))


def resolve_code_language(code_text: str, raw_lang: Any) -> str | None:
    """Resolve and normalize code block language, fixing common VLM misclassifications."""
    lang_val = ""
    if raw_lang is not None:
        lang_val = str(raw_lang.value if hasattr(raw_lang, "value") else raw_lang).strip().lower()
    if lang_val in ("", "unknown", "none", "other"):
        lang_val = ""

    stripped = (code_text or "").strip()
    if re.search(r"(?m)^\[(?:package|dependencies|dev-dependencies|workspace|build-dependencies)\]\s*$", stripped):
        return "toml"
    if re.search(r"(?m)^(?:#\[(?:repr|derive)|pub\s+(?:struct|enum|fn|trait|mod)|impl\b)", stripped):
        return "rust"
    return lang_val or None


class AcademicTextSerializer(MarkdownTextSerializer):
    """Serializer preserving code languages, LaTeX math delimiters and uncertain flags."""

    def _format_heading(
        self,
        text: str,
        item: TitleItem | SectionHeaderItem,
        in_table_cell: bool = False,
    ) -> str:
        if in_table_cell:
            return text
        level = _infer_heading_level(item)
        num_hashes = min(level + 1, 6)
        return f"{num_hashes * '#'} {text}"

    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        is_inline_scope: bool = False,
        in_table_cell: bool = False,
        visited: set[str] | None = None,
        **kwargs: Any,
    ) -> SerializationResult:
        if isinstance(item, CodeItem):
            params = MarkdownParams(**kwargs)
            code_text = item.text or ""
            if params.format_code_blocks and not (is_inline_scope or (params.include_hyperlinks and item.hyperlink)):
                lang = resolve_code_language(code_text, getattr(item, "code_language", None)) or ""
                max_bt = max((len(m.group(0)) for m in re.finditer(r"`+", code_text)), default=0)
                fence = "`" * max(3, max_bt + 1)
                rendered = f"{fence}{lang}\n{code_text}\n{fence}"
                res_parts = [create_ser_result(text=rendered, span_source=item)]
                cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
                if cap_res.text:
                    res_parts.insert(0, cap_res)
                return create_ser_result(
                    text="\n\n".join(r.text for r in res_parts if r.text),
                    span_source=res_parts,
                )

        if isinstance(item, FormulaItem):
            page_no = item.prov[0].page_no if item.prov else "?"
            raw_text = (item.text or "").strip()
            orig_text = (item.orig or "").strip()
            cleaned = strip_outer_math_delimiters(raw_text) if raw_text else ""
            status, reasons = assess_formula_quality(cleaned, orig_text)

            if not cleaned or cleaned == _FORMULA_NOT_DECODED:
                safe_orig = orig_text.replace("-->", "->") if orig_text else ""
                note = (
                    f"<!-- UNCERTAIN_FORMULA (page {page_no}): "
                    f"Primary extractor could not reliably decode expression"
                    + (f'; raw_text="{safe_orig}"' if safe_orig else "")
                    + " — verify against original PDF -->"
                )
                return create_ser_result(text=note, span_source=item)

            if is_inline_scope:
                rendered = rf"\({cleaned}\)"
            else:
                rendered = f"$$\n{cleaned}\n$$"
                if status.startswith("Uncertain"):
                    reason_str = "; ".join(reasons).replace("-->", "->")
                    rendered = (
                        f"<!-- UNCERTAIN_FORMULA (page {page_no}): {reason_str} -->\n{rendered}"
                    )
            return create_ser_result(text=rendered, span_source=item)

        res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            is_inline_scope=is_inline_scope,
            in_table_cell=in_table_cell,
            visited=visited,
            **kwargs,
        )
        if res.text and not isinstance(item, (CodeItem, FormulaItem)):
            normalized = normalize_inline_math(res.text)
            if normalized != res.text:
                return create_ser_result(text=normalized, span_source=res.spans)
        return res


class AcademicTableSerializer(MarkdownTableSerializer):
    """Render simple tables in Markdown and complex tables in HTML to avoid data loss."""

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        if kwargs.get("_nested_in_table"):
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

        is_complex, reasons = analyze_table_complexity(item, doc)
        if not is_complex:
            return super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)

        params = MarkdownParams(**kwargs)
        res_parts: list[SerializationResult] = []
        cap_res = doc_serializer.serialize_captions(item=item, **kwargs)
        cap_after = _caption_goes_after(item, doc, params, standard_after=False)
        if cap_res.text and not cap_after:
            res_parts.append(cap_res)

        html_table = item.export_to_html(doc=doc, add_caption=False)
        if html_table:
            comment = f"<!-- Complex table rendered as HTML ({', '.join(reasons)}) -->\n{html_table}"
            res_parts.append(create_ser_result(text=comment, span_source=item))

        ftn_res = doc_serializer.serialize_footnotes(item=item, **kwargs)
        if ftn_res.text:
            res_parts.append(ftn_res)
        if cap_res.text and cap_after:
            res_parts.append(cap_res)

        return create_ser_result(
            text="\n\n".join(r.text for r in res_parts if r.text),
            span_source=res_parts,
        )


class AcademicPictureSerializer(MarkdownPictureSerializer):
    """Serialize PictureItem with informative alt text and relative asset paths."""

    def _serialize_image_part(
        self,
        item: FloatingItem,
        doc: DoclingDocument,
        image_mode: ImageRefMode,
        image_placeholder: str,
        **kwargs: Any,
    ) -> SerializationResult:
        if (
            image_mode == ImageRefMode.REFERENCED
            and isinstance(item.image, ImageRef)
            and item.image.uri is not None
        ):
            page_no = item.prov[0].page_no if item.prov else "?"
            caption_parts: list[str] = []
            for ref in getattr(item, "captions", []) or []:
                try:
                    resolved = ref.resolve(doc)
                    if getattr(resolved, "text", None):
                        caption_parts.append(resolved.text.strip())
                except Exception:
                    continue
            alt_base = " ".join(caption_parts).replace("\n", " ").strip()
            if len(alt_base) > 90:
                alt_base = alt_base[:87].rstrip() + "..."
            alt = f"Figure (p. {page_no}): {alt_base}" if alt_base else f"Figure (p. {page_no})"
            alt = alt.replace("[", "(").replace("]", ")")
            uri_str = str(item.image.uri)
            return create_ser_result(
                text=f"![{alt}]({self._escape_uri_path(Path(uri_str))})",
                span_source=item,
            )
        return super()._serialize_image_part(
            item=item,
            doc=doc,
            image_mode=image_mode,
            image_placeholder=image_placeholder,
            **kwargs,
        )


def export_academic_markdown(doc: DoclingDocument) -> str:
    """Serialize a DoclingDocument into faithful academic Markdown."""
    serializer = AcademicMarkdownDocSerializer(
        doc=doc,
        text_serializer=AcademicTextSerializer(),
        table_serializer=AcademicTableSerializer(),
        picture_serializer=AcademicPictureSerializer(),
        params=MarkdownParams(
            image_mode=ImageRefMode.REFERENCED,
            escape_html=False,
            escape_underscores=True,
            format_code_blocks=True,
            enable_chart_tables=True,
        ),
    )
    return serializer.serialize().text


def _extract_captions_and_footnotes(item: Any, doc: DoclingDocument) -> tuple[list[str], list[str]]:
    captions: list[str] = []
    footnotes: list[str] = []
    for ref in getattr(item, "captions", []) or []:
        try:
            resolved = ref.resolve(doc)
            txt = getattr(resolved, "text", None)
            if txt:
                captions.append(txt.strip())
        except Exception:
            continue
    for ref in getattr(item, "footnotes", []) or []:
        try:
            resolved = ref.resolve(doc)
            txt = getattr(resolved, "text", None)
            if txt:
                footnotes.append(txt.strip())
        except Exception:
            continue
    return captions, footnotes


def _bbox_to_dict(item: Any) -> dict[str, Any] | None:
    if not getattr(item, "prov", None):
        return None
    prov = item.prov[0]
    b = prov.bbox
    return {
        "page_no": prov.page_no,
        "l": round(float(b.l), 2),
        "t": round(float(b.t), 2),
        "r": round(float(b.r), 2),
        "b": round(float(b.b), 2),
        "coord_origin": str(getattr(b.coord_origin, "value", b.coord_origin)),
    }


def _infer_title_and_authors(doc: DoclingDocument, preflight: PdfPreflightReport) -> dict[str, Any]:
    """Extract document title, authors (best-effort) and abstract without fabrication."""
    title: str | None = None
    abstract: str | None = None
    author_candidates: list[str] = []

    items_list = [item for item, _ in doc.iterate_items()]
    first_section_idx = len(items_list)
    for idx, item in enumerate(items_list):
        if isinstance(item, TitleItem) and title is None and (item.text or "").strip():
            title = item.text.strip()
        elif isinstance(item, SectionHeaderItem):
            header_text = (item.text or "").strip()
            if title is None and idx <= 3 and len(header_text) > 10:
                title = header_text
            if re.match(r"^(?:\d+\.?\s*)?(?:abstract|introduction)\b", header_text, re.IGNORECASE):
                first_section_idx = min(first_section_idx, idx)
                if "abstract" in header_text.lower() and idx + 1 < len(items_list):
                    nxt = items_list[idx + 1]
                    if isinstance(nxt, TextItem) and (nxt.text or "").strip():
                        abstract = nxt.text.strip()
                break

    if title is None and preflight.metadata.get("title"):
        title = preflight.metadata["title"]

    # Author candidates: short text items appearing before Abstract/Introduction on page 1
    for item in items_list[:first_section_idx]:
        if isinstance(item, TextItem) and not isinstance(item, (TitleItem, SectionHeaderItem, FormulaItem, CodeItem)):
            txt = (item.text or "").strip()
            if not txt or txt == title:
                continue
            page_no = item.prov[0].page_no if item.prov else 1
            if page_no == 1 and len(txt) <= 300 and not txt.lower().startswith("abstract"):
                author_candidates.append(txt)
            if txt.lower().startswith("abstract"):
                abstract = abstract or txt

    if not author_candidates and preflight.metadata.get("author"):
        author_candidates.append(preflight.metadata["author"])

    return {
        "title": title,
        "author_lines_extracted": author_candidates,
        "abstract": abstract,
        "pdf_metadata": preflight.metadata,
    }


def build_structured_json(
    doc: DoclingDocument,
    preflight: PdfPreflightReport,
    assets: AssetExtractionResult,
) -> dict[str, Any]:
    """Build the machine-readable structured JSON representation (paper.json)."""
    section_stack: list[tuple[int, str]] = []
    sections: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    code_blocks: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []

    table_ser = AcademicTableSerializer()
    doc_ser = AcademicMarkdownDocSerializer(
        doc=doc,
        text_serializer=AcademicTextSerializer(),
        table_serializer=table_ser,
        picture_serializer=AcademicPictureSerializer(),
        params=MarkdownParams(image_mode=ImageRefMode.REFERENCED),
    )

    eq_idx = 0
    table_idx = 0
    code_idx = 0
    fig_idx = 0

    for block_idx, (item, level) in enumerate(doc.iterate_items(), start=1):
        label = str(getattr(getattr(item, "label", None), "value", getattr(item, "label", "unknown")))
        page_no = item.prov[0].page_no if getattr(item, "prov", None) else None
        bbox = _bbox_to_dict(item)
        captions, footnotes = _extract_captions_and_footnotes(item, doc)

        if isinstance(item, TitleItem):
            sec_title = (item.text or "").strip()
            section_stack = [(0, sec_title)] if sec_title else section_stack
        elif isinstance(item, SectionHeaderItem):
            sec_title = (item.text or "").strip()
            lvl = _infer_heading_level(item)
            section_stack = [s for s in section_stack if s[0] < lvl]
            if sec_title:
                section_stack.append((lvl, sec_title))
                sections.append(
                    {
                        "title": sec_title,
                        "level": lvl,
                        "page_no": page_no,
                        "path": [s[1] for s in section_stack],
                        "self_ref": item.self_ref,
                    }
                )

        section_path = [s[1] for s in section_stack]
        block_entry: dict[str, Any] = {
            "block_id": f"blk_{block_idx:04d}",
            "block_type": label,
            "page_no": page_no,
            "bbox": bbox,
            "section_path": section_path,
            "self_ref": getattr(item, "self_ref", None),
        }

        if isinstance(item, FormulaItem):
            eq_idx += 1
            raw_text = (item.text or "").strip()
            orig_text = (item.orig or "").strip() or None
            cleaned_latex = strip_outer_math_delimiters(raw_text) if raw_text else ""
            status, reasons = assess_formula_quality(cleaned_latex, orig_text)
            eq_num = extract_equation_number(cleaned_latex, orig_text)
            eq_id = f"eq_{eq_idx:03d}"
            eq_record = {
                "equation_id": eq_id,
                "block_id": block_entry["block_id"],
                "page_no": page_no,
                "section_path": section_path,
                "equation_number": eq_num,
                "latex": cleaned_latex,
                "raw_orig": orig_text,
                "verification_status": status,
                "uncertainty_reasons": reasons,
                "nougat_latex": None,
                "nougat_comparison": None,
                "self_ref": item.self_ref,
            }
            equations.append(eq_record)
            block_entry["block_type"] = "formula"
            block_entry["text"] = cleaned_latex
            block_entry["formula"] = eq_record

        elif isinstance(item, CodeItem):
            code_idx += 1
            lang_val = resolve_code_language(item.text or "", getattr(item, "code_language", None))
            code_record = {
                "code_id": f"code_{code_idx:03d}",
                "block_id": block_entry["block_id"],
                "page_no": page_no,
                "section_path": section_path,
                "language": lang_val,
                "code": item.text or "",
                "captions": captions,
                "self_ref": item.self_ref,
            }
            code_blocks.append(code_record)
            block_entry["block_type"] = "code"
            block_entry["text"] = item.text or ""
            block_entry["code_language"] = lang_val
            block_entry["captions"] = captions

        elif isinstance(item, TableItem):
            table_idx += 1
            is_complex, complexity_reasons = analyze_table_complexity(item, doc)
            grid_rows: list[list[str]] = []
            for row in item.data.grid:
                grid_rows.append([(cell.text or "").strip() for cell in row])
            md_repr = item.export_to_markdown(doc=doc)
            html_repr = item.export_to_html(doc=doc, add_caption=False)
            asset_path = assets.table_paths.get(item.self_ref)
            table_record = {
                "table_id": f"table_{table_idx:03d}",
                "block_id": block_entry["block_id"],
                "page_no": page_no,
                "section_path": section_path,
                "num_rows": item.data.num_rows,
                "num_cols": item.data.num_cols,
                "is_complex": is_complex,
                "complexity_reasons": complexity_reasons,
                "rendered_format": "html" if is_complex else "markdown",
                "rows": grid_rows,
                "markdown": md_repr,
                "html": html_repr,
                "captions": captions,
                "footnotes": footnotes,
                "asset_path": asset_path,
                "self_ref": item.self_ref,
            }
            tables.append(table_record)
            block_entry["block_type"] = "table"
            block_entry["table"] = table_record
            block_entry["captions"] = captions
            block_entry["footnotes"] = footnotes

        elif isinstance(item, PictureItem):
            fig_idx += 1
            asset_path = assets.figure_paths.get(item.self_ref)
            fig_record = {
                "figure_id": f"fig_{fig_idx:03d}",
                "block_id": block_entry["block_id"],
                "page_no": page_no,
                "section_path": section_path,
                "asset_path": asset_path,
                "captions": captions,
                "footnotes": footnotes,
                "self_ref": item.self_ref,
            }
            figures.append(fig_record)
            block_entry["block_type"] = "figure"
            block_entry["figure"] = fig_record
            block_entry["captions"] = captions
            block_entry["footnotes"] = footnotes

        else:
            raw_txt = getattr(item, "text", None)
            if raw_txt is not None:
                block_entry["text"] = normalize_inline_math(raw_txt)
            if isinstance(item, SectionHeaderItem):
                block_entry["block_type"] = "section_header"
                block_entry["level"] = getattr(item, "level", level)
            elif isinstance(item, TitleItem):
                block_entry["block_type"] = "title"
                block_entry["level"] = 0
            elif isinstance(item, ListItem):
                block_entry["block_type"] = "list_item"
                block_entry["marker"] = getattr(item, "marker", "")
            elif label == DocItemLabel.FOOTNOTE.value:
                block_entry["block_type"] = "footnote"
            elif label == DocItemLabel.REFERENCE.value:
                block_entry["block_type"] = "reference"
            elif label == DocItemLabel.CAPTION.value:
                block_entry["block_type"] = "caption"
            elif label == DocItemLabel.TEXT.value or label == DocItemLabel.PARAGRAPH.value:
                block_entry["block_type"] = "paragraph"
            if captions:
                block_entry["captions"] = captions
            if footnotes:
                block_entry["footnotes"] = footnotes

        blocks.append(block_entry)

    page_block_counts: dict[int, int] = {}
    for blk in blocks:
        p = blk.get("page_no")
        if isinstance(p, int):
            page_block_counts[p] = page_block_counts.get(p, 0) + 1

    pages_summary = []
    for p_info in preflight.pages:
        pages_summary.append(
            {
                "page_no": p_info.page_no,
                "classification": p_info.classification,
                "has_embedded_text": p_info.has_embedded_text,
                "pdf_text_chars": p_info.text_char_count,
                "extracted_block_count": page_block_counts.get(p_info.page_no, 0),
                "warnings": p_info.warnings,
            }
        )

    metadata = _infer_title_and_authors(doc, preflight)

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "filename": preflight.file_path.name,
            "path": str(preflight.file_path),
            "sha256": preflight.sha256,
            "file_size_bytes": preflight.file_size_bytes,
            "page_count": preflight.page_count,
            "document_kind": preflight.document_kind,
            "has_embedded_text_layer": preflight.has_embedded_text_layer,
        },
        "metadata": metadata,
        "sections": sections,
        "pages": pages_summary,
        "counts": {
            "blocks": len(blocks),
            "sections": len(sections),
            "equations": len(equations),
            "tables": len(tables),
            "code_blocks": len(code_blocks),
            "figures": len(figures),
            "standalone_images": len(assets.other_image_paths),
        },
        "blocks": blocks,
        "equations": equations,
        "tables": tables,
        "code_blocks": code_blocks,
        "figures": figures,
        "standalone_images": assets.other_image_paths,
    }
