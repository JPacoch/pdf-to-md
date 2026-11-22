"""Asset extraction and management for figures, tables and visual PDF assets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf
from docling_core.types.doc import DoclingDocument, PictureItem, TableItem
from docling_core.types.doc.document import BoundingBox, ImageRef, Size
from PIL import Image

logger = logging.getLogger(__name__)

AssetKind = Literal["figure", "table", "image"]


@dataclass(slots=True)
class ExtractedAsset:
    """Metadata for a single extracted visual asset saved under output/assets/."""

    asset_id: str
    kind: AssetKind
    page_no: int
    relative_path: str
    absolute_path: Path
    width: int
    height: int
    caption: str | None = None
    source_ref: str | None = None


@dataclass(slots=True)
class AssetExtractionResult:
    """Summary of all extracted figures, table snapshots and standalone images."""

    assets: list[ExtractedAsset] = field(default_factory=list)
    figure_paths: dict[str, str] = field(default_factory=dict)
    table_paths: dict[str, str] = field(default_factory=dict)
    other_image_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _crop_bbox_from_pdf(
    pdf_doc: pymupdf.Document,
    page_no: int,
    bbox: BoundingBox,
    scale: float,
) -> Image.Image | None:
    """Render a cropped region from the source PDF page using PyMuPDF."""
    if page_no < 1 or page_no > pdf_doc.page_count:
        return None
    page = pdf_doc.load_page(page_no - 1)
    page_height = float(page.rect.height)
    tl_bbox = bbox.to_top_left_origin(page_height=page_height)
    clip = pymupdf.Rect(tl_bbox.l, tl_bbox.t, tl_bbox.r, tl_bbox.b)
    if clip.is_empty or clip.width <= 1 or clip.height <= 1:
        return None
    mat = pymupdf.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _item_caption_text(item: PictureItem | TableItem, doc: DoclingDocument) -> str | None:
    if not getattr(item, "captions", None):
        return None
    parts: list[str] = []
    for ref in item.captions:
        try:
            resolved = ref.resolve(doc)
            txt = getattr(resolved, "text", None)
            if txt:
                parts.append(txt.strip())
        except Exception:
            continue
    return " ".join(parts) if parts else None


def extract_assets(
    doc: DoclingDocument,
    pdf_path: Path,
    output_dir: Path,
    images_scale: float = 2.0,
) -> AssetExtractionResult:
    """Extract figures, table visual snapshots and standalone embedded images.

    Updates ``PictureItem.image`` URIs on ``doc`` to point to relative paths under
    ``assets/figures/`` so Markdown serialization produces working relative links.
    """
    assets_root = output_dir / "assets"
    figures_dir = assets_root / "figures"
    tables_dir = assets_root / "tables"
    images_dir = assets_root / "images"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    result = AssetExtractionResult()
    pdf_doc = pymupdf.open(str(pdf_path))
    figure_pages_with_items: set[int] = set()

    try:
        for idx, picture in enumerate(doc.pictures, start=1):
            page_no = picture.prov[0].page_no if picture.prov else 0
            if page_no:
                figure_pages_with_items.add(page_no)
            filename = f"fig_p{page_no:03d}_{idx:02d}.png"
            rel_path = f"assets/figures/{filename}"
            out_path = figures_dir / filename
            caption = _item_caption_text(picture, doc)

            pil_img: Image.Image | None = None
            try:
                pil_img = picture.get_image(doc)
            except Exception as exc:
                logger.debug("picture.get_image failed for %s: %s", picture.self_ref, exc)

            if pil_img is None and picture.prov:
                pil_img = _crop_bbox_from_pdf(
                    pdf_doc,
                    page_no=picture.prov[0].page_no,
                    bbox=picture.prov[0].bbox,
                    scale=images_scale,
                )

            if pil_img is None:
                result.warnings.append(
                    f"Could not extract image pixels for figure {picture.self_ref} on page {page_no}."
                )
                continue

            pil_img.save(out_path, format="PNG")
            dpi = int(round(72 * images_scale))
            picture.image = ImageRef(
                mimetype="image/png",
                dpi=dpi,
                size=Size(width=float(pil_img.width), height=float(pil_img.height)),
                uri=Path(rel_path),
            )
            result.figure_paths[picture.self_ref] = rel_path
            result.assets.append(
                ExtractedAsset(
                    asset_id=f"fig_{idx:03d}",
                    kind="figure",
                    page_no=page_no,
                    relative_path=rel_path,
                    absolute_path=out_path,
                    width=pil_img.width,
                    height=pil_img.height,
                    caption=caption,
                    source_ref=picture.self_ref,
                )
            )

        for idx, table in enumerate(doc.tables, start=1):
            page_no = table.prov[0].page_no if table.prov else 0
            filename = f"table_p{page_no:03d}_{idx:02d}.png"
            rel_path = f"assets/tables/{filename}"
            out_path = tables_dir / filename
            caption = _item_caption_text(table, doc)

            pil_img = None
            try:
                pil_img = table.get_image(doc)
            except Exception:
                pil_img = None

            if pil_img is None and table.prov:
                pil_img = _crop_bbox_from_pdf(
                    pdf_doc,
                    page_no=table.prov[0].page_no,
                    bbox=table.prov[0].bbox,
                    scale=images_scale,
                )

            if pil_img is None:
                result.warnings.append(
                    f"Could not extract visual snapshot for table {table.self_ref} on page {page_no}."
                )
                continue

            pil_img.save(out_path, format="PNG")
            result.table_paths[table.self_ref] = rel_path
            result.assets.append(
                ExtractedAsset(
                    asset_id=f"table_{idx:03d}",
                    kind="table",
                    page_no=page_no,
                    relative_path=rel_path,
                    absolute_path=out_path,
                    width=pil_img.width,
                    height=pil_img.height,
                    caption=caption,
                    source_ref=table.self_ref,
                )
            )

        # Save raw embedded raster images on pages where Docling found no PictureItems
        seen_xrefs: set[int] = set()
        img_counter = 0
        for page_idx in range(pdf_doc.page_count):
            page_no = page_idx + 1
            page = pdf_doc.load_page(page_idx)
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    extracted = pdf_doc.extract_image(xref)
                except Exception:
                    continue
                width = int(extracted.get("width", 0))
                height = int(extracted.get("height", 0))
                if width < 48 or height < 48:
                    continue
                if page_no in figure_pages_with_items:
                    continue
                img_counter += 1
                ext = str(extracted.get("ext", "png")).lower()
                if ext not in ("png", "jpg", "jpeg", "webp"):
                    ext = "png"
                filename = f"img_p{page_no:03d}_{img_counter:02d}.{ext}"
                rel_path = f"assets/images/{filename}"
                out_path = images_dir / filename
                out_path.write_bytes(extracted["image"])
                result.other_image_paths.append(rel_path)
                result.assets.append(
                    ExtractedAsset(
                        asset_id=f"img_{img_counter:03d}",
                        kind="image",
                        page_no=page_no,
                        relative_path=rel_path,
                        absolute_path=out_path,
                        width=width,
                        height=height,
                        caption=None,
                        source_ref=f"pdf:xref:{xref}",
                    )
                )
    finally:
        pdf_doc.close()

    logger.info(
        "Extracted %d figure(s), %d table snapshot(s) and %d standalone image(s).",
        len(result.figure_paths),
        len(result.table_paths),
        len(result.other_image_paths),
    )
    return result
