from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class PipelineConfig:
    """Pipeline options for PDF ingestion, extraction and validation."""

    input_path: Path = Path("input/paper.pdf")
    output_dir: Path = Path("output")

    ocr_mode: Literal["auto", "always", "off"] = "auto"
    ocr_lang: list[str] = field(default_factory=lambda: ["en"])

    do_table_structure: bool = True
    table_mode: Literal["accurate", "fast"] = "accurate"
    do_formula_enrichment: bool = True
    do_code_enrichment: bool = True
    generate_picture_images: bool = True
    generate_page_images: bool = True
    images_scale: float = 2.0
    document_timeout: float | None = None

    enable_remote_services: bool = False

    verify_nougat: bool = False
    nougat_model_id: str = "facebook/nougat-base"
    nougat_pages: list[int] | None = None
    nougat_max_pages: int = 10

    min_page_chars_warning: int = 50
    short_document_chars_warning: int = 200

    clean_output: bool = True
    log_level: str = "INFO"

    def resolve_paths(self) -> tuple[Path, Path]:
        return self.input_path.expanduser().resolve(), self.output_dir.expanduser().resolve()
