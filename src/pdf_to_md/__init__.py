"""Academic PDF-to-Markdown ingestion pipeline."""

from pdf_to_md.cli import main
from pdf_to_md.config import PipelineConfig
from pdf_to_md.pipeline import PipelineRunResult, run_pipeline

__all__ = ["PipelineConfig", "PipelineRunResult", "main", "run_pipeline"]
