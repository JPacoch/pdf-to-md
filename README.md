# pdf-to-md

Local PDF-to-Markdown conversion pipeline built on Docling and PyMuPDF for academic papers. Extracts structured Markdown, JSON blocks, LaTeX equations, tables, code blocks, images and validation reports without calling external APIs.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download Docling's layout, TableFormer and CodeFormulaV2 weights ahead of time if running offline:

```bash
uv run docling-tools models download layout tableformer code_formula
```

RapidOCR uses the ONNX weights bundled with the `rapidocr` package by default.

## Usage

Put the source PDF at `input/paper.pdf` and run:

```bash
uv run pdf-to-md
```

Common options:

```bash
uv run pdf-to-md \
  --input input/paper.pdf \
  --output output \
  --ocr auto \
  --table-mode accurate \
  --image-scale 2.0
```

- `--ocr {auto,always,off}`: `auto` enables RapidOCR and forces full-page OCR when no text layer is detected.
- `--table-mode {accurate,fast}`: TableFormer mode (default `accurate`).
- `--no-formula-enrichment`: skip Docling formula-to-LaTeX enrichment.
- `--no-code-enrichment`: skip Docling code block enrichment.
- `--verify-nougat`: run local `facebook/nougat-base` verification on flagged or equation-heavy pages.
- `--nougat-pages 2,4-5`: restrict Nougat verification to specific pages.
- `--allow-remote-downloads`: allow Hugging Face downloads if local caches are missing.
- `--no-clean`: keep existing files in `output/` instead of regenerating from scratch.

## Project Layout

```text
src/pdf_to_md/
├── config.py        # PipelineConfig defaults and path resolution
├── preflight.py     # PyMuPDF PDF validation and text-layer detection
├── converter.py     # Docling conversion, OCR wiring and digital code-block recovery
├── assets.py        # Figure, table snapshot and embedded image extraction
├── serializers.py   # Custom Markdown/HTML table serializers and paper.json builder
├── nougat.py        # Optional secondary equation check via facebook/nougat-base
├── validation.py    # Post-conversion checks and validation/ report writers
├── docs_gen.py      # output/AGENTS.md and output/README.md writers
├── pipeline.py      # End-to-end runner
└── cli.py           # pdf-to-md CLI entry point
```

## Output Layout

```text
output/
├── README.md
├── AGENTS.md
├── paper.md
├── paper.json
├── assets/
│   ├── figures/
│   ├── tables/
│   └── images/
└── validation/
    ├── report.json
    ├── report.md
    ├── equations.md
    └── issues.md
```

## Tests

```bash
uv run pytest
```
