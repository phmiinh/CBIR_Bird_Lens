# CBIR Bird Lens

CBIR Bird Lens is a course project for **Bird Image Storage and Retrieval** using the CUB-200-2011 dataset.
This repository is organized as an implementation-first workflow, where each project phase is documented and reproducible.

## Project Objective

Build a complete image retrieval pipeline for bird images, with a focus on:

- Collecting and preparing at least 500 normalized bird images.
- Enforcing visual constraints (perching birds, roughly horizontal/side view).
- Cropping bird ROI from bounding boxes and resizing to a unified format (224x224).
- Storing metadata and preprocessing parameters for each image.
- Preparing data for downstream feature extraction and top-k retrieval experiments.

## Assignment Requirements (Condensed)

The full assignment flow includes:

1. Dataset collection and normalization.
2. Metadata generation.
3. Feature extraction (similarity + difference features).
4. Multimedia storage design.
5. k-NN retrieval (top-5).
6. Intermediate results for report/demo.
7. Evaluation (e.g., Precision@5) and failure-case analysis.

This repository currently implements and documents the **data collection + normalization stage** in production-ready scripts.

## Documentation Map

- [setup.md](setup.md): environment setup, virtual environment, Kaggle API token setup, and dataset download/extraction.
- [pipeline.md](pipeline.md): step-by-step execution pipeline for the normalization stage (sampling, HTML review, keep/skip export, final normalization).

## Repository Structure

- `scripts/download_cub.py`: downloads or extracts CUB-200-2011.
- `scripts/sample_candidates.py`: builds a review batch and interactive HTML board.
- `scripts/normalize_selected.py`: generates the final normalized dataset from reviewed selections.
- `outputs/`: review artifacts and intermediate report examples.
- `data/`: raw and processed datasets.

## Documentation Update Policy

When a new project phase is implemented, update `pipeline.md` with:

1. Goal of the phase.
2. Commands to run.
3. Expected outputs.
4. Verification checklist.
