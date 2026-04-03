# Bird Image Storage and Retrieval

This repository implements a **Content-Based Image Retrieval (CBIR)** system for bird images as a **Multimedia Database** project. The project is not framed as a classification task. The core objective is to store curated bird-image descriptors in a database, retrieve the `top-5` most visually similar images for a query, and evaluate ranking quality using **visual relevance** rather than species prediction alone.

## Project Scope

The current repository is organized around a **DB-first retrieval pipeline**:

1. Curate and normalize a gallery of bird images from CUB-200-2011.
2. Extract an explainable feature set:
   - global HSV histogram
   - regional HSV histogram (`2x2`)
   - color moments
   - LBP histogram
   - HOG descriptor
   - CNN embedding (`ResNet18`, secondary only)
3. Store metadata, preprocessing trace, descriptors, query cache, retrieval logs, judgments, and experiment summaries in **SQLite**.
4. Run DB-backed `top-k` retrieval under multiple experiment configurations.
   - current baseline search strategy: exhaustive `kNN` / linear scan over the gallery
   - storage architecture: **loose coupling** (`images on filesystem`, `metadata + descriptors in SQLite`)
   - query semantics: **approximate similarity ranking** over descriptors, not exact attribute matching and not species classification
5. Evaluate with:
   - **manual relevance** as the main CBIR evaluation (`nDCG@5`, `Precision@5`)
   - **species proxy** as a secondary baseline (`Precision@5`)

## Architecture Choice

The repository follows a **loose-coupling multimedia DBMS architecture**:

- bird image files remain on the filesystem
- SQLite stores metadata, descriptor definitions, feature vectors, query cache, retrieval logs, judgments, and experiment summaries
- the retrieval engine performs **approximate content similarity ranking**

At the project scale of `1000` images, the baseline retrieval method is an **exhaustive linear scan** over the gallery descriptors. This is intentional: it keeps the system transparent, reproducible, and easy to explain in a course project. Advanced indexing is treated as future work, not as a requirement for validating the CBIR pipeline.

The report should frame the system as:
- a multimedia database system for content-based retrieval
- not an image classifier
- not an exact-match DB query system

## Why This Repo Was Refactored

The project was deliberately refocused away from a deep-learning-first narrative. The main deliverables now emphasize:

- what each descriptor captures
- how each descriptor is extracted
- how descriptors are stored and managed in a multimedia database
- how the database is used for similarity search
- how ranking quality is evaluated with graded relevance
- how the report explains the system as a **multimedia retrieval** pipeline rather than a classifier
- how the system fits a **multimedia DBMS loose-coupling architecture** instead of a pure ML training pipeline

## Documentation Map

- [setup.md](/d:/PTIT/Multimedia%20Database/setup.md)
  Environment setup, dependencies, Kaggle access, and CUB download.

- [pipeline.md](/d:/PTIT/Multimedia%20Database/pipeline.md)
  Execution guide for the working pipeline: normalization, feature extraction, database build, retrieval, evaluation, and optional demo UI.

- [plan.md](/d:/PTIT/Multimedia%20Database/plan.md)
  Master refactor roadmap and architecture decisions for the DB-first version of the project.

## Repository Structure

- [scripts/phase1_setup](/d:/PTIT/Multimedia%20Database/scripts/phase1_setup)
  Dataset download and setup utilities.

- [scripts/phase2_normalize](/d:/PTIT/Multimedia%20Database/scripts/phase2_normalize)
  Review-board generation, manual filtering, ROI crop, resize, and metadata export.

- [scripts/phase3_retrieval](/d:/PTIT/Multimedia%20Database/scripts/phase3_retrieval)
  Descriptor extraction, SQLite ingestion, DB-backed retrieval, experiments, relevance labeling, evaluation, and report artifact export.

- [data/processed](/d:/PTIT/Multimedia%20Database/data/processed)
  Curated normalized gallery images and metadata.

- [data/features](/d:/PTIT/Multimedia%20Database/data/features)
  Extracted feature artifacts and the SQLite retrieval database.

## Current Working Target

The refactor is designed for a curated gallery of **1000 normalized images**. If the local workspace still contains only `500` processed images, rerun the Phase 2 review and normalization flow in [pipeline.md](/d:/PTIT/Multimedia%20Database/pipeline.md) to scale the gallery before running the full experiments.
