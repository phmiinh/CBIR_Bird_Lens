# Bird Image Storage and Retrieval

This repository implements a **Content-Based Image Retrieval (CBIR)** system for a Multimedia Database course project. The system stores a curated bird image collection, extracts visual descriptors, manages metadata and feature vectors in a database, and returns the **top-5 most visually similar bird images** for a query image.

The project is framed as a multimedia retrieval system, not as an image classifier. Species labels may be used as metadata or secondary evaluation signals, but the main retrieval objective is visual content similarity.

## Assignment Scope

The implementation is designed around the original course requirements:

- build a bird image dataset with at least 500 images
- normalize all gallery images to the same dimensions and object aspect ratio
- keep images where birds are perching and captured from a consistent horizontal viewing angle
- design visual features that explain both similarity and difference among bird images
- store image metadata and descriptors in a database system
- retrieve the five most similar images for a new query image, including unseen species
- present the system workflow, intermediate retrieval results, demo, and evaluation

## Technical Approach

The system follows a **loose-coupling multimedia DBMS architecture**:

- image files are stored on the filesystem
- metadata, descriptor definitions, feature vectors, query records, retrieval runs, relevance judgments, and experiment summaries are stored in SQLite
- retrieval loads descriptor matrices from SQLite and performs application-layer similarity ranking

At the project scale, retrieval uses an exhaustive kNN-style linear scan over gallery descriptors. This keeps the method transparent, reproducible, and easy to explain in a course report. Approximate nearest-neighbor indexing is left as future work for larger datasets.

## Feature Set

The primary retrieval story is based on handcrafted descriptors:

- `global_hsv_hist`: global color distribution
- `regional_hsv_hist`: spatial color layout over a `2x2` grid
- `color_moments`: compact HSV color statistics
- `lbp_hist`: local feather texture patterns
- `hog_descriptor`: edge, contour, and pose structure
- `silhouette_shape_descriptor`: coarse foreground shape and geometry

A `ResNet18` CNN embedding is included as a secondary semantic baseline and optional fusion signal. It is not used as a classifier in the core retrieval design.

## Retrieval Logic

The retrieval engine computes descriptor-level similarity scores, then ranks gallery images by a weighted fusion score.

Similarity metrics:

- chi-square distance converted to similarity for histogram descriptors
- inverse Euclidean distance for compact numeric descriptors
- cosine similarity for HOG and CNN embeddings

The main demo modes are:

- `calibrated_handcrafted`: primary handcrafted CBIR configuration
- `fusion`: handcrafted-first fusion with secondary CNN support
- `cnn_only`: semantic baseline for comparison

## Technology Stack

- Python
- SQLite
- NumPy
- Pillow
- scikit-image
- PyTorch / TorchVision
- Gradio
- CSV, JSON, and NPY artifacts for reproducibility

## Documentation

- [docs/guideline.md](docs/guideline.md): demo run guide
- [docs/delivery.md](docs/delivery.md): assignment requirements and delivery criteria
- [docs/current_status.md](docs/current_status.md): current implementation status and verification notes
- [docs/roadmap.md](docs/roadmap.md): consolidated setup, command flow, architecture notes, and report planning

## Repository Layout

- `scripts/phase1_setup`: dataset setup utilities
- `scripts/phase2_normalize`: image review, filtering, ROI normalization, and metadata export
- `scripts/phase3_descriptor_extraction`: descriptor extraction
- `scripts/phase4_feature_database`: SQLite database build
- `scripts/phase5_retrieval`: Gradio demo UI and retrieval entrypoints
- `scripts/phase6_experiments`: experiment runs and relevance-judgment preparation
- `scripts/phase7_evaluation`: retrieval metric computation
- `scripts/phase8_report_artifacts`: report-ready exports
- `scripts/shared`: shared descriptor, database, and retrieval utilities
