# CBIR Bird Lens

CBIR Bird Lens is a course project for **Bird Image Storage and Retrieval** on CUB-200-2011.

## Project Goal

Build a complete retrieval pipeline for bird images with:

- dataset collection and normalization,
- metadata and feature extraction,
- top-k similarity retrieval,
- Precision@5 evaluation.

## Assignment Scope (Condensed)

1. Collect and normalize at least 500 images.
2. Keep metadata and preprocessing parameters.
3. Extract similarity and difference features.
4. Store images/features in a multimedia-ready schema.
5. Retrieve top-5 similar images.
6. Report intermediate results and evaluation.

## Documentation

- [setup.md](setup.md): environment setup and dataset download.
- [pipeline.md](pipeline.md): full runbook across Phase 1, Phase 2, and Phase 3.

## Scripts Layout

- `scripts/phase1_setup`
- `scripts/phase2_normalize`
- `scripts/phase3_retrieval`

### Phase 1

- `scripts/phase1_setup/download_cub.py`

### Phase 2

- `scripts/phase2_normalize/cub_utils.py`
- `scripts/phase2_normalize/sample_candidates.py`
- `scripts/phase2_normalize/normalize_selected.py`

### Phase 3

- `scripts/phase3_retrieval/feature_utils.py`
- `scripts/phase3_retrieval/extract_features.py`
- `scripts/phase3_retrieval/demo_ui.py`
- `scripts/phase3_retrieval/retrieve_topk.py`
- `scripts/phase3_retrieval/evaluate_precision_at_k.py`
- `scripts/phase3_retrieval/build_feature_db.py`
