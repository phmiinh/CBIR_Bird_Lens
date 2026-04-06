# Execution Roadmap

This document describes the **working command flow** for the DB-first CBIR project.

Prerequisite:
- complete [setup.md](/d:/PTIT/Multimedia%20Database/setup.md)
- activate the project virtual environment before running any command

Architecture baseline used throughout the pipeline:
- **loose coupling**: image files on filesystem, metadata + descriptors + logs in SQLite
- **approximate similarity retrieval**: ranked content matching, not exact DBMS filtering
- **exhaustive linear scan**: accepted baseline retrieval strategy for the `1000`-image course-project scale

## Phase 1 - Dataset Setup

Download and extract the CUB-200-2011 dataset:

```powershell
python scripts/phase1_setup/download_cub.py
```

Expected output:
- `data/raw/cub2002011/...`

## Phase 2 - Normalization Workflow

Goal:
- review candidate images
- keep only valid perching / side-view samples
- normalize the curated gallery with ROI crop + resize
- export metadata for downstream database ingestion

### Step 2.1 - Generate Review Board Candidates

Create a species-balanced candidate set for manual review:

```powershell
python scripts/phase2_normalize/sample_candidates.py `
  --dataset-root data/raw/cub2002011 `
  --sample-size 1000 `
  --seed 42 `
  --distribution-mode species_balanced `
  --padding-ratio 0.35 `
  --output-dir outputs/review
```

Expected outputs:
- `outputs/review/candidates.csv`
- `outputs/review/gallery.html`
- `outputs/review/previews/...`

### Step 2.2 - Review and Export Decisions

Open:

```text
outputs/review/gallery.html
```

Review rule:
- keep images where the bird is perching, reasonably side-view, visually clear, and not badly occluded

After finishing review:
- click `Download Reviewed CSV`
- place the file in:

```text
outputs/review/reviewed_candidates.csv
```

### Step 2.3 - Normalize the Final Gallery

Normalize the reviewed selection into the processed gallery:

```powershell
python scripts/phase2_normalize/normalize_selected.py `
  --selection-csv outputs/review/reviewed_candidates.csv `
  --dataset-root data/raw/cub2002011 `
  --padding-ratio 0.35 `
  --require-count 1000 `
  --max-images 1000
```

Expected outputs:
- `data/processed/images/...`
- `data/processed/metadata/images.csv`
- `data/processed/metadata/images.jsonl`
- `outputs/intermediate_examples/...`

Notes:
- if your current local gallery still has only `500` images, repeat the review and normalization flow until `1000` curated images are produced

### Step 2.4 - Incremental Expansion from 500 to 1000

If you already have a first reviewed batch with `500` final keeps, do **not** restart from zero. Create a second review batch that excludes every image already reviewed in batch 1.

Assumed existing file:

```text
outputs/review/reviewed_candidates.csv
```

Generate batch 2 without duplicates from batch 1:

```powershell
python scripts/phase2_normalize/sample_candidates.py `
  --dataset-root data/raw/cub2002011 `
  --sample-size 1500 `
  --seed 43 `
  --distribution-mode species_balanced `
  --padding-ratio 0.35 `
  --exclude-reviewed-csv outputs/review/reviewed_candidates.csv `
  --output-dir outputs/review_batch_2
```

Then:
- open `outputs/review_batch_2/gallery.html`
- review the new candidates
- download the CSV
- move it to:

```text
outputs/review_batch_2/reviewed_candidates.csv
```

Important notes:
- batch 2 review only saves your selection decisions as CSV; it does not create final normalized images yet
- this command samples from the **raw CUB dataset** in `data/raw/cub2002011`, not from `data/processed`
- `--exclude-reviewed-csv` removes every `image_id` that already appeared in batch 1, so batch 2 is drawn from the remaining raw dataset
- use a **different output folder** such as `outputs/review_batch_2`; otherwise your browser may show old review state from `outputs/review/gallery.html`
- `1500` is only a recommended candidate pool for batch 2; if you want an even stronger final dataset, increase it to `2000`

Merge batch 1 and batch 2 into one final selection CSV:

```powershell
python scripts/phase2_normalize/merge_review_csvs.py `
  --input-csv outputs/review/reviewed_candidates.csv `
  --input-csv outputs/review_batch_2/reviewed_candidates.csv `
  --output-csv outputs/review/final_reviewed_candidates_1000.csv
```

Normalize the merged final selection:

```powershell
python scripts/phase2_normalize/normalize_selected.py `
  --selection-csv outputs/review/final_reviewed_candidates_1000.csv `
  --dataset-root data/raw/cub2002011 `
  --padding-ratio 0.35 `
  --require-count 1000 `
  --max-images 1000
```

The final normalized gallery is written to:
- `data/processed/images`
- `data/processed/metadata/images.csv`
- `data/processed/metadata/images.jsonl`

## Phase 3 - Descriptor Extraction

Goal:
- extract the full feature inventory required by the DB-first retrieval system

Descriptor set:
- global HSV histogram
- regional HSV histogram (`2x2`)
- color moments
- LBP histogram
- HOG descriptor
- CNN embedding (`ResNet18`, secondary only)

### Step 3.0 - Remove Old Phase 3 Artifacts

Before rebuilding Phase 3 on the final gallery, clear old feature/database/retrieval outputs:

```powershell
python scripts/phase3_descriptor_extraction/clean_phase_outputs.py
```

Actually delete them:

```powershell
python scripts/phase3_descriptor_extraction/clean_phase_outputs.py --yes
```

This removes only:
- `data/features`
- `outputs/retrieval`
- `outputs/eval`
- `outputs/experiments`
- `outputs/manual_relevance`
- `outputs/report_artifacts`
- `outputs/demo_ui_queries`

This does not remove:
- `data/raw`
- `data/processed`
- `outputs/review`
- `outputs/review_batch_2`

Run extraction:

```powershell
python scripts/phase3_descriptor_extraction/extract_features.py `
  --metadata-csv data/processed/metadata/images.csv `
  --processed-root data/processed `
  --output-dir data/features `
  --device auto
```

Optional smoke test on a small subset:

```powershell
python scripts/phase3_descriptor_extraction/extract_features.py `
  --metadata-csv data/processed/metadata/images.csv `
  --processed-root data/processed `
  --output-dir outputs/smoke/features_small `
  --device cpu `
  --limit 5
```

Expected outputs:
- `data/features/images_manifest.csv`
- `data/features/descriptor_table.csv`
- `data/features/global_hsv_hist.npy`
- `data/features/regional_hsv_hist.npy`
- `data/features/color_moments.npy`
- `data/features/lbp_hist.npy`
- `data/features/hog_descriptor.npy`
- `data/features/cnn_embedding.npy`
- `data/features/features.jsonl`
- `data/features/config.json`

## Phase 4 - Build the SQLite Feature Database

Create the DB-first retrieval database:

```powershell
python scripts/phase4_feature_database/build_feature_db.py `
  --features-dir data/features `
  --db-path data/features/cbir_features.sqlite `
  --overwrite
```

Main tables created:
- `images`
- `preprocessing_metadata`
- `feature_types`
- `image_features`
- `queries`
- `query_features`
- `retrieval_runs`
- `retrieval_results`
- `relevance_judgments`
- `experiments`

Architecture note:
- this project uses **loose coupling**
- the bird image files stay on the filesystem
- SQLite stores metadata, descriptors, query cache, retrieval logs, judgments, and experiment summaries
- for the report, map this implementation to functional multimedia DB modules:
  - presentation / result presentation
  - query manager
  - metadata manager
  - storage manager
  - feature extraction + similarity layer

## Phase 5 - DB-Backed Retrieval

Recommended demo path:
- launch a localhost web UI on top of the DB-backed retrieval core
- upload a query image
- inspect the ranked top-5 results and retrieval payload

### Step 5.1 - Launch the Local Retrieval UI

```powershell
python scripts/phase5_retrieval/demo_ui.py `
  --db-path data/features/cbir_features.sqlite `
  --processed-root data/processed `
  --device auto `
  --host 127.0.0.1 `
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

In the UI:
- upload a bird query image
- choose one experiment configuration
- keep `Top K = 5` for the course requirement
- inspect both the visual ranking and the JSON payload

### Step 5.2 - Optional CLI Retrieval

If you still need a scriptable terminal-only retrieval command for reproducible logs, use:

```powershell
python scripts/phase5_retrieval/retrieve_topk.py `
  --db-path data/features/cbir_features.sqlite `
  --processed-root data/processed `
  --experiment-name handcrafted_only `
  --query-image-id 88 `
  --top-k 5 `
  --output-csv outputs/retrieval/topk_results.csv `
  --output-json outputs/retrieval/topk_results.json
```

Available experiment names:
- `handcrafted_only`
- `cnn_only`
- `fusion`
- `ablation_no_regional_color`
- `ablation_no_shape`

Implementation note:
- this is **similarity retrieval**, not exact-match filtering and not species classification
- for the current `1000`-image target, retrieval uses exhaustive similarity comparison over the gallery
- this is the baseline `kNN` strategy for the course project
- advanced indexing structures can be discussed later as future work, but they are not required to validate the CBIR pipeline at this scale

## Phase 6 - Experiments and Relevance Judgments

Goal:
- create a reusable query subset
- run all experiment configurations
- store species-proxy baseline labels
- prepare manual relevance annotation

### Step 6.1 - Run the Registered Experiments

```powershell
python scripts/phase6_experiments/run_experiments.py `
  --db-path data/features/cbir_features.sqlite `
  --processed-root data/processed `
  --query-count 50 `
  --top-k 5 `
  --seed 42 `
  --output-dir outputs/experiments
```

Expected outputs:
- `outputs/experiments/query_subset.csv`
- `outputs/experiments/retrieval_runs.csv`

### Step 6.2 - Export Manual Relevance Template

```powershell
python scripts/phase6_experiments/prepare_manual_relevance.py `
  --db-path data/features/cbir_features.sqlite `
  --top-k 5 `
  --output-csv outputs/manual_relevance/manual_relevance_template.csv
```

Open the exported CSV and fill:
- `relevance_grade = 0` for irrelevant
- `relevance_grade = 1` for partially similar
- `relevance_grade = 2` for highly similar

### Step 6.3 - Import Manual Relevance Judgments

```powershell
python scripts/phase6_experiments/import_relevance_judgments.py `
  --db-path data/features/cbir_features.sqlite `
  --judgments-csv outputs/manual_relevance/manual_relevance_template.csv
```

## Phase 7 - Evaluation

Main evaluation story:
- manual relevance is the **primary** CBIR evaluation
- species proxy is a **secondary baseline**

Metrics:
- manual `nDCG@5`
- manual `Precision@5`
- species-proxy `Precision@5`

Run evaluation:

```powershell
python scripts/phase7_evaluation/evaluate_retrieval.py `
  --db-path data/features/cbir_features.sqlite `
  --judgment-source both `
  --k 5 `
  --output-dir outputs/eval
```

Expected outputs:
- `outputs/eval/manual_per_query.csv`
- `outputs/eval/manual_summary.json`
- `outputs/eval/species_proxy_per_query.csv`
- `outputs/eval/species_proxy_summary.json`

The script also writes summary metrics back into the `experiments` table.

## Phase 8 - Report Artifacts

Export report-ready artifacts:

```powershell
python scripts/phase8_report_artifacts/export_report_artifacts.py `
  --db-path data/features/cbir_features.sqlite `
  --features-dir data/features `
  --output-dir outputs/report_artifacts `
  --example-experiment fusion
```

Expected outputs:
- `outputs/report_artifacts/descriptor_table.csv`
- `outputs/report_artifacts/system_block_diagram.md`
- `outputs/report_artifacts/experiment_summaries.csv`
- `outputs/report_artifacts/example_retrieval_breakdown.json`

Use these outputs to build the report sections that the assignment explicitly asks for:
- system block diagram
- functional architecture / module input-output description
- schema or ERD of the multimedia database
- feature rationale and information value
- database organization and query mechanism
- intermediate results of retrieval
- comparison among experiment settings
- qualitative discussion of success and failure cases

## Optional - Demo UI

The recommended interactive demo path is already defined in **Phase 5.1**.
