# Technical Roadmap

This file consolidates the previous `setup.md`, `roadmap.md`, and `plan.md` documents without rewriting their technical content. The merge is organizational: setup becomes Phase 0, the old execution roadmap remains the command flow, and the old plan remains the architecture/refactor notes.

## Phase 0 - Setup

## Environment Setup

This file covers only:
- Python environment setup
- dependency installation
- Kaggle API access
- CUB-200-2011 download

For the working execution flow, continue with [roadmap.md](/d:/PTIT/Multimedia%20Database/docs/roadmap.md).

Current local workspace note:
- the project virtual environment `.venv` is present
- raw CUB data is present under `data/raw/cub2002011`
- downstream pipeline status is tracked in [docs/current_status.md](/d:/PTIT/Multimedia%20Database/docs/current_status.md)

## 1. Create and Activate the Virtual Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment includes:
- normalization dependencies
- `torch` / `torchvision` for the secondary CNN descriptor
- `scikit-image` for LBP and HOG
- `gradio` for the optional demo UI

## 2. Configure Kaggle Access

Generate `kaggle.json` from your Kaggle account settings, then move it to:

```text
C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json
```

If you already downloaded `kaggle.json` into `Downloads`, copy it with:

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.kaggle"
Copy-Item "D:\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json" -Force
```

Optional verification:

```powershell
python -c "import kaggle; print('Kaggle API ready')"
```

## 3. Download the CUB Dataset

Run:

```powershell
python scripts/phase1_setup/download_cub.py
```

Expected output root:

```text
data/raw/cub2002011
```

The script supports the Kaggle-hosted CUB package used in this repository and resolves the extracted dataset root automatically.

## 4. Confirm the Raw Dataset Layout

You should have a structure similar to:

```text
data/raw/cub2002011/
  CUB_200_2011/
    images.txt
    image_class_labels.txt
    train_test_split.txt
    bounding_boxes.txt
    images/
```

Once this is ready, move to [roadmap.md](/d:/PTIT/Multimedia%20Database/docs/roadmap.md).


---

## Execution Roadmap

This document describes the **working command flow** for the DB-first CBIR project.

Prerequisite:
- complete [setup.md](/d:/PTIT/Multimedia%20Database/docs/roadmap.md)
- activate the project virtual environment before running any command

Architecture baseline used throughout the pipeline:
- **loose coupling**: image files on filesystem, metadata + descriptors + logs in SQLite
- **approximate similarity retrieval**: ranked content matching, not exact DBMS filtering
- **exhaustive linear scan**: accepted baseline retrieval strategy for the `1000`-image course-project scale
- **descriptor matrices loaded from SQLite**: retrieval reads descriptors from SQLite, then scores them in application code

Current workspace status:
- Phase 1-4 are complete in the current workspace.
- Phase 5/6 persisted retrieval experiments exist for `50` queries, `6` experiments, and `top-5` results.
- Phase 7 currently has species-proxy evaluation outputs only.
- Phase 8 report artifacts exist under `outputs/report_artifacts`.
- Manual relevance labels are not filled yet; rerun Phase 7 and Phase 8 after importing manual judgments.

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
  --clean-output `
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
  --clean-output `
  --require-count 1000 `
  --max-images 1000
```

The final normalized gallery is written to:
- `data/processed/images`
- `data/processed/metadata/images.csv`
- `data/processed/metadata/images.jsonl`

Metadata note:
- `width` and `height` in `images.csv` are the original source-image dimensions
- normalized gallery size is recorded separately as `target_width` and `target_height`

## Phase 3 - Descriptor Extraction

Goal:
- extract the full feature inventory required by the DB-first retrieval system
- suppress background influence for handcrafted descriptors by focusing extraction on the bird foreground

Descriptor set:
- global HSV histogram
- regional HSV histogram (`2x2`)
- color moments
- LBP histogram: uniform LBP with `P=8`, `R=1`, so `P + 2 = 10` bins
- HOG descriptor: `224x224` grayscale, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, `orientations=9`, so `13 x 13 x 4 x 9 = 6084`
- silhouette shape descriptor
- CNN embedding (`ResNet18`, secondary only)

Implementation note:
- handcrafted descriptors are extracted in a **foreground-aware** way
- gallery images use the stored bird bbox/crop metadata to estimate a bird-focused mask
- external queries use an estimated foreground mask after query normalization
- after changing this logic, rebuild both `data/features` and `data/features/cbir_features.sqlite`

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
- `data/features/silhouette_shape_descriptor.npy`
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
- the retrieval engine loads descriptor matrices from SQLite and performs exhaustive similarity scoring in application code
- descriptor vectors are stored as JSON text for transparency and reproducibility at the `1000`-image course-project scale; this is not a production storage/indexing strategy
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

Important behavior note:
- the UI is a read-only retrieval demo
- it uses the DB-backed gallery descriptors but does **not** persist query logs to SQLite
- official persisted runs should come from the CLI retrieval path or Phase 6 experiments

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
- `ablation_no_explicit_shape`
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

Important note:
- `manual_per_query.csv` and `manual_summary.json` are generated only after you fill and import the manual relevance CSV from Phase 6.3
- if you have not imported manual judgments yet, Phase 7 will currently produce only the species-proxy evaluation outputs
- in the current workspace, only `species_proxy_per_query.csv` and `species_proxy_summary.json` are present

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

Current workspace note:
- these Phase 8 files have been exported
- the current experiment summaries contain species-proxy metrics only until manual relevance is imported and Phase 7/8 are rerun

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


---

## Refactor Roadmap: DB-First CBIR for Bird Image Retrieval

## Summary

Reposition the repository from a retrieval demo into a **Multimedia Database project** centered on:

- curated normalized gallery data
- explainable feature extraction
- database-managed descriptors
- DB-backed `top-5` retrieval
- evaluation based on **visual relevance** rather than classification

Locked defaults:
- gallery target: `1000` normalized images
- database backbone: `SQLite`
- storage architecture: **loose coupling** (`image files on filesystem`, `metadata + descriptors + logs in SQLite`)
- main feature story: handcrafted descriptors are primary; CNN is secondary for comparison and optional fusion
- main evaluation story: **manual relevance** with graded labels
- baseline evaluation: **species proxy**
- query semantics: **approximate similarity retrieval**, not exact DBMS matching and not image classification

## Execution Order

Implementation should follow this order to avoid refactor drift:

1. Freeze Phase 2 and scale the curated gallery to `1000` images.
2. Implement the full handcrafted feature set.
3. Build the new SQLite schema and ingestion pipeline.
4. Migrate retrieval to be fully DB-backed.
5. Add manual relevance annotation and evaluation pipeline.
6. Run experiments, ablations, and export report artifacts.
7. Reintroduce or refine the demo UI only after DB-backed retrieval is stable.

## Current Implementation Status

As of the latest workspace review, the implementation has advanced beyond the older Phase 3 handoff note:

- Phase 1 raw dataset is present.
- Phase 2 has produced `1000` normalized gallery images.
- Phase 3 has produced all `7` descriptor matrices, including `silhouette_shape_descriptor` with dimension `15`.
- Phase 4 SQLite build exists at `data/features/cbir_features.sqlite`.
- Phase 5/6 retrieval experiments have produced `300` persisted runs and `1500` ranked results.
- Phase 7 currently has species-proxy evaluation only.
- Phase 8 report artifacts have been exported to `outputs/report_artifacts`.
- Manual relevance remains incomplete; the current template has blank `relevance_grade` values, so manual `nDCG@5` and manual `Precision@5` are still pending.

## Key Changes

### 1. Keep Phase 2, but make `1000` images the working target

- Reuse the current review, normalization, and metadata flow.
- Keep species labels only as metadata and baseline evaluation support.
- Treat `data/processed` as the stable boundary for downstream retrieval work.

### 1.5 Align the system explicitly with multimedia DB literature

The implementation and report should present the system as a **multimedia information retrieval system (MIRS)** rather than a classifier. Three architecture choices should be stated clearly:

- **Loose coupling**
  The image files remain on the filesystem while SQLite manages metadata, descriptors, query cache, judgments, and experiment logs.

- **Functional MM-DBMS modules**
  The report should map the implementation to the following modules:
  - presentation / result presentation
  - query manager
  - metadata manager
  - storage manager
  - feature extraction and similarity layer

- **Approximate similarity retrieval**
  Retrieval is based on descriptor comparison and ranked similarity, not exact attribute matching as in a traditional DBMS, and not species classification.

### 2. Lock the full feature set now

Use this descriptor inventory as the official feature set:

- `global_hsv_hist`
  Captures overall color distribution.
  Extraction: fixed-bin HSV histogram over the whole normalized image.
  Comparison: compute **Chi-square distance**, then convert to similarity with `S = 1 / (1 + d)`.

- `regional_hsv_hist`
  Captures spatial color layout.
  Extraction: split the normalized image into a fixed `2x2` grid, compute one HSV histogram per cell, then concatenate.
  Comparison: compute **Chi-square distance**, then convert to similarity with `S = 1 / (1 + d)`.
  Role: primary color-layout descriptor.

- `color_moments`
  Captures coarse color statistics.
  Extraction: mean, standard deviation, and skewness on `H`, `S`, `V`.
  Comparison: Euclidean distance converted to similarity with `S = 1 / (1 + d)`.

- `lbp_hist`
  Captures local feather and texture micro-patterns.
  Extraction: grayscale uniform LBP histogram with `P=8`, `R=1`.
  Dimension: uniform LBP uses `P + 2` histogram bins, so `8 + 2 = 10`.
  Comparison: compute **Chi-square distance**, then convert to similarity with `S = 1 / (1 + d)`.

- `hog_descriptor`
  Captures edge structure, contour, and pose.
  Extraction: `224x224` grayscale HOG with `orientations=9`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, and `block_norm=L2-Hys`.
  Dimension: `224/16 = 14` cells per axis, overlapping `2x2` blocks give `13x13` block positions, and each block has `2x2x9` values, so `13 x 13 x 4 x 9 = 6084`.
  Comparison: cosine similarity.

- `silhouette_shape_descriptor`
  Captures global body silhouette and coarse geometry from the bird foreground mask.
  Extraction: Hu moments plus region properties such as area ratio, aspect ratio, extent, eccentricity, solidity, orientation, major/minor-axis ratio, and compactness.
  Comparison: Euclidean distance converted to similarity with `S = 1 / (1 + d)`.

- `cnn_embedding`
  Captures higher-level semantic similarity.
  Extraction: pooled embedding from pretrained `ResNet18`.
  Comparison: cosine similarity.
  Role: secondary descriptor only.

Default retrieval configurations:

- `handcrafted_only`
  tuned handcrafted stack with a light explicit-shape contribution:
  `regional_hsv_hist 0.261 + global_hsv_hist 0.094 + color_moments 0.195 + lbp_hist 0.118 + hog_descriptor 0.272 + silhouette_shape_descriptor 0.060`

- `cnn_only`
  `cnn_embedding 1.00`

- `fusion`
  handcrafted-first fusion with the handcrafted stack rescaled to `0.80` plus `cnn_embedding 0.20`, for a total weight sum of `1.00`:
  `regional_hsv_hist 0.209 + global_hsv_hist 0.075 + color_moments 0.156 + lbp_hist 0.094 + hog_descriptor 0.218 + silhouette_shape_descriptor 0.048 + cnn_embedding 0.200`

### 3. Make the database the center of the system

Adopt this SQLite schema:

- `images`
  Fields: `image_id`, `source_relative_path`, `processed_relative_path`, `species_id`, `species_name`, `split`, `width`, `height`, `is_perching`, `is_side_view`, `keep`
  Purpose: fast inspection, demo, and reporting without always joining preprocessing tables.

- `preprocessing_metadata`
  Fields: `image_id`, `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`, `crop_left`, `crop_top`, `crop_right`, `crop_bottom`, `target_width`, `target_height`, `padding_ratio`, `keep`, `notes`, `preprocess_params_json`

- `feature_types`
  Fields: `feature_type_id`, `name`, `vector_dim`, `similarity_metric`, `extraction_params_json`, `is_primary`

- `image_features`
  Fields: `image_id`, `feature_type_id`, `vector_json`

Vector-storage tradeoff:

- `vector_json` is a deliberate course-project choice: it is transparent, easy to inspect, reproducible across machines, and compatible with SQLite JSON checks.
- It is not a production-efficient vector layout. In the current artifact, `data/features/cbir_features.sqlite` is about `89.96 MB`; `image_features` has `7000` rows; HOG alone has `6084` dimensions and its JSON payload accounts for roughly `49.6 MB` of stored vector text.
- For larger galleries, keep SQLite as the metadata/system-of-record layer but move dense vectors to `float32` BLOBs, `.npy`/Parquet sidecars, or an ANN/vector-index layer.

- `queries`
  Fields: `query_id`, `query_source_type`, `query_image_path`, `created_at_utc`, `preprocess_params_json`

- `query_features`
  Fields: `query_id`, `feature_type_id`, `vector_json`

- `retrieval_runs`
  Fields: `run_id`, `query_id`, `experiment_id`, `mode`, `top_k`, `created_at_utc`

- `retrieval_results`
  Fields: `run_id`, `rank`, `image_id`, `per_feature_scores_json`, `fused_score`

- `relevance_judgments`
  Fields: `query_id`, `image_id`, `judgment_source`, `relevance_grade`

- `experiments`
  Fields: `experiment_id`, `name`, `feature_set_json`, `weighting_json`, `dataset_version`, `summary_metrics_json`, `notes`

DB-backed retrieval flow:

1. Normalize the query using the same preprocessing rule.
2. Create or reuse a query record in `queries`.
3. Extract and cache descriptors into `query_features`.
4. Fetch gallery descriptors from `image_features`.
5. Compute per-feature distances and convert them to similarities where needed.
6. Fuse scores according to experiment weights.
7. Persist ranked outputs into `retrieval_results`.
8. Persist configuration and summary metrics into `experiments`.

Important implementation note:
- SQLite is the system-of-record for metadata, descriptors, queries, judgments, and experiment logs.
- The current retrieval engine loads descriptor matrices from SQLite and performs an exhaustive similarity scan in application code.
- This is still DB-first and appropriate for the `1000`-image project scale; it should not be described as SQL-native vector retrieval.

Functional module I/O:

| Module | Input | Output | Implementation note |
| --- | --- | --- | --- |
| Query manager | gallery image id or external query image | normalized query record in `queries` | applies the same `224x224` preprocessing contract |
| Feature extraction layer | normalized query/gallery image plus foreground mask | descriptor vectors | 6 handcrafted primary descriptors plus secondary `cnn_embedding` |
| Metadata manager | query/image ids and feature names | feature type ids, experiment config, preprocessing trace | uses `feature_types`, `images`, `preprocessing_metadata`, `experiments` |
| Storage manager | descriptor matrices and query descriptors | `image_features` and `query_features` rows | SQLite is the descriptor system of record |
| Similarity layer | query vector and gallery matrix per descriptor | per-feature similarity scores | exhaustive kNN/linear scan in Python application code |
| Fusion/ranking layer | per-feature scores and experiment weights | ranked top-k results | weights must sum to `1.00`; `fusion` is `0.80` handcrafted + `0.20` CNN |
| Presentation/evaluation | ranked top-k and relevance judgments | CSV/JSON report artifacts, `Precision@5`, `nDCG@5` | manual relevance is primary; species proxy is secondary |

Operational retrieval strategy:

- for the `1000`-image course-project scale, the baseline retrieval engine should use **exhaustive kNN / linear scan** over gallery descriptors loaded from SQLite
- the report should state clearly that this is acceptable at the current scale and keeps the system simple, transparent, and reproducible
- advanced indexing structures such as `k-d tree`, `R-tree` variants, or triangle-inequality filtering should be treated as **future optimization / discussion material**, not the main implementation burden
- the report should explicitly distinguish this approximate similarity search flow from conventional exact-match DBMS querying

Database presentation requirements:

- show the system as a **loose-coupling architecture**
- include one schema/ERD view that separates image files from descriptor metadata storage
- explain that the current retrieval baseline is a transparent exhaustive scan over the descriptor space
- mention advanced indexing only as a scalability extension, not as a mandatory implementation step

### 4. Redesign evaluation around visual relevance

Use two evaluation tracks:

- `manual_relevance` as the main CBIR evaluation
- `species_proxy` as a secondary baseline

Manual relevance protocol:

- use a query subset of `50` queries
- use graded relevance labels:
  - `0`: irrelevant
  - `1`: partially similar
  - `2`: highly similar
- store labels in `relevance_judgments` with `judgment_source='manual'`

Species proxy protocol:

- use same-species as weak binary relevance
- store labels with `judgment_source='species_proxy'`

Metrics:

- main manual-relevance metric: `nDCG@5`
- supplementary manual-relevance metric: `Precision@5`
- secondary baseline metric: `species-proxy Precision@5`

This should be stated explicitly in the docs and report:

- `nDCG@5` is the primary ranking metric because manual relevance is graded
- `Precision@5` is retained because it is easy to read and compare
- species-based evaluation is only a proxy, not the success definition of the system

Minimum experiment set:

- `handcrafted_only`
- `cnn_only`
- `fusion`
- `ablation_no_regional_color`
- `ablation_no_explicit_shape`
- `ablation_no_shape`

### 5. Rewrite docs to match the new project identity

- `plan.md`
  Recreate from scratch as the master roadmap.

- `README.md`
  Reframe the repo as a Multimedia Database CBIR project, explicitly not a classification project.

- `roadmap.md`
  Maintain the executable command flow: setup, normalization, handcrafted features, DB ingestion, DB-backed retrieval, experiments, evaluation, and report artifacts.

- `setup.md`
  Keep only environment, dependency, and dataset acquisition instructions.

### 6. Add report-ready artifacts

The repository should directly support report writing.

Required artifacts:

- system block diagram with modules:
  preprocessing, feature extraction, feature DB, query processing, similarity computation, ranking, evaluation

- descriptor table:
  feature name, captured information, strengths, weaknesses, similarity formula

- example retrieval breakdown:
  one query, per-feature score contributions, fused ranking

- experiment summary exports:
  metrics and ablations tied back to `experiments`

Additional report requirements to make the project academically strong:

- explicitly state that this is a **similarity retrieval** problem, not an image classification problem
- include a short justification for choosing **loose coupling** over a monolithic BLOB-centric design
- include at least one schema/ERD figure showing how image metadata, descriptors, queries, judgments, and experiments are related
- include one functional architecture figure with modules aligned to multimedia DB literature:
  - presentation layer / result presentation
  - query manager
  - metadata manager
  - storage manager
  - feature extraction + similarity layer
- include module-level input/output descriptions and formulas for:
  - Chi-square distance and similarity conversion
  - Euclidean distance and similarity conversion
  - cosine similarity
  - weighted score fusion
- include one short explanation of why multimedia retrieval differs from exact-match DBMS querying:
  - descriptors only represent content approximately
  - relevance is graded and user-dependent
  - ranking quality matters more than exact identity matching
- include intermediate result screenshots:
  - raw image + bbox
  - normalized image
  - descriptor examples
  - per-feature score table
  - final top-5 ranking

## Test Plan

Acceptance criteria:

- `1000` curated normalized images with complete metadata
- all seven descriptors extracted successfully
- `images`, `preprocessing_metadata`, `feature_types`, `image_features`, `queries`, `query_features`, `retrieval_runs`, `retrieval_results`, `relevance_judgments`, and `experiments` populated consistently
- retrieval works for:
  - `handcrafted_only`
  - `cnn_only`
  - `fusion`
- evaluation outputs:
  - manual `nDCG@5`
  - manual `Precision@5`
  - species-proxy `Precision@5`
- comparison outputs exist for both ablation settings
- docs and code use the same terminology and architecture

Current acceptance status:

- complete: `1000` normalized images
- complete: all seven descriptors
- complete: SQLite image, metadata, feature, query, retrieval, experiment, and species-proxy judgment rows
- complete: retrieval runs for `handcrafted_only`, `cnn_only`, `fusion`, and ablations
- complete: species-proxy `Precision@5`
- pending: manual relevance import
- pending: manual `nDCG@5`
- pending: manual `Precision@5`

Critical validation checks:

- the query image is excluded from ranked results when querying from the gallery
- feature row counts match image counts
- `regional_hsv_hist` dimension is stable and documented
- missing manual judgments do not crash evaluation
- every experiment result can be traced back to a stored config in `experiments`

## Assumptions and Defaults

- `1000` images is the target for the refactor
- `2x2` is the default regional color grid in the first pass
- `ResNet18` is the only CNN backbone in the first pass
- SQLite stores vectors as JSON text for transparency and reproducibility
- exhaustive similarity scan is the baseline retrieval method at this scale; indexing is documented as future work
- manual relevance is the official evaluation story
- the demo UI is optional until DB-backed retrieval is stable

## Reference Anchors For The Report

- CUB-200-2011 official dataset page: cite for dataset scale, categories, bounding boxes, part locations, and attributes.
- scikit-image `local_binary_pattern` documentation and Ojala et al. 2002: cite for uniform LBP as a texture descriptor.
- scikit-image `hog` documentation and Dalal-Triggs 2005: cite for HOG cell/block/orientation extraction.
- torchvision `resnet18` documentation: cite for the secondary pretrained ImageNet embedding source.
- SQLite JSON1 and internal-vs-external BLOB documentation: cite for explaining why JSON text vectors are transparent at this scale but not the production-optimal vector layout.



