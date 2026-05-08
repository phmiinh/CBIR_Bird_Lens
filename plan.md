# Refactor Roadmap: DB-First CBIR for Bird Image Retrieval

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
