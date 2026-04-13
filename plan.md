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
  Extraction: grayscale uniform LBP histogram.
  Comparison: compute **Chi-square distance**, then convert to similarity with `S = 1 / (1 + d)`.

- `hog_descriptor`
  Captures edge structure, contour, and pose.
  Extraction: grayscale HOG with a fixed cell/block configuration.
  Comparison: cosine similarity.

- `cnn_embedding`
  Captures higher-level semantic similarity.
  Extraction: pooled embedding from pretrained `ResNet18`.
  Comparison: cosine similarity.
  Role: secondary descriptor only.

Default retrieval configurations:

- `handcrafted_only`
  `regional_hsv_hist 0.30 + global_hsv_hist 0.15 + color_moments 0.10 + lbp_hist 0.20 + hog_descriptor 0.25`

- `cnn_only`
  `cnn_embedding 1.00`

- `fusion`
  `0.80 * handcrafted_score + 0.20 * cnn_score`

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
- all six descriptors extracted successfully
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
