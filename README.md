# AutoFE — Automated Feature Engineering Pipeline

> **Industrial-grade, config-driven feature engineering system built on Featuretools Deep Feature Synthesis.**  
> Works on *any* unknown dataset with minimal human input.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [CLI Usage](#cli-usage)
7. [Python API Usage](#python-api-usage)
8. [Configuration Reference](#configuration-reference)
9. [Pipeline Stages](#pipeline-stages)
10. [Supported File Formats](#supported-file-formats)
11. [Output Artefacts](#output-artefacts)
12. [Advanced Usage](#advanced-usage)
13. [Running Tests](#running-tests)
14. [Troubleshooting](#troubleshooting)
15. [Technology Stack](#technology-stack)

---

## Overview

AutoFE is a **zero-intervention** feature engineering system. Drop in a raw CSV (or JSON, Parquet, Excel, JSONL, Feather) and receive a cleaned, fully engineered, ML-ready feature matrix — along with profiling reports, feature definitions, and metadata JSON.

### What it does automatically

| Stage | What happens |
|-------|-------------|
| **Ingest** | Detects encoding, delimiter, schema, flattens nested JSON |
| **Profile** | Classifies every column (numeric / categorical / text / datetime / id / boolean / constant …) |
| **Clean** | Imputes nulls, removes duplicates, clips outliers, normalises datetimes |
| **Engineer** | Featuretools DFS + frequency/cyclical/TF-IDF/polynomial/statistical features |
| **Select** | Variance filter → correlation filter → mutual information top-K |
| **Export** | Saves CSV / Parquet / Feather + JSON reports |

---

## Architecture

```
AutoFEPipeline (pipeline.py)
│
├── DataLoader         (src/ingestion/loader.py)
│   └── Auto-detects format, encoding, delimiter; chunked reading for large files
│
├── DataProfiler       (src/profiling/profiler.py)
│   └── Column type inference, quality metrics, correlation summary
│
├── DataCleaner        (src/cleaning/cleaner.py)
│   └── Imputation, dedup, outlier clipping, datetime parsing
│
├── FeatureEngineer    (src/engineering/feature_engineer.py)
│   ├── Layer 1 — Featuretools DFS (aggregation + transform primitives)
│   ├── Layer 2 — Manual features (freq-encoding, cyclical, TF-IDF, polynomial)
│   └── Layer 3 — Datetime decomposition
│
├── FeatureSelector    (src/selection/selector.py)
│   └── Variance → Correlation → Mutual Information → Leakage removal
│
└── Exporter           (src/output/exporter.py)
    └── CSV / Parquet / Feather + JSON metadata / reports
```

---

## Project Structure

```
autofe/
├── pipeline.py                  # Main orchestrator + CLI entry point
├── requirements.txt
├── config/
│   └── pipeline_config.yaml     # All tuneable parameters
├── src/
│   ├── ingestion/
│   │   └── loader.py            # Universal data loader
│   ├── profiling/
│   │   └── profiler.py          # Schema inference & data quality
│   ├── cleaning/
│   │   └── cleaner.py           # Adaptive data cleaning
│   ├── engineering/
│   │   └── feature_engineer.py  # Featuretools DFS + advanced features
│   ├── selection/
│   │   └── selector.py          # Feature pruning
│   ├── output/
│   │   └── exporter.py          # Artefact saving
│   └── utils/
│       ├── config.py            # Typed config dataclasses + YAML loader
│       └── logger.py            # Rotating file + colour console logger
├── tests/
│   └── test_pipeline.py         # Unit + integration tests
├── data/
│   └── sample/
│       └── generate_sample.py   # Sample dataset generator
├── examples/
│   └── example_usage.py         # Programmatic usage demo
├── logs/                        # Auto-created; autofe.log written here
└── output/                      # Auto-created; run outputs go here
```

---

## Installation

### 1 — Clone / download the project

```bash
git clone https://github.com/your-org/autofe.git
cd autofe
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Python version**: 3.9 or later is required.  
> **Featuretools** requires Python ≤ 3.11 for the broadest primitive support.

### 4 — Verify installation

```bash
python -c "import featuretools; print('Featuretools', featuretools.__version__)"
```

---

## Quick Start

### Generate sample data and run

```bash
# Step 1 – generate a messy demo CSV
python data/sample/generate_sample.py

# Step 2 – run the full pipeline
python pipeline.py --input data/sample/customer_churn.csv --target churn
```

Output appears in `output/<timestamp>/`:

```
output/20240512_143022/
├── cleaned_data.csv
├── cleaned_data.parquet
├── feature_matrix.csv
├── feature_matrix.parquet
├── selected_features.csv
├── selected_features.parquet
├── feature_definitions.json
├── metadata.json
├── profiling_report.json
└── selected_feature_list.txt
```

---

## CLI Usage

```
python pipeline.py --input <path> [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to file **or** directory of files |
| `--target` | auto-detect | Name of the target / label column |
| `--config` | `config/pipeline_config.yaml` | Path to YAML config file |
| `--output` | from config | Override output directory |
| `--max-depth` | from config | Override DFS maximum depth (1–3) |
| `--no-select` | False | Skip feature selection stage |
| `--formats` | from config | Space-separated: `csv parquet feather` |

### Examples

```bash
# Minimal — auto-detect everything
python pipeline.py --input data/sales.csv

# Specify target, depth 1 for speed, only Parquet output
python pipeline.py \
  --input data/sales.csv \
  --target revenue \
  --max-depth 1 \
  --formats parquet

# Load an entire directory of CSVs
python pipeline.py \
  --input data/monthly_files/ \
  --target churn \
  --output results/monthly_run

# Skip feature selection (keep all generated features)
python pipeline.py \
  --input data/train.parquet \
  --no-select

# Use a custom config file
python pipeline.py \
  --input data/orders.json \
  --config config/my_project.yaml
```

---

## Python API Usage

### Basic programmatic usage

```python
from src.utils.config import PipelineConfig
from pipeline import AutoFEPipeline

# Default config
cfg = PipelineConfig()

pipeline = AutoFEPipeline(cfg)
result_df = pipeline.run("data/my_data.csv", target_col="label")

print(result_df.shape)
print(result_df.head())
```

### Override config programmatically

```python
from src.utils.config import PipelineConfig
from pipeline import AutoFEPipeline

cfg = PipelineConfig()

# Engineering
cfg.engineering.max_depth = 1          # DFS depth (1 = fast, 2 = thorough)
cfg.engineering.max_features = 200     # Hard cap on generated features
cfg.engineering.add_text_features = False

# Cleaning
cfg.cleaning.outlier_method = "zscore"
cfg.cleaning.drop_missing_threshold = 0.5

# Selection
cfg.selection.enabled = True
cfg.selection.mutual_info_top_k = 150
cfg.selection.correlation_threshold = 0.90

# Output
cfg.output.base_dir = "my_outputs"
cfg.output.formats = ["parquet"]

pipeline = AutoFEPipeline(cfg)
df = pipeline.run("data/raw.csv", target_col="survived")
```

### Load config from YAML

```python
from src.utils.config import load_config
from pipeline import AutoFEPipeline

cfg = load_config("config/pipeline_config.yaml")
cfg.output.base_dir = "run_001"

pipeline = AutoFEPipeline(cfg)
pipeline.run("data/orders.parquet", target_col="refunded")
```

### Access intermediate results

```python
from src.utils.config import PipelineConfig
from pipeline import AutoFEPipeline

cfg = PipelineConfig()
pipeline = AutoFEPipeline(cfg)
pipeline.run("data/customers.csv", target_col="churn")

# All intermediate DataFrames are stored on the pipeline object
print("Raw         :", pipeline.raw_df.shape)
print("Cleaned     :", pipeline.clean_df.shape)
print("Features    :", pipeline.feature_matrix.shape)
print("Selected    :", pipeline.selected_features.shape)

# Feature definitions from DFS
for fd in pipeline.metadata["feature_definitions"][:5]:
    print(fd)
```

### Use individual modules

```python
import pandas as pd
from src.utils.config import PipelineConfig
from src.profiling.profiler import DataProfiler
from src.cleaning.cleaner import DataCleaner

cfg = PipelineConfig()
df = pd.read_csv("data/raw.csv")

profiler = DataProfiler(cfg)
profile = profiler.profile(df, target_col="label")

print("Numeric columns :", profile["numeric_columns"])
print("Text columns    :", profile["text_columns"])
print("Suggested target:", profile["suggested_target"])

cleaner = DataCleaner(cfg)
clean_df = cleaner.clean(df, profile)
```

---

## Configuration Reference

The full config lives in `config/pipeline_config.yaml`. Every value can also be set on the `PipelineConfig` dataclass in Python.

### `ingestion`

```yaml
ingestion:
  chunk_size: 50000          # Rows per chunk for large-file reading
  max_sample_rows: 500000    # Hard cap — random-sample if exceeded
  encoding_fallbacks:        # Tried in order when chardet fails
    - utf-8
    - latin-1
    - cp1252
  csv_sep: null              # null = auto-detect delimiter
  excel_sheet: 0             # Sheet index or name
  infer_datetime: true       # Let pandas infer datetime strings
```

### `type_detection`

```yaml
type_detection:
  id_max_cardinality_ratio: 0.95   # unique_values/n > 0.95 → ID column
  cat_max_cardinality: 50          # ≤50 unique values → categorical
  text_avg_token_threshold: 5      # avg tokens ≥5 → free text
```

### `cleaning`

```yaml
cleaning:
  drop_missing_threshold: 0.7      # Drop column if >70% missing
  numeric_impute_strategy: median  # mean | median | most_frequent
  cat_impute_strategy: most_frequent
  outlier_method: iqr              # iqr | zscore | none
  outlier_clip_factor: 3.0         # IQR multiplier OR z-score threshold
  drop_duplicates: true
  remove_constant_cols: true
```

### `engineering`

```yaml
engineering:
  max_depth: 2                     # DFS depth; 1=fast, 2=thorough, 3=slow
  max_features: 500                # Hard cap on total generated features
  add_statistical_features: true   # Row-wise mean/std/min/max/range
  add_frequency_encoding: true     # Categorical → value-count ratio
  add_cyclical_encoding: true      # sin/cos for periodic numeric cols
  add_text_features: true          # TF-IDF for free-text columns
  tfidf_max_features: 50           # Max TF-IDF vocabulary size
  add_polynomial_features: false   # Interaction terms (expensive)
```

### `selection`

```yaml
selection:
  enabled: true
  variance_threshold: 0.01         # Drop features with variance < this
  correlation_threshold: 0.95      # Drop one of each highly-corr pair
  mutual_info_top_k: 300           # Keep top-K by mutual information
  drop_leaky_id_cols: true         # Remove ID-like column names
```

### `output`

```yaml
output:
  base_dir: output                 # Root output directory
  formats: [parquet, csv]          # csv | parquet | feather
  save_cleaned: true
  save_feature_matrix: true
  save_feature_definitions: true
  save_metadata: true
  save_profiling_report: true
  save_selected_features: true
  compress_parquet: true           # snappy compression
```

### `performance`

```yaml
performance:
  n_jobs: -1                       # CPU cores for DFS; -1 = all
  use_dask: false                  # Enable for datasets >10M rows
  cache_entityset: true
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion

- Auto-detects file format from extension
- Uses `chardet` for encoding detection with UTF-8 / latin-1 / cp1252 fallbacks
- Sniffs CSV delimiter (`,` `\t` `;` `|`)
- Flattens nested JSON dicts up to 3 levels deep
- Chunks large files (>200 MB) to avoid OOM
- Concatenates multiple files from a directory
- Sanitises column names (lowercase, underscores)

### Stage 2 — Profiling

- Classifies every column: `numeric`, `categorical`, `boolean`, `text`, `datetime`, `id`, `constant`, `url`, `geospatial`
- Computes per-column missing%, unique%, mean, std, skewness, kurtosis
- Detects duplicate rows
- Identifies high-correlation pairs
- Auto-suggests the target column by name pattern

### Stage 3 — Cleaning

- Drops columns exceeding `drop_missing_threshold` (default 70%)
- Drops constant columns
- Strips whitespace from string columns
- Replaces `inf` / `-inf` with NaN
- Parses datetime strings to `datetime64`
- Converts boolean-like strings (`yes/no`, `true/false`) to 0/1
- Imputes numerics with median (configurable)
- Imputes categoricals with most-frequent (configurable)
- Clips outliers via IQR or z-score
- Removes duplicate rows

### Stage 4 — Feature Engineering

**Layer 1 — Featuretools DFS**  
Builds an `EntitySet` from the cleaned data, auto-detects the primary key, and runs Deep Feature Synthesis to depth `max_depth` using configurable aggregation and transform primitives.

**Layer 2 — Manual features**
- **Frequency encoding**: each category → its proportion in the dataset
- **Cyclical encoding**: month/day/weekday/hour → sin & cos
- **Row statistics**: mean, std, min, max, range across all numeric columns
- **TF-IDF**: top-N vocabulary features from free-text columns
- **Polynomial interactions**: pairwise products of numeric columns (opt-in)

**Layer 3 — Datetime decomposition**  
Extracts year, month, day, weekday, hour, is_weekend from every datetime column.

### Stage 5 — Feature Selection

1. Encode remaining categoricals with `LabelEncoder`
2. Remove duplicate columns (bit-identical)
3. `VarianceThreshold` (default 0.01)
4. Pairwise correlation filter (default 0.95)
5. Mutual information top-K ranking (classif or regression auto-detected)
6. Drop ID-pattern column names to prevent leakage

### Stage 6 — Export

Saves artefacts to `output/<run_timestamp>/`:

| File | Description |
|------|-------------|
| `cleaned_data.*` | Data after cleaning |
| `feature_matrix.*` | All generated features |
| `selected_features.*` | Post-selection features |
| `feature_definitions.json` | Name, primitive, origin, base features |
| `profiling_report.json` | Full per-column statistics |
| `metadata.json` | Run summary, shapes, config snapshot |
| `selected_feature_list.txt` | One feature name per line |

---

## Supported File Formats

| Format | Extensions | Notes |
|--------|-----------|-------|
| CSV | `.csv` `.txt` | Auto delimiter + encoding detection |
| TSV | `.tsv` | Tab-delimited |
| JSON | `.json` | Records / index / split / values / table orient |
| JSONL | `.jsonl` | One JSON object per line |
| Parquet | `.parquet` | Via PyArrow |
| Excel | `.xlsx` `.xls` | Configurable sheet |
| Feather | `.feather` `.ftr` | Fast binary columnar |

---

## Advanced Usage

### Process a directory of files

```bash
python pipeline.py --input data/monthly_logs/ --target conversion
```

All supported files in the directory are concatenated before processing.

### Large dataset (>1M rows)

```yaml
# config/large_dataset.yaml
ingestion:
  chunk_size: 100000
  max_sample_rows: 2000000
engineering:
  max_depth: 1
  max_features: 300
  add_text_features: false
  add_polynomial_features: false
performance:
  n_jobs: -1
  use_dask: false
output:
  formats: [parquet]
```

```bash
python pipeline.py --input data/big.csv --config config/large_dataset.yaml
```

### Custom YAML config per project

```bash
cp config/pipeline_config.yaml config/my_project.yaml
# Edit my_project.yaml …
python pipeline.py --input data/orders.parquet --config config/my_project.yaml
```

### Disable feature selection

```bash
python pipeline.py --input data/train.csv --no-select
```

### Fast mode (DFS depth 1, no text / polynomial features)

```bash
python pipeline.py --input data/raw.csv --max-depth 1
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test class
pytest tests/test_pipeline.py::TestCleaner -v
```

Expected output:

```
tests/test_pipeline.py::TestConfig::test_defaults              PASSED
tests/test_pipeline.py::TestConfig::test_load_missing_file... PASSED
tests/test_pipeline.py::TestProfiler::test_column_types       PASSED
tests/test_pipeline.py::TestProfiler::test_quality_metrics    PASSED
tests/test_pipeline.py::TestProfiler::test_missing_pct        PASSED
tests/test_pipeline.py::TestCleaner::test_removes_constant... PASSED
tests/test_pipeline.py::TestCleaner::test_imputes_numerics    PASSED
tests/test_pipeline.py::TestCleaner::test_no_duplicates       PASSED
tests/test_pipeline.py::TestCleaner::test_inf_handling        PASSED
tests/test_pipeline.py::TestSelector::test_removes_high_corr  PASSED
tests/test_pipeline.py::TestSelector::test_target_preserved   PASSED
tests/test_pipeline.py::TestIntegration::test_smoke           PASSED
```

---

## Troubleshooting

### `ModuleNotFoundError: featuretools`

```bash
pip install featuretools
```

The pipeline will still run without Featuretools — DFS is skipped and all other feature layers remain active.

### `UnicodeDecodeError` on CSV load

The loader tries UTF-8, latin-1, and cp1252 automatically. If your file uses a different encoding:

```yaml
ingestion:
  encoding_fallbacks:
    - utf-8
    - utf-16
    - shift_jis
```

### Memory error on large files

Reduce `max_sample_rows` or enable chunked loading:

```yaml
ingestion:
  chunk_size: 20000
  max_sample_rows: 200000
```

### DFS takes too long

- Set `max_depth: 1`
- Reduce `max_features: 100`
- Increase `n_jobs: -1` (already default)
- Disable expensive primitives in the config `include_primitives` section

### No features selected after selection stage

Lower the thresholds:

```yaml
selection:
  variance_threshold: 0.001
  correlation_threshold: 0.99
  mutual_info_top_k: 500
```

Or disable selection entirely:

```bash
python pipeline.py --input data/raw.csv --no-select
```

### Parquet write fails with mixed types

The exporter auto-converts object columns to strings before Parquet writes. If you still hit issues, use `--formats csv` as a reliable fallback.

---

## Technology Stack

| Library | Role |
|---------|------|
| **Featuretools** | Deep Feature Synthesis (DFS) |
| **Woodwork** | Logical type system for EntitySet |
| **Pandas** | Core dataframe operations |
| **NumPy** | Vectorised math |
| **Scikit-learn** | Imputation, encoding, MI, variance threshold |
| **SciPy** | Skewness, kurtosis |
| **PyArrow** | Parquet read/write |
| **chardet** | Encoding detection |
| **PyYAML** | Config loading |
| **openpyxl / xlrd** | Excel support |

---

## License

MIT — see `LICENSE` for details.

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Add tests in `tests/`
4. Run `pytest tests/ -v` — all tests must pass
5. Open a Pull Request

---

*Built for Kaggle competitions, enterprise ML pipelines, research workflows, and production AI systems.*
