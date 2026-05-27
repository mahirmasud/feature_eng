"""
src/utils/config.py
───────────────────
Load and validate pipeline configuration from YAML / JSON.
All config sections are typed dataclasses so IDE autocomplete works.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────

@dataclass
class IngestionConfig:
    chunk_size: int = 50_000
    max_sample_rows: int = 500_000
    encoding_fallbacks: List[str] = field(
        default_factory=lambda: ["utf-8", "latin-1", "cp1252"]
    )
    csv_sep: Optional[str] = None
    excel_sheet: Any = 0
    json_orient: Optional[str] = None
    infer_datetime: bool = True
    low_memory: bool = False


@dataclass
class TypeDetectionConfig:
    id_max_cardinality_ratio: float = 0.95
    cat_max_cardinality: int = 50
    text_avg_token_threshold: int = 5
    bool_values: List[List] = field(
        default_factory=lambda: [
            [True, False], [0, 1], ["yes", "no"], ["y", "n"]
        ]
    )
    datetime_patterns: List[str] = field(
        default_factory=lambda: [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        ]
    )


@dataclass
class CleaningConfig:
    drop_missing_threshold: float = 0.70
    numeric_impute_strategy: str = "median"
    cat_impute_strategy: str = "most_frequent"
    text_impute_value: str = ""
    outlier_method: str = "iqr"          # iqr | zscore | none
    outlier_clip_factor: float = 3.0
    drop_duplicates: bool = True
    remove_constant_cols: bool = True
    whitespace_strip: bool = True


@dataclass
class EngineeringConfig:
    max_depth: int = 2
    max_features: int = 500
    include_primitives: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "aggregation": [
                "COUNT", "SUM", "MEAN", "STD", "MIN", "MAX",
                "MEDIAN", "NUM_UNIQUE", "PERCENT_TRUE", "SKEW",
            ],
            "transform": [
                "ADD_NUMERIC", "DIVIDE_NUMERIC", "SUBTRACT_NUMERIC",
                "MULTIPLY_NUMERIC", "PERCENTILE", "ABSOLUTE",
                "LOG", "SQUARE", "SQRT",
                "YEAR", "MONTH", "DAY", "WEEKDAY", "HOUR", "MINUTE",
                "IS_WEEKEND",
            ],
        }
    )
    datetime_index: Optional[str] = None
    ignore_columns: List[str] = field(default_factory=list)
    add_statistical_features: bool = True
    add_frequency_encoding: bool = True
    add_cyclical_encoding: bool = True
    add_interaction_features: bool = False
    add_polynomial_features: bool = False
    polynomial_degree: int = 2
    add_text_features: bool = True
    tfidf_max_features: int = 50


@dataclass
class SelectionConfig:
    enabled: bool = True
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    mutual_info_top_k: int = 300
    drop_leaky_id_cols: bool = True
    remove_duplicates: bool = True


@dataclass
class OutputConfig:
    base_dir: str = "output"
    formats: List[str] = field(default_factory=lambda: ["parquet", "csv"])
    save_cleaned: bool = True
    save_feature_matrix: bool = True
    save_feature_definitions: bool = True
    save_metadata: bool = True
    save_profiling_report: bool = True
    save_selected_features: bool = True
    compress_parquet: bool = True


@dataclass
class PerformanceConfig:
    n_jobs: int = -1
    use_dask: bool = False
    dask_npartitions: int = 8
    cache_entityset: bool = True


# ──────────────────────────────────────────────────────────────
# Domain / eCommerce Configs
# ──────────────────────────────────────────────────────────────

@dataclass
class EcommerceConfig:

    classifier_rules: Dict[str, Any] = field(default_factory=dict)

    keywords: Dict[str, Any] = field(default_factory=dict)

    manifest: Dict[str, Any] = field(default_factory=dict)

    metrics: Dict[str, Any] = field(default_factory=dict)

    questions: Dict[str, Any] = field(default_factory=dict)

    roles: Dict[str, Any] = field(default_factory=dict)

    rules: Dict[str, Any] = field(default_factory=dict)

    schema: Dict[str, Any] = field(default_factory=dict)

    validations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    name: str = "AutoFE Pipeline"
    version: str = "1.0.0"
    random_seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"

    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    type_detection: TypeDetectionConfig = field(default_factory=TypeDetectionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    engineering: EngineeringConfig = field(default_factory=EngineeringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    ecommerce: EcommerceConfig = field(default_factory=EcommerceConfig)


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────

def _from_dict(cls, d: dict):
    """Recursively build a dataclass from a dictionary."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return d
    kwargs = {}
    hints = {f.name: f for f in dataclasses.fields(cls)}
    for f in dataclasses.fields(cls):
        if f.name in d:
            val = d[f.name]
            sub_cls = f.type if isinstance(f.type, type) else None
            if sub_cls and hasattr(sub_cls, "__dataclass_fields__"):
                kwargs[f.name] = _from_dict(sub_cls, val or {})
            else:
                kwargs[f.name] = val
        else:
            kwargs[f.name] = f.default if f.default is not dataclasses.MISSING else (
                f.default_factory() if f.default_factory is not dataclasses.MISSING else None
            )
    return cls(**kwargs)


def load_config(path: str | Path) -> PipelineConfig:
    """Load PipelineConfig from a YAML or JSON file (or return defaults)."""
    p = Path(path)
    if not p.exists():
        return PipelineConfig()

    raw = p.read_text(encoding="utf-8")
    data: dict = yaml.safe_load(raw) if p.suffix in {".yaml", ".yml"} else json.loads(raw)

    cfg = PipelineConfig()
    for key, val in data.items():
        if key == "pipeline":
            for k, v in val.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        elif key == "ingestion":
            cfg.ingestion = _from_dict(IngestionConfig, val or {})
        elif key == "type_detection":
            cfg.type_detection = _from_dict(TypeDetectionConfig, val or {})
        elif key == "cleaning":
            cfg.cleaning = _from_dict(CleaningConfig, val or {})
        elif key == "engineering":
            cfg.engineering = _from_dict(EngineeringConfig, val or {})
        elif key == "selection":
            cfg.selection = _from_dict(SelectionConfig, val or {})
        elif key == "output":
            cfg.output = _from_dict(OutputConfig, val or {})
        elif key == "performance":
            cfg.performance = _from_dict(PerformanceConfig, val or {})
        elif key == "ecommerce":
            cfg.ecommerce = _from_dict(EcommerceConfig, val or {})

    return cfg


def load_domain_configs(domain: str = "ecommerce", path: str | None = None):
    """
    Load domain-specific config files (e.g. schema.yml, keywords.yml).
    Automatically looks in ./configs/{domain}/
    """
    d = Path(path) if path else Path("configs") / domain
    cfg = EcommerceConfig()
    for filename in ["schema.yml", "keywords.yml", "manifest.yml", "metrics.yml",
                     "questions.yml", "roles.yml", "rules.yml",
                     "validations.yml", "classifier_rules.yml"]:
        f = d / filename
        if f.exists():
            try:
                raw = f.read_text(encoding="utf-8")
                parsed = yaml.safe_load(raw) or {}
                key = filename.replace(".yml", "")
                setattr(cfg, key, parsed)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    
    return cfg
