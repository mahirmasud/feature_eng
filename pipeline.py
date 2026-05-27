"""
AutoFE – Automated Feature Engineering Pipeline
================================================
Main orchestrator. Run end-to-end or call individual stages.

Usage
-----
  python pipeline.py --input data/sample/titanic.csv --target Survived
  python pipeline.py --input data/ --config config/pipeline_config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.utils.config import load_config, PipelineConfig
from src.utils.logger import get_logger
from src.ingestion.loader import DataLoader
from src.profiling.profiler import DataProfiler
from src.cleaning.cleaner import DataCleaner
from src.engineering.feature_engineer import FeatureEngineer
from src.selection.selector import FeatureSelector
from src.output.exporter import Exporter

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
class AutoFEPipeline:
    """
    End-to-end automated feature engineering pipeline.

    Stages
    ------
    1. Load       – universal ingestion (CSV/JSON/Parquet/Excel/…)
    2. Profile    – schema inference, data quality report
    3. Clean      – imputation, dedup, outlier handling
    4. Engineer   – Featuretools DFS + advanced feature generation
    5. Select     – variance, correlation, MI-based pruning
    6. Export     – save all artefacts
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.loader = DataLoader(config)
        self.profiler = DataProfiler(config)
        self.cleaner = DataCleaner(config)
        self.engineer = FeatureEngineer(config)
        self.selector = FeatureSelector(config)
        self.exporter = Exporter(config)

        # State shared across stages
        self.raw_df: pd.DataFrame | None = None
        self.clean_df: pd.DataFrame | None = None
        self.feature_matrix: pd.DataFrame | None = None
        self.selected_features: pd.DataFrame | None = None
        self.metadata: dict = {}

    # ── public API ────────────────────────────────────────────

    def run(self, input_path: str, target_col: str | None = None) -> pd.DataFrame:
        """Execute the full pipeline and return the final feature matrix."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("AutoFE Pipeline starting")
        logger.info(f"  input  : {input_path}")
        logger.info(f"  target : {target_col or 'auto-detect'}")
        logger.info("=" * 60)

        # 1 – Load
        self.raw_df = self._stage("LOAD", self.loader.load, input_path)

        # 2 – Profile
        profile = self._stage("PROFILE", self.profiler.profile, self.raw_df, target_col)
        self.metadata["profile"] = profile

        # Resolve target column
        if target_col is None:
            target_col = profile.get("suggested_target")
            if target_col:
                logger.info(f"Auto-selected target column: '{target_col}'")

        # 3 – Clean
        self.clean_df = self._stage(
            "CLEAN", self.cleaner.clean, self.raw_df, profile
        )

        # 4 – Engineer
        self.feature_matrix, feat_defs = self._stage(
            "ENGINEER", self.engineer.engineer, self.clean_df, profile, target_col
        )
        self.metadata["feature_definitions"] = feat_defs

        # 5 – Select
        if self.cfg.selection.enabled:
            self.selected_features, sel_meta = self._stage(
                "SELECT",
                self.selector.select,
                self.feature_matrix,
                target_col,
            )
            self.metadata["selection"] = sel_meta
        else:
            self.selected_features = self.feature_matrix

        # 6 – Export
        self._stage(
            "EXPORT",
            self.exporter.export,
            raw_df=self.raw_df,
            clean_df=self.clean_df,
            feature_matrix=self.feature_matrix,
            selected_df=self.selected_features,
            metadata=self.metadata,
        )

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        logger.info(
            f"Final feature matrix: {self.selected_features.shape[0]} rows × "
            f"{self.selected_features.shape[1]} features"
        )
        logger.info("=" * 60)
        return self.selected_features

    # ── internals ─────────────────────────────────────────────

    def _stage(self, name: str, fn, *args, **kwargs):
        logger.info(f"[Stage: {name}] starting …")
        t = time.time()
        try:
            result = fn(*args, **kwargs)
            logger.info(f"[Stage: {name}] done ({time.time()-t:.1f}s)")
            return result
        except Exception as exc:
            logger.error(f"[Stage: {name}] FAILED – {exc}", exc_info=True)
            raise


# ──────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AutoFE – Automated Feature Engineering Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input file or directory")
    p.add_argument("--target", default=None, help="Target column name")
    p.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to YAML configuration file",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Override output directory from config",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Override DFS max depth",
    )
    p.add_argument(
        "--no-select",
        action="store_true",
        help="Skip feature selection stage",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        default=None,
        choices=["csv", "parquet", "feather"],
        help="Override output formats",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.output:
        cfg.output.base_dir = args.output
    if args.max_depth is not None:
        cfg.engineering.max_depth = args.max_depth
    if args.no_select:
        cfg.selection.enabled = False
    if args.formats:
        cfg.output.formats = args.formats

    pipeline = AutoFEPipeline(cfg)
    pipeline.run(args.input, target_col=args.target)


if __name__ == "__main__":
    main()
