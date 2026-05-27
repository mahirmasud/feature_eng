"""
src/output/exporter.py
───────────────────────
Save all pipeline artefacts to the output directory:
  - Cleaned dataset
  - Full feature matrix
  - Selected feature matrix
  - Feature definitions (JSON)
  - Pipeline metadata (JSON)
  - Profiling report (JSON)
  - Selected feature list (TXT)

Supports CSV, Parquet, and Feather formats.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
class Exporter:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.oc = cfg.output
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(self.oc.base_dir) / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.out_dir}")

    # ── public ────────────────────────────────────────────────

    def export(
        self,
        raw_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        feature_matrix: pd.DataFrame,
        selected_df: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> None:
        if self.oc.save_cleaned:
            self._save_df(clean_df, "cleaned_data")

        if self.oc.save_feature_matrix:
            self._save_df(feature_matrix, "feature_matrix")

        if self.oc.save_selected_features:
            self._save_df(selected_df, "selected_features")

        if self.oc.save_feature_definitions:
            self._save_json(
                metadata.get("feature_definitions", []),
                "feature_definitions.json",
            )

        if self.oc.save_metadata:
            self._save_metadata(metadata, raw_df, selected_df)

        if self.oc.save_profiling_report:
            self._save_json(
                metadata.get("profile", {}),
                "profiling_report.json",
            )

        self._save_feature_list(selected_df)
        logger.info(f"All artefacts saved to {self.out_dir}")

    # ── helpers ───────────────────────────────────────────────

    def _save_df(self, df: pd.DataFrame, name: str) -> None:
        """Save a DataFrame in all configured formats."""
        for fmt in self.oc.formats:
            path = self.out_dir / f"{name}.{fmt}"
            try:
                if fmt == "csv":
                    df.to_csv(path, index=False)
                elif fmt == "parquet":
                    compression = "snappy" if self.oc.compress_parquet else None
                    # Parquet can't store certain dtypes
                    df_clean = self._parquet_safe(df)
                    df_clean.to_parquet(path, index=False, compression=compression)
                elif fmt == "feather":
                    df_clean = self._parquet_safe(df)
                    df_clean.to_feather(path)
                logger.info(f"Saved {name} → {path} ({df.shape})")
            except Exception as e:
                logger.warning(f"Could not save {name} as {fmt}: {e}")

    def _parquet_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to parquet-compatible dtypes."""
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Ensure UTC-naive
                df[col] = df[col].dt.tz_localize(None) if hasattr(df[col].dt, "tz_localize") else df[col]
            elif df[col].dtype == "object":
                df[col] = df[col].astype(str)
        return df

    def _save_json(self, data: Any, filename: str) -> None:
        path = self.out_dir / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {filename}")
        except Exception as e:
            logger.warning(f"Could not save {filename}: {e}")

    def _save_metadata(
        self,
        metadata: Dict[str, Any],
        raw_df: pd.DataFrame,
        selected_df: pd.DataFrame,
    ) -> None:
        summary = {
            "run_id": self.run_id,
            "pipeline_version": self.cfg.version,
            "raw_shape": list(raw_df.shape),
            "selected_shape": list(selected_df.shape),
            "n_features_generated": metadata.get("profile", {}).get("shape", [0, 0])[1],
            "n_features_selected": selected_df.shape[1],
            "selection_meta": metadata.get("selection", {}),
            "config": {
                "max_depth": self.cfg.engineering.max_depth,
                "max_features": self.cfg.engineering.max_features,
                "outlier_method": self.cfg.cleaning.outlier_method,
            },
        }
        self._save_json(summary, "metadata.json")

    def _save_feature_list(self, df: pd.DataFrame) -> None:
        path = self.out_dir / "selected_feature_list.txt"
        with open(path, "w") as f:
            for col in df.columns:
                f.write(col + "\n")
        logger.info(f"Saved selected_feature_list.txt ({len(df.columns)} features)")
