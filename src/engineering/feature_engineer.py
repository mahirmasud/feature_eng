"""
src/engineering/feature_engineer.py
─────────────────────────────────────
Feature generation in three layers:

Layer 1 – Featuretools Deep Feature Synthesis (DFS)
    Builds an EntitySet, infers relationships, and runs DFS to
    create aggregation and transform features automatically.

Layer 2 – Advanced manual features
    Frequency encoding, cyclical encoding, statistical features,
    TF-IDF text features, polynomial features.

Layer 3 – Datetime decomposition
    Year / month / day / weekday / hour / is_weekend derived cols.

All generated feature names and origins are recorded in a
feature_definitions dict for explainability.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import featuretools as ft
    # Verify it's actually usable (woodwork compatibility check)
    _es_test = ft.EntitySet(id="__test__")
    del _es_test
    FT_AVAILABLE = True
    logger.info(f"Featuretools {ft.__version__} loaded successfully")
except Exception as _ft_err:
    FT_AVAILABLE = False
    logger.warning(
        f"Featuretools unavailable ({type(_ft_err).__name__}: {_ft_err}) "
        "– DFS disabled; using manual features only"
    )

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
class FeatureEngineer:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.ec = cfg.engineering
        self._feat_defs: List[Dict] = []

    # ── public ────────────────────────────────────────────────

    def engineer(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        logger.info(f"Feature engineering: input {df.shape}")
        self._feat_defs = []

        col_types = profile.get("column_types", {})

        # Keep target aside
        target_series = None
        if target_col and target_col in df.columns:
            target_series = df[target_col].copy()
            df = df.drop(columns=[target_col])
            col_types.pop(target_col, None)

        # ── Layer 1: DFS ──────────────────────────────────────
        if FT_AVAILABLE:
            df = self._run_dfs(df, profile, target_col)
        else:
            logger.info("Skipping DFS (featuretools not available)")

        # ── Layer 2: Advanced manual features ─────────────────
        df = self._datetime_features(df, profile.get("datetime_columns", []))
        df = self._frequency_encoding(df, profile)
        df = self._cyclical_encoding(df)
        if self.ec.add_statistical_features:
            df = self._statistical_features(df, profile)
        if self.ec.add_text_features:
            df = self._text_features(df, profile.get("text_columns", []))
        if self.ec.add_polynomial_features:
            df = self._polynomial_features(df, profile)

        # Re-attach target
        if target_series is not None:
            df[target_col] = target_series.values

        # Cap features
        if df.shape[1] > self.ec.max_features + (1 if target_col else 0):
            feat_cols = [c for c in df.columns if c != target_col]
            keep = feat_cols[: self.ec.max_features]
            df = df[keep + ([target_col] if target_col else [])]
            logger.info(f"Capped to {self.ec.max_features} features")

        logger.info(f"Feature engineering: output {df.shape}")
        return df, self._feat_defs

    # ── Layer 1: Featuretools DFS ─────────────────────────────

    def _run_dfs(
        self, df: pd.DataFrame, profile: Dict, target_col: Optional[str]
    ) -> pd.DataFrame:
        try:
            logger.info("Building Featuretools EntitySet …")
            es = ft.EntitySet(id="autofe_data")

            # Detect or create index
            id_cols = profile.get("id_columns", [])
            idx_col = id_cols[0] if id_cols else None

            if idx_col and idx_col in df.columns:
                df_ft = df.copy()
            else:
                df_ft = df.copy()
                idx_col = "_row_id"
                df_ft[idx_col] = range(len(df_ft))

            # Ensure index is unique
            if df_ft[idx_col].duplicated().any():
                idx_col = "_row_id"
                df_ft[idx_col] = range(len(df_ft))

            # Detect datetime index
            dt_col = self.ec.datetime_index
            if dt_col is None:
                dt_cols = profile.get("datetime_columns", [])
                dt_col = dt_cols[0] if dt_cols else None

            logical_types = self._build_logical_types(df_ft, profile)

            add_kwargs: Dict[str, Any] = {
                "dataframe": df_ft,
                "dataframe_name": "main",
                "index": idx_col,
                "logical_types": logical_types,
            }
            if dt_col and dt_col in df_ft.columns:
                add_kwargs["time_index"] = dt_col

            es = es.add_dataframe(**add_kwargs)

            # Build primitive lists
            agg_prims = self._resolve_primitives(
                self.ec.include_primitives.get("aggregation", []), "aggregation"
            )
            trans_prims = self._resolve_primitives(
                self.ec.include_primitives.get("transform", []), "transform"
            )

            logger.info(
                f"Running DFS (depth={self.ec.max_depth}, "
                f"{len(agg_prims)} agg + {len(trans_prims)} transform prims) …"
            )

            ignore_cols = self.ec.ignore_columns or []
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="main",
                agg_primitives=agg_prims,
                trans_primitives=trans_prims,
                max_depth=self.ec.max_depth,
                ignore_columns={"main": ignore_cols},
                verbose=False,
                n_jobs=self.cfg.performance.n_jobs,
            )

            # Record feature origins
            for fd in feature_defs:
                self._feat_defs.append({
                    "name": fd.get_name(),
                    "primitive": str(fd.primitive),
                    "origin": "featuretools_dfs",
                    "base_features": [b.get_name() for b in fd.base_features],
                })

            # Merge DFS output back; drop the synthetic index if we added it
            feature_matrix = feature_matrix.reset_index()
            if "_row_id" in feature_matrix.columns:
                feature_matrix = feature_matrix.drop(columns=["_row_id"])

            logger.info(
                f"DFS generated {feature_matrix.shape[1]} columns "
                f"from {df.shape[1]} input columns"
            )
            return feature_matrix

        except Exception as exc:
            logger.warning(f"DFS failed ({exc}); falling back to manual features only")
            return df

    def _build_logical_types(
        self, df: pd.DataFrame, profile: Dict
    ) -> Dict[str, Any]:
        """Map column types to Featuretools logical types."""
        if not FT_AVAILABLE:
            return {}

        col_types = profile.get("column_types", {})
        mapping = {}
        for col in df.columns:
            ct = col_types.get(col, "unknown")
            if ct == "categorical":
                mapping[col] = "Categorical"
            elif ct == "text":
                mapping[col] = "NaturalLanguage"
            elif ct == "boolean":
                mapping[col] = "BooleanNullable"
            elif ct == "datetime":
                mapping[col] = "Datetime"
            # numeric / id / etc.: let Featuretools infer
        return mapping

    def _resolve_primitives(self, names: List[str], kind: str) -> List:
        """Convert string names to Featuretools primitive objects."""
        if not FT_AVAILABLE:
            return []
        prims = []
        all_prims = ft.primitives.get_aggregation_primitives() if kind == "aggregation" else ft.primitives.get_transform_primitives()
        available = {p.__name__.upper(): p for p in all_prims}
        for name in names:
            cls = available.get(name.upper().replace(" ", "_"))
            if cls:
                try:
                    prims.append(cls())
                except Exception:
                    pass
            else:
                logger.debug(f"Primitive '{name}' not found – skipping")
        return prims

    # ── Layer 2: Manual features ──────────────────────────────

    def _datetime_features(
        self, df: pd.DataFrame, dt_cols: List[str]
    ) -> pd.DataFrame:
        for col in dt_cols:
            if col not in df.columns:
                continue
            s = df[col]
            if not pd.api.types.is_datetime64_any_dtype(s):
                try:
                    s = pd.to_datetime(s, format="mixed", errors="coerce")
                except Exception:
                    try:
                        s = pd.to_datetime(s, errors="coerce")
                    except Exception:
                        continue

            prefix = f"{col}_"
            df[f"{prefix}year"] = s.dt.year
            df[f"{prefix}month"] = s.dt.month
            df[f"{prefix}day"] = s.dt.day
            df[f"{prefix}weekday"] = s.dt.weekday
            df[f"{prefix}hour"] = s.dt.hour if s.dt.hour.nunique() > 1 else None
            df[f"{prefix}is_weekend"] = s.dt.weekday.isin([5, 6]).astype(int)
            # Drop null hour column if it appeared
            if f"{prefix}hour" in df.columns and df[f"{prefix}hour"].isna().all():
                df.drop(columns=[f"{prefix}hour"], inplace=True)

            self._feat_defs.append({
                "name": f"{prefix}*", "primitive": "datetime_decomposition",
                "origin": "manual", "base_features": [col],
            })
            logger.debug(f"Datetime decomposition: {col}")
        return df

    def _frequency_encoding(
        self, df: pd.DataFrame, profile: Dict
    ) -> pd.DataFrame:
        if not self.ec.add_frequency_encoding:
            return df
        cat_cols = profile.get("categorical_columns", [])
        for col in cat_cols:
            if col not in df.columns:
                continue
            freq = df[col].value_counts(normalize=True)
            new_col = f"{col}_freq"
            df[new_col] = df[col].map(freq).astype(float)
            self._feat_defs.append({
                "name": new_col, "primitive": "frequency_encoding",
                "origin": "manual", "base_features": [col],
            })
        return df

    def _cyclical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.ec.add_cyclical_encoding:
            return df
        cyclical_map = {
            "month": 12, "day": 31, "weekday": 7,
            "hour": 24, "minute": 60, "second": 60,
        }
        new_cols: Dict[str, pd.Series] = {}
        for col in df.columns:
            base = col.split("_")[-1]
            if base in cyclical_map:
                period = cyclical_map[base]
                s = pd.to_numeric(df[col], errors="coerce")
                new_cols[f"{col}_sin"] = np.sin(2 * np.pi * s / period)
                new_cols[f"{col}_cos"] = np.cos(2 * np.pi * s / period)
                self._feat_defs.append({
                    "name": f"{col}_sin/cos", "primitive": "cyclical_encoding",
                    "origin": "manual", "base_features": [col],
                })
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def _statistical_features(
        self, df: pd.DataFrame, profile: Dict
    ) -> pd.DataFrame:
        num_cols = [
            c for c in profile.get("numeric_columns", [])
            if c in df.columns
        ]
        if len(num_cols) < 2:
            return df

        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
        df["__stat_row_mean"] = num_df.mean(axis=1)
        df["__stat_row_std"] = num_df.std(axis=1)
        df["__stat_row_min"] = num_df.min(axis=1)
        df["__stat_row_max"] = num_df.max(axis=1)
        df["__stat_row_range"] = df["__stat_row_max"] - df["__stat_row_min"]
        self._feat_defs.append({
            "name": "__stat_row_*", "primitive": "row_statistics",
            "origin": "manual", "base_features": num_cols,
        })
        return df

    def _text_features(
        self, df: pd.DataFrame, text_cols: List[str]
    ) -> pd.DataFrame:
        if not TFIDF_AVAILABLE:
            return df
        for col in text_cols:
            if col not in df.columns:
                continue
            texts = df[col].fillna("").astype(str)
            if texts.str.strip().eq("").all():
                continue
            try:
                tfidf = TfidfVectorizer(
                    max_features=self.ec.tfidf_max_features,
                    strip_accents="unicode",
                    min_df=2,
                    sublinear_tf=True,
                )
                mat = tfidf.fit_transform(texts)
                vocab = tfidf.get_feature_names_out()
                tdf = pd.DataFrame(
                    mat.toarray(),
                    columns=[f"{col}_tfidf_{w}" for w in vocab],
                    index=df.index,
                )
                df = pd.concat([df, tdf], axis=1)
                self._feat_defs.append({
                    "name": f"{col}_tfidf_*", "primitive": "tfidf",
                    "origin": "manual", "base_features": [col],
                })
                logger.debug(f"TF-IDF: {col} → {len(vocab)} features")
            except Exception as e:
                logger.warning(f"TF-IDF failed for {col}: {e}")
        return df

    def _polynomial_features(
        self, df: pd.DataFrame, profile: Dict
    ) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        num_cols = [
            c for c in profile.get("numeric_columns", [])
            if c in df.columns
        ][:10]  # limit input columns for memory safety
        if len(num_cols) < 2:
            return df
        try:
            poly = PolynomialFeatures(
                degree=self.ec.polynomial_degree,
                include_bias=False,
                interaction_only=True,
            )
            arr = poly.fit_transform(df[num_cols].fillna(0))
            names = poly.get_feature_names_out(num_cols)
            # Only add interaction columns (skip originals)
            new_names = [n for n in names if " " in n]
            new_idx = [list(names).index(n) for n in new_names]
            poly_df = pd.DataFrame(
                arr[:, new_idx], columns=new_names, index=df.index
            )
            df = pd.concat([df, poly_df], axis=1)
            self._feat_defs.append({
                "name": "poly_interactions", "primitive": "polynomial_features",
                "origin": "manual", "base_features": num_cols,
            })
        except Exception as e:
            logger.warning(f"Polynomial features failed: {e}")
        return df