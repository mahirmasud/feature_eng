"""
tests/test_pipeline.py
───────────────────────
Unit tests for core pipeline modules.
Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PipelineConfig, load_config
from src.profiling.profiler import DataProfiler
from src.cleaning.cleaner import DataCleaner
from src.selection.selector import FeatureSelector


# ── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return PipelineConfig()


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age":      np.random.randint(18, 80, n).astype(float),
        "income":   np.random.exponential(50_000, n),
        "gender":   np.random.choice(["M", "F", "Other"], n),
        "city":     np.random.choice(["Dhaka", "Chittagong", "Sylhet"], n),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="2D"),
        "notes":    ["user note " + str(i) for i in range(n)],
        "const_col": "same_value",
        "target":   np.random.randint(0, 2, n),
    })


# ── config ────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.engineering.max_depth == 2
        assert cfg.cleaning.drop_missing_threshold == 0.70

    def test_load_missing_file_returns_defaults(self):
        cfg = load_config("nonexistent.yaml")
        assert isinstance(cfg, PipelineConfig)


# ── profiler ──────────────────────────────────────────────────

class TestProfiler:
    def test_column_types(self, cfg, sample_df):
        profiler = DataProfiler(cfg)
        profile = profiler.profile(sample_df)
        ct = profile["column_types"]
        assert ct["age"] == "numeric"
        assert ct["gender"] == "categorical"
        assert ct["const_col"] == "constant"

    def test_quality_metrics(self, cfg, sample_df):
        profiler = DataProfiler(cfg)
        profile = profiler.profile(sample_df)
        q = profile["quality"]
        assert q["total_rows"] == 200
        assert "column_stats" in q

    def test_missing_pct(self, cfg, sample_df):
        sample_df.loc[:99, "age"] = np.nan
        profiler = DataProfiler(cfg)
        profile = profiler.profile(sample_df)
        miss = profile["quality"]["column_stats"]["age"]["missing_pct"]
        assert miss == pytest.approx(50.0, abs=1.0)


# ── cleaner ───────────────────────────────────────────────────

class TestCleaner:
    def test_removes_constant_cols(self, cfg, sample_df):
        profiler = DataProfiler(cfg)
        profile = profiler.profile(sample_df)
        cleaner = DataCleaner(cfg)
        cleaned = cleaner.clean(sample_df, profile)
        assert "const_col" not in cleaned.columns

    def test_imputes_numerics(self, cfg, sample_df):
        sample_df.loc[:10, "age"] = np.nan
        profiler = DataProfiler(cfg)
        profile = profiler.profile(sample_df)
        cleaner = DataCleaner(cfg)
        cleaned = cleaner.clean(sample_df, profile)
        assert cleaned["age"].isna().sum() == 0

    def test_no_duplicates(self, cfg, sample_df):
        dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        profiler = DataProfiler(cfg)
        profile = profiler.profile(dup)
        cleaner = DataCleaner(cfg)
        cleaned = cleaner.clean(dup, profile)
        assert cleaned.duplicated().sum() == 0

    def test_inf_handling(self, cfg, sample_df):
        sample_df.loc[0, "income"] = float("inf")
        profiler = DataProfiler(cfg)
        profile = profiler.profile(sample_df)
        cleaner = DataCleaner(cfg)
        cleaned = cleaner.clean(sample_df, profile)
        assert not np.isinf(cleaned["income"]).any()


# ── selector ──────────────────────────────────────────────────

class TestSelector:
    def _make_feat_df(self):
        np.random.seed(0)
        n = 100
        a = np.random.randn(n)
        return pd.DataFrame({
            "feat_a": a,
            "feat_b": a * 1.001,   # near-duplicate
            "feat_c": np.random.randn(n),
            "const":  np.ones(n),
            "target": np.random.randint(0, 2, n),
        })

    def test_removes_high_corr(self, cfg):
        df = self._make_feat_df()
        cfg.selection.variance_threshold = 0.0
        cfg.selection.correlation_threshold = 0.95
        cfg.selection.mutual_info_top_k = 999
        sel = FeatureSelector(cfg)
        result, meta = sel.select(df, "target")
        # feat_b should be dropped (nearly identical to feat_a)
        assert "feat_b" not in result.columns or "feat_a" not in result.columns

    def test_target_preserved(self, cfg):
        df = self._make_feat_df()
        sel = FeatureSelector(cfg)
        result, _ = sel.select(df, "target")
        assert "target" in result.columns


# ── integration smoke test ────────────────────────────────────

class TestIntegration:
    def test_smoke(self, cfg, sample_df, tmp_path):
        """Run a trimmed pipeline end-to-end without DFS."""
        cfg.output.base_dir = str(tmp_path / "output")
        cfg.engineering.max_depth = 1
        cfg.engineering.add_polynomial_features = False
        cfg.performance.n_jobs = 1

        from src.profiling.profiler import DataProfiler
        from src.cleaning.cleaner import DataCleaner
        from src.engineering.feature_engineer import FeatureEngineer
        from src.selection.selector import FeatureSelector

        profiler = DataProfiler(cfg)
        cleaner = DataCleaner(cfg)
        engineer = FeatureEngineer(cfg)
        selector = FeatureSelector(cfg)

        profile = profiler.profile(sample_df, "target")
        cleaned = cleaner.clean(sample_df, profile)
        feat_matrix, _ = engineer.engineer(cleaned, profile, "target")
        selected, meta = selector.select(feat_matrix, "target")

        assert selected.shape[0] == cleaned.shape[0]
        assert "target" in selected.columns
        assert selected.shape[1] >= 2
