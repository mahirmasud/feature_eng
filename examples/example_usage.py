"""
examples/example_usage.py
──────────────────────────
Demonstrates how to use AutoFE both programmatically
and as a CLI tool.

Run from the project root:
  python examples/example_usage.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 1. Generate sample data ───────────────────────────────────
def generate_data():
    from data.sample.generate_sample import make_customer_churn
    df = make_customer_churn(500)
    p = Path("data/sample/demo.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Demo data saved → {p}  shape={df.shape}")
    return p


# ── 2. Programmatic usage ─────────────────────────────────────
def run_programmatic(csv_path: Path):
    from src.utils.config import PipelineConfig
    from pipeline import AutoFEPipeline

    cfg = PipelineConfig()
    cfg.engineering.max_depth = 1          # keep it fast for demo
    cfg.engineering.max_features = 100
    cfg.output.base_dir = "output/demo"
    cfg.output.formats = ["csv"]           # only CSV for demo
    cfg.performance.n_jobs = 1

    pipeline = AutoFEPipeline(cfg)
    result = pipeline.run(str(csv_path), target_col="churn")

    print("\n" + "=" * 50)
    print(f"Result shape : {result.shape}")
    print(f"Sample columns : {list(result.columns[:8])}")
    print("=" * 50)
    return result


# ── 3. Config override example ────────────────────────────────
def run_with_config(csv_path: Path):
    from src.utils.config import load_config
    from pipeline import AutoFEPipeline

    # Load from YAML and override specific fields
    cfg = load_config("config/pipeline_config.yaml")
    cfg.engineering.max_depth = 1
    cfg.selection.enabled = False          # skip selection
    cfg.output.base_dir = "output/demo_no_selection"

    pipeline = AutoFEPipeline(cfg)
    result = pipeline.run(str(csv_path), target_col="churn")
    print(f"\nNo-selection run → {result.shape[1]} features")


if __name__ == "__main__":
    csv_path = generate_data()
    run_programmatic(csv_path)
    # Uncomment to test config-driven run:
    # run_with_config(csv_path)
