"""
src/ingestion/loader.py
────────────────────────
Universal data loader supporting CSV, TSV, JSON, JSONL,
Parquet, Excel, Feather.  Handles encoding detection,
delimiter sniffing, nested JSON flattening, and chunked
loading for large files.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator, List

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

import pandas as pd

from src.utils.config import PipelineConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED = {".csv", ".tsv", ".txt", ".json", ".jsonl",
              ".parquet", ".xlsx", ".xls", ".feather", ".ftr"}


# ──────────────────────────────────────────────────────────────
class DataLoader:
    """
    Load any supported file (or directory of files) into a
    single pandas DataFrame.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._ic = cfg.ingestion

    # ── public ────────────────────────────────────────────────

    def load(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.is_dir():
            return self._load_directory(p)
        return self._load_file(p)

    # ── directory ─────────────────────────────────────────────

    def _load_directory(self, directory: Path) -> pd.DataFrame:
        files = sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in _SUPPORTED
        )
        if not files:
            raise ValueError(f"No supported files found in {directory}")
        logger.info(f"Loading {len(files)} file(s) from directory {directory}")
        frames = [self._load_file(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Combined shape: {df.shape}")
        return df

    # ── single file ───────────────────────────────────────────

    def _load_file(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        logger.info(f"Loading {path} ({suffix})")
        loaders = {
            ".csv": self._load_csv,
            ".tsv": self._load_tsv,
            ".txt": self._load_csv,
            ".json": self._load_json,
            ".jsonl": self._load_jsonl,
            ".parquet": self._load_parquet,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".feather": self._load_feather,
            ".ftr": self._load_feather,
        }
        loader = loaders.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported file format: {suffix}")
        df = loader(path)
        df = self._post_process(df)
        logger.info(f"Loaded shape: {df.shape}")
        return df

    # ── format-specific ───────────────────────────────────────

    def _detect_encoding(self, path: Path) -> str:
        if not CHARDET_AVAILABLE:
            logger.debug("chardet not available; falling back to utf-8")
            return "utf-8"

        raw = path.read_bytes()[:100_000]
        result = chardet.detect(raw)
        enc = result.get("encoding") or "utf-8"
        logger.debug(f"Detected encoding: {enc} (confidence {result.get('confidence', 0):.2f})")
        return enc

    def _detect_delimiter(self, path: Path, encoding: str) -> str:
        sample = path.read_text(encoding=encoding, errors="replace")[:4096]
        counts = {sep: sample.count(sep) for sep in [",", "\t", ";", "|"]}
        best = max(counts, key=counts.get)
        logger.debug(f"Detected delimiter: {repr(best)}")
        return best

    def _load_csv(self, path: Path) -> pd.DataFrame:
        enc = self._detect_encoding(path)
        sep = self._ic.csv_sep or self._detect_delimiter(path, enc)

        file_size = path.stat().st_size
        if file_size > 200 * 1024 * 1024:   # > 200 MB → chunked
            logger.info(f"Large file ({file_size/1e6:.0f} MB) – loading in chunks")
            return self._chunked_csv(path, sep, enc)

        return pd.read_csv(
            path,
            sep=sep,
            encoding=enc,
            low_memory=self._ic.low_memory,
            on_bad_lines="warn",
        )

    def _chunked_csv(self, path: Path, sep: str, enc: str) -> pd.DataFrame:
        chunks: List[pd.DataFrame] = []
        for chunk in pd.read_csv(
            path,
            sep=sep,
            encoding=enc,
            chunksize=self._ic.chunk_size,
            on_bad_lines="warn",
            low_memory=True,
        ):
            chunks.append(chunk)
            if sum(len(c) for c in chunks) >= self._ic.max_sample_rows:
                logger.warning(
                    f"Capped at {self._ic.max_sample_rows} rows for memory safety"
                )
                break
        return pd.concat(chunks, ignore_index=True)

    def _load_tsv(self, path: Path) -> pd.DataFrame:
        enc = self._detect_encoding(path)
        return pd.read_csv(path, sep="\t", encoding=enc, on_bad_lines="warn")

    def _load_json(self, path: Path) -> pd.DataFrame:
        orient = self._ic.json_orient
        try:
            if orient:
                df = pd.read_json(path, orient=orient)
            else:
                df = pd.read_json(path)
        except ValueError:
            # Fallback: read as raw Python dict and normalize
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.json_normalize(data, max_level=3)
            elif isinstance(data, dict):
                df = pd.json_normalize([data], max_level=3)
            else:
                raise ValueError(f"Unsupported JSON structure in {path}")
        return self._flatten_nested(df)

    def _load_jsonl(self, path: Path) -> pd.DataFrame:
        records = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSONL line")
        df = pd.json_normalize(records, max_level=3)
        return self._flatten_nested(df)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _load_excel(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path, sheet_name=self._ic.excel_sheet)

    def _load_feather(self, path: Path) -> pd.DataFrame:
        return pd.read_feather(path)

    # ── helpers ───────────────────────────────────────────────

    def _flatten_nested(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten any remaining object/dict columns one level deep."""
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(10)
            if sample.apply(lambda x: isinstance(x, dict)).all():
                try:
                    expanded = pd.json_normalize(df[col].tolist())
                    expanded.columns = [f"{col}.{c}" for c in expanded.columns]
                    df = df.drop(columns=[col]).join(expanded)
                    logger.debug(f"Flattened nested column: {col}")
                except Exception:
                    pass
        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sanitise column names and basic types."""
        # Clean column names
        df.columns = (
            pd.Index(df.columns)
            .str.strip()
            .str.replace(r"[\s\.\-\/\\]", "_", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
            .str.lower()
        )
        # Remove fully empty rows/columns
        df.dropna(how="all", axis=0, inplace=True)
        df.dropna(how="all", axis=1, inplace=True)

        # Trim to max_sample_rows
        if len(df) > self._ic.max_sample_rows:
            df = df.sample(n=self._ic.max_sample_rows, random_state=42).reset_index(drop=True)
            logger.warning(f"Sampled to {self._ic.max_sample_rows} rows")

        return df