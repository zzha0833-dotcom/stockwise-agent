from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from stockwise.schemas import DatasetProfile

CANONICAL_COLUMNS = [
    "date",
    "store_id",
    "item_id",
    "sales",
    "sell_price",
    "promo_flag",
    "event_name",
]
M5_ZENODO_RECORD = "https://zenodo.org/api/records/10203108"
M5_REQUIRED_FILES = {"calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"}


class DataValidationError(ValueError):
    pass


def _stable_seed(*parts: str, seed: int = 42) -> int:
    digest = hashlib.sha256(("|".join(parts) + f"|{seed}").encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def generate_synthetic_retail_data(
    *, items: int = 8, stores: int = 2, days: int = 196, seed: int = 42
) -> pl.DataFrame:
    """Create a deterministic retail dataset with seasonality, promotions and intermittent demand."""
    rng = np.random.default_rng(seed)
    start = date(2025, 1, 1)
    rows: list[dict[str, object]] = []
    for store_idx in range(stores):
        store_id = f"STORE_{store_idx + 1:02d}"
        store_factor = 0.85 + 0.15 * store_idx
        for item_idx in range(items):
            item_id = f"ITEM_{item_idx + 1:03d}"
            base = 5.0 + item_idx * 0.8
            intermittent = item_idx % 5 == 0
            base_price = 3.5 + item_idx * 0.65
            for day_idx in range(days):
                current = start + timedelta(days=day_idx)
                weekend = current.weekday() >= 5
                promo = day_idx % 31 in (0, 1, 2)
                event = "Promotion" if promo else ("Holiday" if day_idx % 89 == 0 else "")
                price = base_price * (0.88 if promo else 1.0) * (1 + 0.01 * math.sin(day_idx / 17))
                weekly = 1.25 if weekend else 0.92
                trend = 1 + day_idx * 0.0008
                demand = base * store_factor * weekly * trend * (1.6 if promo else 1.0)
                demand += rng.normal(0, max(0.8, demand * 0.12))
                if intermittent and rng.random() < 0.58:
                    demand = 0
                sales = max(0, int(round(demand)))
                rows.append(
                    {
                        "date": current,
                        "store_id": store_id,
                        "item_id": item_id,
                        "sales": float(sales),
                        "sell_price": round(float(price), 2),
                        "promo_flag": bool(promo),
                        "event_name": event,
                    }
                )
    return pl.DataFrame(rows).sort(["store_id", "item_id", "date"])


def normalize_retail_frame(frame: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    normalized = frame.select(CANONICAL_COLUMNS).with_columns(
        pl.col("date").cast(pl.String).str.to_date(strict=False),
        pl.col("store_id").cast(pl.String),
        pl.col("item_id").cast(pl.String),
        pl.col("sales").cast(pl.Float64, strict=False),
        pl.col("sell_price").cast(pl.Float64, strict=False),
        pl.col("promo_flag")
        .cast(pl.String)
        .str.to_lowercase()
        .is_in(["true", "1", "yes", "y"]),
        pl.col("event_name").cast(pl.String).fill_null(""),
    )
    return normalized.sort(["store_id", "item_id", "date"])


def load_retail_csv(path: Path) -> pl.DataFrame:
    return normalize_retail_frame(pl.read_csv(path, try_parse_dates=True, infer_schema_length=10_000))


def validate_and_profile(frame: pl.DataFrame) -> DatasetProfile:
    frame = normalize_retail_frame(frame)
    if frame.is_empty():
        raise DataValidationError("Dataset is empty")
    if frame["date"].null_count() > 0:
        raise DataValidationError("Some date values could not be parsed")
    missing_sales = frame["sales"].null_count()
    negative_sales = frame.filter(pl.col("sales") < 0).height
    duplicate_keys = frame.select(
        pl.struct(["date", "store_id", "item_id"]).is_duplicated().sum()
    ).item()
    if missing_sales:
        raise DataValidationError("Sales contains missing or non-numeric values")
    if negative_sales:
        raise DataValidationError("Sales cannot contain negative values")
    if duplicate_keys:
        raise DataValidationError("Duplicate date/store/item rows are not allowed")

    stats = frame.select(
        pl.col("sales").mean().alias("mean"),
        pl.col("sales").std().fill_null(0).alias("std"),
    ).row(0, named=True)
    threshold = float(stats["mean"] or 0) + 4 * float(stats["std"] or 0)
    anomaly_count = frame.filter(pl.col("sales") > threshold).height if threshold > 0 else 0
    warnings: list[str] = []
    zero_ratio = frame.select((pl.col("sales") == 0).mean()).item()
    if zero_ratio > 0.5:
        warnings.append("More than half of observations have zero demand")
    if anomaly_count:
        warnings.append(f"Detected {anomaly_count} high-demand outliers")
    if frame["sell_price"].null_count():
        warnings.append("Missing prices will be forward-filled per series")

    return DatasetProfile(
        rows=frame.height,
        stores=frame["store_id"].n_unique(),
        items=frame["item_id"].n_unique(),
        series=frame.select(pl.struct(["store_id", "item_id"]).n_unique()).item(),
        start_date=str(frame["date"].min()),
        end_date=str(frame["date"].max()),
        missing_sales=missing_sales,
        negative_sales=negative_sales,
        duplicate_keys=int(duplicate_keys),
        zero_demand_ratio=round(float(zero_ratio), 4),
        anomaly_count=anomaly_count,
        warnings=warnings,
    )


def subset_frame(
    frame: pl.DataFrame, *, item_limit: int, stores: list[str] | None = None
) -> pl.DataFrame:
    frame = normalize_retail_frame(frame)
    selected_stores = stores or sorted(frame["store_id"].unique().to_list())[:3]
    available_items = sorted(
        frame.filter(pl.col("store_id").is_in(selected_stores))["item_id"].unique().to_list()
    )
    selected_items = available_items[:item_limit]
    result = frame.filter(
        pl.col("store_id").is_in(selected_stores) & pl.col("item_id").is_in(selected_items)
    )
    if result.is_empty():
        raise DataValidationError("The requested store/item subset is empty")
    return result


def write_dataset(frame: pl.DataFrame, path: Path) -> DatasetProfile:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_retail_frame(frame)
    profile = validate_and_profile(normalized)
    if path.suffix.lower() == ".parquet":
        normalized.write_parquet(path, compression="zstd")
    else:
        normalized.write_csv(path)
    return profile


def load_dataset(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return normalize_retail_frame(pl.read_parquet(path))
    return load_retail_csv(path)


def download_m5(target_dir: Path, *, accept_source_terms: bool = False) -> list[Path]:
    """Download the three M5 source files from the public Zenodo record.

    The caller must explicitly acknowledge the upstream data terms. Files are never committed.
    """
    if not accept_source_terms:
        raise ValueError("Pass accept_source_terms=True after reviewing the M5 source terms")
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata = httpx.get(M5_ZENODO_RECORD, timeout=60).raise_for_status().json()
    available = {file["key"]: file for file in metadata.get("files", [])}
    absent = M5_REQUIRED_FILES - set(available)
    if absent:
        raise RuntimeError(f"Zenodo record is missing: {', '.join(sorted(absent))}")
    downloaded: list[Path] = []
    for name in sorted(M5_REQUIRED_FILES):
        destination = target_dir / name
        if destination.exists() and destination.stat().st_size > 0:
            downloaded.append(destination)
            continue
        url = available[name]["links"]["self"]
        with httpx.stream("GET", url, timeout=None, follow_redirects=True) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    handle.write(chunk)
        downloaded.append(destination)
    return downloaded


def prepare_m5_subset(
    source_dir: Path,
    output_path: Path,
    *,
    item_limit: int = 30,
    stores: Iterable[str] | None = None,
) -> DatasetProfile:
    required = {name: source_dir / name for name in M5_REQUIRED_FILES}
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing M5 files: {', '.join(sorted(missing))}")

    sales_scan = pl.scan_csv(required["sales_train_evaluation.csv"])
    selected_stores = list(stores or ["CA_1", "TX_1", "WI_1"])
    selected_items = (
        sales_scan.filter(pl.col("store_id").is_in(selected_stores))
        .select("item_id")
        .unique()
        .sort("item_id")
        .head(item_limit)
        .collect(engine="streaming")["item_id"]
        .to_list()
    )
    sales = (
        sales_scan.filter(
            pl.col("store_id").is_in(selected_stores)
            & pl.col("item_id").is_in(selected_items)
        )
        .collect(engine="streaming")
        .unpivot(
            index=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            variable_name="d",
            value_name="sales",
        )
    )
    calendar = pl.read_csv(required["calendar.csv"], try_parse_dates=True).select(
        "d", "date", "wm_yr_wk", "event_name_1"
    )
    prices = pl.read_csv(required["sell_prices.csv"]).filter(
        pl.col("store_id").is_in(selected_stores) & pl.col("item_id").is_in(selected_items)
    )
    merged = (
        sales.join(calendar, on="d", how="left")
        .join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        .sort(["store_id", "item_id", "date"])
        .with_columns(
            pl.col("sell_price")
            .forward_fill()
            .backward_fill()
            .over(["store_id", "item_id"])
            .alias("sell_price"),
            pl.col("event_name_1").fill_null("").alias("event_name"),
            pl.col("event_name_1").is_not_null().alias("promo_flag"),
        )
        .select(CANONICAL_COLUMNS)
    )
    profile = write_dataset(merged, output_path)
    manifest = {
        "source": M5_ZENODO_RECORD,
        "stores": selected_stores,
        "items": selected_items,
        "profile": profile.model_dump(),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return profile
