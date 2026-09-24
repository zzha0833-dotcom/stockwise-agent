from __future__ import annotations

import polars as pl
import pytest

from stockwise.data import (
    DataValidationError,
    normalize_retail_frame,
    prepare_m5_subset,
    validate_and_profile,
)


def test_profile_is_deterministic(retail_frame):
    profile = validate_and_profile(retail_frame)
    assert profile.items == 2
    assert profile.stores == 1
    assert profile.series == 2
    assert profile.rows == 280
    assert profile.start_date == "2025-01-01"


def test_missing_required_column_is_rejected(retail_frame):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        normalize_retail_frame(retail_frame.drop("sales"))


def test_negative_sales_are_rejected(retail_frame):
    broken = retail_frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(-1.0))
        .otherwise(pl.col("sales"))
        .alias("sales")
    )
    with pytest.raises(DataValidationError, match="negative"):
        validate_and_profile(broken)


def test_m5_subset_is_normalized_without_large_intermediate_files(tmp_path):
    source = tmp_path / "m5"
    source.mkdir()
    pl.DataFrame(
        {
            "id": ["ITEM_001_CA_1_evaluation"],
            "item_id": ["ITEM_001"],
            "dept_id": ["FOODS_1"],
            "cat_id": ["FOODS"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            "d_1": [3],
            "d_2": [5],
        }
    ).write_csv(source / "sales_train_evaluation.csv")
    pl.DataFrame(
        {
            "d": ["d_1", "d_2"],
            "date": ["2026-01-01", "2026-01-02"],
            "wm_yr_wk": [1, 1],
            "event_name_1": [None, "Promotion"],
        }
    ).write_csv(source / "calendar.csv")
    pl.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["ITEM_001"],
            "wm_yr_wk": [1],
            "sell_price": [4.5],
        }
    ).write_csv(source / "sell_prices.csv")

    output = tmp_path / "processed" / "m5_subset.parquet"
    profile = prepare_m5_subset(source, output, item_limit=1, stores=["CA_1"])
    result = pl.read_parquet(output)

    assert profile.rows == 2
    assert result.columns == [
        "date",
        "store_id",
        "item_id",
        "sales",
        "sell_price",
        "promo_flag",
        "event_name",
    ]
    assert result["promo_flag"].to_list() == [False, True]
    assert output.with_suffix(".manifest.json").exists()
