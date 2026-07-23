"""
Data loading layer -- updated for the REAL Olist schema (9 tables).

Priority order:
  1. Real Olist CSVs found in data/raw/ (exact Kaggle filenames).
  2. Synthetic data generated on the fly (src.data_generation), cached in
     data/synthetic_cache/ so data/raw/ is reserved exclusively for real data.

Required tables (core metrics depend on these):
  - olist_customers_dataset
  - olist_orders_dataset
  - olist_order_items_dataset
  - olist_order_payments_dataset
  - olist_order_reviews_dataset       (review_score lives here, not on orders)
  - olist_products_dataset            (product_category_name lives here, keyed by product_id)
  - product_category_name_translation (maps product_category_name -> English)

Optional tables (loaded if present, not required -- available for future
enrichment such as segmenting by seller state or delivery distance):
  - olist_geolocation_dataset
  - olist_sellers_dataset
"""
from __future__ import annotations
import pandas as pd

import config
from src.logging_utils import get_logger
from src import data_generation

logger = get_logger(__name__, config.LOGS_DIR)

REQUIRED_FILES = {
    "olist_customers_dataset": "olist_customers_dataset.csv",
    "olist_orders_dataset": "olist_orders_dataset.csv",
    "olist_order_items_dataset": "olist_order_items_dataset.csv",
    "olist_order_payments_dataset": "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset": "olist_order_reviews_dataset.csv",
    "olist_products_dataset": "olist_products_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv",
}

OPTIONAL_FILES = {
    "olist_geolocation_dataset": "olist_geolocation_dataset.csv",
    "olist_sellers_dataset": "olist_sellers_dataset.csv",
}


def _files_available(file_map: dict, base_dir) -> bool:
    return all((base_dir / fname).exists() for fname in file_map.values())


def _real_data_available() -> bool:
    return _files_available(REQUIRED_FILES, config.DATA_RAW_DIR)


def _synthetic_cache_available() -> bool:
    return _files_available(REQUIRED_FILES, config.DATA_SYNTHETIC_DIR)


def _load_from(base_dir) -> dict[str, pd.DataFrame]:
    tables = {}
    for key, fname in REQUIRED_FILES.items():
        tables[key] = pd.read_csv(base_dir / fname)
    for key, fname in OPTIONAL_FILES.items():
        path = base_dir / fname
        if path.exists():
            tables[key] = pd.read_csv(path)
        else:
            logger.info(f"Optional file {fname} not found -- skipping (not required for core metrics).")
    return tables


def load_raw_tables() -> dict[str, pd.DataFrame]:
    if _real_data_available():
        logger.info("Real Olist CSVs detected in data/raw/ -- loading actual dataset.")
        tables = _load_from(config.DATA_RAW_DIR)
        source = "real"
    elif _synthetic_cache_available():
        logger.info("Real Olist CSVs not found -- loading cached synthetic dataset from data/synthetic_cache/.")
        tables = _load_from(config.DATA_SYNTHETIC_DIR)
        source = "synthetic"
    else:
        logger.info("Real Olist CSVs not found -- generating synthetic dataset with identical schema.")
        tables = data_generation.generate_all()
        data_generation.write_to_disk(tables, out_dir=config.DATA_SYNTHETIC_DIR)
        source = "synthetic"

    # normalize timestamp dtypes regardless of source
    orders = tables["olist_orders_dataset"]
    for col in ("order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date"):
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")
    tables["olist_orders_dataset"] = orders

    tables["_source"] = source
    return tables


def _build_category_lookup(items: pd.DataFrame, products: pd.DataFrame,
                            translation: pd.DataFrame) -> pd.DataFrame:
    """orders -> order_items -> products (product_id) -> English category name."""
    cat = items[["order_id", "order_item_id", "product_id"]].merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    ).merge(
        translation, on="product_category_name", how="left"
    )
    # fall back to the original (Portuguese) name if no English translation exists
    cat["product_category"] = cat["product_category_name_english"].fillna(cat["product_category_name"])
    primary_category = (
        cat.sort_values(["order_id", "order_item_id"])
        .groupby("order_id", as_index=False)
        .first()[["order_id", "product_category"]]
    )
    return primary_category


def _build_review_lookup(reviews: pd.DataFrame) -> pd.DataFrame:
    """A small fraction of orders have >1 review; average the score per order."""
    return reviews.groupby("order_id", as_index=False)["review_score"].mean()


def build_analysis_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join orders <- customers, orders <- payments (summed), orders <- category,
    orders <- review score, plus an on-time-delivery flag using the estimated
    delivery date (now available in the real schema)."""
    orders = tables["olist_orders_dataset"].copy()
    customers = tables["olist_customers_dataset"]
    items = tables["olist_order_items_dataset"]
    payments = tables["olist_order_payments_dataset"]
    reviews = tables["olist_order_reviews_dataset"]
    products = tables["olist_products_dataset"]
    translation = tables["product_category_name_translation"]

    aov = payments.groupby("order_id", as_index=False)["payment_value"].sum()
    primary_category = _build_category_lookup(items, products, translation)
    review_scores = _build_review_lookup(reviews)
    n_items = items.groupby("order_id", as_index=False).size().rename(columns={"size": "n_items"})

    df = (
        orders
        .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
        .merge(aov, on="order_id", how="left")
        .merge(primary_category, on="order_id", how="left")
        .merge(review_scores, on="order_id", how="left")
        .merge(n_items, on="order_id", how="left")
    )
    df["payment_value"] = df["payment_value"].fillna(0.0)
    df["n_items"] = df["n_items"].fillna(0).astype(int)

    # On-time delivery: only defined for delivered orders that also have an
    # estimate to compare against. This is the guardrail businesses actually
    # use, rather than a raw delivery-days average.
    has_both_dates = df["order_delivered_customer_date"].notna() & df["order_estimated_delivery_date"].notna()
    df["delivered_on_time"] = pd.NA
    df.loc[has_both_dates, "delivered_on_time"] = (
        df.loc[has_both_dates, "order_delivered_customer_date"]
        <= df.loc[has_both_dates, "order_estimated_delivery_date"]
    )

    n_missing_review = df["review_score"].isna().sum()
    n_missing_category = df["product_category"].isna().sum()
    logger.info(f"Built analysis table: {len(df):,} rows, {df['order_id'].nunique():,} unique orders "
                f"(source={tables.get('_source')}); "
                f"{n_missing_review:,} orders missing a review score, "
                f"{n_missing_category:,} missing a product category")
    return df
