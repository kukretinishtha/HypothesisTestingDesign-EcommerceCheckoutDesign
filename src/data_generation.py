"""
Synthetic data generator matching the REAL Olist Brazilian E-commerce schema
(7 tables used by the pipeline; geolocation/sellers are optional and not
generated here since no core metric depends on them).

This exists so the pipeline is fully runnable out of the box. If you have the
real Kaggle Olist CSVs, drop them into `data/raw/` using the exact filenames
in `data_loader.REQUIRED_FILES` and the loader will use those instead --
no code changes needed.

Column names, dtypes and table relationships mirror the public dataset:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import config
from src.logging_utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR)


def _rng() -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED)


def generate_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    states = rng.choice(config.STATES, size=n, p=config.STATE_WEIGHTS)
    return pd.DataFrame({
        "customer_id": [f"cust_{i:07d}" for i in range(n)],
        "customer_unique_id": [f"uniq_{i:07d}" for i in range(n)],
        "customer_zip_code_prefix": rng.integers(1000, 99999, size=n),
        "customer_city": [f"city_{s.lower()}" for s in states],
        "customer_state": states,
    })


def generate_orders(n: int, customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    purchase_ts = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365, size=n), unit="D"
    ) + pd.to_timedelta(rng.integers(0, 86400, size=n), unit="s")

    customer_ids = rng.choice(customers["customer_id"], size=n, replace=True)

    # Ground-truth order_status distribution BEFORE any treatment effect is applied.
    status_options = ["delivered", "shipped", "canceled", "unavailable", "processing"]
    status_probs = [0.78, 0.09, 0.05, 0.04, 0.04]
    order_status = rng.choice(status_options, size=n, p=status_probs)

    order_approved_at = purchase_ts + pd.to_timedelta(rng.integers(0, 48, size=n), unit="h")
    order_delivered_carrier_date = purchase_ts + pd.to_timedelta(rng.integers(1, 5, size=n), unit="D")

    # Delivery time: only defined for delivered orders, lognormal ~ Olist's real skew
    delivery_days = rng.lognormal(mean=2.35, sigma=0.45, size=n).round(1)
    order_delivered_customer_date = purchase_ts + pd.to_timedelta(delivery_days, unit="D")
    order_delivered_customer_date = order_delivered_customer_date.where(order_status == "delivered", pd.NaT)
    order_delivered_carrier_date = pd.Series(order_delivered_carrier_date).where(
        pd.Series(order_status).isin(["delivered", "shipped"]), pd.NaT
    ).values

    # Estimated delivery date: business promise, roughly delivery_days_median + noise
    estimated_offset_days = rng.lognormal(mean=2.6, sigma=0.3, size=n)
    order_estimated_delivery_date = purchase_ts + pd.to_timedelta(estimated_offset_days, unit="D")

    orders = pd.DataFrame({
        "order_id": [f"order_{i:07d}" for i in range(n)],
        "customer_id": customer_ids,
        "order_status": order_status,
        "order_purchase_timestamp": purchase_ts,
        "order_approved_at": order_approved_at,
        "order_delivered_carrier_date": order_delivered_carrier_date,
        "order_delivered_customer_date": order_delivered_customer_date,
        "order_estimated_delivery_date": order_estimated_delivery_date,
    })
    return orders


def generate_products(rng: np.random.Generator, n_products: int = 3000) -> pd.DataFrame:
    categories_pt = [f"{c}_pt" for c in config.CATEGORIES]  # pretend these are the raw PT names
    cats = rng.choice(categories_pt, size=n_products)
    return pd.DataFrame({
        "product_id": [f"prod_{i:06d}" for i in range(n_products)],
        "product_category_name": cats,
        "product_name_lenght": rng.integers(10, 70, size=n_products),
        "product_description_lenght": rng.integers(50, 3000, size=n_products),
        "product_photos_qty": rng.integers(1, 6, size=n_products),
        "product_weight_g": rng.integers(50, 15000, size=n_products),
        "product_length_cm": rng.integers(5, 100, size=n_products),
        "product_height_cm": rng.integers(5, 100, size=n_products),
        "product_width_cm": rng.integers(5, 100, size=n_products),
    })


def generate_category_translation() -> pd.DataFrame:
    return pd.DataFrame({
        "product_category_name": [f"{c}_pt" for c in config.CATEGORIES],
        "product_category_name_english": config.CATEGORIES,
    })


def generate_order_items(orders: pd.DataFrame, products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(orders)
    n_items_per_order = rng.poisson(lam=1.3, size=n).clip(min=1)
    rows = []
    product_ids = products["product_id"].values
    for order_id, n_items in zip(orders["order_id"], n_items_per_order):
        chosen_products = rng.choice(product_ids, size=n_items)
        prices = rng.lognormal(mean=4.0, sigma=0.7, size=n_items)
        freights = np.round(prices * rng.uniform(0.05, 0.2, size=n_items), 2)
        shipping_limit = pd.Timestamp("2023-06-01")
        for j, (pid, price, freight) in enumerate(zip(chosen_products, prices, freights)):
            rows.append((order_id, j + 1, pid, f"seller_{rng.integers(0, 500):04d}",
                         shipping_limit, round(price, 2), freight))
    items = pd.DataFrame(rows, columns=[
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ])
    return items


def generate_payments(orders: pd.DataFrame, order_items: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    item_totals = (order_items.assign(item_total=order_items["price"] + order_items["freight_value"])
                   .groupby("order_id", as_index=False)["item_total"].sum())
    base = pd.DataFrame({"order_id": orders["order_id"]}).merge(item_totals, on="order_id", how="left")
    base["item_total"] = base["item_total"].fillna(0.0)

    payment_type = rng.choice(
        ["credit_card", "boleto", "voucher", "debit_card"],
        size=len(base), p=[0.74, 0.19, 0.04, 0.03],
    )
    installments = np.where(payment_type == "credit_card", rng.integers(1, 12, size=len(base)), 1)
    return pd.DataFrame({
        "order_id": base["order_id"],
        "payment_sequential": 1,
        "payment_type": payment_type,
        "payment_installments": installments,
        "payment_value": base["item_total"].round(2),
    })


def generate_reviews(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(orders)
    review_score = np.clip(rng.normal(loc=4.0, scale=1.0, size=n).round().astype(int), 1, 5)
    creation = orders["order_purchase_timestamp"] + pd.to_timedelta(rng.integers(3, 20, size=n), unit="D")
    answered = creation + pd.to_timedelta(rng.integers(1, 5, size=n), unit="D")
    return pd.DataFrame({
        "review_id": [f"rev_{i:07d}" for i in range(n)],
        "order_id": orders["order_id"],
        "review_score": review_score,
        "review_comment_title": "",
        "review_comment_message": "",
        "review_creation_date": creation,
        "review_answer_timestamp": answered,
    })


def generate_all(n_orders: int = config.N_ORDERS,
                  n_customers: int = config.N_CUSTOMERS) -> dict[str, pd.DataFrame]:
    rng = _rng()
    logger.info(f"Generating synthetic Olist-schema data: {n_orders:,} orders, {n_customers:,} customers")
    customers = generate_customers(n_customers, rng)
    orders = generate_orders(n_orders, customers, rng)
    products = generate_products(rng)
    translation = generate_category_translation()
    order_items = generate_order_items(orders, products, rng)
    payments = generate_payments(orders, order_items, rng)
    reviews = generate_reviews(orders, rng)
    tables = {
        "olist_customers_dataset": customers,
        "olist_orders_dataset": orders,
        "olist_order_items_dataset": order_items,
        "olist_order_payments_dataset": payments,
        "olist_order_reviews_dataset": reviews,
        "olist_products_dataset": products,
        "product_category_name_translation": translation,
    }
    return tables


def write_to_disk(tables: dict[str, pd.DataFrame], out_dir: Path = config.DATA_RAW_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Wrote {path.name} ({len(df):,} rows)")


if __name__ == "__main__":
    tables = generate_all()
    write_to_disk(tables)
