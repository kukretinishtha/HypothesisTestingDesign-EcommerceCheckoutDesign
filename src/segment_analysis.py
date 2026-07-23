"""Follow-up / segment analyses (Step 10 of the brief):
  - premium vs non-premium customers
  - by state (Simpson's paradox check)
  - by product category
  - late vs on-time delivery interaction with treatment
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

import config
from src.stats_tests import test_conversion
from src.logging_utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR)


@dataclass
class SegmentResult:
    segment_dimension: str
    segment_value: str
    n_control: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    absolute_lift: float
    p_value: float
    significant: bool


def _run_segment_test(sub_df: pd.DataFrame, dimension: str, value: str) -> SegmentResult | None:
    if sub_df["group"].nunique() < 2 or len(sub_df) < 30:
        return None
    if sub_df.loc[sub_df.group == "control"].empty or sub_df.loc[sub_df.group == "treatment"].empty:
        return None
    # skip degenerate segments where the outcome has no variance at all
    # (e.g. an outcome that is constant by construction within the filtered subset)
    if sub_df["converted"].nunique() < 2:
        return None
    r = test_conversion(sub_df)
    return SegmentResult(
        segment_dimension=dimension, segment_value=value,
        n_control=r.n_control, n_treatment=r.n_treatment,
        rate_control=r.rate_control, rate_treatment=r.rate_treatment,
        absolute_lift=r.absolute_lift, p_value=r.chi2_p_value, significant=r.significant,
    )


def by_premium_status(df: pd.DataFrame) -> list[SegmentResult]:
    df = df.copy()
    df["is_premium"] = df["payment_value"] > config.PREMIUM_THRESHOLD_BRL
    out = []
    for is_premium, label in [(True, "Premium (>500 BRL)"), (False, "Non-premium (<=500 BRL)")]:
        sub = df[df["is_premium"] == is_premium]
        res = _run_segment_test(sub, "Customer Tier", label)
        if res:
            out.append(res)
    return out


def by_state(df: pd.DataFrame, top_n_states: int = 5) -> list[SegmentResult]:
    out = []
    top_states = df["customer_state"].value_counts().head(top_n_states).index
    for state in top_states:
        sub = df[df["customer_state"] == state]
        res = _run_segment_test(sub, "State", state)
        if res:
            out.append(res)
    return out


def by_category(df: pd.DataFrame) -> list[SegmentResult]:
    out = []
    for cat in df["product_category"].dropna().unique():
        sub = df[df["product_category"] == cat]
        res = _run_segment_test(sub, "Product Category", cat)
        if res:
            out.append(res)
    return out


def by_delivery_speed(df: pd.DataFrame) -> list[SegmentResult]:
    """
    Note: this segment is restricted to delivered orders, which makes the
    'converted' outcome constant (all delivered=1) and unusable for a
    conversion chi-square test. Instead we use review_score (satisfaction)
    as the outcome -- a more meaningful question anyway: "does the new
    checkout's satisfaction impact differ between late and on-time
    deliveries?" -- and map it onto the same SegmentResult shape so it
    plots alongside the conversion-based segments.
    """
    from src.stats_tests import test_continuous_metric

    df = df.copy()
    delivered = df[df["order_status"] == "delivered"].copy()
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    median_days = delivered["delivery_days"].median()
    delivered["speed_bucket"] = delivered["delivery_days"].apply(
        lambda d: "Late (> median)" if d > median_days else "On-time (<= median)"
    )
    out = []
    for bucket in delivered["speed_bucket"].unique():
        sub = delivered[delivered["speed_bucket"] == bucket]
        if sub["group"].nunique() < 2 or len(sub) < 30:
            continue
        r = test_continuous_metric(sub, "review_score")
        out.append(SegmentResult(
            segment_dimension="Delivery Speed (review score, 1-5)",
            segment_value=bucket,
            n_control=r.n_control, n_treatment=r.n_treatment,
            rate_control=r.mean_control, rate_treatment=r.mean_treatment,
            absolute_lift=r.absolute_diff, p_value=r.p_value, significant=r.significant,
        ))
    return out


def run_all_segments(df: pd.DataFrame) -> pd.DataFrame:
    all_results: list[SegmentResult] = []
    all_results += by_premium_status(df)
    all_results += by_state(df)
    all_results += by_category(df)
    all_results += by_delivery_speed(df)

    records = [r.__dict__ for r in all_results]
    out_df = pd.DataFrame(records)
    logger.info(f"Segment analysis complete: {len(out_df)} segment cells evaluated")
    return out_df
