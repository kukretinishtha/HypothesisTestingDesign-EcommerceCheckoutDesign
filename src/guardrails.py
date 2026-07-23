"""Guardrail metrics: signals that must NOT regress even if the primary
metric improves. Each one is evaluated independently and rolled up into a
pass/fail table used by the ship/no-ship decision."""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

import config
from src.stats_tests import test_continuous_metric, test_conversion
from src.logging_utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR)

MAX_ACCEPTABLE_ON_TIME_RATE_DECREASE = 0.01  # 1 pp drop in on-time delivery rate


@dataclass
class GuardrailResult:
    name: str
    control_value: float
    treatment_value: float
    diff: float
    p_value: float
    threshold_breached: bool
    verdict: str


def evaluate_guardrails(df: pd.DataFrame) -> list[GuardrailResult]:
    results = []

    # 1. Delivery time (continuous, delivered orders only)
    delivered = df[df["order_status"] == "delivered"].copy()
    delivered["delivery_days"] = (
        delivered["order_delivered_customer_date"] - delivered["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    r = test_continuous_metric(delivered, "delivery_days")
    breached = r.absolute_diff > config.MAX_ACCEPTABLE_DELIVERY_DAYS_INCREASE and r.significant
    results.append(GuardrailResult(
        name="Delivery Time (days)", control_value=r.mean_control, treatment_value=r.mean_treatment,
        diff=r.absolute_diff, p_value=r.p_value, threshold_breached=breached,
        verdict="BREACHED" if breached else "OK",
    ))

    # 1b. On-time delivery rate (categorical) -- uses order_estimated_delivery_date,
    # the guardrail businesses actually gate on rather than raw delivery days.
    on_time_df = df[df["delivered_on_time"].notna()].copy()
    if len(on_time_df) > 0:
        on_time_df["delivered_on_time"] = on_time_df["delivered_on_time"].astype(bool).astype(int)
        r = test_conversion(on_time_df, outcome_col="delivered_on_time")
        breached = (-r.absolute_lift) > MAX_ACCEPTABLE_ON_TIME_RATE_DECREASE and r.significant
        results.append(GuardrailResult(
            name="On-Time Delivery Rate", control_value=r.rate_control, treatment_value=r.rate_treatment,
            diff=r.absolute_lift, p_value=r.chi2_p_value, threshold_breached=breached,
            verdict="BREACHED" if breached else "OK",
        ))

    # 2. Cancellation rate (categorical)
    df = df.copy()
    df["is_cancelled"] = (df["order_status"] == "canceled").astype(int)
    r = test_conversion(df, outcome_col="is_cancelled")
    breached = r.absolute_lift > config.MAX_ACCEPTABLE_CANCEL_RATE_INCREASE and r.significant
    results.append(GuardrailResult(
        name="Cancellation Rate", control_value=r.rate_control, treatment_value=r.rate_treatment,
        diff=r.absolute_lift, p_value=r.chi2_p_value, threshold_breached=breached,
        verdict="BREACHED" if breached else "OK",
    ))

    # 3. Payment / fulfillment failure rate (order_status in {unavailable})
    df["is_payment_failure"] = (df["order_status"] == "unavailable").astype(int)
    r = test_conversion(df, outcome_col="is_payment_failure")
    breached = r.absolute_lift > config.MAX_ACCEPTABLE_PAYMENT_FAILURE_INCREASE and r.significant
    results.append(GuardrailResult(
        name="Payment Failure Rate", control_value=r.rate_control, treatment_value=r.rate_treatment,
        diff=r.absolute_lift, p_value=r.chi2_p_value, threshold_breached=breached,
        verdict="BREACHED" if breached else "OK",
    ))

    # 4. Review score (continuous, 1-5 stars) -- now joined from the separate
    # reviews table; orders without a review are dropped by test_continuous_metric's dropna().
    r = test_continuous_metric(df, "review_score")
    breached = r.absolute_diff < config.MIN_ACCEPTABLE_REVIEW_SCORE_DROP and r.significant
    results.append(GuardrailResult(
        name="Review Score (stars)", control_value=r.mean_control, treatment_value=r.mean_treatment,
        diff=r.absolute_diff, p_value=r.p_value, threshold_breached=breached,
        verdict="BREACHED" if breached else "OK",
    ))

    for res in results:
        logger.info(f"Guardrail [{res.name}]: control={res.control_value:.3f} "
                    f"treatment={res.treatment_value:.3f} diff={res.diff:+.3f} "
                    f"p={res.p_value:.4f} -> {res.verdict}")
    return results
