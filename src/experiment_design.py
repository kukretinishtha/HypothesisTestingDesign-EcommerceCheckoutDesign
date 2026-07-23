"""
Experiment design mechanics:
  - user-level randomization (not order-level, matching the notes)
  - Sample Ratio Mismatch (SRM) check
  - A/A test validation
  - injection of a KNOWN ground-truth treatment effect (since Olist has no
    real experiment flag) so downstream statistical tests can be validated
    against a known answer -- this is clearly separated from real analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import config
from src.logging_utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR)


def assign_user_level_treatment(df: pd.DataFrame, user_col: str = "customer_id",
                                 p_treatment: float = config.TREATMENT_ALLOCATION,
                                 seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Randomize at the USER level: every order from the same customer gets the
    same arm, avoiding cross-arm contamination for repeat customers."""
    rng = np.random.default_rng(seed)
    unique_users = df[user_col].unique()
    assignment = rng.choice(["control", "treatment"], size=len(unique_users),
                             p=[1 - p_treatment, p_treatment])
    user_map = pd.Series(assignment, index=unique_users, name="group")
    out = df.merge(user_map.rename("group"), left_on=user_col, right_index=True, how="left")
    return out


def srm_check(df: pd.DataFrame, group_col: str = "group",
              expected_ratio: float = config.TREATMENT_ALLOCATION,
              alpha: float = config.SRM_ALPHA) -> dict:
    """Sample Ratio Mismatch check: verifies observed allocation matches the
    planned split using a chi-square goodness-of-fit test. A significant
    result (p < alpha, conventionally 0.001) means randomization is broken
    and results should NOT be trusted until fixed."""
    counts = df[group_col].value_counts()
    n_control = int(counts.get("control", 0))
    n_treatment = int(counts.get("treatment", 0))
    n_total = n_control + n_treatment

    expected_control = n_total * (1 - expected_ratio)
    expected_treatment = n_total * expected_ratio

    chi2, p_value = chi2_contingency(
        [[n_control, n_treatment], [expected_control, expected_treatment]]
    )[:2]

    passed = p_value >= alpha
    result = {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "observed_ratio_treatment": n_treatment / n_total,
        "expected_ratio_treatment": expected_ratio,
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "alpha": alpha,
        "srm_detected": not passed,
        "verdict": "PASS - no SRM detected" if passed else "FAIL - SRM detected, investigate randomization",
    }
    log_fn = logger.info if passed else logger.warning
    log_fn(f"SRM check: control={n_control:,} treatment={n_treatment:,} "
           f"p={p_value:.4f} -> {result['verdict']}")
    return result


def aa_test(df: pd.DataFrame, metric_col: str, group_col: str, n_simulations: int = 1000,
            seed: int = config.RANDOM_SEED) -> dict:
    """A/A test: repeatedly re-randomize labels within a SINGLE arm (or shuffle
    existing labels) and confirm the false-positive rate of the test statistic
    is close to alpha. Validates that the test setup itself isn't biased
    before trusting the real A/B result."""
    rng = np.random.default_rng(seed)
    values = df[metric_col].dropna().values
    n = len(values)
    n_a = n // 2

    false_positives = 0
    p_values = []
    for _ in range(n_simulations):
        shuffled = rng.permutation(values)
        a, b = shuffled[:n_a], shuffled[n_a:]
        # Welch t-test as the generic engine (works for binary or continuous)
        from scipy.stats import ttest_ind
        _, p = ttest_ind(a, b, equal_var=False)
        p_values.append(p)
        if p < config.ALPHA:
            false_positives += 1

    observed_fpr = false_positives / n_simulations
    result = {
        "n_simulations": n_simulations,
        "alpha": config.ALPHA,
        "observed_false_positive_rate": observed_fpr,
        "expected_false_positive_rate": config.ALPHA,
        "verdict": (
            "PASS - false positive rate matches alpha, test setup is valid"
            if abs(observed_fpr - config.ALPHA) < 0.03
            else "WARNING - false positive rate deviates from alpha, investigate test/metric variance"
        ),
    }
    logger.info(f"A/A test on '{metric_col}': observed FPR={observed_fpr:.3f} "
                f"(expected ~{config.ALPHA}) -> {result['verdict']}")
    return result


def inject_ground_truth_effects(df: pd.DataFrame,
                                 conversion_lift: float = config.TRUE_TREATMENT_LIFT_ABS,
                                 seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Olist has no real experiment column, so per the project brief we simulate
    one with a KNOWN ground-truth effect. This function is the ONLY place
    that manufactures signal -- everything downstream (stats_tests, metrics,
    visualization) treats the result as if it were an observed real experiment.

    Effect model:
      - conversion ('delivered' outcome): treatment arm gets a `conversion_lift`
        absolute bump, applied by flipping a fraction of non-converted orders.
      - AOV: untouched (null effect), so the pipeline demonstrates correctly
        detecting "no significant difference" on a secondary metric.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["converted"] = (df["order_status"] == "delivered").astype(int)

    mask = (df["group"] == "treatment") & (df["converted"] == 0)
    idx = df.index[mask]
    flip = rng.random(len(idx)) < conversion_lift
    flip_idx = idx[flip]
    df.loc[flip_idx, "converted"] = 1
    # keep order_status coherent for anyone inspecting guardrails downstream
    df.loc[flip_idx, "order_status"] = "delivered"

    logger.info(f"Injected ground-truth effect: flipped {flip.sum():,} treatment orders "
                f"to converted (target lift={conversion_lift:.1%})")
    return df
