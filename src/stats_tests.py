"""
Statistical engine: hypothesis tests, effect sizes, confidence intervals,
power analysis, and CUPED variance reduction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from scipy.stats import chi2_contingency, ttest_ind, norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

import config
from src.logging_utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR)


# --------------------------------------------------------------------------
# Effect sizes
# --------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(b) - np.mean(a)) / pooled_sd if pooled_sd > 0 else 0.0


def cramers_v(contingency_table: np.ndarray) -> float:
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    r, k = contingency_table.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1))))


# --------------------------------------------------------------------------
# Primary metric: conversion rate (categorical, two-proportion test)
# --------------------------------------------------------------------------
@dataclass
class ConversionTestResult:
    n_control: int
    n_treatment: int
    conversions_control: int
    conversions_treatment: int
    rate_control: float
    rate_treatment: float
    absolute_lift: float
    relative_lift_pct: float
    chi2_statistic: float
    chi2_p_value: float
    z_statistic: float
    z_p_value: float
    cramers_v: float
    ci_low_control: float
    ci_high_control: float
    ci_low_treatment: float
    ci_high_treatment: float
    ci_low_diff: float
    ci_high_diff: float
    significant: bool


def test_conversion(df: pd.DataFrame, group_col: str = "group",
                     outcome_col: str = "converted", alpha: float = config.ALPHA) -> ConversionTestResult:
    control = df.loc[df[group_col] == "control", outcome_col]
    treatment = df.loc[df[group_col] == "treatment", outcome_col]

    n_c, n_t = len(control), len(treatment)
    conv_c, conv_t = int(control.sum()), int(treatment.sum())
    rate_c, rate_t = conv_c / n_c, conv_t / n_t

    table = np.array([[n_c - conv_c, conv_c], [n_t - conv_t, conv_t]])
    chi2, chi2_p, _, _ = chi2_contingency(table)

    z_stat, z_p = proportions_ztest([conv_c, conv_t], [n_c, n_t])

    ci_c = proportion_confint(conv_c, n_c, alpha=alpha, method="wilson")
    ci_t = proportion_confint(conv_t, n_t, alpha=alpha, method="wilson")

    diff = rate_t - rate_c
    se_diff = np.sqrt(rate_c * (1 - rate_c) / n_c + rate_t * (1 - rate_t) / n_t)
    z_crit = norm.ppf(1 - alpha / 2)
    ci_diff = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    result = ConversionTestResult(
        n_control=n_c, n_treatment=n_t,
        conversions_control=conv_c, conversions_treatment=conv_t,
        rate_control=rate_c, rate_treatment=rate_t,
        absolute_lift=diff,
        relative_lift_pct=(diff / rate_c * 100) if rate_c > 0 else float("nan"),
        chi2_statistic=float(chi2), chi2_p_value=float(chi2_p),
        z_statistic=float(z_stat), z_p_value=float(z_p),
        cramers_v=cramers_v(table),
        ci_low_control=ci_c[0], ci_high_control=ci_c[1],
        ci_low_treatment=ci_t[0], ci_high_treatment=ci_t[1],
        ci_low_diff=ci_diff[0], ci_high_diff=ci_diff[1],
        significant=bool(chi2_p < alpha),
    )
    logger.info(f"Conversion test: control={rate_c:.4f} treatment={rate_t:.4f} "
                f"abs_lift={diff:+.4f} p={chi2_p:.5f} significant={result.significant}")
    return result


# --------------------------------------------------------------------------
# Secondary metric: AOV (continuous, Welch t-test)
# --------------------------------------------------------------------------
@dataclass
class ContinuousTestResult:
    metric_name: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    std_control: float
    std_treatment: float
    absolute_diff: float
    relative_diff_pct: float
    t_statistic: float
    p_value: float
    cohens_d: float
    ci_low_diff: float
    ci_high_diff: float
    significant: bool


def test_continuous_metric(df: pd.DataFrame, metric_col: str, group_col: str = "group",
                            alpha: float = config.ALPHA) -> ContinuousTestResult:
    control = df.loc[df[group_col] == "control", metric_col].dropna().values
    treatment = df.loc[df[group_col] == "treatment", metric_col].dropna().values

    t_stat, p_val = ttest_ind(control, treatment, equal_var=False)
    mean_c, mean_t = np.mean(control), np.mean(treatment)
    diff = mean_t - mean_c

    se_diff = np.sqrt(np.var(control, ddof=1) / len(control) + np.var(treatment, ddof=1) / len(treatment))
    z_crit = norm.ppf(1 - alpha / 2)
    ci_diff = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    result = ContinuousTestResult(
        metric_name=metric_col,
        n_control=len(control), n_treatment=len(treatment),
        mean_control=float(mean_c), mean_treatment=float(mean_t),
        std_control=float(np.std(control, ddof=1)), std_treatment=float(np.std(treatment, ddof=1)),
        absolute_diff=float(diff),
        relative_diff_pct=float(diff / mean_c * 100) if mean_c != 0 else float("nan"),
        t_statistic=float(t_stat), p_value=float(p_val),
        cohens_d=cohens_d(control, treatment),
        ci_low_diff=ci_diff[0], ci_high_diff=ci_diff[1],
        significant=bool(p_val < alpha),
    )
    logger.info(f"Continuous test [{metric_col}]: control={mean_c:.2f} treatment={mean_t:.2f} "
                f"diff={diff:+.2f} p={p_val:.5f} significant={result.significant}")
    return result


# --------------------------------------------------------------------------
# CUPED variance reduction
# --------------------------------------------------------------------------
def cuped_adjust(df: pd.DataFrame, metric_col: str, covariate_col: str,
                  group_col: str = "group") -> pd.Series:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data): reduces metric
    variance by regressing out a pre-experiment covariate correlated with the
    outcome, increasing statistical power without changing the estimand.

    theta = Cov(Y, X) / Var(X), computed on the POOLED sample (both arms) to
    avoid introducing bias.
    """
    y = df[metric_col].values
    x = df[covariate_col].values
    valid = ~np.isnan(y) & ~np.isnan(x)
    theta = np.cov(y[valid], x[valid])[0, 1] / np.var(x[valid], ddof=1)
    x_bar = np.mean(x[valid])
    adjusted = y - theta * (x - x_bar)
    logger.info(f"CUPED adjustment on '{metric_col}' using covariate '{covariate_col}': theta={theta:.4f}, "
                f"variance reduced from {np.var(y[valid]):.4f} to {np.var(adjusted[valid]):.4f} "
                f"({(1 - np.var(adjusted[valid]) / np.var(y[valid])) * 100:.1f}% reduction)")
    return pd.Series(adjusted, index=df.index, name=f"{metric_col}_cuped")


# --------------------------------------------------------------------------
# Power analysis
# --------------------------------------------------------------------------
def required_sample_size(baseline_rate: float, mde_absolute: float,
                          alpha: float = config.ALPHA, power: float = config.TARGET_POWER) -> dict:
    """Sample size per arm needed to detect an absolute MDE on a baseline
    conversion rate, using a normal-approximation two-proportion power calc."""
    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute
    pooled_sd = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    # standardized effect size for statsmodels' NormalIndPower (Cohen's h approx via pooled sd)
    effect_size = abs(p2 - p1) / pooled_sd

    analysis = NormalIndPower()
    n_per_arm = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)

    result = {
        "baseline_rate": p1,
        "mde_absolute": mde_absolute,
        "target_rate": p2,
        "effect_size": float(effect_size),
        "alpha": alpha,
        "target_power": power,
        "required_n_per_arm": int(np.ceil(n_per_arm)),
        "required_n_total": int(np.ceil(n_per_arm) * 2),
    }
    logger.info(f"Power analysis: need n={result['required_n_per_arm']:,}/arm "
                f"to detect {mde_absolute:.1%} abs lift on {baseline_rate:.1%} baseline "
                f"(alpha={alpha}, power={power})")
    return result


def achieved_power(n_per_arm: int, baseline_rate: float, observed_lift: float,
                    alpha: float = config.ALPHA) -> float:
    """Post-hoc: power the experiment actually had, given the sample size collected."""
    p1 = baseline_rate
    p2 = baseline_rate + observed_lift
    pooled_sd = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    effect_size = abs(p2 - p1) / pooled_sd if pooled_sd > 0 else 0.0
    analysis = NormalIndPower()
    power = analysis.solve_power(effect_size=effect_size, nobs1=n_per_arm, alpha=alpha, ratio=1.0)
    return float(np.clip(power, 0, 1))
