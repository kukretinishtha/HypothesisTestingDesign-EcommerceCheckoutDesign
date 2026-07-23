"""All chart generation. Every function saves a PNG to outputs/figures/ and
returns the path, so the report generator can reference them directly."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

import config
from src.logging_utils import get_logger

logger = get_logger(__name__, config.LOGS_DIR)

# --- consistent business-storytelling theme ---
COLOR_CONTROL = "#94A3B8"     # slate grey
COLOR_TREATMENT = "#2563EB"   # brand blue
COLOR_GOOD = "#16A34A"        # green
COLOR_BAD = "#DC2626"         # red
COLOR_NEUTRAL = "#64748B"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E1",
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.6,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path.name}")
    return path


def plot_conversion_comparison(conv_result) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    groups = ["Control\n(old checkout)", "Treatment\n(new checkout)"]
    rates = [conv_result.rate_control, conv_result.rate_treatment]
    ci_low = [conv_result.ci_low_control, conv_result.ci_low_treatment]
    ci_high = [conv_result.ci_high_control, conv_result.ci_high_treatment]
    errs = [[r - lo for r, lo in zip(rates, ci_low)], [hi - r for r, hi in zip(rates, ci_high)]]
    colors = [COLOR_CONTROL, COLOR_TREATMENT]

    bars = ax.bar(groups, rates, color=colors, width=0.55, yerr=errs, capsize=6,
                   error_kw={"linewidth": 1.5, "ecolor": "#1E293B"})
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.1%}",
                ha="center", fontweight="bold", fontsize=12)

    sig_txt = "significant" if conv_result.significant else "not significant"
    ax.set_title(f"Conversion Rate: Control vs Treatment\n"
                 f"Absolute lift {conv_result.absolute_lift:+.2%} "
                 f"(p={conv_result.chi2_p_value:.4f}, {sig_txt})")
    ax.set_ylabel("Conversion Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(0, max(rates) * 1.3)
    return _save(fig, "01_conversion_comparison")


def plot_aov_distribution(df: pd.DataFrame, aov_result) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    control = df.loc[df.group == "control", "payment_value"]
    treatment = df.loc[df.group == "treatment", "payment_value"]
    clip_val = df["payment_value"].quantile(0.99)

    ax.hist(control.clip(upper=clip_val), bins=50, alpha=0.6, color=COLOR_CONTROL,
            label=f"Control (mean={aov_result.mean_control:.0f} BRL)", density=True)
    ax.hist(treatment.clip(upper=clip_val), bins=50, alpha=0.6, color=COLOR_TREATMENT,
            label=f"Treatment (mean={aov_result.mean_treatment:.0f} BRL)", density=True)
    ax.axvline(aov_result.mean_control, color=COLOR_CONTROL, linestyle="--", linewidth=2)
    ax.axvline(aov_result.mean_treatment, color=COLOR_TREATMENT, linestyle="--", linewidth=2)

    sig_txt = "significant" if aov_result.significant else "not significant"
    ax.set_title(f"Average Order Value Distribution\n"
                 f"Diff {aov_result.absolute_diff:+.2f} BRL (p={aov_result.p_value:.4f}, {sig_txt})")
    ax.set_xlabel("Payment Value (BRL, clipped at 99th pct)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    return _save(fig, "02_aov_distribution")


def plot_guardrails(guardrail_results) -> Path:
    fig, axes = plt.subplots(1, len(guardrail_results), figsize=(4 * len(guardrail_results), 5))
    if len(guardrail_results) == 1:
        axes = [axes]
    for ax, g in zip(axes, guardrail_results):
        vals = [g.control_value, g.treatment_value]
        color_treat = COLOR_BAD if g.threshold_breached else COLOR_GOOD
        bars = ax.bar(["Control", "Treatment"], vals, color=[COLOR_CONTROL, color_treat], width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                    fontweight="bold", fontsize=10)
        ax.set_title(g.name, fontsize=11)
        ax.set_xlabel(g.verdict, fontsize=10,
                      color=COLOR_BAD if g.threshold_breached else COLOR_GOOD, fontweight="bold")
    fig.suptitle("Guardrail Metrics: Control vs Treatment", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    return _save(fig, "03_guardrails")


def plot_power_curve(baseline_rate: float, mde: float, alpha: float = config.ALPHA) -> Path:
    from statsmodels.stats.power import NormalIndPower
    analysis = NormalIndPower()
    sample_sizes = np.linspace(200, 20000, 200)
    p1, p2 = baseline_rate, baseline_rate + mde
    pooled_sd = np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    effect_size = abs(p2 - p1) / pooled_sd
    powers = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=1.0)
              for n in sample_sizes]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sample_sizes, powers, color=COLOR_TREATMENT, linewidth=2.5)
    ax.axhline(config.TARGET_POWER, color=COLOR_NEUTRAL, linestyle="--", label=f"Target power ({config.TARGET_POWER:.0%})")
    n_required = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=config.TARGET_POWER, ratio=1.0)
    ax.axvline(n_required, color=COLOR_BAD, linestyle=":", label=f"Required n/arm ≈ {int(np.ceil(n_required)):,}")
    ax.set_title(f"Power Curve: Detecting {mde:.1%} Absolute Lift\non {baseline_rate:.0%} Baseline Conversion")
    ax.set_xlabel("Sample Size per Arm")
    ax.set_ylabel("Statistical Power")
    ax.legend(frameon=False)
    return _save(fig, "04_power_curve")


def plot_effect_size_forest(conv_result, aov_result) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["Conversion\n(abs. lift)", "AOV\n(BRL diff)"]
    diffs = [conv_result.absolute_lift, aov_result.absolute_diff]
    ci_lows = [conv_result.ci_low_diff, aov_result.ci_low_diff]
    ci_highs = [conv_result.ci_high_diff, aov_result.ci_high_diff]
    y_pos = np.arange(len(metrics))

    for i, (d, lo, hi) in enumerate(zip(diffs, ci_lows, ci_highs)):
        color = COLOR_GOOD if lo > 0 or hi < 0 else COLOR_NEUTRAL
        ax.plot([lo, hi], [i, i], color=color, linewidth=3)
        ax.plot(d, i, "o", color=color, markersize=10)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_title("Effect Sizes with 95% Confidence Intervals\n(CI crossing 0 = not statistically significant)")
    ax.set_xlabel("Treatment − Control")
    return _save(fig, "05_effect_size_forest")


def plot_segment_results(segment_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(segment_df))))
    segment_df = segment_df.sort_values("absolute_lift")
    labels = segment_df["segment_dimension"] + ": " + segment_df["segment_value"].astype(str)
    colors = [COLOR_GOOD if sig else COLOR_NEUTRAL for sig in segment_df["significant"]]
    ax.barh(labels, segment_df["absolute_lift"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Segment-Level Treatment Effect (Step 10 Follow-ups)\n"
                 "green = statistically significant at α=0.05 (conversion lift, except Delivery Speed = review-score diff)")
    ax.set_xlabel("Treatment − Control")
    fig.tight_layout()
    return _save(fig, "06_segment_lift")


def plot_srm_and_aa(srm_result: dict, aa_result: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    labels = ["Control", "Treatment"]
    observed = [srm_result["n_control"], srm_result["n_treatment"]]
    expected_total = sum(observed)
    expected = [expected_total * (1 - srm_result["expected_ratio_treatment"]),
                expected_total * srm_result["expected_ratio_treatment"]]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, observed, width, label="Observed", color=COLOR_TREATMENT)
    ax.bar(x + width / 2, expected, width, label="Expected", color=COLOR_CONTROL)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"SRM Check (p={srm_result['p_value']:.4f})\n{srm_result['verdict']}", fontsize=11)
    ax.legend(frameon=False)

    ax = axes[1]
    fpr = aa_result["observed_false_positive_rate"]
    expected_fpr = aa_result["expected_false_positive_rate"]
    ax.bar(["Observed FPR", "Expected (α)"], [fpr, expected_fpr],
           color=[COLOR_TREATMENT, COLOR_CONTROL])
    ax.set_title(f"A/A Test Validation\n{aa_result['verdict']}", fontsize=11)
    ax.set_ylabel("False Positive Rate")

    fig.suptitle("Experiment Setup Validation", fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, "00_setup_validation")


def plot_cuped_variance_reduction(before_var: float, after_var: float) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    pct_reduction = (1 - after_var / before_var) * 100
    bars = ax.bar(["Before CUPED", "After CUPED"], [before_var, after_var],
                  color=[COLOR_CONTROL, COLOR_TREATMENT], width=0.5)
    for bar, v in zip(bars, [before_var, after_var]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"CUPED Variance Reduction\n{pct_reduction:.1f}% lower variance -> more statistical power")
    ax.set_ylabel("Variance")
    return _save(fig, "07_cuped_variance_reduction")


def plot_ship_decision(checklist: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    items = list(checklist.keys())
    passed = [checklist[k] for k in items]
    colors = [COLOR_GOOD if p else COLOR_BAD for p in passed]
    y_pos = np.arange(len(items))
    ax.barh(y_pos, [1] * len(items), color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(items)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for i, p in enumerate(passed):
        ax.text(0.5, i, "PASS" if p else "FAIL", ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
    overall = all(passed)
    ax.set_title(f"Ship / Don't Ship Checklist\nDecision: {'SHIP' if overall else 'DO NOT SHIP'}",
                 fontsize=14, fontweight="bold", color=COLOR_GOOD if overall else COLOR_BAD)
    fig.tight_layout()
    return _save(fig, "08_ship_decision")
