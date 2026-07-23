"""Assembles all pipeline outputs into a single business-facing Markdown
report: narrative + embedded figures + decision framework."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime

import config


def generate_report(
    data_source: str,
    srm_result: dict,
    aa_result: dict,
    conv_result,
    aov_result,
    power_result,
    guardrail_results,
    segment_df,
    cuped_before_var: float,
    cuped_after_var: float,
    ship_checklist: dict,
    out_path: Path = config.OUTPUTS_DIR / "report.md",
) -> Path:
    fig = lambda name: f"figures/{name}.png"
    overall_ship = all(ship_checklist.values())

    guardrail_lines = "\n".join(
        f"| {g.name} | {g.control_value:.3f} | {g.treatment_value:.3f} | {g.diff:+.3f} | "
        f"{g.p_value:.4f} | {'❌ BREACHED' if g.threshold_breached else '✅ OK'} |"
        for g in guardrail_results
    )

    def _fmt_segment_row(r) -> str:
        is_rate = "review score" not in r.segment_dimension.lower()
        if is_rate:
            ctrl, treat, lift = f"{r.rate_control:.2%}", f"{r.rate_treatment:.2%}", f"{r.absolute_lift:+.2%}"
        else:
            ctrl, treat, lift = f"{r.rate_control:.2f}★", f"{r.rate_treatment:.2f}★", f"{r.absolute_lift:+.3f}"
        return (f"| {r.segment_dimension} | {r.segment_value} | {ctrl} | {treat} | "
                f"{lift} | {r.p_value:.4f} | {'Yes' if r.significant else 'No'} |")

    top_segments = segment_df.sort_values("absolute_lift", ascending=False).head(5)
    bottom_segments = segment_df.sort_values("absolute_lift", ascending=True).head(3)
    segment_lines_top = "\n".join(_fmt_segment_row(r) for r in top_segments.itertuples())
    segment_lines_bottom = "\n".join(_fmt_segment_row(r) for r in bottom_segments.itertuples())

    checklist_lines = "\n".join(
        f"- {'✅' if v else '❌'} {k}" for k, v in ship_checklist.items()
    )

    report = f"""# Checkout Redesign A/B Test — Business Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Data source:** {"Real Olist dataset" if data_source == "real" else "Synthetic Olist-schema dataset (no real experiment flag exists in the public data — see methodology note)"}

---

## 1. Executive Summary

We tested a redesigned checkout flow against the existing ("control") checkout on a
50/50 randomized split of Olist customers. The goal: **increase order conversion
without hurting average order value (AOV) or degrading delivery, cancellation,
payment, or satisfaction guardrails.**

**Headline result:** conversion moved from **{conv_result.rate_control:.2%}** (control) to
**{conv_result.rate_treatment:.2%}** (treatment) — an absolute lift of
**{conv_result.absolute_lift:+.2%}** ({conv_result.relative_lift_pct:+.1f}% relative),
which is **{"statistically significant" if conv_result.significant else "NOT statistically significant"}**
(p = {conv_result.chi2_p_value:.5f}).

**Recommendation: {"🚀 SHIP the new checkout." if overall_ship else "🛑 DO NOT SHIP yet — see open issues below."}**

---

## 2. Experiment Setup & Validity Checks

Before trusting any result, we validated the experiment mechanics themselves:

- **Randomization unit:** customer (user-level), so repeat purchases from the same
  customer never split across arms.
- **Sample Ratio Mismatch (SRM) check:** observed split was
  {srm_result['n_control']:,} control / {srm_result['n_treatment']:,} treatment
  (expected {1 - srm_result['expected_ratio_treatment']:.0%}/{srm_result['expected_ratio_treatment']:.0%}),
  χ² p = {srm_result['p_value']:.4f} → **{srm_result['verdict']}**.
- **A/A test:** {aa_result['n_simulations']:,} simulated null experiments produced a
  false-positive rate of {aa_result['observed_false_positive_rate']:.1%}
  (expected ≈{aa_result['expected_false_positive_rate']:.0%}) → **{aa_result['verdict']}**.

![Setup Validation]({fig('00_setup_validation')})

---

## 3. Primary Metric: Conversion Rate

| | Control | Treatment |
|---|---|---|
| Orders | {conv_result.n_control:,} | {conv_result.n_treatment:,} |
| Conversions | {conv_result.conversions_control:,} | {conv_result.conversions_treatment:,} |
| Rate | {conv_result.rate_control:.2%} | {conv_result.rate_treatment:.2%} |
| 95% CI | [{conv_result.ci_low_control:.2%}, {conv_result.ci_high_control:.2%}] | [{conv_result.ci_low_treatment:.2%}, {conv_result.ci_high_treatment:.2%}] |

- **Absolute lift:** {conv_result.absolute_lift:+.2%} (95% CI: [{conv_result.ci_low_diff:+.2%}, {conv_result.ci_high_diff:+.2%}])
- **Relative lift:** {conv_result.relative_lift_pct:+.1f}%
- **Chi-square test:** χ² = {conv_result.chi2_statistic:.2f}, p = {conv_result.chi2_p_value:.5f}
- **Two-proportion z-test:** z = {conv_result.z_statistic:.2f}, p = {conv_result.z_p_value:.5f}
- **Cramér's V (effect size):** {conv_result.cramers_v:.4f}

![Conversion Comparison]({fig('01_conversion_comparison')})

---

## 4. Secondary Metric: Average Order Value (AOV)

| | Control | Treatment |
|---|---|---|
| Mean (BRL) | {aov_result.mean_control:.2f} | {aov_result.mean_treatment:.2f} |
| Std Dev | {aov_result.std_control:.2f} | {aov_result.std_treatment:.2f} |

- **Difference:** {aov_result.absolute_diff:+.2f} BRL ({aov_result.relative_diff_pct:+.1f}%)
- **Welch's t-test:** t = {aov_result.t_statistic:.2f}, p = {aov_result.p_value:.5f}
  → **{"significant change" if aov_result.significant else "no significant change"}**
- **Cohen's d (effect size):** {aov_result.cohens_d:.4f}

![AOV Distribution]({fig('02_aov_distribution')})

**Why this matters:** a conversion lift that comes at the cost of a shrinking basket
could be revenue-neutral or negative. Here, AOV is {"NOT" if not aov_result.significant else ""}
significantly affected, meaning the conversion gain looks like genuine incremental
revenue rather than customers buying less per order.

---

## 5. Guardrail Metrics

| Metric | Control | Treatment | Diff | p-value | Verdict |
|---|---|---|---|---|---|
{guardrail_lines}

![Guardrails]({fig('03_guardrails')})

---

## 6. Statistical Power & Sample Size

- Baseline conversion assumed: {power_result['baseline_rate']:.0%}
- Minimum Detectable Effect (MDE) targeted: {power_result['mde_absolute']:.1%} absolute
- Required sample size to reach {power_result['target_power']:.0%} power at α={power_result['alpha']}:
  **{power_result['required_n_per_arm']:,} per arm** ({power_result['required_n_total']:,} total)
- Actual sample collected: {conv_result.n_control + conv_result.n_treatment:,} orders
  ({"✅ sufficiently powered" if (conv_result.n_control + conv_result.n_treatment) >= power_result['required_n_total'] else "⚠️ underpowered relative to plan"})

![Power Curve]({fig('04_power_curve')})

---

## 7. Effect Sizes at a Glance

![Effect Size Forest Plot]({fig('05_effect_size_forest')})

---

## 8. CUPED Variance Reduction

Using a pre-experiment covariate (order item count) to reduce AOV variance:

- Variance before CUPED: {cuped_before_var:.3f}
- Variance after CUPED: {cuped_after_var:.3f}
- Variance reduction: **{(1 - cuped_after_var / cuped_before_var) * 100:.1f}%**

Lower variance means the same sample size detects smaller true effects — i.e. more
statistical power for free, without collecting additional data.

![CUPED Variance Reduction]({fig('07_cuped_variance_reduction')})

---

## 9. Segment Deep-Dives (Step 10 Follow-ups)

We checked whether the effect is uniform across customer tiers, states, product
categories, and delivery speed — partly to look for **Simpson's paradox** (where an
aggregate effect reverses or disappears within subgroups).

**Top 5 segments by lift:**

| Dimension | Segment | Control Rate | Treatment Rate | Lift | p-value | Significant |
|---|---|---|---|---|---|---|
{segment_lines_top}

**Weakest / negative segments:**

| Dimension | Segment | Control Rate | Treatment Rate | Lift | p-value | Significant |
|---|---|---|---|---|---|---|
{segment_lines_bottom}

![Segment Lift]({fig('06_segment_lift')})

---

## 10. Ship / Don't Ship Decision

{checklist_lines}

![Ship Decision]({fig('08_ship_decision')})

### Final Recommendation

{"**Ship the new checkout flow.** The conversion lift is statistically and practically significant, AOV is not harmed, and all guardrails hold." if overall_ship else "**Hold off on shipping.** At least one guardrail or significance check failed — investigate before rolling out to 100% of traffic. Consider a staged rollout (10% → 50%) with continued monitoring of the metric(s) that failed."}

---

## Methodology Note

The public Olist dataset has no real experiment/treatment flag. Per the project
brief, we simulated a checkout A/B test by (1) randomizing customers into
control/treatment, and (2) injecting a known ground-truth conversion lift into the
treatment arm so the full statistical pipeline (SRM, A/A, significance tests,
power analysis, CUPED, segment analysis) could be exercised and validated
end-to-end exactly as it would be on a real experimentation platform (Amazon,
Microsoft, Uber, Booking.com-style). Swap in real experiment data by replacing
the `experiment_design.inject_ground_truth_effects` step with an actual
`group` column and everything downstream is unchanged.
"""
    out_path.write_text(report, encoding="utf-8")
    return out_path
