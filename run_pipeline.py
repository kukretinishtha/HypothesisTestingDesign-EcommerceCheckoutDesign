#!/usr/bin/env python3
"""
End-to-end orchestrator for the Olist Checkout A/B Testing pipeline.

Usage:
    python run_pipeline.py

Produces:
    outputs/figures/*.png   - all business-storytelling charts
    outputs/tables/*.csv    - all numeric results, ready for BI tools / slides
    outputs/report.md       - narrative business report stitching everything together
    logs/pipeline.log       - full run log
"""
from __future__ import annotations
import json
import sys
import traceback
import pandas as pd

import config
from src.logging_utils import get_logger
from src import data_loader, experiment_design, stats_tests, guardrails, segment_analysis
from src import visualization, report_generator

logger = get_logger("pipeline", config.LOGS_DIR)


def save_table(df_or_dict, name: str) -> None:
    path = config.TABLES_DIR / f"{name}.csv"
    if isinstance(df_or_dict, dict):
        pd.DataFrame([df_or_dict]).to_csv(path, index=False)
    elif isinstance(df_or_dict, list):
        pd.DataFrame([r.__dict__ if hasattr(r, "__dict__") else r for r in df_or_dict]).to_csv(path, index=False)
    elif hasattr(df_or_dict, "__dict__") and not isinstance(df_or_dict, pd.DataFrame):
        pd.DataFrame([df_or_dict.__dict__]).to_csv(path, index=False)
    else:
        df_or_dict.to_csv(path, index=False)
    logger.info(f"Saved table: {path.name}")


def main() -> int:
    logger.info("=" * 70)
    logger.info("STARTING OLIST CHECKOUT A/B TESTING PIPELINE")
    logger.info("=" * 70)

    try:
        # ------------------------------------------------------------------
        # Step 1-3: Load data, build analysis table
        # ------------------------------------------------------------------
        raw_tables = data_loader.load_raw_tables()
        source = raw_tables["_source"]
        df = data_loader.build_analysis_table(raw_tables)

        # ------------------------------------------------------------------
        # Step 1 (design): user-level randomization
        # ------------------------------------------------------------------
        df = experiment_design.assign_user_level_treatment(df)

        # ------------------------------------------------------------------
        # Validity checks: SRM + A/A test (on payment_value as a neutral,
        # pre-treatment-independent continuous metric)
        # ------------------------------------------------------------------
        srm_result = experiment_design.srm_check(df)
        aa_result = experiment_design.aa_test(df, metric_col="payment_value", group_col="group")
        save_table(srm_result, "00_srm_check")
        save_table(aa_result, "00_aa_test")

        # ------------------------------------------------------------------
        # Step 2: inject ground-truth treatment effect (simulation layer,
        # since Olist has no real experiment flag)
        # ------------------------------------------------------------------
        df = experiment_design.inject_ground_truth_effects(df)

        # ------------------------------------------------------------------
        # Step 4: Primary metric - conversion
        # ------------------------------------------------------------------
        conv_result = stats_tests.test_conversion(df)
        save_table(conv_result, "01_conversion_test")

        # ------------------------------------------------------------------
        # Step 5: Secondary metric - AOV
        # ------------------------------------------------------------------
        aov_result = stats_tests.test_continuous_metric(df, "payment_value")
        save_table(aov_result, "02_aov_test")

        # ------------------------------------------------------------------
        # Step 6: Guardrail metrics
        # ------------------------------------------------------------------
        guardrail_results = guardrails.evaluate_guardrails(df)
        save_table(guardrail_results, "03_guardrails")

        # ------------------------------------------------------------------
        # Step 7: Power analysis
        # ------------------------------------------------------------------
        power_result = stats_tests.required_sample_size(
            baseline_rate=conv_result.rate_control, mde_absolute=config.MDE_ABSOLUTE
        )
        save_table(power_result, "04_power_analysis")

        # ------------------------------------------------------------------
        # Step 8: CUPED variance reduction on AOV using n_items as covariate
        # ------------------------------------------------------------------
        cuped_series = stats_tests.cuped_adjust(df, "payment_value", "n_items")
        df["payment_value_cuped"] = cuped_series
        cuped_before_var = df["payment_value"].var()
        cuped_after_var = df["payment_value_cuped"].var()
        save_table({
            "variance_before": cuped_before_var,
            "variance_after": cuped_after_var,
            "pct_reduction": (1 - cuped_after_var / cuped_before_var) * 100,
        }, "05_cuped_variance_reduction")

        # ------------------------------------------------------------------
        # Step 10: Segment / follow-up analyses
        # ------------------------------------------------------------------
        segment_df = segment_analysis.run_all_segments(df)
        save_table(segment_df, "06_segment_analysis")

        # ------------------------------------------------------------------
        # Step 9: Ship / Don't Ship decision framework
        # ------------------------------------------------------------------
        n_total = conv_result.n_control + conv_result.n_treatment
        ship_checklist = {
            "Conversion lift is statistically significant": conv_result.significant,
            "Conversion lift is practically meaningful (>= 1pp absolute)": conv_result.absolute_lift >= 0.01,
            "AOV is not significantly worse": not (aov_result.significant and aov_result.absolute_diff < 0),
            "No guardrail metric breached": not any(g.threshold_breached for g in guardrail_results),
            "Experiment adequately powered (sample >= required)": n_total >= power_result["required_n_total"],
            "No Sample Ratio Mismatch detected": not srm_result["srm_detected"],
        }
        save_table(ship_checklist, "07_ship_decision_checklist")

        with open(config.TABLES_DIR / "ship_decision_checklist.json", "w") as f:
            json.dump(ship_checklist, f, indent=2)

        # ------------------------------------------------------------------
        # Visualizations
        # ------------------------------------------------------------------
        visualization.plot_srm_and_aa(srm_result, aa_result)
        visualization.plot_conversion_comparison(conv_result)
        visualization.plot_aov_distribution(df, aov_result)
        visualization.plot_guardrails(guardrail_results)
        visualization.plot_power_curve(conv_result.rate_control, config.MDE_ABSOLUTE)
        visualization.plot_effect_size_forest(conv_result, aov_result)
        visualization.plot_segment_results(segment_df)
        visualization.plot_cuped_variance_reduction(cuped_before_var, cuped_after_var)
        visualization.plot_ship_decision(ship_checklist)

        # ------------------------------------------------------------------
        # Business report
        # ------------------------------------------------------------------
        report_path = report_generator.generate_report(
            data_source=source,
            srm_result=srm_result,
            aa_result=aa_result,
            conv_result=conv_result,
            aov_result=aov_result,
            power_result=power_result,
            guardrail_results=guardrail_results,
            segment_df=segment_df,
            cuped_before_var=cuped_before_var,
            cuped_after_var=cuped_after_var,
            ship_checklist=ship_checklist,
        )

        # ------------------------------------------------------------------
        # Persist the full analysis table for downstream BI / notebook use
        # ------------------------------------------------------------------
        analysis_table_path = config.TABLES_DIR / "full_analysis_table.csv"
        df.to_csv(analysis_table_path, index=False)
        logger.info(f"Saved full analysis table: {analysis_table_path.name} ({len(df):,} rows)")

        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETE. Report: {report_path}")
        logger.info("=" * 70)
        return 0

    except Exception:
        logger.error("Pipeline failed with an unhandled exception:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
