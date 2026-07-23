# Olist Checkout A/B Testing Pipeline

Production-grade, end-to-end experimentation pipeline for the "new checkout
increases conversion without hurting AOV" hypothesis, built on the Olist
Brazilian E-commerce dataset.

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py
```

That's it. One command runs the entire pipeline and regenerates every table,
chart, and the business report from scratch.

## Using the REAL Olist dataset instead of synthetic data

This environment couldn't reach Kaggle to download the real CSVs, so the
pipeline ships with a synthetic data generator (`src/data_generation.py`)
that reproduces the **exact same schema** as the real dataset, so everything
downstream (loading, joining, testing, plotting) works identically either way.

To use the real data:
1. Download the dataset from Kaggle: `olistbr/brazilian-ecommerce`
2. Place these 4 files in `data/raw/` (exact filenames):
   - `olist_customers_dataset.csv`
   - `olist_orders_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_order_payments_dataset.csv`
3. Run `python run_pipeline.py` again — the loader auto-detects real files
   in `data/raw/` and prioritizes them over the synthetic cache.

Note: the real dataset has no experiment/treatment flag, so
`src/experiment_design.py` still randomizes customers into control/treatment
and injects a known ground-truth conversion lift — this is documented
in the generated report's "Methodology Note" section and is the intended
design per the project brief (Olist has no real A/B test to recover).

## What gets produced

```
outputs/
├── report.md                 <- business-facing narrative report (start here)
├── figures/                  <- 9 PNG charts, all referenced in report.md
│   ├── 00_setup_validation.png     (SRM check + A/A test)
│   ├── 01_conversion_comparison.png
│   ├── 02_aov_distribution.png
│   ├── 03_guardrails.png
│   ├── 04_power_curve.png
│   ├── 05_effect_size_forest.png
│   ├── 06_segment_lift.png
│   ├── 07_cuped_variance_reduction.png
│   └── 08_ship_decision.png
└── tables/                   <- every numeric result as CSV, for BI tools / slides
    ├── 00_srm_check.csv
    ├── 00_aa_test.csv
    ├── 01_conversion_test.csv
    ├── 02_aov_test.csv
    ├── 03_guardrails.csv
    ├── 04_power_analysis.csv
    ├── 05_cuped_variance_reduction.csv
    ├── 06_segment_analysis.csv
    ├── 07_ship_decision_checklist.csv
    └── full_analysis_table.csv   <- full order-level table with group assignment, all metrics
```

`logs/pipeline.log` has the full run log (every step, every statistic, every
save) for auditability.

## Architecture

```
config.py                  <- every tunable parameter (seed, alpha, MDE, thresholds...)
run_pipeline.py             <- orchestrator; run this
src/
├── logging_utils.py        <- shared console + file logging
├── data_generation.py      <- synthetic Olist-schema data (fallback only)
├── data_loader.py          <- real-data-first loader, builds the analysis table
├── experiment_design.py    <- user-level randomization, SRM check, A/A test,
│                              ground-truth effect injection
├── stats_tests.py          <- chi-square, two-proportion z-test, Welch t-test,
│                              Cohen's d, Cramér's V, confidence intervals,
│                              power analysis, CUPED
├── guardrails.py           <- delivery time, cancellation, payment failure, review score
├── segment_analysis.py     <- premium/state/category/delivery-speed cuts,
│                              Simpson's paradox check
├── visualization.py        <- all chart generation (saved to outputs/figures/)
└── report_generator.py     <- stitches everything into outputs/report.md
```

Every module is independently testable and importable — e.g. you can run
just `src/stats_tests.py`'s functions in a notebook against your own
DataFrame if you want to extend the analysis.

## Design decisions worth knowing about

- **Randomization is user-level**, not order-level, so a repeat customer's
  orders never get split across both arms (a common real bug).
- **SRM check** runs before any hypothesis test — if it fails, the pipeline
  still runs (for visibility) but the ship-decision checklist will flag it.
- **A/A test** (1,000 simulated null experiments) validates that the testing
  machinery itself produces a false-positive rate matching alpha before any
  real result is trusted.
- **CUPED** reduces AOV variance using order item count as a pre-experiment
  covariate — this is a real technique used at Booking.com/Microsoft/Netflix
  to shrink required sample sizes.
- **Ship/No-Ship** is a hard, auditable checklist (see `config.py` for
  thresholds), not a single p-value — matches how mature experimentation
  platforms gate rollout decisions.
