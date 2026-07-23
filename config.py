"""
Central configuration for the Olist Checkout A/B Testing pipeline.

All tunable parameters live here so the rest of the codebase never hardcodes
magic numbers. Change assumptions in ONE place.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"              # place REAL Olist CSVs here
DATA_SYNTHETIC_DIR = ROOT_DIR / "data" / "synthetic_cache"  # auto-generated fallback data lives here, never mixed with real data
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = ROOT_DIR / "logs"

for _d in (DATA_RAW_DIR, DATA_SYNTHETIC_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Synthetic data generation (used only if real Olist CSVs are absent)
# --------------------------------------------------------------------------
N_ORDERS = 60_000
N_CUSTOMERS = 45_000
STATES = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "GO", "ES"]
STATE_WEIGHTS = [0.42, 0.13, 0.12, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03, 0.03]
CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
    "computers_accessories", "housewares", "watches_gifts", "telephony",
    "auto", "toys", "electronics", "fashion_bags_accessories",
]

# --------------------------------------------------------------------------
# Experiment design assumptions
# --------------------------------------------------------------------------
TREATMENT_ALLOCATION = 0.5          # planned 50/50 split
BASELINE_CONVERSION = 0.78          # "delivered" rate in control (proxy for checkout completion)
TRUE_TREATMENT_LIFT_ABS = 0.06      # ground-truth absolute lift injected into treatment (simulation only)
AOV_TREATMENT_SHIFT = 0.0           # treatment does NOT move AOV in the simulated ground truth (null effect)
AOV_NOISE_SD_MULTIPLIER = 1.0

PREMIUM_THRESHOLD_BRL = 500.0       # premium customer segment cutoff used in step 10 segmentation

# --------------------------------------------------------------------------
# Statistical parameters
# --------------------------------------------------------------------------
ALPHA = 0.05
TARGET_POWER = 0.80
MDE_ABSOLUTE = 0.02                 # minimum detectable effect (2 pp) used for power analysis
SRM_ALPHA = 0.001                   # stricter threshold for Sample Ratio Mismatch checks (convention)

# --------------------------------------------------------------------------
# Guardrail metric bounds (business-defined "do not regress" thresholds)
# --------------------------------------------------------------------------
MAX_ACCEPTABLE_DELIVERY_DAYS_INCREASE = 0.5     # days
MAX_ACCEPTABLE_CANCEL_RATE_INCREASE = 0.01      # 1 pp
MAX_ACCEPTABLE_PAYMENT_FAILURE_INCREASE = 0.01  # 1 pp
MIN_ACCEPTABLE_REVIEW_SCORE_DROP = -0.1         # stars

@dataclass(frozen=True)
class PipelineConfig:
    random_seed: int = RANDOM_SEED
    alpha: float = ALPHA
    target_power: float = TARGET_POWER
    mde_absolute: float = MDE_ABSOLUTE
    treatment_allocation: float = TREATMENT_ALLOCATION

CFG = PipelineConfig()
