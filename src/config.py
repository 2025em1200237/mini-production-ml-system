"""Project-wide constants. Everything that is a modelling decision lives here
so the reports, the training code and the serving code cannot drift apart."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# The raw PaySim file is ~470 MB and is not shipped with the repo.
# Put it in ./data/paysim.csv (a sibling ../data/paysim.csv also works).
DATA_CANDIDATES = [ROOT / "data" / "paysim.csv", ROOT.parent / "data" / "paysim.csv"]
SAMPLE_PATH = ROOT / "data" / "sample_transactions.csv"

RANDOM_STATE = 42

# ---- problem scope ---------------------------------------------------------
# In PaySim, fraud only ever happens on TRANSFER and CASH_OUT (checked in EDA).
SCOPE_TYPES = ("TRANSFER", "CASH_OUT")
OUT_OF_SCOPE_TYPES = ("PAYMENT", "CASH_IN", "DEBIT")
KNOWN_TYPES = SCOPE_TYPES + OUT_OF_SCOPE_TYPES

# ---- temporal split (step = 1 hour, 1..743) --------------------------------
TRAIN_END_STEP = 450   # train: step <= 450
VAL_END_STEP = 600     # val: 451..600, test: 601..743

# ---- columns ---------------------------------------------------------------
TARGET = "isFraud"
RAW_FEATURE_COLS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]
BALANCE_COLS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
# Columns deliberately never used as inputs (see the leakage audit in data_pipeline.py).
FORBIDDEN_COLS = {"isFraud", "isFlaggedFraud", "nameOrig", "nameDest"}

# engineered feature groups (built in src/features.py)
#
# Feature availability matters. newbalanceOrig / newbalanceDest are the balances
# AFTER the transaction has been applied, so at authorisation time they do not
# exist yet. In PaySim they also expose how the simulator injects fraud (the
# balance books do not reconcile for fraud rows), which makes the problem
# near-trivial. The serving model therefore uses only "pre_auth" features. A
# "post_settlement" model is trained purely as an ablation to show the effect.
CAT_FEATURES = ["type"]
NUM_ALL = [
    "amount_log", "old_org_log", "old_dest_log", "amount_to_balance", "amount_to_dest_log",
    "hour_of_day",                                           # known before settlement
    "new_org_log", "new_dest_log", "err_orig", "err_dest",   # need post-transaction balances
]
FLAG_ALL = ["org_would_empty", "dest_empty_before",          # known before settlement
            "org_emptied", "dest_unchanged"]                 # need post-transaction balances
ALL_FEATURES = NUM_ALL + FLAG_ALL + CAT_FEATURES             # what add_features() returns

POST_SETTLEMENT_ONLY = ["new_org_log", "new_dest_log", "err_orig", "err_dest",
                        "org_emptied", "dest_unchanged"]
FEATURE_SETS = {
    "pre_auth": {
        "num": [f for f in NUM_ALL if f not in POST_SETTLEMENT_ONLY],
        "flag": [f for f in FLAG_ALL if f not in POST_SETTLEMENT_ONLY],
    },
    "post_settlement": {"num": NUM_ALL, "flag": FLAG_ALL},
    # Hard-mode ablation. In PaySim ~98% of fraud drains the sender's account
    # exactly (amount == oldbalanceOrg) and no legitimate transaction does, so
    # the sender's balance alone nearly solves the task. This set hides it.
    "sender_blind": {
        "num": ["amount_log", "old_dest_log", "amount_to_dest_log", "hour_of_day"],
        "flag": ["dest_empty_before"],
    },
}
# the serving model's inputs (used by monitoring, leakage audit, tests)
NUM_FEATURES = FEATURE_SETS["pre_auth"]["num"]
FLAG_FEATURES = FEATURE_SETS["pre_auth"]["flag"]
MODEL_FEATURES = NUM_FEATURES + FLAG_FEATURES + CAT_FEATURES
# fields a caller must supply for a full-quality score
REQUIRED_FOR_MODEL = ["oldbalanceOrg", "oldbalanceDest"]
ABLATION_FILE = "ablation_post_settlement.joblib"
BLIND_FILE = "ablation_sender_blind.joblib"

# ---- business cost model ---------------------------------------------------
# Missed fraud costs the transaction amount. Every flagged transaction costs an
# analyst review. A flagged legitimate transaction also costs customer friction,
# modelled as a fraction of its amount. These are assumptions, not PaySim facts,
# so evaluate.py reports a sensitivity table over the friction rate.
REVIEW_COST = 25.0
FRICTION_RATE = 0.001
FRICTION_RATES_SENSITIVITY = [0.0005, 0.001, 0.005, 0.01]

# ---- heuristic baseline ----------------------------------------------------
HEURISTIC_MIN_AMOUNT = 200_000.0   # rule: large transfer that would empty the sender's account

# ---- serving ---------------------------------------------------------------
MODEL_FILE = "model_pipeline.joblib"
BASELINE_FILE = "baseline_logreg.joblib"
THRESHOLD_FILE = "threshold.json"
REFERENCE_STATS_FILE = "reference_stats.json"
REFERENCE_SAMPLE_FILE = "reference_sample.csv.gz"

# ---- monitoring thresholds -------------------------------------------------
PSI_WARN = 0.10
PSI_ALERT = 0.25
KS_ALERT = 0.10
