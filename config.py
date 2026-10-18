"""Configuration used by the minimal point-query experiments."""

###############################################################################
# Reproducibility and scheduler output
###############################################################################
SEED = 62
SHOW_SCHEDULER_PROGRESS = False

###############################################################################
# Available local datasets
###############################################################################
DATASET = "adult"
DATASET_SPECS = {
    "adult": {
        "source": "local",
        "filename": "adult12b.txt",
        "attributes": ("a", "d", "f"),
        "attribute_domains": (8, 8, 8),
        "n_records": 32_562,
    },
}
_DATASET_KEY = DATASET.lower()
if _DATASET_KEY not in DATASET_SPECS:
    raise ValueError(f"Unknown DATASET {DATASET!r}; choose from {tuple(DATASET_SPECS)}.")
_DATASET_SPEC = DATASET_SPECS[_DATASET_KEY]
DATASET_N_RECORDS = {
    name: int(spec["n_records"]) for name, spec in DATASET_SPECS.items()
}
N_RECORDS = 30_000
DP_RATIO = 0.1
DROP_NA = True
REMOVE_UNKNOWN = True
UNKNOWN_VALUE = "?"
ATTRIBUTES = list(_DATASET_SPEC["attributes"])
ATTRIBUTE_DOMAINS = list(_DATASET_SPEC["attribute_domains"])
DOMAIN_SIZE = 1
for _domain in ATTRIBUTE_DOMAINS:
    DOMAIN_SIZE *= _domain

###############################################################################
# Point-query microbatches
###############################################################################
VARIABLE_MICROBATCH_SIZE = True
MICROBATCH_SIZE = 20
MICROBATCH_SIZE_MIN = 20
MICROBATCH_SIZE_MAX = 50

###############################################################################
# Privacy and scheduler defaults
###############################################################################
EPS_DP = 1.0
EPS_LDP = 1.0
OUE_CLIP_NEGATIVE_ESTIMATES = False
CP = 2000
BETA = 0.1
EPSILON_TOL = 1e-10
SMART_TARGET_VARIANCE = 120_000.0
SMART_MIN_EPSILON = 0.01
BUDGET_POLICY = "hdmm_flat"
N_RUNS = 10
VERBOSE = True
