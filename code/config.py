"""Все зафиксированные константы протокола UA-Mondrian (protocol/План статьи.docx).

Никаких «магических» чисел вне этого модуля: параметры, сетки, сиды,
роли атрибутов, иерархии и гиперпараметры моделей задаются только здесь.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("UA_DATA_DIR", ROOT_DIR / "data"))
LOG_DIR = CODE_DIR / "logs"
RESULT_DIR = CODE_DIR / "results"
FIG_DIR = CODE_DIR / "figures"

# --- протокол ---------------------------------------------------------------
K_GRID = (5, 10, 25, 50, 100)
ELL = 2
ALPHA = 0.5
BETA = 0.5

REPEATS = (0, 1)
FOLDS = (1, 2, 3, 4, 5)
SPLIT_SEED_BASE = 20261029
CLF_SEED_BASE = 42


def split_seed(repeat: int, fold: int) -> int:
    """Seed разбиения: 20261029 + 1000*repeat + fold (протокол 2.4)."""
    return SPLIT_SEED_BASE + 1000 * repeat + fold


def clf_seed(repeat: int, fold: int) -> int:
    """Seed классификатора: 42 + 1000*repeat + fold (одинаков в паре методов)."""
    return CLF_SEED_BASE + 1000 * repeat + fold

TEST_FRACTION = 0.2
TRAIN_FRACTION = 0.8
NB_NOISE_RATIO = TEST_FRACTION / TRAIN_FRACTION  # n_test/n_train = 1/4
NB_SE_FACTOR = 0.1 + NB_NOISE_RATIO               # 1/10 + 1/4 = 0.35
H1_DELTA0 = 0.01
H1_ALPHA = 0.05
H1_DF = 9

# --- модели ----------------------------------------------------------------
RF_PARAMS = {"n_estimators": 200, "random_state": None, "n_jobs": -1}
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": None,
    "n_jobs": 4,
    "tree_method": "hist",
    "verbosity": 0,
}

# --- метки классов (порядок зафиксирован) ----------------------------------
ADULT_LABELS = ["<=50K", ">50K"]
ADULT_POSITIVE = ">50K"
BANK_LABELS = ["no", "yes"]
BANK_POSITIVE = "yes"

# --- наборы данных ---------------------------------------------------------
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

ADULT_QI = [
    "age", "workclass", "education", "marital-status", "relationship",
    "race", "sex", "hours-per-week", "native-country",
]
ADULT_NUMERIC = ["age", "hours-per-week"]
ADULT_SA = "occupation"
ADULT_Y = "income"
ADULT_P_AUX = ["education-num", "capital-gain", "capital-loss"]

BANK_QI = ["age", "job", "marital", "education", "default", "housing", "loan"]
BANK_NUMERIC = ["age"]
BANK_SA = "balance"
BANK_Y = "y"
BANK_P_AUX = [
    "contact", "day", "month", "duration", "campaign",
    "pdays", "previous", "poutcome",
]
BANK_Q4 = 4

CMC_COLUMNS = [
    "wife_age", "wife_edu", "husband_edu", "num_children", "wife_religion",
    "wife_working", "husband_occ", "sol_index", "media_exposure",
    "contraceptive_method",
]
CMC_A_QI = [
    "wife_age", "wife_edu", "husband_edu", "num_children", "wife_religion",
    "wife_working", "husband_occ", "sol_index",
]
CMC_A_SA = "media_exposure"
CMC_B_QI = [
    "wife_age", "wife_edu", "husband_edu", "num_children", "media_exposure",
    "wife_working", "husband_occ", "sol_index",
]
CMC_B_SA = "wife_religion"
CMC_Y = "contraceptive_method"

# --- R_rep: экспертные веса воспроизводимости (Low 0.2 / Medium 0.5 / High 0.8)
R_REP_ADULT = {
    "age": 0.8,
    "workclass": 0.5,
    "education": 0.8,
    "marital-status": 0.8,
    "occupation": 0.5,
    "relationship": 0.5,
    "race": 0.5,
    "sex": 0.8,
    "hours-per-week": 0.2,
    "native-country": 0.5,
}
R_REP_BANK = {
    "age": 0.8,
    "job": 0.5,
    "marital": 0.8,
    "education": 0.8,
    "default": 0.2,
    "housing": 0.5,
    "loan": 0.5,
}

# --- иерархии категориальных QI -------------------------------------------
# Спецификация: список групп верхнего уровня; группа = (имя, [значения])
# либо (имя, [подгруппы]) для двухуровневой иерархии.
ADULT_HIER = {
    "workclass": [
        ("private_self", ["Private", "Self-emp-not-inc", "Self-emp-inc"]),
        ("government", ["Federal-gov", "Local-gov", "State-gov"]),
        ("unpaid", ["Without-pay", "Never-worked"]),
    ],
    "education": [
        ("primary", ["Preschool", "1st-4th", "5th-6th", "7th-8th"]),
        ("secondary", ["9th", "10th", "11th", "12th", "HS-grad"]),
        ("postsecondary", ["Some-college", "Assoc-voc", "Assoc-acdm"]),
        ("tertiary", ["Bachelors", "Masters", "Prof-school", "Doctorate"]),
    ],
    "marital-status": [
        ("married", ["Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"]),
        ("formerly_married", ["Divorced", "Separated", "Widowed"]),
        ("never_married", ["Never-married"]),
    ],
    "relationship": [
        ("spouse", ["Wife", "Husband"]),
        ("child", ["Own-child"]),
        ("non_family", ["Not-in-family", "Other-relative", "Unmarried"]),
    ],
    "race": [
        ("white", ["White"]),
        ("non_white", ["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]),
    ],
    "sex": [
        ("male", ["Male"]),
        ("female", ["Female"]),
    ],
    "native-country": [
        ("north_america", [
            "United-States", "Puerto-Rico", "Canada", "Outlying-US(Guam-USVI-etc)",
            "Cuba", "Honduras", "Jamaica", "Mexico", "Dominican-Republic",
            "Haiti", "Guatemala", "Nicaragua", "El-Salvador", "Trinadad&Tobago",
        ]),
        ("south_america", ["Ecuador", "Columbia", "Peru"]),
        ("europe", [
            "England", "Germany", "Greece", "Italy", "Poland", "Portugal",
            "Ireland", "France", "Hungary", "Scotland", "Yugoslavia",
            "Holand-Netherlands",
        ]),
        ("asia", [
            "Cambodia", "India", "Japan", "South", "China", "Iran",
            "Philippines", "Vietnam", "Laos", "Taiwan", "Thailand", "Hong-Kong",
        ]),
    ],
}

BANK_HIER = {
    "job": [
        ("white_collar", ["admin.", "management", "entrepreneur", "self-employed"]),
        ("blue_collar", ["blue-collar", "technician", "services", "housemaid"]),
        ("inactive", ["unemployed", "retired", "student"]),
        ("unknown", ["unknown"]),
    ],
    "marital": [
        ("married", ["married"]),
        ("not_married", ["divorced", "single"]),
    ],
    "education": [
        ("known", ["primary", "secondary", "tertiary"]),
        ("unknown_level", ["unknown"]),
    ],
    "default": [("no", ["no"]), ("yes", ["yes"])],
    "housing": [("no", ["no"]), ("yes", ["yes"])],
    "loan": [("no", ["no"]), ("yes", ["yes"])],
}

# --- методы ---------------------------------------------------------------
METHODS_MAIN = [
    "B2", "B4", "B4m", "A1", "A2", "UA-Global", "UA-Local",
]
METHODS_ML_ONLY = ["Raw", "B1"]

# идентификаторы методов в журналах: B4 и B4m обязаны быть различимы;
# значения записаны в журналы и results/pilot_summary.json — не менять
JOURNAL_IDS = {
    "Raw": "J-Raw-v1.0",
    "B1": "J-ZeroRule-v1.0",
    "B2": "J-B2-Mondrian-v1.0",
    "B4": "J-B4-InfoGainMondrian-v1.0",
    "B4m": "J-B4m-MatchedIG-UAframework-v1.0",
    "A1": "J-A1-UtilityOnly-v1.0",
    "A2": "J-A2-VulnerabilityOnly-v1.0",
    "UA-Global": "J-UA-Global-v1.0",
    "UA-Local": "J-UA-Local-v1.0",
}

# какие методы считают приватность/атаки (Raw/B1 — только ML utility)
PRIVACY_METHODS = set(METHODS_MAIN)


def set_xgboost_runtime_env() -> None:
    """На macOS xgboost требует libomp из Homebrew (keg-only)."""
    extra = "/opt/homebrew/opt/libomp/lib"
    cur = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if extra not in cur.split(":"):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{extra}:{cur}" if cur else extra
