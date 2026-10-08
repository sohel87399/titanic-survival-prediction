"""
Central configuration for Titanic ML Playground.
All magic numbers, paths, and default values live here.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
TITANIC_LOCAL = DATA_DIR / "titanic.csv"
MODELS_DIR = ROOT_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Target & ID columns ────────────────────────────────────────────────────────
TARGET_COL = "Survived"
ID_COL = "PassengerId"
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

# ── Feature lists ──────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "FarePerPerson",
]
CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "AgeBin",
    "IsAlone",
    "HasCabin",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Engineered feature toggles (display names → internal names) ────────────────
ENGINEERED_FEATURES = {
    "Title": "Title",
    "FamilySize": "FamilySize",
    "IsAlone": "IsAlone",
    "FarePerPerson": "FarePerPerson",
    "AgeBin": "AgeBin",
    "HasCabin": "HasCabin",
}

# ── Class labels ───────────────────────────────────────────────────────────────
CLASS_NAMES = ["Not Survived", "Survived"]

# ── Default train/test split ───────────────────────────────────────────────────
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

# ── Cross-validation ───────────────────────────────────────────────────────────
DEFAULT_CV_FOLDS = 5

# ── Colour palette (Plotly-friendly) ──────────────────────────────────────────
PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
]

MODEL_COLORS = {
    "Logistic Regression": PALETTE[0],
    "K-Nearest Neighbors": PALETTE[1],
    "Decision Tree": PALETTE[2],
    "Random Forest": PALETTE[3],
    "Gradient Boosting": PALETTE[4],
    "XGBoost": PALETTE[5],
    "SVM": PALETTE[6],
    "Neural Network": PALETTE[7],
    "Voting Ensemble": "#7f7f7f",
    "Stacking Ensemble": "#bcbd22",
}

# ── Metric display names ───────────────────────────────────────────────────────
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

# ── AgeBin labels ──────────────────────────────────────────────────────────────
AGE_BINS = [0, 12, 18, 35, 60, 100]
AGE_LABELS = ["Child", "Teen", "Adult", "Middle-aged", "Senior"]

# ── Title groups ──────────────────────────────────────────────────────────────
TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",
    "Countess": "Rare",
    "Ms": "Miss",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Mme": "Mrs",
    "Capt": "Rare",
    "Sir": "Rare",
}

# ── Hyperparameter defaults ────────────────────────────────────────────────────
HP_DEFAULTS: dict = {
    "Logistic Regression": {
        "C": 1.0,
        "max_iter": 200,
        "penalty": "l2",
        "solver": "lbfgs",
    },
    "K-Nearest Neighbors": {
        "n_neighbors": 5,
        "weights": "uniform",
        "metric": "minkowski",
    },
    "Decision Tree": {
        "max_depth": 5,
        "min_samples_split": 2,
        "criterion": "gini",
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
    "XGBoost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "subsample": 0.8,
    },
    "SVM": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
    },
    "Neural Network": {
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "learning_rate_init": 0.001,
        "max_iter": 200,
    },
}
