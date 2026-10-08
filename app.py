"""
app.py
------
Titanic ML Playground — main Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

# ── Make sure project root is on path ─────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CLASS_NAMES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    ENGINEERED_FEATURES,
    HP_DEFAULTS,
    METRIC_NAMES,
    MODEL_COLORS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PALETTE,
    TARGET_COL,
)
from src.data import get_missing_summary, load_raw_data, sample_data
from src.features import engineer_features, filter_feature_lists, get_feature_names_out
from src.metrics import (
    build_leaderboard,
    compute_all_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_learning_curve_data,
    get_pr_curve,
    get_roc_curve,
    get_validation_curve_data,
)
from src.models import (
    build_pipeline,
    cross_validate_model,
    gb_staged_scores,
    rf_accuracy_sweep,
    train_model,
    xgb_eval_results,
)
from src.plots import (
    plot_age_distribution,
    plot_all_model_predictions,
    plot_class_balance,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_fare_distribution,
    plot_feature_importance,
    plot_learning_curve,
    plot_metric_comparison,
    plot_missing_values,
    plot_mlp_loss_curve,
    plot_permutation_importance,
    plot_pr_curve,
    plot_roc_curve,
    plot_roc_curves_comparison,
    plot_search_results,
    plot_staged_accuracy,
    plot_survival_by_class,
    plot_survival_by_sex,
    plot_survival_gauge,
    plot_training_time,
    plot_validation_curve,
)

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic ML Playground",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Inter font (OpenAI's typeface) ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, .stText,
    .stButton > button, .stSelectbox, .stMultiSelect,
    .stSlider, .stNumberInput, .stTextInput, .stExpander,
    .stTabs [data-baseweb="tab"], h1, h2, h3, h4, h5, h6, p, div, span {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Global background & surface ── */
    .stApp {
        background: #0d0d0d;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #111111 !important;
        border-right: 1px solid #2a2a2a;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ececec;
        letter-spacing: -0.02em;
    }

    /* ── Dividers ── */
    hr {
        border: none;
        border-top: 1px solid #2a2a2a !important;
        margin: 1rem 0 !important;
    }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #111111;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #2a2a2a;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #888;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 6px 14px;
        transition: all 0.2s ease;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #10a37f !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: #1e1e1e;
        color: #ececec;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* ── Headings ── */
    h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.03em !important; color: #ececec !important; }
    h2 { font-size: 1.4rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; color: #d4d4d4 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; color: #c4c4c4 !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #10a37f;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: #888 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #ececec !important;
        letter-spacing: -0.02em;
    }

    /* ── Custom metric-card class ── */
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        transition: border-color 0.2s, transform 0.15s;
    }
    .metric-card:hover { border-color: #10a37f; transform: translateY(-1px); }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #ececec; letter-spacing: -0.02em; }
    .metric-card .label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
    .best-model { border-color: #10a37f !important; background: #0d2b22 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #10a37f !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 10px 20px !important;
        letter-spacing: 0.01em;
        transition: background 0.2s, transform 0.15s !important;
    }
    .stButton > button:hover {
        background: #0d8a6b !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        background: transparent !important;
        color: #10a37f !important;
        border: 1px solid #10a37f !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stDownloadButton > button:hover {
        background: #0d2b22 !important;
    }

    /* ── Input widgets ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #ececec !important;
        font-size: 0.875rem !important;
    }
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #10a37f !important;
        border-color: #10a37f !important;
    }

    /* ── Dataframes & tables ── */
    .stDataFrame {
        border: 1px solid #2a2a2a !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    .stDataFrame thead th {
        background: #1a1a1a !important;
        color: #888 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 1px solid #2a2a2a !important;
    }
    .stDataFrame tbody tr:hover td {
        background: #1e1e1e !important;
    }

    /* ── Expanders ── */
    .stExpander {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
    }
    .stExpander header {
        font-weight: 600 !important;
        color: #ececec !important;
        font-size: 0.9rem !important;
    }

    /* ── Alerts & info boxes ── */
    .stInfo, [data-testid="stInfo"] {
        background: #0d1f2d !important;
        border: 1px solid #1a4a6b !important;
        border-radius: 10px !important;
        color: #7ab3d4 !important;
    }
    .stSuccess, [data-testid="stSuccess"] {
        background: #0d2b22 !important;
        border: 1px solid #10a37f !important;
        border-radius: 10px !important;
        color: #4fd1a5 !important;
    }
    .stWarning, [data-testid="stWarning"] {
        background: #2b1f0d !important;
        border: 1px solid #a37f10 !important;
        border-radius: 10px !important;
        color: #d1a54f !important;
    }
    .stError, [data-testid="stError"] {
        background: #2b0d0d !important;
        border: 1px solid #a31010 !important;
        border-radius: 10px !important;
        color: #d14f4f !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10a37f, #0d8a6b) !important;
        border-radius: 4px;
    }

    /* ── Plotly chart containers ── */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Code blocks ── */
    .stCodeBlock {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d0d0d; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #10a37f; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (cached) ──────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading dataset…")
def get_raw_data() -> pd.DataFrame:
    """Load and cache the raw Titanic dataset."""
    return load_raw_data()


# ── Session state helpers ──────────────────────────────────────────────────────


def init_session_state() -> None:
    defaults = {
        "trained_pipelines": {},
        "train_metrics": {},
        "cv_results": {},
        "train_times": {},
        "roc_data": {},
        "last_X_test": None,
        "last_y_test": None,
        "last_X_train": None,
        "last_y_train": None,
        "feature_names": [],
        "tuning_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/320px-RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🚢 Titanic ML Playground")
    st.markdown("---")

    # ── Model selection ────────────────────────────────────────────────────────
    st.subheader("📋 Models")
    MODEL_LIST = [
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "SVM",
        "Neural Network",
    ]
    selected_models = st.multiselect(
        "Select algorithms",
        MODEL_LIST,
        default=["Logistic Regression", "Random Forest"],
        help="Choose which models to train.",
    )

    use_ensemble = st.checkbox("Add Voting Ensemble", value=False, help="Train a soft-voting ensemble of the selected models.")
    ensemble_type = "Voting"
    if use_ensemble:
        ensemble_type = st.radio("Ensemble type", ["Voting", "Stacking"], horizontal=True)

    st.markdown("---")

    # ── Data controls ──────────────────────────────────────────────────────────
    st.subheader("📊 Data")
    raw_df = get_raw_data()
    n_max = len(raw_df)

    n_records = st.slider(
        "Number of records",
        min_value=50,
        max_value=n_max,
        value=n_max,
        step=50,
        help="Randomly sample this many rows from the dataset.",
    )
    test_size = st.slider(
        "Test split (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data reserved for testing.",
    ) / 100.0
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=DEFAULT_RANDOM_STATE,
        step=1,
        help="Seed for reproducible splits and model initialization.",
    )
    stratified_split = st.checkbox(
        "Stratified split",
        value=True,
        help="Preserve class proportions in train/test split.",
    )
    do_cv = st.checkbox("Cross-validation", value=True, help="Run k-fold CV after training.")
    cv_folds = st.slider("CV folds", 3, 10, 5, help="Number of folds for cross-validation.") if do_cv else 5

    st.markdown("---")

    # ── Feature selection ──────────────────────────────────────────────────────
    st.subheader("🔧 Features")
    base_numeric = ["Age", "Fare", "SibSp", "Parch"]
    base_categorical = ["Pclass", "Sex", "Embarked"]
    engineered = list(ENGINEERED_FEATURES.keys())

    sel_base = st.multiselect(
        "Base features",
        base_numeric + base_categorical,
        default=base_numeric + base_categorical,
        help="Original dataset columns to include.",
    )
    sel_eng = st.multiselect(
        "Engineered features",
        engineered,
        default=engineered,
        help="Derived features computed during preprocessing.",
    )
    selected_features = sel_base + sel_eng

    # Validate at least one feature selected
    if not selected_features:
        st.warning("⚠️ Select at least one feature.")
        selected_features = ["Age", "Fare"]

    st.markdown("---")

    # ── Hyperparameter controls ────────────────────────────────────────────────
    st.subheader("⚙️ Hyperparameters")
    hp_params: dict[str, dict] = {}

    for model_name in selected_models:
        with st.expander(f"**{model_name}**", expanded=False):
            defaults = HP_DEFAULTS[model_name]

            if model_name == "Logistic Regression":
                hp_params[model_name] = {
                    "C": st.select_slider("C (Regularization)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=defaults["C"], help="Inverse regularization strength. Smaller = stronger regularization.", key=f"lr_C"),
                    "max_iter": st.slider("max_iter", 50, 1000, defaults["max_iter"], 50, help="Maximum number of solver iterations.", key=f"lr_iter"),
                    "penalty": st.selectbox("penalty", ["l2", "l1", "elasticnet", "none"], index=0, help="Regularization penalty type.", key=f"lr_pen"),
                }

            elif model_name == "K-Nearest Neighbors":
                hp_params[model_name] = {
                    "n_neighbors": st.slider("n_neighbors", 1, 30, defaults["n_neighbors"], help="Number of nearest neighbours.", key="knn_k"),
                    "weights": st.selectbox("weights", ["uniform", "distance"], help="Weight function: uniform or distance-based.", key="knn_w"),
                    "metric": st.selectbox("metric", ["minkowski", "euclidean", "manhattan"], help="Distance metric for neighbour search.", key="knn_m"),
                }

            elif model_name == "Decision Tree":
                hp_params[model_name] = {
                    "max_depth": st.slider("max_depth", 1, 20, defaults["max_depth"], help="Maximum tree depth (None = unlimited).", key="dt_d"),
                    "min_samples_split": st.slider("min_samples_split", 2, 20, defaults["min_samples_split"], help="Min samples to split an internal node.", key="dt_mss"),
                    "criterion": st.selectbox("criterion", ["gini", "entropy", "log_loss"], help="Impurity measure for splits.", key="dt_c"),
                }

            elif model_name == "Random Forest":
                hp_params[model_name] = {
                    "n_estimators": st.slider("n_estimators", 10, 500, defaults["n_estimators"], 10, help="Number of trees in the forest.", key="rf_n"),
                    "max_depth": st.slider("max_depth", 1, 20, defaults["max_depth"], help="Max depth of each tree.", key="rf_d"),
                    "min_samples_leaf": st.slider("min_samples_leaf", 1, 20, defaults["min_samples_leaf"], help="Min samples required at a leaf node.", key="rf_msl"),
                    "max_features": st.selectbox("max_features", ["sqrt", "log2", None], help="Features considered at each split.", key="rf_mf"),
                }

            elif model_name == "Gradient Boosting":
                hp_params[model_name] = {
                    "n_estimators": st.slider("n_estimators", 10, 500, defaults["n_estimators"], 10, help="Number of boosting rounds.", key="gb_n"),
                    "learning_rate": st.select_slider("learning_rate", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5], value=defaults["learning_rate"], help="Step size shrinkage to prevent overfitting.", key="gb_lr"),
                    "max_depth": st.slider("max_depth", 1, 10, defaults["max_depth"], help="Max depth of individual trees.", key="gb_d"),
                }

            elif model_name == "XGBoost":
                hp_params[model_name] = {
                    "n_estimators": st.slider("n_estimators", 10, 500, defaults["n_estimators"], 10, help="Number of gradient boosted trees.", key="xgb_n"),
                    "learning_rate": st.select_slider("learning_rate", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5], value=defaults["learning_rate"], help="Learning rate / step size.", key="xgb_lr"),
                    "max_depth": st.slider("max_depth", 1, 10, defaults["max_depth"], help="Max depth of a tree.", key="xgb_d"),
                    "subsample": st.slider("subsample", 0.5, 1.0, defaults["subsample"], 0.05, help="Fraction of samples used per round.", key="xgb_ss"),
                }

            elif model_name == "SVM":
                hp_params[model_name] = {
                    "C": st.select_slider("C", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=defaults["C"], help="Regularization parameter.", key="svm_C"),
                    "kernel": st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], help="Type of kernel function.", key="svm_k"),
                    "gamma": st.selectbox("gamma", ["scale", "auto"], help="Kernel coefficient for rbf/poly/sigmoid.", key="svm_g"),
                }

            elif model_name == "Neural Network":
                hl_str = st.text_input(
                    "hidden_layer_sizes (comma-separated)",
                    value="100",
                    help="e.g. '128,64' creates two hidden layers of size 128 and 64.",
                    key="mlp_hl",
                )
                try:
                    hidden = tuple(int(x) for x in hl_str.split(",") if x.strip())
                    if not hidden:
                        hidden = (100,)
                except ValueError:
                    hidden = (100,)
                hp_params[model_name] = {
                    "hidden_layer_sizes": hidden,
                    "activation": st.selectbox("activation", ["relu", "tanh", "logistic"], help="Activation function for hidden layers.", key="mlp_act"),
                    "learning_rate_init": st.select_slider("learning_rate_init", [0.0001, 0.001, 0.01, 0.05], value=0.001, help="Initial learning rate for SGD/Adam.", key="mlp_lr"),
                    "max_iter": st.slider("max_iter", 50, 500, 200, 50, help="Maximum epochs.", key="mlp_iter"),
                }

    st.markdown("---")

    # ── Train button ───────────────────────────────────────────────────────────
    auto_retrain = st.checkbox("Auto-retrain on change", value=False, help="Retrain whenever any control changes.")
    train_button = st.button("🚀 Train Models", type="primary", use_container_width=True)


# ── Training logic ─────────────────────────────────────────────────────────────


def run_training(
    selected_models: list[str],
    hp_params: dict[str, dict],
    selected_features: list[str],
    engineered_features: list[str],
    n_records: int,
    test_size: float,
    random_seed: int,
    stratified: bool,
    do_cv: bool,
    cv_folds: int,
    use_ensemble: bool,
    ensemble_type: str,
) -> None:
    """Train all selected models and store results in session state."""
    raw_df = get_raw_data()
    df = sample_data(raw_df, n_records, random_state=int(random_seed))
    df = engineer_features(df, selected=engineered_features if engineered_features else None)
    df = df.dropna(subset=[TARGET_COL])

    # Prepare X, y
    available = [f for f in selected_features if f in df.columns]
    X = df[available].copy()
    y = df[TARGET_COL].copy()

    # Split
    strat_arg = y if stratified else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_seed), stratify=strat_arg
    )

    st.session_state["last_X_train"] = X_train
    st.session_state["last_y_train"] = y_train
    st.session_state["last_X_test"] = X_test
    st.session_state["last_y_test"] = y_test

    pipelines: dict[str, Pipeline] = {}
    train_metrics: dict[str, dict] = {}
    cv_results_store: dict[str, dict] = {}
    train_times: dict[str, float] = {}
    roc_data: dict[str, tuple] = {}

    progress = st.progress(0, text="Starting training…")
    total = len(selected_models)

    for idx, model_name in enumerate(selected_models):
        progress.progress((idx) / total, text=f"Training {model_name}…")
        with st.spinner(f"Training {model_name}…"):
            try:
                params = hp_params.get(model_name, HP_DEFAULTS[model_name])
                pipe = build_pipeline(model_name, params, available)
                fitted_pipe, elapsed = train_model(pipe, X_train, y_train)

                pipelines[model_name] = fitted_pipe
                train_times[model_name] = elapsed

                metrics = compute_all_metrics(fitted_pipe, X_test, y_test)
                metrics["Training Time (s)"] = round(elapsed, 4)
                train_metrics[model_name] = metrics

                try:
                    fpr, tpr, auc = get_roc_curve(fitted_pipe, X_test, y_test)
                    roc_data[model_name] = (fpr, tpr, auc)
                except Exception:
                    pass

                if do_cv:
                    try:
                        cv_res = cross_validate_model(
                            build_pipeline(model_name, params, available),
                            X, y, cv=cv_folds, stratified=stratified
                        )
                        cv_results_store[model_name] = cv_res
                        train_metrics[model_name]["CV Accuracy"] = (
                            f"{cv_res['cv_accuracy_mean']:.3f} ± {cv_res['cv_accuracy_std']:.3f}"
                        )
                    except Exception as e:
                        cv_results_store[model_name] = {}

            except Exception as e:
                st.error(f"❌ {model_name} failed: {e}")

    # Ensemble
    if use_ensemble and len(pipelines) >= 2:
        progress.progress(0.95, text=f"Building {ensemble_type} ensemble…")
        try:
            from src.models import build_stacking_ensemble, build_voting_ensemble

            if ensemble_type == "Voting":
                ens = build_voting_ensemble(pipelines, voting="soft")
            else:
                ens = build_stacking_ensemble(pipelines)

            t0 = time.perf_counter()
            ens.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0

            ens_name = f"{ensemble_type} Ensemble"
            pipelines[ens_name] = ens
            train_times[ens_name] = elapsed
            metrics = compute_all_metrics(ens, X_test, y_test)
            metrics["Training Time (s)"] = round(elapsed, 4)
            train_metrics[ens_name] = metrics
            try:
                fpr, tpr, auc = get_roc_curve(ens, X_test, y_test)
                roc_data[ens_name] = (fpr, tpr, auc)
            except Exception:
                pass
        except Exception as e:
            st.warning(f"Ensemble failed: {e}")

    # Store feature names for explainability
    if pipelines:
        first_pipe = next(iter(pipelines.values()))
        try:
            if hasattr(first_pipe, "named_steps"):
                preprocessor = first_pipe.named_steps.get("preprocessor")
                if preprocessor is not None:
                    fn = get_feature_names_out(preprocessor)
                    st.session_state["feature_names"] = fn
        except Exception:
            st.session_state["feature_names"] = available

    st.session_state["trained_pipelines"] = pipelines
    st.session_state["train_metrics"] = train_metrics
    st.session_state["cv_results"] = cv_results_store
    st.session_state["train_times"] = train_times
    st.session_state["roc_data"] = roc_data

    progress.progress(1.0, text="✅ Training complete!")
    time.sleep(0.5)
    progress.empty()
    st.success(f"✅ Trained {len(pipelines)} model(s) successfully!")


# ── Trigger training ───────────────────────────────────────────────────────────
if train_button or auto_retrain:
    if not selected_models:
        st.sidebar.error("Please select at least one model.")
    else:
        run_training(
            selected_models=selected_models,
            hp_params=hp_params,
            selected_features=selected_features,
            engineered_features=sel_eng,
            n_records=n_records,
            test_size=test_size,
            random_seed=int(random_seed),
            stratified=stratified_split,
            do_cv=do_cv,
            cv_folds=cv_folds,
            use_ensemble=use_ensemble,
            ensemble_type=ensemble_type,
        )


# ── Main tabs ──────────────────────────────────────────────────────────────────

trained_pipelines: dict[str, Pipeline] = st.session_state["trained_pipelines"]
train_metrics: dict[str, dict] = st.session_state["train_metrics"]
train_times: dict[str, float] = st.session_state["train_times"]
roc_data: dict[str, tuple] = st.session_state["roc_data"]
X_test = st.session_state["last_X_test"]
y_test = st.session_state["last_y_test"]
X_train = st.session_state["last_X_train"]
y_train = st.session_state["last_y_train"]
feature_names = st.session_state["feature_names"]

tabs = st.tabs([
    "📈 Overview",
    "🎯 Model Results",
    "⚖️ Compare Models",
    "📉 Training Insights",
    "🔍 Explainability",
    "📋 Parameters",
    "🔮 Predict",
    "🔧 Tuning",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📈 Dataset Overview")
    raw_df = get_raw_data()

    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", len(raw_df))
    col2.metric("Columns", len(raw_df.columns))
    col3.metric("Survivors", int(raw_df["Survived"].sum()))
    col4.metric("Survival Rate", f"{raw_df['Survived'].mean():.1%}")

    st.markdown("### Dataset Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.markdown("### Missing Values")
    miss_df = get_missing_summary(raw_df)
    if not miss_df.empty:
        col_m1, col_m2 = st.columns([1, 2])
        col_m1.dataframe(miss_df, use_container_width=True)
        col_m2.plotly_chart(plot_missing_values(raw_df), use_container_width=True)
    else:
        st.info("No missing values.")

    st.markdown("### Class Balance")
    col_b1, col_b2 = st.columns(2)
    col_b1.plotly_chart(plot_class_balance(raw_df), use_container_width=True)

    st.markdown("### EDA Charts")
    col_e1, col_e2 = st.columns(2)
    col_e1.plotly_chart(plot_survival_by_sex(raw_df), use_container_width=True)
    col_e2.plotly_chart(plot_survival_by_class(raw_df), use_container_width=True)

    col_e3, col_e4 = st.columns(2)
    col_e3.plotly_chart(plot_age_distribution(raw_df), use_container_width=True)
    col_e4.plotly_chart(plot_fare_distribution(raw_df), use_container_width=True)

    st.plotly_chart(plot_correlation_heatmap(raw_df), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Results
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🎯 Model Results")

    if not trained_pipelines:
        st.info("👈 Train at least one model using the sidebar to see results here.")
    else:
        model_select = st.selectbox("Select model", list(trained_pipelines.keys()))
        pipeline = trained_pipelines[model_select]
        metrics = train_metrics.get(model_select, {})

        # Metric cards
        st.markdown("### Performance Metrics")
        m_cols = st.columns(5)
        for i, metric in enumerate(METRIC_NAMES):
            val = metrics.get(metric, float("nan"))
            m_cols[i].metric(metric, f"{val:.4f}" if not np.isnan(val) else "N/A")

        cv_acc = metrics.get("CV Accuracy", "N/A")
        st.metric("CV Accuracy (mean ± std)", cv_acc)

        st.markdown("---")

        # Confusion matrix
        st.markdown("### Confusion Matrix")
        cm_norm = st.toggle("Normalize", value=False, key="cm_norm")
        cm = get_confusion_matrix(pipeline, X_test, y_test)
        st.plotly_chart(plot_confusion_matrix(cm, normalized=cm_norm), use_container_width=True)

        # Classification report
        with st.expander("Classification Report"):
            report = get_classification_report(pipeline, X_test, y_test, target_names=CLASS_NAMES)
            st.code(report)

        # ROC + PR curves
        col_roc, col_pr = st.columns(2)
        try:
            fpr, tpr, auc = get_roc_curve(pipeline, X_test, y_test)
            col_roc.plotly_chart(
                plot_roc_curve(fpr, tpr, auc, model_select, MODEL_COLORS.get(model_select, PALETTE[0])),
                use_container_width=True,
            )
        except Exception as e:
            col_roc.warning(f"ROC curve unavailable: {e}")

        try:
            prec, rec, ap = get_pr_curve(pipeline, X_test, y_test)
            col_pr.plotly_chart(
                plot_pr_curve(prec, rec, ap, model_select, MODEL_COLORS.get(model_select, PALETTE[1])),
                use_container_width=True,
            )
        except Exception as e:
            col_pr.warning(f"PR curve unavailable: {e}")

        # Download model
        st.markdown("---")
        st.markdown("### Download")
        buf = io.BytesIO()
        joblib.dump(pipeline, buf)
        st.download_button(
            "⬇️ Download Model (.joblib)",
            data=buf.getvalue(),
            file_name=f"{model_select.replace(' ', '_')}.joblib",
            mime="application/octet-stream",
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compare Models
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("⚖️ Compare Models")

    if not trained_pipelines:
        st.info("👈 Train models first.")
    else:
        leaderboard = build_leaderboard(train_metrics)

        # Highlight best model
        if not leaderboard.empty and "F1" in leaderboard.columns:
            best_model = leaderboard.iloc[0]["Model"]
            st.success(f"🏆 Best model by F1: **{best_model}** ({leaderboard.iloc[0]['F1']:.4f})")

        st.markdown("### Leaderboard")
        st.dataframe(
            leaderboard.style.highlight_max(
                subset=[c for c in METRIC_NAMES if c in leaderboard.columns],
                color="#d4edda",
            ),
            use_container_width=True,
        )

        # Download leaderboard
        csv = leaderboard.to_csv(index=False)
        st.download_button("⬇️ Download Results (.csv)", data=csv, file_name="leaderboard.csv", mime="text/csv")

        st.markdown("### Metric Comparison")
        metrics_to_show = st.multiselect(
            "Metrics to compare",
            [m for m in METRIC_NAMES if m in leaderboard.columns],
            default=["Accuracy", "F1", "ROC-AUC"],
        )
        if metrics_to_show:
            st.plotly_chart(plot_metric_comparison(leaderboard, metrics_to_show), use_container_width=True)

        st.markdown("### ROC Curves (All Models)")
        if roc_data:
            st.plotly_chart(plot_roc_curves_comparison(roc_data), use_container_width=True)
        else:
            st.info("No ROC data available yet.")

        st.markdown("### Training Time")
        st.plotly_chart(plot_training_time(train_times), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Training Insights
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📉 Training Insights")

    if not trained_pipelines:
        st.info("👈 Train models first.")
    else:
        # MLP loss curve
        if "Neural Network" in trained_pipelines:
            st.markdown("### MLP Loss Curve")
            mlp = trained_pipelines["Neural Network"].named_steps["classifier"]
            if hasattr(mlp, "loss_curve_"):
                val_loss = mlp.validation_scores_ if hasattr(mlp, "validation_scores_") else None
                # Convert validation scores to loss-like
                val_loss_list = None
                if val_loss is not None:
                    val_loss_list = [1 - v for v in val_loss]
                st.plotly_chart(
                    plot_mlp_loss_curve(mlp.loss_curve_, val_loss_list),
                    use_container_width=True,
                )
            else:
                st.warning("MLP loss curve not available (model may not have converged).")

        # GB staged accuracy
        if "Gradient Boosting" in trained_pipelines and X_train is not None:
            st.markdown("### Gradient Boosting — Accuracy vs Rounds")
            with st.spinner("Computing staged scores…"):
                try:
                    gb_params = hp_params.get("Gradient Boosting", HP_DEFAULTS["Gradient Boosting"])
                    avail = [f for f in selected_features if f in X_train.columns]
                    rounds, tr_acc, te_acc = gb_staged_scores(X_train, y_train, X_test, y_test, gb_params, avail)
                    st.plotly_chart(plot_staged_accuracy(rounds, tr_acc, te_acc, "Gradient Boosting"), use_container_width=True)
                except Exception as e:
                    st.warning(f"GB staged scores unavailable: {e}")

        # XGBoost eval results
        if "XGBoost" in trained_pipelines and X_train is not None:
            st.markdown("### XGBoost — Loss vs Rounds")
            with st.spinner("Computing XGBoost eval results…"):
                try:
                    xgb_params = hp_params.get("XGBoost", HP_DEFAULTS["XGBoost"])
                    avail = [f for f in selected_features if f in X_train.columns]
                    rounds, tr_loss, te_loss = xgb_eval_results(X_train, y_train, X_test, y_test, xgb_params, avail)
                    st.plotly_chart(plot_staged_accuracy(rounds, tr_loss, te_loss, "XGBoost (Logloss)"), use_container_width=True)
                except Exception as e:
                    st.warning(f"XGBoost eval results unavailable: {e}")

        # RF accuracy sweep
        if "Random Forest" in trained_pipelines and X_train is not None:
            st.markdown("### Random Forest — Accuracy vs n_estimators")
            with st.spinner("Computing RF sweep…"):
                try:
                    rf_params = hp_params.get("Random Forest", HP_DEFAULTS["Random Forest"])
                    avail = [f for f in selected_features if f in X_train.columns]
                    max_est = min(rf_params.get("n_estimators", 100), 200)
                    n_vals, tr_acc, te_acc = rf_accuracy_sweep(X_train, y_train, X_test, y_test, rf_params, avail, max_estimators=max_est, step=10)
                    st.plotly_chart(plot_staged_accuracy(n_vals, tr_acc, te_acc, "Random Forest"), use_container_width=True)
                except Exception as e:
                    st.warning(f"RF sweep unavailable: {e}")

        # Learning curve
        st.markdown("### Learning Curve")
        lc_model = st.selectbox("Model for learning curve", list(trained_pipelines.keys()), key="lc_model")
        if lc_model and X_train is not None:
            with st.spinner("Computing learning curve…"):
                try:
                    avail = [f for f in selected_features if f in X_train.columns]
                    all_X = pd.concat([X_train, X_test])
                    all_y = pd.concat([y_train, y_test])
                    # Build a fresh unfitted pipeline
                    lc_params = hp_params.get(lc_model, HP_DEFAULTS.get(lc_model, {}))
                    lc_pipe = build_pipeline(lc_model if lc_model in HP_DEFAULTS else list(HP_DEFAULTS.keys())[0],
                                             lc_params, avail)
                    sizes, tr_s, te_s = get_learning_curve_data(lc_pipe, all_X[avail], all_y, cv=5)
                    st.plotly_chart(plot_learning_curve(sizes, tr_s, te_s), use_container_width=True)
                except Exception as e:
                    st.warning(f"Learning curve unavailable: {e}")

        # Validation curve
        st.markdown("### Validation Curve")
        vc_model = st.selectbox("Model for validation curve", [m for m in trained_pipelines if m in HP_DEFAULTS], key="vc_model")
        if vc_model and X_train is not None:
            param_options = {
                "Logistic Regression": ("classifier__C", [0.001, 0.01, 0.1, 1, 10, 100]),
                "K-Nearest Neighbors": ("classifier__n_neighbors", list(range(1, 21))),
                "Decision Tree": ("classifier__max_depth", list(range(1, 15))),
                "Random Forest": ("classifier__n_estimators", [10, 30, 50, 100, 150, 200]),
                "Gradient Boosting": ("classifier__learning_rate", [0.01, 0.05, 0.1, 0.2, 0.5]),
                "XGBoost": ("classifier__max_depth", list(range(1, 10))),
                "SVM": ("classifier__C", [0.001, 0.01, 0.1, 1, 10, 100]),
                "Neural Network": ("classifier__learning_rate_init", [0.0001, 0.001, 0.01, 0.05]),
            }
            if vc_model in param_options:
                param_name, param_range = param_options[vc_model]
                with st.spinner("Computing validation curve…"):
                    try:
                        avail = [f for f in selected_features if f in X_train.columns]
                        vc_params = hp_params.get(vc_model, HP_DEFAULTS[vc_model])
                        vc_pipe = build_pipeline(vc_model, vc_params, avail)
                        all_X = pd.concat([X_train, X_test])
                        all_y = pd.concat([y_train, y_test])
                        tr_s, te_s = get_validation_curve_data(vc_pipe, all_X[avail], all_y, param_name, param_range)
                        st.plotly_chart(plot_validation_curve(param_range, tr_s, te_s, param_name.split("__")[-1]), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Validation curve unavailable: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Explainability
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔍 Explainability")

    if not trained_pipelines or X_test is None:
        st.info("👈 Train models first.")
    else:
        exp_model = st.selectbox("Select model", list(trained_pipelines.keys()), key="exp_model")
        pipeline = trained_pipelines[exp_model]
        classifier = pipeline.named_steps.get("classifier") if hasattr(pipeline, "named_steps") else pipeline
        fn = feature_names if feature_names else [f"Feature {i}" for i in range(50)]

        # Tree-based feature importance
        st.markdown("### Feature Importance (tree-based models)")
        if hasattr(classifier, "feature_importances_"):
            imp = classifier.feature_importances_
            if len(imp) == len(fn):
                st.plotly_chart(plot_feature_importance(imp, fn, exp_model), use_container_width=True)
            else:
                st.info(f"Feature importance shape mismatch ({len(imp)} vs {len(fn)} feature names).")
        else:
            st.info("This model does not expose `feature_importances_`.")

        # Permutation importance
        st.markdown("### Permutation Importance")
        with st.spinner("Computing permutation importance…"):
            try:
                perm = permutation_importance(
                    pipeline, X_test, y_test, n_repeats=10, random_state=DEFAULT_RANDOM_STATE, n_jobs=-1
                )
                mean_imp = perm.importances_mean
                std_imp = perm.importances_std

                # Use original feature names (columns of X_test)
                orig_fn = list(X_test.columns)
                st.plotly_chart(
                    plot_permutation_importance(mean_imp, std_imp, orig_fn, exp_model),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Permutation importance failed: {e}")

        # SHAP
        st.markdown("### SHAP Summary Plot (optional)")
        shap_requested = st.checkbox("Compute SHAP values (may be slow)", value=False)
        if shap_requested:
            try:
                import shap

                preprocessor = pipeline.named_steps.get("preprocessor")
                X_test_t = preprocessor.transform(X_test) if preprocessor else X_test.values

                if hasattr(classifier, "predict_proba"):
                    explainer = shap.Explainer(classifier, X_test_t)
                    shap_vals = explainer(X_test_t)
                    fig_shap, ax = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots()
                    shap.summary_plot(shap_vals.values, X_test_t, feature_names=fn, show=False)
                    st.pyplot(fig_shap)
                else:
                    st.info("SHAP not available for this model type.")
            except ImportError:
                st.info("Install `shap` (`pip install shap`) to enable SHAP plots.")
            except Exception as e:
                st.warning(f"SHAP failed: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — Parameters
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("📋 Model Parameters")

    if not trained_pipelines:
        st.info("👈 Train models first.")
    else:
        for model_name, pipeline in trained_pipelines.items():
            with st.expander(f"**{model_name}**", expanded=False):
                if hasattr(pipeline, "get_params"):
                    params_dict = pipeline.get_params()
                    # Show in a nice table
                    params_df = pd.DataFrame(
                        {"Parameter": list(params_dict.keys()), "Value": [str(v) for v in params_dict.values()]}
                    )
                    st.dataframe(params_df, use_container_width=True)

                # Pipeline steps
                if hasattr(pipeline, "steps"):
                    st.markdown("**Pipeline Steps:**")
                    for step_name, step_obj in pipeline.steps:
                        st.markdown(f"- `{step_name}`: `{type(step_obj).__name__}`")

                # Feature list
                st.markdown(f"**Features used ({len(selected_features)}):**")
                st.code(", ".join(selected_features))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — Predict
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("🔮 Passenger Survival Prediction")

    if not trained_pipelines:
        st.info("👈 Train models first.")
    else:
        st.markdown("Enter passenger details to get a survival prediction.")

        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                p_pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2, help="1=First, 2=Second, 3=Third")
                p_sex = st.selectbox("Sex", ["male", "female"])
                p_age = st.slider("Age", 1, 80, 30)
            with col2:
                p_fare = st.slider("Fare", 0.0, 500.0, 32.0, step=1.0)
                p_sibsp = st.slider("SibSp (siblings/spouses)", 0, 8, 0)
                p_parch = st.slider("Parch (parents/children)", 0, 6, 0)
            with col3:
                p_embarked = st.selectbox("Embarked", ["S", "C", "Q"], help="Port of embarkation: Southampton, Cherbourg, Queenstown")
                p_cabin = st.checkbox("Has Cabin info", value=False)

            predict_btn = st.form_submit_button("🔮 Predict Survival", type="primary")

        if predict_btn:
            family_size = p_sibsp + p_parch + 1
            fare_per_person = p_fare / family_size

            # Build a passenger row
            passenger_raw = pd.DataFrame([{
                "Pclass": p_pclass,
                "Sex": p_sex,
                "Age": float(p_age),
                "Fare": p_fare,
                "SibSp": p_sibsp,
                "Parch": p_parch,
                "Embarked": p_embarked,
                "Name": "Passenger, Mr. Test",
                "Cabin": "C123" if p_cabin else None,
                "Ticket": "12345",
                "PassengerId": 9999,
                "Survived": 0,
            }])

            passenger_df = engineer_features(passenger_raw, selected=sel_eng if sel_eng else None)
            avail = [f for f in selected_features if f in passenger_df.columns]
            passenger_X = passenger_df[avail]

            st.markdown("---")
            proba_dict: dict[str, float] = {}

            for model_name, pipe in trained_pipelines.items():
                try:
                    if hasattr(pipe, "predict_proba"):
                        prob = pipe.predict_proba(passenger_X)[0][1]
                    else:
                        prob = float(pipe.predict(passenger_X)[0])
                    proba_dict[model_name] = prob
                except Exception as e:
                    st.warning(f"{model_name}: prediction failed ({e})")

            if proba_dict:
                # Gauge for selected model
                gauge_model = st.selectbox("Show gauge for", list(proba_dict.keys()), key="gauge_model")
                st.plotly_chart(plot_survival_gauge(proba_dict[gauge_model], gauge_model), use_container_width=True)

                # Bar chart for all models
                st.plotly_chart(plot_all_model_predictions(proba_dict), use_container_width=True)

                # Summary table
                st.dataframe(
                    pd.DataFrame(
                        {"Model": list(proba_dict.keys()), "Survival Probability": [f"{v:.2%}" for v in proba_dict.values()]}
                    ),
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 8 — Hyperparameter Tuning
# ════════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("🔧 Hyperparameter Tuning")

    if not trained_pipelines or X_train is None:
        st.info("👈 Train at least one model first.")
    else:
        tune_model = st.selectbox("Model to tune", [m for m in trained_pipelines if m in HP_DEFAULTS], key="tune_model")
        search_type = st.radio("Search strategy", ["RandomizedSearchCV", "GridSearchCV"], horizontal=True)
        n_iter_rand = st.slider("n_iter (RandomizedSearch only)", 10, 100, 20, help="Number of parameter combinations to try.")
        tune_cv = st.slider("CV folds for tuning", 3, 10, 5)

        # Parameter grids
        PARAM_GRIDS: dict[str, dict] = {
            "Logistic Regression": {"classifier__C": [0.01, 0.1, 1, 10, 100], "classifier__penalty": ["l2", "l1"]},
            "K-Nearest Neighbors": {"classifier__n_neighbors": list(range(1, 20, 2)), "classifier__weights": ["uniform", "distance"]},
            "Decision Tree": {"classifier__max_depth": [3, 5, 7, 10, None], "classifier__criterion": ["gini", "entropy"]},
            "Random Forest": {"classifier__n_estimators": [50, 100, 200], "classifier__max_depth": [3, 5, 7, None]},
            "Gradient Boosting": {"classifier__n_estimators": [50, 100, 200], "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2], "classifier__max_depth": [2, 3, 5]},
            "XGBoost": {"classifier__n_estimators": [50, 100, 200], "classifier__learning_rate": [0.01, 0.1, 0.2], "classifier__max_depth": [3, 5, 7]},
            "SVM": {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["rbf", "linear"], "classifier__gamma": ["scale", "auto"]},
            "Neural Network": {"classifier__hidden_layer_sizes": [(50,), (100,), (100, 50)], "classifier__activation": ["relu", "tanh"]},
        }

        st.markdown("**Parameter grid:**")
        grid = PARAM_GRIDS.get(tune_model, {})
        st.json(grid)

        if st.button("🔍 Start Tuning", type="primary"):
            avail = [f for f in selected_features if f in X_train.columns]
            base_params = hp_params.get(tune_model, HP_DEFAULTS[tune_model])
            base_pipe = build_pipeline(tune_model, base_params, avail)

            progress_bar = st.progress(0, text="Running search…")
            with st.spinner("Tuning in progress…"):
                try:
                    if search_type == "RandomizedSearchCV":
                        searcher = RandomizedSearchCV(
                            base_pipe, grid, n_iter=n_iter_rand, cv=tune_cv,
                            scoring="f1", n_jobs=-1, random_state=DEFAULT_RANDOM_STATE, refit=True, verbose=0,
                        )
                    else:
                        searcher = GridSearchCV(
                            base_pipe, grid, cv=tune_cv,
                            scoring="f1", n_jobs=-1, refit=True, verbose=0,
                        )
                    all_X = pd.concat([X_train, X_test])
                    all_y = pd.concat([y_train, y_test])
                    searcher.fit(all_X[avail], all_y)
                    progress_bar.progress(1.0, text="Done!")

                    st.success(f"✅ Best F1: {searcher.best_score_:.4f}")
                    st.json(searcher.best_params_)

                    cv_results_df = pd.DataFrame(searcher.cv_results_)
                    st.dataframe(cv_results_df[["params", "mean_test_score", "std_test_score", "rank_test_score"]].sort_values("rank_test_score"), use_container_width=True)
                    st.plotly_chart(plot_search_results(cv_results_df), use_container_width=True)

                    # Store tuned model
                    st.session_state["trained_pipelines"][f"{tune_model} (Tuned)"] = searcher.best_estimator_
                    st.session_state["tuning_results"] = cv_results_df
                    st.info("Tuned model added to the trained models. Switch to other tabs to inspect it.")

                except Exception as e:
                    st.error(f"Tuning failed: {e}")
            progress_bar.empty()
