"""
src/plots.py
------------
Plotly chart builders for every dashboard tab.

All functions return ``plotly.graph_objects.Figure`` objects so Streamlit
can render them with ``st.plotly_chart(..., use_container_width=True)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CLASS_NAMES, MODEL_COLORS, PALETTE


# ── Helpers ───────────────────────────────────────────────────────────────────


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply consistent layout to a figure."""
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    return fig


# ── Overview / EDA charts ─────────────────────────────────────────────────────


def plot_class_balance(df: pd.DataFrame) -> go.Figure:
    """Pie chart of Survived vs Not Survived."""
    counts = df["Survived"].value_counts().sort_index()
    labels = [CLASS_NAMES[i] for i in counts.index]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values,
            marker_colors=[PALETTE[1], PALETTE[0]],
            hole=0.4,
            textinfo="label+percent",
        )
    )
    return _apply_theme(fig, "Class Balance")


def plot_survival_by_sex(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: survival rate by sex."""
    grp = df.groupby("Sex")["Survived"].mean().reset_index()
    grp["Survived"] *= 100
    fig = px.bar(
        grp,
        x="Sex",
        y="Survived",
        color="Sex",
        color_discrete_sequence=PALETTE,
        labels={"Survived": "Survival Rate (%)"},
        text_auto=".1f",
    )
    return _apply_theme(fig, "Survival Rate by Sex")


def plot_survival_by_class(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: survival rate by passenger class."""
    grp = df.groupby("Pclass")["Survived"].mean().reset_index()
    grp["Survived"] *= 100
    grp["Pclass"] = grp["Pclass"].astype(str)
    fig = px.bar(
        grp,
        x="Pclass",
        y="Survived",
        color="Pclass",
        color_discrete_sequence=PALETTE,
        labels={"Survived": "Survival Rate (%)", "Pclass": "Passenger Class"},
        text_auto=".1f",
    )
    return _apply_theme(fig, "Survival Rate by Passenger Class")


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Overlapping histograms: age distribution for survivors vs not."""
    fig = go.Figure()
    for val, name, color in zip([0, 1], CLASS_NAMES, [PALETTE[1], PALETTE[0]]):
        subset = df[df["Survived"] == val]["Age"].dropna()
        fig.add_trace(
            go.Histogram(
                x=subset,
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
            )
        )
    fig.update_layout(barmode="overlay")
    return _apply_theme(fig, "Age Distribution by Survival")


def plot_fare_distribution(df: pd.DataFrame) -> go.Figure:
    """Box plots: fare distribution by survival."""
    fig = px.box(
        df,
        x="Survived",
        y="Fare",
        color="Survived",
        color_discrete_map={0: PALETTE[1], 1: PALETTE[0]},
        labels={"Survived": "Survived (0=No, 1=Yes)"},
        points="outliers",
    )
    return _apply_theme(fig, "Fare Distribution by Survival")


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of numeric features."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
        )
    )
    return _apply_theme(fig, "Feature Correlation Heatmap")


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of missing-value percentages."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    pct = (missing / len(df) * 100).round(1)
    fig = go.Figure(
        go.Bar(
            x=pct.values,
            y=pct.index.tolist(),
            orientation="h",
            marker_color=PALETTE[1],
            text=[f"{v}%" for v in pct.values],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Missing (%)", yaxis_title="Column")
    return _apply_theme(fig, "Missing Values")


# ── Model Results charts ──────────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    normalized: bool = False,
    class_names: list[str] = CLASS_NAMES,
) -> go.Figure:
    """Interactive confusion matrix heatmap.

    Parameters
    ----------
    cm:
        Raw confusion matrix (2×2).
    normalized:
        Whether to show row-normalized percentages.
    class_names:
        Display labels for 0 and 1.
    """
    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        z = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".2%"
        title_suffix = " (Normalized)"
    else:
        z = cm.astype(float)
        fmt = "d"
        title_suffix = " (Counts)"

    text = np.array([[f"{v:.2%}" if normalized else str(int(v)) for v in row] for row in z])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"Pred: {c}" for c in class_names],
            y=[f"True: {c}" for c in class_names],
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Value: %{text}<extra></extra>",
            showscale=True,
        )
    )
    return _apply_theme(fig, f"Confusion Matrix{title_suffix}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    model_name: str = "",
    color: str = PALETTE[0],
) -> go.Figure:
    """Single ROC curve."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"{model_name} (AUC={auc:.3f})",
            line=dict(color=color, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", width=1, dash="dash"),
        )
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return _apply_theme(fig, "ROC Curve")


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    model_name: str = "",
    color: str = PALETTE[0],
) -> go.Figure:
    """Single Precision-Recall curve."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"{model_name} (AP={ap:.3f})",
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return _apply_theme(fig, "Precision-Recall Curve")


# ── Compare Models charts ─────────────────────────────────────────────────────


def plot_metric_comparison(leaderboard: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Grouped bar chart comparing all models across multiple metrics."""
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        if metric not in leaderboard.columns:
            continue
        fig.add_trace(
            go.Bar(
                name=metric,
                x=leaderboard["Model"],
                y=leaderboard[metric],
                marker_color=PALETTE[i % len(PALETTE)],
                text=leaderboard[metric].round(3),
                textposition="outside",
            )
        )
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1]))
    return _apply_theme(fig, "Model Metric Comparison")


def plot_roc_curves_comparison(
    roc_data: dict[str, tuple],
) -> go.Figure:
    """Overlay ROC curves for multiple models.

    Parameters
    ----------
    roc_data:
        Mapping of model_name → (fpr, tpr, auc).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", width=1, dash="dash"),
        )
    )
    for name, (fpr, tpr, auc) in roc_data.items():
        color = MODEL_COLORS.get(name, PALETTE[0])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={auc:.3f})",
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return _apply_theme(fig, "ROC Curves — All Models")


def plot_training_time(model_times: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of training times."""
    names = list(model_times.keys())
    times = [model_times[n] for n in names]
    colors = [MODEL_COLORS.get(n, PALETTE[0]) for n in names]
    fig = go.Figure(
        go.Bar(
            x=times,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{t:.3f}s" for t in times],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Training Time (s)")
    return _apply_theme(fig, "Training Time Comparison")


# ── Training Insights charts ──────────────────────────────────────────────────


def plot_mlp_loss_curve(
    train_loss: list[float],
    val_loss: list[float] | None = None,
) -> go.Figure:
    """MLP loss per epoch."""
    epochs = list(range(1, len(train_loss) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, mode="lines", name="Train Loss", line=dict(color=PALETTE[0]))
    )
    if val_loss:
        fig.add_trace(
            go.Scatter(x=epochs, y=val_loss, mode="lines", name="Val Loss", line=dict(color=PALETTE[1]))
        )
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    return _apply_theme(fig, "MLP Loss Curve")


def plot_staged_accuracy(
    rounds: list[int],
    train_accs: list[float],
    test_accs: list[float],
    model_name: str = "Model",
) -> go.Figure:
    """Accuracy vs boosting rounds / n_estimators."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=rounds, y=train_accs, mode="lines", name="Train", line=dict(color=PALETTE[0]))
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=test_accs, mode="lines", name="Test", line=dict(color=PALETTE[1]))
    )
    fig.update_layout(xaxis_title="Rounds / n_estimators", yaxis_title="Accuracy")
    return _apply_theme(fig, f"{model_name} — Accuracy vs Rounds")


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
) -> go.Figure:
    """Learning curve: accuracy vs training set size."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores,
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color=PALETTE[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores,
            mode="lines+markers",
            name="CV Accuracy",
            line=dict(color=PALETTE[1]),
        )
    )
    fig.update_layout(xaxis_title="Training Set Size", yaxis_title="Accuracy")
    return _apply_theme(fig, "Learning Curve")


def plot_validation_curve(
    param_range: list,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    param_name: str = "Parameter",
) -> go.Figure:
    """Validation curve: accuracy vs a single hyperparameter value."""
    x_vals = [str(v) for v in param_range]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=train_scores,
            mode="lines+markers",
            name="Train",
            line=dict(color=PALETTE[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=test_scores,
            mode="lines+markers",
            name="CV",
            line=dict(color=PALETTE[1]),
        )
    )
    fig.update_layout(xaxis_title=param_name, yaxis_title="Accuracy")
    return _apply_theme(fig, f"Validation Curve — {param_name}")


# ── Explainability charts ─────────────────────────────────────────────────────


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list[str],
    model_name: str = "",
    top_n: int = 20,
) -> go.Figure:
    """Horizontal bar chart of feature importances (tree-based models)."""
    idx = np.argsort(importance)[-top_n:]
    fig = go.Figure(
        go.Bar(
            x=importance[idx],
            y=[feature_names[i] for i in idx],
            orientation="h",
            marker_color=PALETTE[3],
        )
    )
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature")
    return _apply_theme(fig, f"Feature Importance — {model_name} (Top {top_n})")


def plot_permutation_importance(
    importance_mean: np.ndarray,
    importance_std: np.ndarray,
    feature_names: list[str],
    model_name: str = "",
    top_n: int = 20,
) -> go.Figure:
    """Bar chart with error bars for permutation importance."""
    idx = np.argsort(importance_mean)[-top_n:]
    fig = go.Figure(
        go.Bar(
            x=importance_mean[idx],
            y=[feature_names[i] for i in idx],
            orientation="h",
            error_x=dict(type="data", array=importance_std[idx], visible=True),
            marker_color=PALETTE[4],
        )
    )
    fig.update_layout(xaxis_title="Mean Accuracy Decrease", yaxis_title="Feature")
    return _apply_theme(fig, f"Permutation Importance — {model_name} (Top {top_n})")


# ── Predict tab ───────────────────────────────────────────────────────────────


def plot_survival_gauge(probability: float, model_name: str = "") -> go.Figure:
    """Gauge chart showing survival probability from 0 to 1."""
    pct = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 32}},
            delta={"reference": 50, "suffix": "%"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": PALETTE[0] if probability >= 0.5 else PALETTE[1]},
                "steps": [
                    {"range": [0, 40], "color": "#ffcccc"},
                    {"range": [40, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            title={"text": f"Survival Probability<br><span style='font-size:14px'>{model_name}</span>"},
        )
    )
    fig.update_layout(height=280)
    return _apply_theme(fig)


def plot_all_model_predictions(proba_dict: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of survival probabilities across all models."""
    names = list(proba_dict.keys())
    probs = [proba_dict[n] for n in names]
    colors = [PALETTE[0] if p >= 0.5 else PALETTE[1] for p in probs]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis=dict(range=[0, 1.1], title="Survival Probability"),
        yaxis_title="Model",
    )
    return _apply_theme(fig, "Survival Probability — All Models")


# ── Hyperparameter tuning ─────────────────────────────────────────────────────


def plot_search_results(cv_results: pd.DataFrame, score_col: str = "mean_test_score") -> go.Figure:
    """Scatter plot of hyperparameter search results."""
    df = cv_results.sort_values(score_col, ascending=False).reset_index(drop=True)
    fig = px.scatter(
        df,
        x=df.index,
        y=score_col,
        color=score_col,
        color_continuous_scale="Viridis",
        labels={score_col: "CV Score", "index": "Candidate"},
        hover_data=[c for c in df.columns if c.startswith("param_")],
    )
    return _apply_theme(fig, "Hyperparameter Search Results")
