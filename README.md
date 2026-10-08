# 🚢 Titanic ML Playground

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://titanic-survival-prediction-bndkb979xavgherjmjuzyy.streamlit.app/)

An interactive machine-learning dashboard inspired by TensorFlow Playground — train, tune, compare, and explain 8 classifiers on the Titanic dataset, all from a live UI with zero code.

### 🌐 Live Demo

**Try it now: [titanic-survival-prediction.streamlit.app](https://titanic-survival-prediction-bndkb979xavgherjmjuzyy.streamlit.app/)**

No installation needed. Open the link, adjust the sliders, and the models retrain instantly.

---

## ✨ Features at a Glance

| Feature | Details |
|---|---|
| **8 ML Models** | Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting, XGBoost, SVM, MLP |
| **Ensemble** | Soft-voting or Stacking ensemble of any selected models |
| **Live Controls** | Every slider/dropdown triggers immediate retraining |
| **7 Dashboard Tabs** | Overview · Results · Compare · Insights · Explainability · Parameters · Predict |
| **Hyperparameter Tuning** | GridSearchCV / RandomizedSearchCV with progress bar |
| **Explainability** | Feature importance, permutation importance, optional SHAP |
| **Offline-first** | Dataset cached locally after first download |
| **Dockerised** | Single `docker-compose up` command |

---

## 📐 Architecture

```
titanic-ml-playground/
├── app.py                  # Streamlit entry point (all tabs + sidebar)
├── config.py               # Central config — paths, defaults, palettes
├── src/
│   ├── data.py             # Download, cache, load, sample
│   ├── features.py         # Feature engineering + sklearn Pipeline/ColumnTransformer
│   ├── models.py           # Model factory, training, ensemble, sweep utilities
│   ├── metrics.py          # Metric computation, leaderboard, learning/validation curves
│   └── plots.py            # All Plotly chart builders (pure functions → Figure)
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_models.py
├── data/                   # Auto-created; cached titanic.csv lives here
├── saved_models/           # Auto-created; downloaded .joblib models live here
├── requirements.txt
├── pyproject.toml          # ruff + pytest config
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

**Data flow:**

```
URL / local CSV
      │
      ▼
  src/data.py          ← download, cache, sample
      │
      ▼
  src/features.py      ← engineer_features() → build_preprocessor() (ColumnTransformer)
      │
      ▼
  src/models.py        ← build_pipeline() → train_model() / cross_validate_model()
      │
      ▼
  src/metrics.py       ← compute_all_metrics(), leaderboard, curves
      │
      ▼
  src/plots.py         ← Plotly Figure builders
      │
      ▼
  app.py (Streamlit)   ← sidebar controls + 7 tabs
```

---

## 🚀 Quick Start (local)

> Prefer not to install anything? Use the [live demo](https://titanic-survival-prediction-bndkb979xavgherjmjuzyy.streamlit.app/).

### Prerequisites

- Python 3.11+
- pip

### 1. Clone

```bash
git clone https://github.com/your-username/titanic-ml-playground.git
cd titanic-ml-playground
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.  
On the first run it downloads `titanic.csv` (~60 KB) and caches it to `data/titanic.csv`.  
All subsequent runs work **offline**.

---

## 🐳 Docker

### Build & run with docker-compose

```bash
docker-compose up --build
```

Open **http://localhost:8501**.  
The `data/` and `saved_models/` directories are mounted as volumes so the dataset and models persist between container restarts.

### Standalone Docker

```bash
docker build -t titanic-ml-playground .
docker run -p 8501:8501 -v $(pwd)/data:/app/data titanic-ml-playground
```

---

## 🧪 Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Run a single test file
pytest tests/test_features.py -v
```

---

## 🔍 Lint

```bash
ruff check src/ config.py tests/
```

---

## 🖼️ Screenshots

> _Add screenshots here after first run._

| Tab | Description |
|---|---|
| Overview | EDA charts: class balance, survival by sex/class/age/fare, correlation heatmap |
| Model Results | Metric cards, confusion matrix, ROC + PR curves |
| Compare Models | Sortable leaderboard, grouped bar chart, overlaid ROC curves, training times |
| Training Insights | MLP loss curve, GB/XGBoost per-round curves, learning curve, validation curve |
| Explainability | Feature importance, permutation importance, SHAP summary |
| Parameters | Full `get_params()` dict per model |
| Predict | Gauge chart + bar chart of survival probability across all models |

---

## ⚙️ Configuration

All constants, paths, and defaults are in `config.py`.  
Notable settings:

| Key | Default | Description |
|---|---|---|
| `TITANIC_URL` | GitHub raw CSV | Remote dataset URL |
| `TITANIC_LOCAL` | `data/titanic.csv` | Local cache path |
| `DEFAULT_TEST_SIZE` | 0.20 | Train/test split |
| `DEFAULT_RANDOM_STATE` | 42 | Global random seed |
| `AGE_BINS` | `[0,12,18,35,60,100]` | Age discretisation boundaries |
| `TITLE_MAP` | dict | Raw title → normalised title mapping |
| `HP_DEFAULTS` | dict | Default hyperparameters per model |

---

## 🧩 Preprocessing Pipeline

```
Raw DataFrame
    │
    ├── engineer_features()          # Title, FamilySize, IsAlone, FarePerPerson, AgeBin, HasCabin
    │
    └── ColumnTransformer
            ├── Numeric pipeline     # SimpleImputer(median) → StandardScaler
            └── Categorical pipeline # SimpleImputer(most_frequent) → OneHotEncoder
```

- Transformers are **fit on training data only** — no leakage.
- The full pipeline (preprocessor + classifier) is fitted as one unit.

---

## 📦 CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **ruff** — lint `src/`, `config.py`, `tests/`
2. **pytest** — run all unit tests with coverage

Triggers on every push to `main`/`master`/`develop` and all pull requests.

---

## 📄 License

MIT
