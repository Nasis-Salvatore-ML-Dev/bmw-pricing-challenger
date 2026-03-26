# BMW Pricing Challenge — Production ML Pricing System

> **Replacing manual dealership pricing with a sub-200ms REST API backed by a machine learning model, a full CI/CD pipeline, and a drift monitoring strategy — deployed on Google Cloud Run.**

---

## Business Problem

Used-car dealerships face a systematic pricing problem: vehicles priced too high sit on lots and erode margins; vehicles priced too low are sold below their market value. This project builds an end-to-end ML system to replace intuition-based pricing with data-driven predictions.

| Error Type   | Current Annual Loss | Target Annual Loss | Projected Savings |
| ------------ | ------------------- | ------------------ | ----------------- |
| Underpricing | €480K               | €288K              | €192K             |
| Overpricing  | €324K               | €194K              | €130K             |
| **Total**    | **€804K**           | **€482K**          | **€322K**         |

---

## System Architecture

```
GitHub Push
    │
    ├── CI (GitHub Actions)
    │     lint → test → ✅
    │
    ├── CD (GitHub Actions + Cloud Build)
    │     build image → push to Artifact Registry → deploy to Cloud Run
    │
    └── Load Test (Locust, staging branch only)
          10 concurrent users → HTML report artifact
```

**Stack:** Python 3.10/3.11 · FastAPI · scikit-learn · XGBoost · Docker · GitHub Actions · Google Cloud Run · Google Artifact Registry · Locust

---

## Data & Feature Engineering

**Dataset:** BMW Pricing Challenge (Kaggle) — 3,874 rows, 17 raw features.

### Data Leakage — Critical Finding

An early version of the pipeline achieved R² = 0.99. Investigation revealed three features computed directly from the target (`depreciation_rate`, `price_per_km`, `price_segment`) were included in training. These were removed entirely. All preprocessing and target encoding is fit exclusively on the training split and applied to validation/test sets.

### Features Engineered

| Feature                                            | Rationale                                                |
| -------------------------------------------------- | -------------------------------------------------------- |
| `car_age` (registration → sale date)               | Primary depreciation driver                              |
| BMW series extraction (1-series … X7)              | Captures brand hierarchy                                 |
| Luxury tier flag                                   | Proxy for buyer willingness to pay                       |
| `age_mileage_interaction`                          | Non-linear depreciation effect                           |
| `mileage_per_power`                                | Efficiency proxy correlated with use intensity           |
| `annual_mileage`                                   | Normalises mileage by age                                |
| Registration season                                | Seasonal supply/demand signal                            |
| Target encoding (fuel, colour, car type)           | Replaces raw categoricals; fit on train only             |
| Rare-category grouping (hybrid + electric → other) | Prevents model unreliability on underrepresented classes |
| Segment consolidation                              | Increases per-group sample size for generalisation       |

### Target Variable

`price` exhibits a right-skewed distribution (skewness: 3.51, mean/median ratio: 1.12). A `log1p` transformation is applied before training to stabilise variance across the price range. All reported metrics are computed after inverting the transformation (`np.expm1`) to ensure business interpretability in euros.

---

## Model Development

### Why Random Forest over XGBoost?

Both algorithms were trained and evaluated on identical splits. XGBoost showed no statistically meaningful improvement (MAE 1,980 vs. 1,945; R² 0.765 vs. 0.776). With a dataset of ~3,900 rows, the marginal complexity of gradient boosting does not justify the additional tuning overhead. Random Forest was selected for its robustness, interpretability, and deployment simplicity.

> **On dataset size:** The business metrics (Tail Rate ≤15%, TC-APE ≤6.5%) represent aspirational targets derived from stakeholder requirements. With 3,874 training examples and 17 features, the model hits a performance ceiling that additional algorithmic complexity cannot overcome. Closing this gap in a production setting would require richer data (e.g. market-level supply/demand, regional pricing, dealer network data) — a data acquisition problem, not a modelling one.

### Hyperparameter Tuning

Strategy: `RandomizedSearchCV` with 5-fold cross-validation.

```python
{
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [10, 20, 30, 40, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}
```

Best configuration found: `n_estimators=100`, `max_depth=40`, `min_samples_split=10`.

Validation strategy: stratified time-based split (60% train / 20% validation / 20% test) to respect temporal ordering and prevent future data leaking into training.

### Model Performance

| Metric    | Value  | Target   | Status          |
| --------- | ------ | -------- | --------------- |
| MAE       | €1,946 | < €1,500 | ⚠️ Acceptable   |
| RMSE      | €2,875 | < €2,200 | ⚠️ Acceptable   |
| R²        | 0.776  | > 0.88   | ⚠️ Data ceiling |
| MAPE      | 20.48% | < 3.5%   | ⚠️ Data ceiling |
| Tail Rate | 71.77% | ≤ 15%    | ❌ Data ceiling |
| TC-APE    | 27.54% | ≤ 6.5%   | ❌ Data ceiling |

The gap between achieved and target performance is a deliberate and honest acknowledgement that the dataset does not contain sufficient information to meet the original business thresholds. The system is deployed at achievable industry-standard performance (€1,946 MAE) rather than overfitting to the training data to hit headline numbers.

---

## API Design

**Framework:** FastAPI with Uvicorn. The model is loaded once at container startup and reused for all subsequent requests.

### Endpoints

| Endpoint         | Method | Description                                          |
| ---------------- | ------ | ---------------------------------------------------- |
| `/`              | GET    | Service info and version                             |
| `/health`        | GET    | Liveness check (used by Cloud Run and load balancer) |
| `/predict`       | POST   | Single car prediction                                |
| `/predict/batch` | POST   | Batch prediction (up to N cars)                      |
| `/metrics`       | GET    | Prediction throughput and error counters             |

### Input Validation

All inputs are validated via Pydantic before reaching the model. Validated constraints include: non-negative mileage, positive engine power, `sold_at` ≥ `registration_date`, and enum membership for categorical fields (fuel type, car type, paint colour).

### Known Limitation — Latency

The current p50 latency is ~180ms against a target of <20ms. This is attributable to Random Forest inference time (100 trees) on a single vCPU. Identified mitigations not yet implemented due to dataset scope:

- **ONNX Runtime export** — typically yields 5-10x inference speedup with no model retraining
- **Increased CPU allocation** — Cloud Run supports up to 8 vCPUs per instance
- **Lighter model** — reducing `n_estimators` to 20-30 trades ~0.5% accuracy for ~5x inference speed

---

## CI/CD Pipeline

### Continuous Integration (`ci.yml`)

Triggered on every push to `main` and `develop`, and on all pull requests to `main`.

- Python matrix: 3.10 and 3.11
- `make lint` → flake8, pylint, black, isort
- `make test` → pytest with coverage

### Continuous Deployment (`cd.yml`)

Triggered on push to `main` or `staging`.

1. Authenticate to GCP via service account credentials (stored as GitHub secrets)
2. Build Docker image tagged with `github.sha`
3. Push to Google Artifact Registry (`europe-west1`)
4. Deploy new revision to Cloud Run

```
gcloud run deploy bmw-pricing-api \
  --allow-unauthenticated \
  --memory=1Gi --cpu=1 \
  --no-cpu-throttling \
  --min-instances=1 \
  --concurrency=20
```

> **On keyless authentication:** The current pipeline uses a long-lived service account key (GCP_SA_KEY). The production-grade approach is Workload Identity Federation, which eliminates stored credentials entirely by exchanging GitHub's OIDC token for a short-lived GCP token per run. This is the intended next step before any production promotion.

> **On model evaluation gates:** A mature MLOps pipeline includes a CI step that evaluates the newly trained model against a fixed golden dataset and blocks deployment if metrics regress beyond a threshold. This is not implemented here given the single-model scope of the project, but is the natural next extension.

### Load Testing (`load-test.yml`)

Triggered on push to `staging`.

- Tool: Locust
- Scenario: 10 concurrent users, ramp rate 2/s, 30s duration
- Metrics collected: p50/p95/p99 latency, throughput, error rate
- Report: HTML artifact attached to each workflow run

**Load test results (10 concurrent users, 29s):**

| Metric      | Result     | Target      |
| ----------- | ---------- | ----------- |
| p50 Latency | 180ms      | < 20ms      |
| p95 Latency | 240ms      | < 50ms      |
| p99 Latency | 270ms      | < 100ms     |
| Throughput  | 4.36 req/s | > 100 req/s |
| Error Rate  | 0%         | < 0.1%      |

The throughput gap reflects model inference time, not infrastructure capacity. Cloud Run scales horizontally to absorb concurrent load; the bottleneck is per-request CPU time.

---

## Monitoring & Drift Detection

To ensure predictions remain reliable after deployment, the following signals are monitored:

| Signal                       | Alert Threshold | Action                  |
| ---------------------------- | --------------- | ----------------------- |
| Avg input mileage            | ±30% shift      | Investigate data source |
| Avg model prediction         | ±20% shift      | Check for market shift  |
| KL Divergence (predictions)  | > 0.3           | Trigger retraining      |
| % unknown `model_key` values | > 10%           | Update training data    |

**Retraining is triggered when any 2 of the following 6 conditions are met:**

1. Model age > 90 days
2. MAE increase > 15% vs. deployment baseline
3. KL Divergence > 0.3
4. Prediction distribution shift > 20%
5. New labelled data > 10% of original training set size
6. Business metric (Tail Rate) drops > 5 percentage points

> **On model versioning:** The current deployment bakes the model artifact into the Docker image, providing implicit versioning via git SHA tags in Artifact Registry. A dedicated model registry (MLflow or Vertex AI Model Registry) would add richer metadata — training metrics, data lineage, and promotion history — and is the recommended next step for a multi-model or multi-team environment.

---

## Running Locally

```bash
# Clone and set up environment
git clone https://github.com/yourusername/bmw-pricing-challenger.git
cd bmw-pricing-challenger
python3 -m venv .venv && source .venv/bin/activate
make install

# Run API
python -m src.api.app

# In a second terminal — run integration tests
python -m scripts.deployment.test_api

# Run load test locally
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --headless -u 10 -r 2 --run-time 30s
```

---

## Project Structure

```
bmw-pricing-challenger/
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── model/
│   │   ├── train.py            # Training pipeline
│   │   └── preprocess_data.py  # Feature engineering
│   └── notebooks/
│       ├── EDA.ipynb
│       ├── tuning.ipynb
│       └── 02-xgboost-optimization.ipynb
├── tests/
│   ├── unit/
│   │   └── test_api.py
│   └── load/
│       └── locustfile.py
├── .github/workflows/
│   ├── ci.yml
│   ├── cd.yml
│   └── load-test.yml
├── Dockerfile
├── Makefile
├── requirements-dev.txt
└── README.md
```

---

## What I Would Do Next (Honest Roadmap)

| Priority  | Item                         | Rationale                                      |
| --------- | ---------------------------- | ---------------------------------------------- |
| 🔴 High   | Workload Identity Federation | Eliminate long-lived credentials               |
| 🔴 High   | Model evaluation gate in CD  | Prevent silent metric regression               |
| 🟡 Medium | ONNX Runtime export          | Reduce p50 latency from 180ms → ~20ms          |
| 🟡 Medium | MLflow Model Registry        | Track model versions with metrics              |
| 🟡 Medium | Canary traffic splitting     | Safe rollout of new model revisions            |
| 🟢 Low    | Optuna HPO                   | More sample-efficient than RandomizedSearchCV  |
| 🟢 Low    | Richer dataset               | Only path to meeting original business targets |

---

## References

- Dataset: [BMW Pricing Challenge — Kaggle](https://www.kaggle.com/)
- Bergstra & Bengio (2012): _Random Search for Hyper-Parameter Optimization_
- Chen & Guestrin (2016): _XGBoost: A Scalable Tree Boosting System_
