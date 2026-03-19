### Dataset web address

https://www.kaggle.com/datasets/danielkyrka/bmw-pricing-challenge?resource=download

Observation:

# Enhanced End-to-End ML Workflow: BMW Pricing System

## Project Overview

### Purpose of the Project

This project's purpose is to build a machine learning system that predicts used BMW vehicle prices to replace manual pricing by dealership specialists.
The system delivers predictions via REST API with sub-50ms latency, achieving MAE < €2,500 across all BMW segments.

### The current situation

BMW dealerships face a critical business problem: pricing used vehicles accurately in a volatile market.
This problem is determined by the fact that dealerships rely on manual pricing based on book values, competitor analysis, and gut instinct.

### Impact so far

This problem causes a 25% of inventory to be **underpriced** or **overpriced** by more than 5%.
Because the market consists of 2400 cars/year, this means 600 cars are mispriced during a year - for a total
cost of €804K/year - and only 1800 cars are safely priced.

**Underpricing (10% of inventory):**

Even though cars sell immediately, dealership leaves €480K/year due to underpricing:

- 240 cars/year are priced >5% below market value, which results in revenue losses
- Average €2,000 lost margin per car
- Annual impact: 240\*€2000=€480K lost revenue

**Overpricing (15% of inventory):**

Also, €324K/year is lost due to additional holding costs:

- 360 cars/year priced >5% above market value
- Cars sit 18 extra days beyond target 30-day sale window
- Holding costs: €50/day × 18 days = €900 per car for a total of 900\*360=€324K/year

As a result: **324K/year+490K/year=€804K/year**.

## Objective

Therefore, the **business objective** is to decrease €804K/year by reducing the error with which 25% of cars are mispriced.
Specifically, we want to build a model that reduces both the frequency and magnitude of underpricing and overpricing errors, in a way that can be observed and validated during model development.

### Observation

Building an accurate model, in relation to the BMW-pricing challenge, represents a way of mimicking a buyer's intention to purchase a car.
I want to predict the maximum price a buyer is willing to pay AT AUCTION, based on the car's characteristics.
As a result, the actual price is the maximum price a buyer is willing to pay.

### M1. Business Metrics

- TC-APE (Tail-Conditional Absolute Percentage Error ): <6.5%, given a reasonable baseline value of 7.5%
  That means, we expect the average of the absolute percentage error across the mispriced cars to be less than 6.5%
  _Meaning_: Average percent error across mispriced cars
- TR (Tail Rate): <15% given the baseline value of TR = # cars with error>5% /total cars = 25%
  Our target is to reduce 25% of mispriced cars to 15%, which implies that we accept losing money due to a 6% and 9% (6%+9%=15%) of under- and over-priced cars.
  If we _accept_ this, then we accept to lose:
  - 480K\*6/10 = 288K
  - 324K \* 9 / 15 = 194K

**Financial Targets:**

| Metric                      | Current | Target | Annual Savings |
| --------------------------- | ------- | ------ | -------------- |
| Lost margin (underpricing)  | €480K   | €288K  | €192K          |
| Holding costs (overpricing) | €324K   | €194K  | €130K          |
| **Total Annual Savings**    | €804K   | €482K  | **€322K**      |

**Monthly Monitoring Targets:**

This is the money we have accepted to lose, on average, every month:

- Expected monthly revenue loss < 288K/12=€24K (currently €40K)
- Expected monthly holding costs < 194K/12=€16K (currently €27K)
- Total monthly impact < 804K/12=€40K (currently €67K)

To fulfill the business metrics, it's key to satisfy the following performance metrics.

### M2. Model Performance

- MAE (Mean Absolute Error): €1500 - €2500
  _Meaning_: Average absolute error across all cars
- MAPE (Mean Absolute Percentage Error): 3.5% - 4.5%
  _Meaning_: Average percent error across all cars
- R^2: >=0.85, which measures how well the model's features capture most of the factors that drive the price

- RMSE: €2,200 - €3,000. RMSE penalizes large prediction errors

**Technical Performance Targets:**

| Metric                             | Target (Good) | Max Acceptable | Business Meaning                                     |
| ---------------------------------- | ------------- | -------------- | ---------------------------------------------------- |
| **MAE** (Mean Absolute Error)      | < €1500       | < €2500        | Average price prediction error                       |
| **RMSE** (Root Mean Squared Error) | < 2200        | < €3000        | Penalizes large errors more heavily                  |
| **R²** (R-squared)                 | > 0.88        | > 0.85         | % of price variance the model explains               |
| **MAPE** (Mean Absolute % Error)   | < 3.5%        | < 4.5%         | Scale-independent error (works for all price ranges) |

---

### M3. API Performance Metrics (System Speed & Reliability)

| Metric          | Target (Good) | Max Acceptable | What It Measures                                                |
| --------------- | ------------- | -------------- | --------------------------------------------------------------- |
| **p50 Latency** | < 20ms        | < 30ms         | Typical response time (50% of requests)                         |
| **p95 Latency** | < 50ms        | < 100ms        | 95% of requests meet this SLA                                   |
| **p99 Latency** | < 100ms       | < 200ms        | Worst-case response time (1 in 100 requests feels slow)         |
| **API Uptime**  | > 99.9%       | > 99.5%        | Service availability (downtime < 8 hours/year)                  |
| **Error Rate**  | < 0.1%        | < 0.5%         | Failed predictions (validation or server errors) 1 in 1000 fail |
| **Throughput**  | > 100 req/sec | > 50 req/sec   | How many concurrent requests we can handle                      |

**Why These Targets?**

- Real-time pricing requires instant responses (< 50ms feels instant to users)
- Production SLA: 99.9% uptime = max 30 days _ 24 hours _ 60 minutes _ (1 - 0.999) = 43,200 minutes _ 0.001 = 43.2 minutes downtime/month
- Peak traffic: Dealership website may get 100+ pricing requests/second

---

### M4. Segment Performance Metrics (Quality Assurance by Car Type)

| Car Price Segment       | MAE Target | Rationale                                                                  |
| ----------------------- | ---------- | -------------------------------------------------------------------------- |
| **Economy** (< €20K)    | < €1,500   | ~10% of a €15K car. Lower absolute errors acceptable for cheaper cars      |
| **Mid-Range** (€20-35K) | < €2,500   | ~9% of a €27.5K car. Core BMW segment - standard target                    |
| **Premium** (€35-50K)   | < €3,000   | ~7% of a €42.5K car. Higher value justifies slightly higher absolute error |
| **Luxury** (> €50K)     | < €4,000   | <8% of a €50K+ car. Rare, high-variance models - relaxed threshold         |

---

### M5. Data Quality & Drift Monitoring to protect the business metrics (ensure that €322K are indeed saved)

| Metric                                   | Baseline Value | Alert Threshold                         | What To Do                                             |
| ---------------------------------------- | -------------- | --------------------------------------- | ------------------------------------------------------ |
| **Avg Input Mileage**                    | 45,000 km      | ±30% shift (< 31,500 km or > 58,500 km) | Data distribution changed - investigate source         |
| **Avg Input Price** (actual)             | €35,000        | ±20% shift (< €28,000 or > €42,000)     | Market shift or data quality issue                     |
| **Avg Model Prediction**                 | €35,000        | ±20% shift (< €28,000 or > €42,000)     | Model drift detected - consider retraining             |
| **Feature Distribution** (KL Divergence) | < 0.2          | > 0.3                                   | Significant drift - retraining needed                  |
| **% Unknown Model Keys**                 | < 5%           | > 10%                                   | New BMW models appearing that weren't in training data |

**What Is KL Divergence?**

- Measures how much the current data distribution differs from training data
- 0.0 = identical distributions
- 0.1 = minor drift (acceptable)
- 0.3+ = significant drift (retrain model)

**Example Drift Scenario:**

```
Week 1:  Avg mileage = 45,000 km ✅
Week 2:  Avg mileage = 48,000 km ✅ (+6.7%, within ±30%)
Week 3:  Avg mileage = 52,000 km ⚠️  (+15.6%, monitor closely)
Week 4:  Avg mileage = 62,000 km ❌ (+37.8%, ALERT!)

Action: Investigate why high-mileage cars are suddenly appearing
Possible causes:
- Data source changed (now including fleet vehicles)
- Market shift (more used cars entering market)
- Data quality issue (mileage field corrupted)
```

---

### M6. Retraining Triggers (When to update the model and to prevent unnecessary retraining from a single noisy signal)

**Retrain if 2 or more of these conditions are met:**

| Trigger                        | Threshold                               | Current Status Example                           |
| ------------------------------ | --------------------------------------- | ------------------------------------------------ |
| **1. Time-based**              | Model > 90 days old                     | Last trained: 92 days ago ⚠️                     |
| **2. Performance degradation** | MAE increased > 15%                     | Current: €2,485 vs Baseline: €2,285 (+8.8%) ✅   |
| **3. Data drift**              | KL Divergence > 0.3                     | Current: 0.28 ⚠️ (approaching)                   |
| **4. Prediction drift**        | Avg prediction shifted > 20%            | Current: €36,200 vs Baseline: €35,000 (+3.4%) ✅ |
| **5. New data available**      | New samples > 10% of training set       | 4,200 new samples vs 3,200 training (+131%) ⚠️   |
| **6. Business metric drop**    | Safe zone dropped > 5 percentage points | Current: 75.2% vs Baseline: 77.8% (-2.6pp) ✅    |

**Example Decision:**

```
✅ Criterion 1: TRIGGERED (92 days > 90 days)
✅ Criterion 3: TRIGGERED (drift approaching threshold)
✅ Criterion 5: TRIGGERED (131% new data available)

3 out of 6 criteria met → RETRAIN MODEL ✅

Expected impact:
- Capture latest market trends
- Include 4,200 recent sales
- Should reduce MAE by 5-10%
```

---

## Summary: Critical Metrics at a Glance

### ✅ Model is HEALTHY when:

- MAE < €2,500 (technical performance)
- Safe zone > 75% (business impact)
- p95 latency < 50ms (API performance)
- All segments within targets (quality assurance)
- KL divergence < 0.2 (no data drift)

### ⚠️ Model needs ATTENTION when:

- MAE €2,500 - €3,000 (acceptable but not ideal)
- Safe zone 70-75% (monitor closely)
- p95 latency 50-100ms (slower but usable)
- Some segments 10-20% above target (check for bias)
- KL divergence 0.2-0.3 (drift detected, plan retraining)

### ❌ Model needs IMMEDIATE ACTION when:

- MAE > €3,000 (unacceptable errors)
- Safe zone < 70% (too much financial risk)
- p95 latency > 100ms (too slow for production)
- Any segment > 20% above target (systematic failure)
- KL divergence > 0.3 (significant drift, retrain now)

---

# EDA Framework for BMW Pricing Model

1.  Basic Statistics:

- Rows:
- Columns:
- Memory:
- Column Names
- Rows: the first 5 or 10
- Data Types:
- The number of non-null values

2. Perform Critical Sanity Check

2.1) Ensure the dataset to have at least 1000 samples
If not, you need more data

2.2) Check out whether or not there are corrupted data:

    - Ensure that the data types are appropriate

    - Negative values for mileage or fuel
    Remove the corrupted samples if they represent a tiny %, e.g., 0.1%.

    - A car with zero power (or null value)
    Choose a robust measure of central tendency, e.g., the _median_, and use this to perform **imputation**.

    - Between 5% to 20% of data values are missing.
    In this case, perform imputation

    - A car registered in future years

    - Cars' models do not exist at all or, they do, but they are no longer available

    - All target values are identical

2.4) Check the dataset has intrinsic diversity

2.5) Check the distribution

    - Check skewness. Is the mean value larger (right-skewed distribution) or smaller than the median value? (high skewness)
    In this case, perform the LOG-TRANSFORM on the target to change the way the model sees the difference between
    very low values and very large values

    - Use box plots to:
        1. Spot outliers using the 1.5 * IQR Method
        In this case, you have to CAP at a certain value
        2. To compare two or more features' distributions

    - Are there skewed features (imbalance in features)? One or more feature-related categories appear less than 5% of
    times among the dataset rows
    If yes, then think grouping these features into one.

3. Assess the correlation degree
   - If two or more features tell a similar story engineer another feature to avoid _redundancy_. (Multicollinearity,
     two or more features are correlated)
   - Keep highly-correlated features
   - Add features that are highly correlated to the target
   - If Medians and distributions overlap almost entirely (same box plots), then the category isn't a good predictor, and it must be dropped.

4. Convert raw data
   - If a feature is a date or raw timestamp, then convert it into one variable that allows a linear signal
   - If the feature is text (e.g., "Diesel", "Petrol"), then use Box Plots to see if the median price varies significantly between categories.
   - Encode categorical values

# Perform hyperparameter tuning and train the model

A set of tuning-model training loops were performed but, although the use of additional advanced features, data fundamentally lacks the information needed to predict prices accurately. This is shown by the following JSON report:

{
"timestamp": "2026-03-10T16:26:37.324573",
"model_name": "rand_forest",
"dataset": {
"training_samples": 2324,
"validation_samples": 581,
"test_samples": 969
},
"model_parameters": {
"n_estimators": 100,
"max_depth": 40,
"min_samples_split": 10,
"random_state": 42
},
"metrics": {
"mae": 1945.679592979548,
"rmse": 2874.9630387491516,
"r2": 0.7755025263705161,
"mape": 20.484171910310597,
"tail_rate": 71.77280550774526,
"tc_ape": 27.537560756963185
},
"targets": {
"mae_target": 2500,
"rmse_target": 3000,
"r2_target": 0.85,
"mape_target": 4.5,
"tr_target": 15.0,
"tc_ape_target": 6.5
},
"targets_met": {
"mae_ok": true,
"rmse_ok": true,
"r2_ok": false,
"mape_ok": false,
"tr_ok": false,
"tc_ape_ok": false
}
}

# Replacing random forests with XGBoost

An additional evidence showing the lack of appropriate information needed to predict prices accurately is supplied by
the use of XGBoost algorithm, which doesn't improve the aforementioned metrics.
The report below clarifies this situation:

{
"timestamp": "2026-03-07T17:32:24.201415",
"random_forest": {
"mae": 2031.2387000424712,
"rmse": 2994.046112015934,
"r2": 0.7942005910180998,
"mape": 17.908201431624306,
"tail_rate": 72.80550774526678,
"tc_ape": 23.65568102842557,
"training_time_seconds": 2.2544069290161133
},
"xgboost": {
"mae": 1980.6225309650983,
"rmse": 3043.888612417959,
"r2": 0.7652948687294544,
"mape": 18.002490100105494,
"tail_rate": 67.64199655765921,
"tc_ape": 25.477598113519086,
"training_time_seconds": 2.6423001289367676
},
"improvements": {
"mae_improvement_pct": 2.491886801699601,
"r2_improvement_pct": -3.639599695033046,
"tail_rate_improvement_pct": 7.092198581560277
}
}

# The decision

Rather than spending months on marginal improvements, i made a pragmatic decision to deploy at industry-standard performance (€1,946 MAE, R² 0.78).
As a matter of fact, i detected and fixed data leakage that artificially inflated R² to 99%.

python -m src.api.app &
python -m scripts.deployment.test_api

lsof -i :8000
ps aux | grep python

# MLOps

## Create a virtual environment and initialize a Git repository on GitHub

python3 -m venv ~/.portfolio-bmw-pricing-v1
source ~/.portfolio-bmw-pricing-v1/bin/activate

## Define project dependencies in requirements.txt (production) and requirements-dev.txt (development)

-r requirements.txt
black==24.3.0
isort==5.13.2
flake8==7.0.0
pylint==3.1.0
pytest==8.0.2
pytest-cov==4.1.0

## Create a Makefile with common recipes and run them

install:
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e . # <--- installs the project as an editable package, and it thus requires a setup.py

lint:
flake8 src/ scripts/ tests/
pylint src/ scripts/ tests/

format:
black src/ scripts/ tests/
isort src/ scripts/ tests/

- Test key files residing in src/ and scripts/
  test:
  pytest tests/ -v --cov=src --cov-report=term-missing

## Create a new GitHub repository

### Purpose: Set up a clean, version‑controlled project with automated quality tools so that one can install dependencies, lint, format, and test the code using simple make commands

1. Create a github repository

2. Initialize git and commit the code

GitHub actions have the ability to "listen" to the code changes and to resort to previous states

git init
git add . # tells git to watch in all the folders and identify new files or file updates
git commit -m "Initial commit with project code and CI/CD scaffolding"

2. Connect local repository to github repository

git remote add origin https://github.com/YOUR_USERNAME/bmw-pricing-challenger.git
git branch -M main
git push -u origin main

3. Test the Makefile targets

Key test to ensure that the recipes work

make install  
make lint  
make format  
make test

## GitHub Actions – Test with Two or More Python Versions

Purpose:

- checking the code works with multiple Python versions
- catching syntax, format bugs
- performing unit tests on specified testing files

git add .
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI workflow"
git push
