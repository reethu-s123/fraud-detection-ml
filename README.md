# Credit Card Fraud Detection System

A production-ready, Python-based machine learning system that detects fraudulent credit card transactions in near real-time by learning normal spending patterns from historical data and automatically flagging anomalous transactions.

## 🎯 Objectives

✅ **Real-world Problem**: Address a critical challenge faced by banks and fintech companies  
✅ **Handle Class Imbalance**: Manage highly imbalanced data (fraud rate ~0.2%)  
✅ **Multi-Model Approach**: Combine supervised and unsupervised learning  
✅ **Production-Ready**: Include API, monitoring, and deployment configurations  
✅ **Fraud-Optimized Metrics**: Focus on precision, recall, ROC-AUC, and cost-based evaluation  

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DATA INGESTION                          │
│        (Kaggle Credit Card Dataset / Custom Data)        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              DATA PREPROCESSING                          │
│  • Handle missing values                                 │
│  • Feature scaling (StandardScaler)                      │
│  • Train-test split                                      │
│  • Handle imbalance (SMOTE, RandomOverSampler)          │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬──────────────┐
         │                       │              │
         ▼                       ▼              ▼
   ┌─────────────┐      ┌──────────────┐  ┌─────────────┐
   │  Supervised │      │  Supervised  │  │ Unsupervised│
   │   Models    │      │   Models     │  │   Anomaly   │
   ├─────────────┤      ├──────────────┤  │  Detection  │
   │• Logistic   │      │• XGBoost     │  ├─────────────┤
   │  Regression │      │• Random      │  │• Isolation  │
   │             │      │  Forest      │  │  Forest     │
   │             │      │              │  │• Autoenoder │
   └──────┬──────┘      └──────┬───────┘  └──────┬──────┘
          │                    │                 │
          └────────────┬───────┴─────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │    MODEL EVALUATION          │
        │  • Precision/Recall/F1       │
        │  • ROC-AUC                   │
        │  • Cost-based metrics        │
        │  • Confusion matrix          │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   REAL-TIME API SERVICE      │
        │  • Single transaction scoring│
        │  • Batch prediction          │
        │  • Risk score calculation    │
        │  • Decision threshold config │
        └──────────────────────────────┘
```

## 📁 Project Structure

```
fraud-detection-ml/
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data ingestion & validation
│   ├── preprocessor.py          # Feature engineering & scaling
│   ├── model_builder.py         # ML model implementations
│   ├── evaluator.py             # Evaluation metrics & analysis
│   ├── fraud_detector.py        # Real-time scoring interface
│   └── utils.py                 # Utility functions
├── api/
│   ├── __init__.py
│   └── app.py                   # Flask REST API
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_model_builder.py
│   ├── test_evaluator.py
│   └── test_api.py
├── data/
│   ├── raw/                     # Original dataset
│   ├── processed/               # Preprocessed data
│   └── models/                  # Trained model artifacts
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_hyperparameter_tuning.ipynb
├── train_pipeline.py            # Main training script
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/reethu-s123/fraud-detection-ml.git
cd fraud-detection-ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get Data

Download the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud):

```bash
# Using Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
mv creditcard.csv data/raw/
```

### 3. Train Models

```bash
python train_pipeline.py
```

### 4. Start API Server

```bash
python api/app.py
```

API runs on `http://localhost:5000`

### 5. Test Predictions

```bash
# Single transaction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "Time": 1.0,
      "Amount": 100.0,
      "V1": -1.358, "V2": -0.072, "V3": 2.536,
      "V4": 1.377, "V5": -0.338, "V6": 0.462,
      "V7": 0.241, "V8": 0.865, "V9": 0.184,
      "V10": 0.458, "V11": 1.233, "V12": -0.374,
      "V13": 0.585, "V14": -0.269, "V15": 0.830,
      "V16": 0.817, "V17": -0.411, "V18": 0.266,
      "V19": -0.225, "V20": 0.060, "V21": 0.024,
      "V22": 0.038, "V23": 0.136, "V24": -0.289,
      "V25": 0.261, "V26": -0.113, "V27": 0.081,
      "V28": -0.046
    }
  }'

# Batch transactions
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{...}, {...}]}'
```

## 🎯 Key Features

### 1. **Handling Class Imbalance**

- ✅ SMOTE (Synthetic Minority Oversampling Technique)
- ✅ RandomOverSampler / RandomUnderSampler
- ✅ XGBoost's `scale_pos_weight` parameter
- ✅ Class weighting in Logistic Regression & Random Forest
- ✅ Cost-sensitive learning (penalize missed frauds)

### 2. **Multi-Model Approach**

#### Supervised Learning:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Feature importance, robustness
- **XGBoost**: State-of-the-art gradient boosting performance

#### Unsupervised Anomaly Detection:
- **Isolation Forest**: Detects novel fraud patterns

### 3. **Fraud-Optimized Evaluation**

```
Metrics:
├── Precision: False alarm rate minimization
├── Recall: Fraud detection rate
├── F1-Score: Balanced performance
├── ROC-AUC: Discrimination ability
├── Cost-Based: Penalize missed frauds (500x cost)
├── Confusion Matrix: True/False Positive/Negative breakdown
└── PR-Curve: Precision-Recall tradeoff
```

### 4. **Real-time API**

```
Endpoints:
├── POST /predict              - Single transaction
├── POST /predict_batch        - Multiple transactions
├── POST /set_threshold        - Adjust decision boundary
├── GET  /model_info           - Model metadata
├── GET  /feature_importance   - Feature importance scores
└── GET  /health               - Health check
```

## 📊 Expected Performance

| Metric | Target | Status |
|--------|--------|--------|
| Precision | >80% | ✅ Balanced false alarms |
| Recall | >75% | ✅ Catches most frauds |
| ROC-AUC | >0.95 | ✅ Strong discrimination |
| Latency | <100ms | ✅ Real-time requirement |
| Training Time | <10min | ✅ Rapid iteration |

## 🔧 Configuration

Edit config parameters in `train_pipeline.py`:

```python
# Data settings
test_size = 0.2
random_state = 42

# Preprocessing
scaler_type = 'standard'  # standard, minmax, robust
imbalance_strategy = 'smote'  # smote, oversample, undersample

# Model hyperparameters
xgboost_params = {
    'n_estimators': 100,
    'max_depth': 7,
    'learning_rate': 0.1,
    'scale_pos_weight': 200  # Handle imbalance
}

# Evaluation
cost_false_negative = 500  # Cost of missing fraud
cost_false_positive = 1    # Cost of false alarm
```

## 📈 Model Training Pipeline

```
1. Load Data
   ├─ Validate schema
   ├─ Check for missing values
   └─ Print dataset statistics

2. Explore Data
   ├─ Class distribution
   ├─ Feature correlations
   ├─ Anomaly detection preview
   └─ Visualization

3. Preprocess
   ├─ Scaling (StandardScaler)
   ├─ Train-test split (80-20)
   ├─ Handle imbalance on training set only
   └─ Save preprocessor for inference

4. Build Models
   ├─ Logistic Regression (baseline)
   ├─ Random Forest
   ├─ XGBoost
   └─ Isolation Forest

5. Evaluate
   ├─ Individual model metrics
   ├─ Ensemble comparison
   ├─ Cost-based analysis
   └─ Feature importance

6. Save Artifacts
   ├─ Trained models
   ├─ Preprocessor
   ├─ Scaler
   ├─ Feature names
   └─ Performance reports
```

## 🔍 Handling Class Imbalance in Depth

### Problem
- Frauds: ~0.17% of transactions
- Normal: ~99.83% of transactions
- **Standard models bias towards majority class**

### Solutions Implemented

**1. Resampling (on training set only)**
```python
from imblearn.over_sampling import SMOTE

# SMOTE: Create synthetic minority examples
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

**2. Cost-Sensitive Learning**
```python
# XGBoost
model = xgb.XGBClassifier(
    scale_pos_weight=200,  # 200x weight for fraud class
    max_depth=7,
    learning_rate=0.1
)

# Logistic Regression
model = LogisticRegression(
    class_weight='balanced',  # Auto-weight inversely to class frequency
    max_iter=1000
)
```

**3. Appropriate Metrics**
```python
# Don't use accuracy! Use:
- Precision: TP / (TP + FP)  - False alarm rate
- Recall: TP / (TP + FN)     - Detection rate
- F1: Harmonic mean of precision & recall
- ROC-AUC: Probability ranking quality
- PR-AUC: Precision-recall curve (better for imbalanced)
```

## 🚀 Advanced Features

### Feature Importance Analysis
```python
importances = model.feature_importances_
top_features = sorted(zip(feature_names, importances), 
                     key=lambda x: x[1], reverse=True)[:10]
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 7, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

## 📚 Learning Resources

- [Imbalanced Learning with Python](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Credit Card Fraud Detection Paper](https://arxiv.org/abs/1402.6352)

## 📧 Support

For questions or issues, please open a GitHub issue.

---

**Happy fraud detection! 🛡️**
