# Chinese Real Estate Demand Forecasting

Time series forecasting of housing transaction amounts for 96 sectors across 12 future months using EWGM (Exponentially Weighted Geometric Mean).

## Problem

Predict monthly housing transaction amounts for 96 geographic sectors in China from August 2024 to July 2025, using historical data from January 2019 to July 2024.

**Competition Metric**: Two-stage MAPE with outlier penalty
- Stage 1: Score = 0 if >30% of predictions have >100% error
- Stage 2: Scaled MAPE on predictions with ≤100% error

## Approach

### Final Model: EWGM with POI Features
- **Score**: 0.56569 (baseline: 0.56006)
- **Method**: Exponentially weighted geometric mean with Point of Interest feature enhancement
- **Key Parameters**: 
  - 6-month lookback window
  - Alpha = 0.5 (exponential decay)
  - December seasonal multiplier (0.85-1.4x)
  - POI weight = 0.1

### Why EWGM?
- Geometric mean handles zeros naturally (common in sparse sectors)
- Exponential weighting captures recent market trends
- Simple enough to avoid overfitting on limited data (67 months)
- Robust to outliers (critical for competition scoring function)

## Models Attempted

| Model | Validation | Actual Score | Correlation | Result |
|-------|-----------|--------------|-------------|---------|
| Basic EWGM | - | 0.56006 | 1.00 | ✓ Baseline |
| EWGM + POI | - | 0.56569 | 0.98 | ✓ Best |
| Prophet | 0.615 | 0.18415 | 0.50 | ✗ Predicted mean reversion |
| Sector-Specific Alphas | 0.615 | - | 0.50 | ✗ Overfitting |
| Optuna | 0.594 | - | 0.47 | ✗ Overfitting |
| XGBoost | 0.90 CV | - | 0.05 | ✗ Learned broken patterns |
| Lasso | - | - | 0.02 | ✗ No relationships |
| EWGM Ensemble | 0.9651 | 0.53930 | 0.97 | ✗ Seasonality conflict |

## Key Learnings

1. **Simplicity wins in declining markets** - Complex ML models learned historical patterns that no longer apply (Chinese property crisis post-2021)
2. **Validation is unreliable without test feedback** - Models with 0.9+ validation scores and low correlation (<0.5) consistently failed
3. **Low correlation = high risk** - Any model <0.85 correlation with baseline diverged too much and failed
4. **December seasonality conflicts with trends** - Month-specific models captured declining Decembers, fighting against seasonal multipliers
5. **The metric matters** - Two-stage MAPE with outlier penalties requires robust predictions; extreme values get instant score=0

## Repository Structure

```
├── notebooks/
│   ├── 12_DAN_Submission_EWGM_POI.ipynb        # Final baseline model (0.56569)
│   ├── 13_EWGM_Ensemble.ipynb                   # Ensemble experiments
│   └── 09_Less_Variables_XGBoost.ipynb         # XGBoost experiments (failed)
├── data/
│   ├── train/                                  # Historical transaction data
│   └── sample_submission.csv                   # Submission format
└── output/
    └── submissions/                            # Generated predictions
```

## Usage

```python
# Generate predictions
python notebooks/EWGM_POI.ipynb

# Modify parameters in CONFIG dictionary
CONFIG = {
    'n_lags': 6,        # Lookback window
    'alpha': 0.5,       # Exponential decay
    'poi_weight': 0.1,  # POI feature weight
}
```

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```

## Competition Context

Part of a Kaggle-style competition for Chinese real estate forecasting. The challenge highlighted how structural market changes (property crisis) make historical patterns unreliable, favoring simple trend-following approaches over sophisticated pattern learning.

**Key Constraint**: Limited submissions meant optimization without feedback loop, making conservative approaches (high baseline correlation) essential.