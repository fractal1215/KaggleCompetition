# Chinese Real Estate Demand Forecasting

Time series forecasting of housing transaction amounts for 96 sectors across 12 future months using EWGM (Exponentially Weighted Geometric Mean).

## Problem

Predict monthly housing transaction amounts for 96 geographic sectors in China from August 2024 to July 2025, using historical data from January 2019 to July 2024.

**Competition Metric**: Two-stage MAPE with outlier penalty
- Stage 1: Score = 0 if >30% of predictions have >100% error
- Stage 2: Scaled MAPE on predictions with ≤100% error

## Approach

### Final Model: Optimized EWGM Ensemble with Validation-Tuned Weights
- **Score**: 0.57682 (baseline: 0.56006, +2.99% improvement)
- **Method**: Two-stage ensemble combining trend-following with seasonal patterns
  - **Method 1**: Weighted Geometric Mean with exponential decay (α=0.5, 6-month lookback)
  - **Method 2**: EWGM with December seasonality bump (α=0.5, 7-month lookback, multiplier clipped 0.85-1.4x)
  - **Ensemble**: 0.3 × Method1 + 0.6 × Method2

### Details
- **Non-normalized weights (sum=0.9)**: Both methods over-predict in the declining market; 10% haircut improves accuracy
- **Heavy seasonality weighting (2:1 ratio)**: December patterns are more stable than recent trends in a crashing market
- **Why EWGM works**: 
  - Geometric mean handles zeros naturally (common in sparse sectors)
  - Exponential weighting (α=0.5) emphasizes recent months while smoothing volatility
  - Simple enough to avoid overfitting on limited data (67 training months)
  - Robust to outliers (critical for competition's two-stage MAPE scoring)

## Models Attempted

| Model | Custom MAPE | Actual Score | Correlation | Result |
|-------|-----------|--------------|-------------|---------|
| Basic EWGM | 0.7817 | 0.56006 | 1.00 | ✓ Baseline |
| EWGM + POI | 0.7539 | 0.56569 | 0.98 | ✓ Second Best |
| Prophet | 0.615 | 0.18415 | 0.50 | ✗ Predicted mean reversion |
| EWGM Sector-Specific Alphas | 0.3849 | - | 0.50 | ✗ Overfitting |
| Optuna XGBosst | 0.594 | 0.42311 | 0.47 | ✗ Overfitting |
| XGBoost | 0.9076 | 0.44336 | 0.05 | ✗ Learned broken patterns |
| Lasso | - | 0.18415 | 0.02 | ✗ No relationships |
| EWGM Ensemble | 0.9651 | 0.56110 | 0.97 | ✗ Seasonality conflict |
| EWGM Ensemble KAGGLE | - | 0.57682 | - | ✓ Best |

## Key Learnings

1. **Simplicity wins in declining markets** - Complex ML models learned historical patterns that no longer apply (Chinese property crisis post-2021)
2. **Validation is unreliable without test feedback** - Models with 0.9+ validation scores and low correlation (<0.5) consistently failed
3. **Low correlation = high risk** - Any model <0.85 correlation with baseline diverged too much and failed
4. **December seasonality conflicts with trends** - Month-specific models captured declining Decembers, fighting against seasonal multipliers
5. **The metric matters** - Two-stage MAPE with outlier penalties requires robust predictions; extreme values get instant score=0

## Repository Structure

```
├── notebooks/
│   ├── 09_Less_Variables_XGBoost.ipynb         # XGBoost experiments (failed)
│   ├── 12_DAN_Submission_EWGM_POI.ipynb        # Final baseline model (0.56569)
│   ├── 13_EWGM_Ensemble.ipynb                  # Ensemble experiments
│   └── 14_EWGM_Kaggle_Ensemble.ipynb           # Highest Scoring submission (0.57682)
├── data/
│   ├── train/                                  # Historical transaction data
│   └── sample_submission.csv                   # Submission format
└── output/
    └── submissions/                            # Generated predictions
```

## Usage

```python
# Method 1: Weighted Geometric Mean Configuration
CONFIG_METHOD1 = {
    'n_lags': 6,          # Number of months to look back
    'alpha': 0.5,         # Exponential decay parameter (0 < alpha < 1)
    't2': 6,              # Months to check for baseline condition (zero-handling)
}

# Method 2: Seasonality Bump Configuration
CONFIG_METHOD2 = {
    'n_lags': 7,          # Number of months to look back
    'alpha': 0.5,         # Exponential decay parameter
    't2': 6,              # Months to check for baseline condition
    'clip_low': 0.85,     # Lower bound for December multiplier
    'clip_high': 1.40,    # Upper bound for December multiplier
}

# Ensemble Configuration
CONFIG_ENSEMBLE = {
    'weight_method1': 0.3,    # Weight for Weighted Geometric Mean
    'weight_method2': 0.6,    # Weight for Seasonality Bump
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