# Chinese Real Estate Demand Forecasting

Time series forecasting of housing transaction amounts for 96 sectors across 12 future months using EWGM (Exponentially Weighted Geometric Mean).

## Problem

Predict monthly housing transaction amounts for 96 geographic sectors in China from August 2024 to July 2025, using historical data from January 2019 to July 2024.

**Competition Metric**: Two-stage MAPE with outlier penalty
- Stage 1: Score = 0 if >30% of predictions have >100% error
- Stage 2: Scaled MAPE on predictions with ≤100% error

## Approach

### Final Model: Optimized EWGM Ensemble with Aggressive Market Correction
- **Score**: 0.57974 (baseline: 0.56006, +3.52% improvement)
- **Method**: Two-stage ensemble combining trend-following with seasonal patterns
  - **Method 1**: Weighted Geometric Mean with exponential decay (α=0.5, 6-month lookback)
  - **Method 2**: EWGM with December seasonality bump (α=0.5, 7-month lookback, multiplier clipped 0.85-1.4x)
  - **Ensemble**: **0.2 × Method1 + 0.65 × Method2** (discovered via systematic weight search)

### Key Insights
- **Aggressive market correction (sum=0.85)**: 15% haircut applied to all predictions reflects accelerating market decline
  - Original Kaggle weights (0.3, 0.6, sum=0.90) were too optimistic
  - Systematic testing revealed sum=0.85 as optimal sweet spot
  - More aggressive (sum=0.80) and less aggressive (sum=0.90) both underperform
- **Seasonality dominance (3.25:1 ratio)**: December patterns remain stable even as volumes crash
  - Weight ratio w2/w1 = 3.25 (seasonality is 3.25× more predictive than recent trends)
  - In declining markets, relative seasonal behavior persists while absolute volumes collapse
- **Why EWGM works**: 
  - Geometric mean handles zeros naturally (common in sparse sectors)
  - Exponential weighting (α=0.5) emphasizes recent months while smoothing volatility
  - Simple enough to avoid overfitting on limited data (67 training months)
  - Robust to outliers (critical for competition's two-stage MAPE scoring)
- **What failed**: Complex ML (XGBoost, Prophet, Lasso) overfit to broken historical patterns with near-zero correlation to baseline; simple trend-following + seasonality wins

### Weight Optimization Process
1. **Initial model**: Kaggle baseline with (0.3, 0.6) → 0.57682
2. **Systematic search**: Manually tested weight combinations exploring different sums and ratios
3. **Discovery**: Sum=0.85 outperforms due to market acceleration requiring aggressive discount
4. **Current best**: (0.2, 0.65) with sum=0.85 → 0.57974

### Weight Search Results
| w1 | w2 | Sum | Ratio | Score | Notes |
|----|----|-----|-------|-------|-------|
| 0.30 | 0.60 | 0.90 | 2.00 | 0.57682 | Kaggle baseline |
| 0.25 | 0.65 | 0.90 | 2.60 | 0.57695 | Same sum, higher seasonality |
| **0.20** | **0.65** | **0.85** | **3.25** | **0.57974** | **Optimal: aggressive haircut + strong seasonality** |
| 0.20 | 0.60 | 0.80 | 3.00 | 0.57714 | Too aggressive |
| 0.15 | 0.55 | 0.70 | 3.67 | 0.56599 | Far too pessimistic |

## Models Attempted

| Model | Custom MAPE | Actual Score | Correlation | Result |
|-------|-----------|--------------|-------------|---------|
| Basic EWGM | 0.7817 | 0.56006 | 1.00 | ✓ Baseline |
| EWGM + POI | 0.7539 | 0.56569 | 0.98 | ✓ Third Best |
| **EWGM Ensemble (Optimized)** | - | **0.57974** | - | **✓ Best** |
| EWGM Ensemble (Kaggle) | - | 0.57682 | - | ✓ Second Best |
| Prophet | 0.615 | 0.18415 | 0.50 | ✗ Predicted mean reversion |
| EWGM Sector-Specific Alphas | 0.3849 | - | 0.50 | ✗ Overfitting |
| Optuna XGBoost | 0.594 | 0.42311 | 0.47 | ✗ Overfitting |
| XGBoost | 0.9076 | 0.44336 | 0.05 | ✗ Learned broken patterns |
| Lasso | - | 0.18415 | 0.02 | ✗ No relationships |
| EWGM Ensemble (Seasonality Conflict) | 0.9651 | 0.56110 | 0.97 | ✗ December trends fought multipliers |

## Key Learnings

1. **Simplicity wins in declining markets** - Complex ML models learned historical patterns that no longer apply (Chinese property crisis post-2021)
2. **Non-normalized ensemble weights reveal market dynamics** - Sum<1.0 indicates systematic over-prediction; optimal sum=0.85 suggests 15% haircut needed
3. **Seasonality persists through crashes** - December patterns remained predictable even as absolute volumes collapsed, requiring 3.25:1 seasonality:trend ratio
4. **Validation is unreliable without test feedback** - Models with 0.9+ validation scores and low correlation (<0.5) consistently failed
5. **Low correlation = high risk** - Any model <0.85 correlation with baseline diverged too much and failed
6. **The metric matters** - Two-stage MAPE with outlier penalties requires robust predictions; extreme values get instant score=0

## Repository Structure

```
├── notebooks/
│   ├── 09_Less_Variables_XGBoost.ipynb         # XGBoost experiments (failed)
│   ├── 12_DAN_Submission_EWGM_POI.ipynb        # Baseline model (0.56569)
│   ├── 13_EWGM_Ensemble.ipynb                  # Initial ensemble experiments
│   └── 14_EWGM_Kaggle_Ensemble.ipynb           # Optimized ensemble (0.57974)
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

# Ensemble Configuration (Optimized)
CONFIG_ENSEMBLE = {
    'weight_method1': 0.2,     # Weight for Weighted Geometric Mean
    'weight_method2': 0.65,    # Weight for Seasonality Bump
    # Sum = 0.85 (15% market correction applied)
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

**Key Constraint**: Limited daily submissions (5/day) meant optimization required systematic manual search rather than automated grid search, making each test valuable for understanding the weight-score landscape.

---

**Final Insight**: The winning 0.85 weight sum reveals that both methods (pure trend-following and seasonality-adjusted) systematically over-predict by ~15% in the accelerating decline, while the 3.25:1 ratio shows December patterns are far more reliable than recent trajectories in a crashing market.