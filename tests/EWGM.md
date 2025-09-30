# =====================================================
# Utility Functions
# =====================================================

def build_month_codes():
    """Create a mapping from month abbreviations to numeric values."""
    return {m: i for i, m in enumerate(
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start=1)}


def split_test_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ID column into month text and sector components."""
    parts = df['id'].str.split('_', expand=True)
    df['month_text'], df['sector'] = parts[0], parts[1]
    return df


def add_time_and_sector_fields(df: pd.DataFrame, month_codes: dict) -> pd.DataFrame:
    """Add parsed year, month, time index, and sector_id to dataframe."""
    if 'sector' in df.columns:
        df['sector_id'] = df['sector'].str[7:].astype(int)

    if 'month' in df.columns:  # test data
        df['year'] = df['month'].str[:4].astype(int)
        df['month'] = df['month'].str[5:].map(month_codes)
    else:  # train data
        df['year'] = df['month_text'].str[:4].astype(int)
        df['month'] = df['month_text'].str[5:].map(month_codes)

    df['time'] = (df['year'] - 2019) * 12 + df['month'] - 1
    return df


def load_competition_data():
    """Load competition training and test datasets."""
    path = '/Users/nikola/Python/KaggleCompetition/data'
    train = pd.read_csv(f'{path}/train/new_house_transactions.csv')
    test = pd.read_csv(f'{path}/test.csv')
    return train, test


# =====================================================
# Data Transformation
# =====================================================

def build_amount_matrix(train: pd.DataFrame, month_codes: dict) -> pd.DataFrame:
    """Pivot training data into [time x sector_id] transaction matrix."""
    train = add_time_and_sector_fields(train.copy(), month_codes)
    pivot = train.pivot_table(
        index='time', columns='sector_id',
        values='amount_new_house_transactions', fill_value=0
    )

    # Ensure all 96 sectors are present
    all_sectors = np.arange(1, 97)
    pivot = pivot.reindex(columns=all_sectors, fill_value=0)

    return pivot


# =====================================================
# Modeling Helpers
# =====================================================

def compute_december_multipliers(a_tr, eps=1e-9, min_dec_obs=1, clip_low=0.8, clip_high=1.5):
    """Compute sector-level December multipliers from training data."""
    is_dec = (a_tr.index % 12 == 11)
    dec_means = a_tr[is_dec].mean()
    nondec_means = a_tr[~is_dec].mean()
    dec_counts = a_tr[is_dec].count()

    raw_mult = dec_means / (nondec_means + eps)
    overall_mult = float(dec_means.mean() / (nondec_means.mean() + eps))

    raw_mult = raw_mult.where(dec_counts >= min_dec_obs, overall_mult)
    raw_mult = raw_mult.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return raw_mult.clip(clip_low, clip_high).to_dict()


def apply_december_bump_row(pred_row: pd.Series, sector_to_mult: dict) -> pd.Series:
    """Apply December adjustment to a prediction row."""
    return pred_row.multiply(pd.Series(sector_to_mult)).fillna(pred_row)


def ewgm_per_sector(a_tr, sector, n_lags, alpha):
    """Exponential weighted geometric mean for one sector."""
    recent = a_tr[sector].tail(n_lags).values
    if len(recent) < n_lags or (recent <= 0).all():
        return 0.0

    weights = np.array([alpha**(n_lags - 1 - i) for i in range(n_lags)])
    weights /= weights.sum()

    mask = recent > 0
    if not mask.any():
        return 0.0

    log_vals = np.log(recent[mask] + 1e-12)
    pos_w = weights[mask] / weights[mask].sum()
    return float(np.exp(np.sum(pos_w * log_vals)))


def predict_one_step(a_hist, n_lags, alpha):
    """Predict next-step values for all sectors."""
    return pd.Series({
        sector: ewgm_per_sector(a_hist, sector, n_lags, alpha)
        if a_hist[sector].tail(n_lags).min() > 0 else 0.0
        for sector in a_hist.columns
    })


def evaluate_params(a_tr_full, n_lags, alpha, t2, clip_low, clip_high, val_len=6):
    """Evaluate parameters via rolling-origin backtest."""
    times = a_tr_full.index
    if len(times) < max(n_lags + 1, t2 + 1) + val_len:
        return 1e12

    rmses = []
    for t in times[-val_len:]:
        a_hist = a_tr_full.loc[a_tr_full.index < t]
        if len(a_hist) < max(n_lags, t2):
            continue

        y_true = a_tr_full.loc[t]
        y_pred = predict_one_step(a_hist, n_lags, alpha)

        if t % 12 == 11:  # December bump
            mult = compute_december_multipliers(a_hist, clip_low=clip_low, clip_high=clip_high)
            y_pred = apply_december_bump_row(y_pred, mult)

        rmses.append(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    return float(np.mean(rmses)) if rmses else 1e12


def predict_horizon(a_tr, n_lags, alpha, t2):
    """Forecast horizon [67..78]."""
    idx = np.arange(67, 79)
    preds = pd.DataFrame(index=idx, columns=a_tr.columns, dtype=float)

    for sector in a_tr.columns:
        if (a_tr[sector].tail(t2).min() == 0) or (a_tr[sector].sum() == 0):
            preds[sector] = 0.0
        else:
            preds[sector] = ewgm_per_sector(a_tr, sector, n_lags, alpha)

    preds.index.name = 'time'
    return preds


# =====================================================
# Submission
# =====================================================

def build_submission_df(a_pred, test_raw, month_codes):
    """Format predictions into competition submission file."""
    test = add_time_and_sector_fields(split_test_id_column(test_raw.copy()), month_codes)
    lookup = a_pred.stack().rename('pred').reset_index().rename(columns={'level_1': 'sector_id'})
    merged = test.merge(lookup, on=['time', 'sector_id'], how='left')
    merged['pred'] = merged['pred'].fillna(0.0)

    return merged[['id', 'pred']].rename(columns={'pred': 'new_house_transaction_amount'})


def generate_submission_with_december_bump(n_lags=6, alpha=0.5, t2=6, clip_low=0.85, clip_high=1.4):
    """End-to-end pipeline for submission with December bump."""
    month_codes = build_month_codes()
    train, test = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)
    a_pred = predict_horizon(a_tr, n_lags, alpha, t2)

    # Apply December bump
    mult = compute_december_multipliers(a_tr, clip_low=clip_low, clip_high=clip_high)
    for t in a_pred.index[a_pred.index % 12 == 11]:
        a_pred.loc[t] = apply_december_bump_row(a_pred.loc[t], mult)

    sub = build_submission_df(a_pred, test, month_codes)
    sub.to_csv('/Users/nikola/Python/KaggleCompetition/output/submission.csv', index=False)
    return a_tr, a_pred, sub


# =====================================================
# Optuna Optimization
# =====================================================

def optuna_objective(trial, a_tr):
    """Objective for Optuna hyperparameter tuning."""
    n_lags = trial.suggest_int('n_lags', 3, 12)
    alpha = trial.suggest_float('alpha', 0.20, 0.95)
    t2 = trial.suggest_int('t2', 3, 9)
    clip_low = trial.suggest_float('clip_low', 0.70, 0.95)
    clip_high = trial.suggest_float('clip_high', 1.10, 1.80)

    if clip_low >= clip_high:
        clip_low = max(0.70, clip_high - 0.05)

    return evaluate_params(a_tr, n_lags, alpha, t2, clip_low, clip_high)


def run_optuna_search(a_tr, n_trials=1000, seed=1337):
    """Run Optuna search and return the study."""
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(partial(optuna_objective, a_tr=a_tr), n_trials=n_trials, show_progress_bar=False)
    return study


# =====================================================
# Main
# =====================================================

def main():
    month_codes = build_month_codes()
    train, _ = load_competition_data()
    a_tr = build_amount_matrix(train, month_codes)

    # Run Optuna search
    study = run_optuna_search(a_tr, n_trials=512, seed=1337)
    best = study.best_params

    # Generate submission
    generate_submission_with_december_bump(**best)
    print("Best parameters:", best)
    print("Submission saved to /Users/nikola/Python/KaggleCompetition/output/submission.csv")


if __name__ == "__main__":
    main()
    