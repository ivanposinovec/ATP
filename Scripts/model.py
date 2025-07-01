import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from Scripts.functions import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from skopt import gp_minimize
from skopt.space import Integer, Real
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_rows',600)

games = pd.read_csv('games.csv')
games.set_index('Unnamed: 0', inplace=True)
games['tourney_date'] = pd.to_datetime(games['tourney_date'])
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games = games[(games['season'] >= 2009) & (games['favored_odds'].notna()) & (games['underdog_odds'].notna())].copy()

features = pd.read_csv('features.csv')
features = features.set_index('Unnamed: 0')

features_to_keep = [feature for feature in features.columns if '_diff' in feature] + ['inactive_match', 'uncertainty']
games = pd.concat([games, features[features_to_keep]], axis=1)

games.sort_values('uncertainty')[['favored', 'underdog', 'tourney_name', 'season', 'round', 'uncertainty']]

games = games[(games['underdog_odds'] <= 29) & (games['favored_odds'] >= 1) & (games['underdog_odds'] >= 1) & (((games['favored_odds'] <= games['underdog_odds']) & (games['favored_odds'] < 2)) | ((games['underdog_odds'] <= games['favored_odds']) & (games['underdog_odds'] < 2)))].reset_index(drop=True)
implied_probs = np.array([get_no_vig_odds(odds1=favored_odds_row, odds2=underdog_odds_row, odds0=None) for favored_odds_row, underdog_odds_row in zip(games['favored_odds'], games['underdog_odds'])])
games['favored_prob_implied'], games['underdog_prob_implied'] = 1 / implied_probs[:, 0], 1 / implied_probs[:, 1]

games['age_diff'] = games['favored_age'] - games['underdog_age']
games['height_diff'] = games['favored_ht'] - games['underdog_ht']
games['hand_match'] = np.where((games['favored_hand'] == 'R') & (games['underdog_hand'] == 'R'), 3,
                            np.where((games['favored_hand'] == 'R') & (games['underdog_hand'] == 'L'), 2,
                                    np.where((games['favored_hand'] == 'L') & (games['underdog_hand'] == 'R'), 1, 0)))

games['favored_entry'].replace({'SE':'Q', 'LL':'Q', 'PR':'WC', 'W':'WC', 'ALT':np.nan, 'Alt':np.nan}, inplace=True)
games['underdog_entry'].replace({'SE':'Q', 'LL':'Q', 'PR':'WC', 'W':'WC', 'ALT':np.nan, 'Alt':np.nan}, inplace=True)
games['entry_match'] = np.where((games['favored_entry'].isna()) & (games['underdog_entry'].isna()), 8,
                            np.where((games['favored_entry'].isna()) & (games['underdog_entry'] == 'Q'), 7,
                                    np.where((games['favored_entry'].isna()) & (games['underdog_entry'] == 'WC'), 6,
                                        np.where((games['favored_entry'] == 'Q') & (games['underdog_entry'].isna()), 5,
                                            np.where((games['favored_entry'] == 'Q') & (games['underdog_entry'] == 'Q'), 4,
                                                np.where((games['favored_entry'] == 'Q') & (games['underdog_entry'] == 'WC'), 3,
                                                    np.where((games['favored_entry'] == 'WC') & (games['underdog_entry'].isna()), 2,
                                                        np.where((games['favored_entry'] == 'WC') & (games['underdog_entry'] == 'Q'), 1, 0))))))))
games['home_match'] = np.where((games['favored_ioc'] == games['tourney_ioc']) & (games['underdog_ioc'] == games['tourney_ioc']), 3,
                            np.where((games['favored_ioc'] == games['tourney_ioc']) & (games['underdog_ioc'] != games['tourney_ioc']), 2,
                                    np.where((games['favored_ioc'] != games['tourney_ioc']) & (games['underdog_ioc'] == games['tourney_ioc']), 1, 0)))


# Feature selection
features_names = ['week',  'elo_diff', 'elo_surface_diff',  'log_elo_diff', 'log_elo_surface_diff', 'age_diff', 'height_diff', 'surface_speed'] + features_to_keep
features_names = [feature for feature in features_names if feature not in ['inactivity_diff', 'inactive_match']]
games[features_names] = games[features_names].replace([np.inf, -np.inf], np.nan)
games = games.dropna(subset = ['favored_odds', 'underdog_odds', 'hand_match', 'entry_match'] + features_names).reset_index(drop = True)

# One Hot Encoding
enc = OneHotEncoder()
dummies = pd.DataFrame(enc.fit_transform(games[['tourney_series', 'surface', 'surface_cluster', 'round', 'best_of', 'hand_match', 'entry_match', 'home_match', 'inactive_match']]).toarray(),
                    columns = enc.get_feature_names_out(['tourney_series', 'surface', 'surface_cluster', 'round', 'best_of', 'hand_match', 'entry_match', 'home_match', 'inactive_match']))
#dummies.drop(columns = [dummies.filter(like='series_').columns[-1]] + [dummies.filter(like='surface_').columns[-1]] + [dummies.filter(like='round_').columns[-1]] + [dummies.filter(like='best_of_').columns[-1]] +
#            [dummies.filter(like='hand_match_').columns[-1]] + [dummies.filter(like='entry_match_').columns[-1]] + [dummies.filter(like='home_match_').columns[-1]] + [dummies.filter(like='inactive_match_').columns[-1]], axis = 1, inplace = True)
dummies_features = dummies.columns.tolist()
games = pd.concat([games, dummies], axis=1)

# Interaction terms
# Create interaction terms between selected features
interaction_pairs = [
    ('elo_diff', 'surface_speed'),
    ('elo_diff', 'age_diff'),
    ('elo_diff', 'height_diff'),
    ('elo_diff', 'win_pct_diff'),
    ('elo_surface_diff', 'surface_speed'),
    ('elo_surface_diff', 'age_diff'),
    ('elo_surface_diff', 'height_diff'),
    ('elo_surface_diff', 'win_pct_surface_diff'),
    ('log_elo_diff', 'surface_speed'),
    ('log_elo_diff', 'age_diff'),
    ('log_elo_diff', 'height_diff'),
    ('log_elo_diff', 'win_pct_diff'),
    ('log_elo_surface_diff', 'surface_speed'),
    ('log_elo_surface_diff', 'age_diff'),
    ('log_elo_surface_diff', 'height_diff'),
    ('log_elo_surface_diff', 'win_pct_surface_diff'),
    ('elo_diff', 'elo_surface_diff'),
    ('log_elo_diff', 'log_elo_surface_diff')
]

for f1, f2 in interaction_pairs:
    col_name = f"{f1}_x_{f2}"
    games[col_name] = games[f1] * games[f2]
    features_names.append(col_name)

# Drop uncertain values
uncertainty_threshold = games['uncertainty'].quantile(0.95)
games = games[(games['uncertainty'] <= uncertainty_threshold)].reset_index(drop=True)

# Prepare X, y and odds df
X = games[features_names+dummies_features]
y = games['favored_win']
odds = games[['favored', 'underdog', 'season', 'tourney_date', 'tourney_name', 'round', 'comment', 'game_url', 'favored_win', 'favored_prob_implied', 'underdog_prob_implied', 'favored_mean_opening_odds', 'underdog_mean_opening_odds',
            'favored_mean_closing_odds', 'underdog_mean_closing_odds', 'favored_max_opening_odds', 'underdog_max_opening_odds', 'favored_max_closing_odds', 'underdog_max_closing_odds',
            'favored_min_opening_odds', 'underdog_min_opening_odds', 'favored_min_closing_odds', 'underdog_min_closing_odds', 'favored_pinnacle_opening_odds', 'underdog_pinnacle_opening_odds',
            'favored_pinnacle_closing_odds', 'underdog_pinnacle_closing_odds']].copy()

#--------------------------------------------------------------------------- Elastic Net ---------------------------------------------------------------------------#
models = {}
log_loss_test_results = pd.DataFrame()
for fold_number in range(0, 6):
    model = 'enet'
    train_lower = 2009
    train_upper = 2018 + fold_number - 1
    val_lower = 2018 + fold_number
    val_upper = 2020 + fold_number - 1
    train_val_lower = train_lower
    train_val_upper = val_upper
    test_season = 2020 + fold_number
    
    print(f'\nTraining fold {fold_number+1}/{len(range(0,6))} - Train Seasons: {train_lower}-{train_upper} - Val Seasons: {val_lower}-{val_upper} - Test Season: {test_season}')
    
    train_index = games[(games['season'] >= train_lower) & (games['season'] <= (train_upper))].index
    val_index = games[(games['season'] >= val_lower) & (games['season'] <= val_upper)].index
    train_val_index = games[(games['season'] >= train_val_lower) & (games['season'] <= train_val_upper)].index
    test_index = games[(games['season'] == test_season)].index
    X_train, X_val, X_train_val, X_test = X.loc[train_index], X.loc[val_index], X.loc[train_val_index], X.loc[test_index]
    y_train, y_val, y_train_val, y_test = y.loc[train_index], y.loc[val_index], y.loc[train_val_index], y.loc[test_index]
    odds_train, odds_val, odds_train_val, odds_test = odds.loc[train_index], odds.loc[val_index], odds.loc[train_val_index], odds.loc[test_index]
    
    # Plot distribution of each feature
    """
    for feature in features_names:
        plt.figure(figsize=(10, 6))
        plt.hist(X_train_val[feature].replace([np.inf, -np.inf], np.nan).dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
    """
    
    # High leverage values removal
    features_to_remove_high_leverage = [feature for feature in X_train_val.columns if feature not in dummies_features+['uncertainty']]
    indexes_to_drop = []
    for col in features_to_remove_high_leverage:
        Q1 = X_train_val[col].quantile(0.2)
        Q3 = X_train_val[col].quantile(0.8)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        indexes_to_drop.append(list(X_train_val[(X_train_val[col] <= lower_bound) | (X_train_val[col] >= upper_bound)].index))
        print(f'{col} Number of indexes to drop: {len(list(X_train_val[(X_train_val[col] <= lower_bound) | (X_train_val[col] >= upper_bound)].index))}')
    
    indexes_to_drop = list(set([index for sublist in indexes_to_drop for index in sublist]))
    indexes = [index for index in X_train_val.index if index not in indexes_to_drop]
    print(f'Number of indexes to drop: {len(indexes_to_drop)}')
    print(f'Number of indexes kept: {len(indexes)}')
    
    X_train = X_train.loc[[index for index in indexes if index in X_train.index]]
    y_train = y.loc[[index for index in indexes if index in X_train.index]]
    odds_train = odds.loc[[index for index in indexes if index in X_train.index]]
    
    X_val = X_val.loc[[index for index in indexes if index in X_val.index]]
    y_val = y.loc[[index for index in indexes if index in X_val.index]]
    odds_val = odds.loc[[index for index in indexes if index in X_val.index]]
    
    X_train_val = X_train_val.loc[indexes]
    y_train_val = y_train_val.loc[indexes]
    odds_train_val = odds_train_val.loc[indexes]
    
    # Scaling
    features_to_scale = [feature for feature in X_train_val.columns if X_train_val[feature].nunique() > 2]
    features_not_to_scale = [feature for feature in X_train_val.columns if feature not in features_to_scale]
    scaler = StandardScaler()
    scaler.fit(X_train_val[features_to_scale])
    
    X_train_scaled = pd.concat([X_train[features_not_to_scale], pd.DataFrame(scaler.transform(X_train[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_train.index)], axis = 1)
    X_val_scaled = pd.concat([X_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_val[features_to_scale]), columns = X_val[features_to_scale].columns, index = X_val.index)], axis = 1)
    X_train_val_scaled = pd.concat([X_train_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_train_val[features_to_scale]), columns = X_train_val[features_to_scale].columns, index = X_train_val.index)], axis = 1)
    X_test_scaled = pd.concat([X_test[features_not_to_scale], pd.DataFrame(scaler.transform(X_test[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_test.index)], axis = 1)
    
    models[f'{test_season}'] = {}
    models[f'{test_season}']['data'] = {'X_train_scaled':X_train_scaled, 'X_val_scaled':X_val_scaled, 'X_train_val':X_train_val_scaled, 'X_test_scaled':X_test_scaled,
                                        'y_train':y_train, 'y_val':y_val, 'y_train_val':y_train_val, 'y_test':y_test,
                                        'odds_train':odds_train, 'odds_val':odds_val, 'odds_train_val':odds_train_val, 'odds_test':odds_test}
    search_space = [
        Real(0.1, 1, name='l1_ratio'),
        Real(0.01, 5, name='C'),
    ]
    
    def objective(params):
        l1_ratio, C = params
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            random_state=42,
            fit_intercept=True,
            max_iter=10_000,
            l1_ratio=l1_ratio,
            C=C
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict_proba(X_val_scaled)
        return log_loss(y_val, y_pred)
    
    # Run Bayesian Optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=50,
        random_state=42,
        verbose=True
    )
    print("Best MSE:", result.fun)
    print("Best parameters:", result.x)
    l1_ratio, C = result.x
    best_logit = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            random_state=42,
            fit_intercept=True,
            max_iter=10_000,
            l1_ratio=l1_ratio,
            C=C
        )
    best_logit.fit(X_train_val_scaled, y_train_val)
    
    odds_train_val[f'favored_prob_{model}'] = best_logit.predict_proba(X_train_val_scaled)[:, 1]
    odds_test[f'favored_prob_{model}'] = best_logit.predict_proba(X_test_scaled)[:, 1]
    odds_train_val[f'underdog_prob_{model}'] = best_logit.predict_proba(X_train_val_scaled)[:, 0]
    odds_test[f'underdog_prob_{model}'] = best_logit.predict_proba(X_test_scaled)[:, 0]
    coefficients = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Coefficient': best_logit.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)
    print(coefficients)
    selected_features = list(coefficients[coefficients['Coefficient'] != 0]['Feature'])
    
    # Log loss
    log_loss_test_results.loc[len(log_loss_test_results), ['model', 'season', 'log_loss']] = ['Odds', test_season, log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_implied"])]
    log_loss_test_results.loc[len(log_loss_test_results), ['model', 'season', 'log_loss']] = ['Logit', test_season, log_loss(odds_test["favored_win"].astype(int), odds_test[f"favored_prob_{model}"])]
    
    models[f'{(2020 + fold_number)}']['model'] = {'model_name':'logit', 'hyperparams':{'l1_ratio':l1_ratio, 'C':C}, 'trained_model':best_logit, 'coefficients':coefficients, 'selected_features':selected_features}
    
    # Pinnacle Odds and ML Model
    cl_fit = fit_second_stage(odds_train_val['favored_win'], odds_train_val[f'underdog_prob_{model}'], odds_train_val[f'favored_prob_{model}'], odds_train_val['underdog_prob_implied'], odds_train_val['favored_prob_implied'])
    print(cl_fit.summary())
    
    train_second_stage_probs = predict_second_stage(cl_fit.params, odds_train_val[f'underdog_prob_{model}'], odds_train_val[f'favored_prob_{model}'], odds_train_val['underdog_prob_implied'], odds_train_val['favored_prob_implied'])
    odds_train_val['underdog_prob_2st'], odds_train_val['favored_prob_2st'] = train_second_stage_probs[:,0], train_second_stage_probs[:,1]
    
    test_second_stage_probs = predict_second_stage(cl_fit.params, odds_test[f'underdog_prob_{model}'], odds_test[f'favored_prob_{model}'], odds_test['underdog_prob_implied'], odds_test['favored_prob_implied'])
    odds_test['underdog_prob_2st'], odds_test['favored_prob_2st'] = test_second_stage_probs[:,0], test_second_stage_probs[:,1]
    
    log_loss_test_results.loc[len(log_loss_test_results), ['model', 'season', 'log_loss']] = [f'{model} & Odds', test_season, log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_2st"])]
    
    print(f"Train Log Loss (2009 - {2018+fold_number}): {log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_prob_{model}']):.5f}")
    print(f"2nd Stage Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_2st']):.5f}")
    print(f"Odds Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_implied']):.5f}")
    
    print(f"Test Log Loss ({test_season}): {log_loss(odds_test['favored_win'], odds_test[f'favored_prob_{model}']):.5f}")
    print(f"2nd Stage Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_2st']):.5f}")
    print(f"Odds Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_implied']):.5f}")
    
    models[f'{test_season}']['predictions'] = {'train_odds_log_loss': log_loss(odds_train_val["favored_win"], odds_train_val[f"favored_prob_implied"]),
                                                'train_log_loss':log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_prob_{model}']),
                                                'train_2st_log_loss':log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_2st']),
                                                'test_odds_log_loss': log_loss(odds_test["favored_win"], odds_test[f"favored_prob_implied"]),
                                                'test_log_loss':log_loss(odds_test['favored_win'], odds_test[f'favored_prob_{model}']),
                                                'test_2st_log_loss':log_loss(odds_test['favored_win'], odds_test['favored_prob_2st']),
                                                'train_predictions':odds_train_val, 'test_predictions':odds_test}

6608
#--------------------------------------------------------------------------- XGBoost ---------------------------------------------------------------------------#
models = {}
for fold_number in range(0, 6):
    model = 'xgb'
    train_lower = 2009
    train_upper = 2018 + fold_number - 1
    val_lower = 2018 + fold_number
    val_upper = 2020 + fold_number - 1
    train_val_lower = train_lower
    train_val_upper = val_upper
    test_season = 2020 + fold_number
    
    print(f'Training fold {fold_number+1}/{len(range(0,6))} - Train Seasons: {train_lower}-{train_upper} - Val Seasons: {val_lower}-{val_upper}  - Test Season: {test_season}')
    
    train_index = games[(games['season'] >= train_lower) & (games['season'] <= (train_upper))].index
    val_index = games[(games['season'] >= val_lower) & (games['season'] <= val_upper)].index
    train_val_index = games[(games['season'] >= train_val_lower) & (games['season'] <= train_val_upper)].index
    test_index = games[(games['season'] == test_season)].index
    X_train, X_val, X_train_val, X_test = X.loc[train_index], X.loc[val_index], X.loc[train_val_index], X.loc[test_index]
    y_train, y_val, y_train_val, y_test = y.loc[train_index], y.loc[val_index], y.loc[train_val_index], y.loc[test_index]
    odds_train, odds_val, odds_train_val, odds_test = odds.loc[train_index], odds.loc[val_index], odds.loc[train_val_index], odds.loc[test_index]
    
    # High leverage values removal
    features_to_remove_high_leverage = [feature for feature in X_train_val.columns if feature not in dummies_features+['uncertainty']]
    indexes_to_drop = []
    for col in features_to_remove_high_leverage:
        Q1 = X_train_val[col].quantile(0.2)
        Q3 = X_train_val[col].quantile(0.8)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        indexes_to_drop.append(list(X_train_val[(X_train_val[col] <= lower_bound) | (X_train_val[col] >= upper_bound)].index))
        print(f'{col} Number of indexes to drop: {len(list(X_train_val[(X_train_val[col] <= lower_bound) | (X_train_val[col] >= upper_bound)].index))}')
    
    indexes_to_drop = list(set([index for sublist in indexes_to_drop for index in sublist]))
    indexes = [index for index in X_train_val.index if index not in indexes_to_drop]
    print(f'Number of indexes to drop: {len(indexes_to_drop)}')
    print(f'Number of indexes kept: {len(indexes)}')
    
    X_train = X_train.loc[[index for index in indexes if index in X_train.index]]
    y_train = y.loc[[index for index in indexes if index in X_train.index]]
    odds_train = odds.loc[[index for index in indexes if index in X_train.index]]
    
    X_val = X_val.loc[[index for index in indexes if index in X_val.index]]
    y_val = y.loc[[index for index in indexes if index in X_val.index]]
    odds_val = odds.loc[[index for index in indexes if index in X_val.index]]
    
    X_train_val = X_train_val.loc[indexes]
    y_train_val = y_train_val.loc[indexes]
    odds_train_val = odds_train_val.loc[indexes]
    
    # Scaling
    features_to_scale = [feature for feature in X_train_val.columns if X_train_val[feature].nunique() > 2]
    features_not_to_scale = [feature for feature in X_train_val.columns if feature not in features_to_scale]
    scaler = StandardScaler()
    scaler.fit(X_train_val[features_to_scale])
    
    X_train_scaled = pd.concat([X_train[features_not_to_scale], pd.DataFrame(scaler.transform(X_train[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_train.index)], axis = 1)
    X_val_scaled = pd.concat([X_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_val[features_to_scale]), columns = X_val[features_to_scale].columns, index = X_val.index)], axis = 1)
    X_train_val_scaled = pd.concat([X_train_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_train_val[features_to_scale]), columns = X_train_val[features_to_scale].columns, index = X_train_val.index)], axis = 1)
    X_test_scaled = pd.concat([X_test[features_not_to_scale], pd.DataFrame(scaler.transform(X_test[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_test.index)], axis = 1)
    
    models[f'{test_season}'] = {}
    models[f'{test_season}']['data'] = {'X_train_scaled':X_train_scaled, 'X_val_scaled':X_val_scaled, 'X_train_val':X_train_val_scaled, 'X_test_scaled':X_test_scaled,
                                                'y_train':y_train, 'y_val':y_val, 'y_train_val':y_train_val, 'y_test':y_test,
                                                'odds_train':odds_train, 'odds_val':odds_val, 'odds_train_val':odds_train_val, 'odds_test':odds_test}
    search_space = [
        Integer(50, 1000, name='n_estimators'),
        Integer(2, 8, name='max_depth'),
        Real(0.001, 0.1, name= 'learning_rate'),
        Real(0.5, 1, name= 'subsample'),
        Real(0.5, 1, name= 'colsample_bytree'),
    ]
    
    def objective(params):
        n_estimators, max_depth, learning_rate, subsample, colsample = params
        model = XGBClassifier(n_estimators = n_estimators,
                            max_depth = max_depth,
                            learning_rate = learning_rate,
                            subsample = subsample,
                            colsample_bytree = colsample,
                            random_state=42, eval_metric='logloss')

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict_proba(X_val_scaled)
        return log_loss(y_val, y_pred)
    
    # Run Bayesian Optimization
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=100,
        random_state=42,
        verbose=True
    )
    print("Best MSE:", result.fun)
    print("Best parameters:", result.x)
    n_estimators, max_depth, learning_rate, subsample, colsample = result.x
    best_xgb = XGBClassifier(n_estimators = n_estimators,
                            max_depth = max_depth,
                            learning_rate = learning_rate,
                            subsample = subsample,
                            colsample_bytree = colsample,
                            random_state=42, eval_metric='logloss')
    best_xgb.fit(X_train_val_scaled, y_train_val)
    
    odds_train_val[f'favored_prob_{model}'] = best_xgb.predict_proba(X_train_val_scaled)[:, 1]
    odds_test[f'favored_prob_{model}'] = best_xgb.predict_proba(X_test_scaled)[:, 1]
    odds_train_val[f'underdog_prob_{model}'] = best_xgb.predict_proba(X_train_val_scaled)[:, 0]
    odds_test[f'underdog_prob_{model}'] = best_xgb.predict_proba(X_test_scaled)[:, 0]
    
    # Log loss
    log_loss_test_results.loc[len(log_loss_test_results), ['model', 'season', 'log_loss']] = ['Odds', test_season, log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_implied"])]
    log_loss_test_results.loc[len(log_loss_test_results), ['model', 'season', 'log_loss']] = ['XGB', test_season, log_loss(odds_test["favored_win"].astype(int), odds_test[f"favored_prob_{model}"])]
    
    models[f'{(2020 + fold_number)}']['model'] = {'model_name':model,
                                                'hyperparams':{'n_estimators':n_estimators, 'max_depth':max_depth, 'learning_rate':learning_rate, 'subsample':subsample, 'colsample_bytree':colsample},
                                                'trained_model':best_xgb}
    
    # Pinnacle Odds and ML Model
    cl_fit = fit_second_stage(odds_train_val['favored_win'], odds_train_val[f'underdog_prob_{model}'], odds_train_val[f'favored_prob_{model}'], odds_train_val['underdog_prob_implied'], odds_train_val['favored_prob_implied'])
    print(cl_fit.summary())
    
    train_second_stage_probs = predict_second_stage(cl_fit.params, odds_train_val[f'underdog_prob_{model}'], odds_train_val[f'favored_prob_{model}'], odds_train_val['underdog_prob_implied'], odds_train_val['favored_prob_implied'])
    odds_train_val['underdog_prob_2st'], odds_train_val['favored_prob_2st'] = train_second_stage_probs[:,0], train_second_stage_probs[:,1]
    
    test_second_stage_probs = predict_second_stage(cl_fit.params, odds_test[f'underdog_prob_{model}'], odds_test[f'favored_prob_{model}'], odds_test['underdog_prob_implied'], odds_test['favored_prob_implied'])
    odds_test['underdog_prob_2st'], odds_test['favored_prob_2st'] = test_second_stage_probs[:,0], test_second_stage_probs[:,1]
    
    print(f"Train Log Loss (2009 - {val_upper}): {log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_prob_{model}']):.5f}")
    print(f"2nd Stage Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_2st']):.5f}")
    print(f"Odds Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_implied']):.5f}")
    print(f"Test Log Loss ({test_season}): {log_loss(odds_test['favored_win'], odds_test[f'favored_prob_{model}']):.5f}")
    print(f"2nd Stage Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_2st']):.5f}")
    print(f"Odds Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_implied']):.5f}")
    
    log_loss_test_results.loc[len(log_loss_test_results), ['model', 'season', 'log_loss']] = [f'{model} & Odds', test_season, log_loss(odds_test["favored_win"], odds_test["favored_prob_2st"])]
    
    models[f'{(2020 + fold_number)}']['predictions'] = {'train_odds_log_loss': log_loss(odds_train_val["favored_win"], odds_train_val[f"favored_prob_implied"]),
                                                        'train_log_loss':log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_prob_{model}']),
                                                        'train_2st_log_loss':log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_2st']),
                                                        'test_odds_log_loss': log_loss(odds_test["favored_win"], odds_test[f"favored_prob_implied"]),
                                                        'test_log_loss':log_loss(odds_test['favored_win'], odds_test[f'favored_prob_{model}']),
                                                        'test_2st_log_loss':log_loss(odds_test['favored_win'], odds_test['favored_prob_2st']),
                                                        'train_predictions':odds_train_val, 'test_predictions':odds_test}


for model in models:
    print(f'\n ===== {model} =====')
    print(f"Test Odds Log Loss: {models[model]['predictions']['test_odds_log_loss']}")
    print(f"Test 1st Stage Log Loss: {models[model]['predictions']['test_log_loss']}")
    print(f"Test 2nd Stage Log Loss: {models[model]['predictions']['test_2st_log_loss']}")


# Betting
bets = pd.concat([models[str(year)]['predictions']['test_predictions'] for year in range(2020, 2026)], ignore_index=True)
model = 'xgb'
odds_str = 'mean_opening'
expected_return_favored = (bets[f'favored_{odds_str}_odds']-1)*bets[f'favored_prob_{model}'] - 1*(1-bets[f'favored_prob_{model}'])
expected_return_underdog = (bets[f'underdog_{odds_str}_odds']-1)*bets[f'underdog_prob_{model}'] - 1*(1-bets[f'underdog_prob_{model}'])
bets['bet'] = np.where(np.maximum(expected_return_favored, expected_return_underdog) < 0, 'no_bet', np.where(expected_return_favored > expected_return_underdog, 'favored', 'underdog'))
bets['expected_return'] = np.where(bets['bet'] == 'no_bet', 0, np.where(expected_return_favored > expected_return_underdog, expected_return_favored, expected_return_underdog))

bets['probs'] = np.where(bets['bet'] == 'favored', bets[f'favored_prob_{model}'], bets[f'underdog_prob_{model}'])
bets['odds'] = np.where(bets['bet'] == 'favored', bets[f'favored_{odds_str}_odds'], bets[f'underdog_{odds_str}_odds'])

bets['kelly_fraction_fixed'] = kelly_fraction_calc(bets['odds'], bets['probs']).apply(lambda x: max(x,0))

bets['result_kelly_fixed'] = np.where(bets['bet'] == 'no_bet', 0,
                                        np.where((bets['bet'] == 'favored') & (bets['favored_win']== 1), (bets[f'favored_{odds_str}_odds']-1)*bets['kelly_fraction_fixed'],
                                                np.where((bets['bet'] == 'underdog') & (bets['favored_win']== 0), (bets[f'underdog_{odds_str}_odds']-1)*bets['kelly_fraction_fixed'],
                                                        -bets['kelly_fraction_fixed'] )))	

total_bets_kelly = len(bets)
bets_placed_kelly = (bets['bet'] != 'no_bet').sum()
wins_kelly = (pd.to_numeric(bets['result_kelly_fixed'], errors='coerce') > 0).sum()
hit_rate_kelly = wins_kelly/bets_placed_kelly
earnings_kelly = pd.to_numeric(bets['result_kelly_fixed'], errors='coerce').fillna(0).sum()
spent_kelly = (bets['kelly_fraction_fixed']).sum()
return_kelly = earnings_kelly/spent_kelly 
earnings_per_bet_kelly = earnings_kelly/bets_placed_kelly
earnings_per_win_kelly = earnings_kelly/wins_kelly

predictions = {'bets':bets,
            'stats':{'earnings':earnings_kelly, 'spent':spent_kelly, 'return':return_kelly,'per_bet':earnings_per_bet_kelly, 'per_win':earnings_per_win_kelly}, 
            'other_stats':{'total':total_bets_kelly, 'placed':bets_placed_kelly, "wins":wins_kelly, 'hit_rate':hit_rate_kelly}}
print(predictions)

bets.sort_values('expected_return', ascending=False)[['favored', 'underdog', 'tourney_name', 'season', 'favored_win', 'favored_prob_implied', 'underdog_prob_implied', 'favored_mean_closing_odds', 'underdog_mean_closing_odds', 'bet', 'probs', 'expected_return']].head(10)

games[(games['favored'] == 'Bradley Klahn') & (games['underdog'] == 'Felipe Meligeni Alves')][['favored', 'underdog', 'favored_win', 'favored_odds', 'underdog_odds', 'favored_elo', 'underdog_elo', 'favored_elo_surface', 'underdog_elo_surface']]


# Calculate return by odds buckets
bins = [1, 1.5, 2, 3, 5, 10, 30]
labels = ['1-1.5', '1.5-2', '2-3', '3-5', '5-10', '10-30']
bets['odds_bucket'] = pd.cut(bets['odds'], bins=bins, labels=labels, right=True)

bucket_stats = []
for bucket in labels:
    bucket_bets = bets[bets['odds_bucket'] == bucket]
    placed = (bucket_bets['bet'] != 'no_bet').sum()
    wins = (pd.to_numeric(bucket_bets['result_kelly_fixed'], errors='coerce') > 0).sum()
    earnings = pd.to_numeric(bucket_bets['result_kelly_fixed'], errors='coerce').fillna(0).sum()
    spent = bucket_bets['kelly_fraction_fixed'].sum()
    roi = earnings / spent if spent > 0 else 0
    bucket_stats.append({
        'odds_bucket': bucket,
        'bets_placed': placed,
        'wins': wins,
        'hit_rate': wins / placed if placed > 0 else 0,
        'earnings': earnings,
        'spent': spent,
        'roi': roi
    })

print(pd.DataFrame(bucket_stats))



thresholds = list(np.arange(0, 0.2, 0.01))
results_by_threshold = []
for index, threshold in enumerate(thresholds):
    full_bets_threshold = bets[bets['expected_return'] > threshold].copy()
    
    results = {'threshold':f'{threshold*100:.1f}%', 'bets_placed':(full_bets_threshold['bet'] != 'No Bet').sum(), 'wins':(pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce') > 0).sum(),
            'hit_rate':(pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce') > 0).sum()/(full_bets_threshold['bet'] != 'No Bet').sum(),
            'earnings':pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce').fillna(0).sum(), 'spent':(full_bets_threshold['kelly_fraction_fixed']).sum(),
            'expected_return':f'{pd.to_numeric(full_bets_threshold['expected_return'], errors='coerce').mean()*100:.2f}%',
            'return':f'{pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce').fillna(0).sum()/(full_bets_threshold['kelly_fraction_fixed']).sum()*100:.2f}%'}
    results_by_threshold.append(results)
print(pd.DataFrame(results_by_threshold))





thresholds = list(np.arange(1.1, 2.1, 0.1))
results_by_threshold = []
for index, threshold in enumerate(thresholds):
    full_bets_threshold = bets[bets['odds'] <= threshold].copy()
    
    results = {'threshold':f'{threshold:.1f}', 'bets_placed':(full_bets_threshold['bet'] != 'No Bet').sum(), 'wins':(pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce') > 0).sum(),
            'hit_rate':(pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce') > 0).sum()/(full_bets_threshold['bet'] != 'No Bet').sum(),
            'earnings':pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce').fillna(0).sum(), 'spent':(full_bets_threshold['kelly_fraction_fixed']).sum(),
            'expected_return':f'{pd.to_numeric(full_bets_threshold['expected_return'], errors='coerce').mean()*100:.2f}%',
            'return':f'{pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce').fillna(0).sum()/(full_bets_threshold['kelly_fraction_fixed']).sum()*100:.2f}%'}
    results_by_threshold.append(results)
print(pd.DataFrame(results_by_threshold))




bets_results = bets[bets['expected_return'] > 0.0].copy()
bets_results = bets[bets['odds'] <= 1.8].copy()
capital = 1
full_model_capital = pd.DataFrame({'date':pd.to_datetime(bets_results['tourney_date'].iloc[0])-timedelta(days=1), f'capital':capital}, index=[0])
for date, group in bets_results.groupby('tourney_date'):
    bets_results.loc[group.index, 'capital'] = capital
    
    total_number_of_props = len(group)
    bets_results.loc[group.index, 'kelly_bet'] = bets_results.loc[group.index, 'kelly_fraction_fixed']*bets_results.loc[group.index, 'capital']/total_number_of_props
    bets_results.loc[group.index, 'result_kelly_bet'] = bets_results.loc[group.index, 'result_kelly_fixed']*bets_results.loc[group.index, 'capital']/total_number_of_props
    
    earnings = bets_results.loc[group.index, 'result_kelly_bet'].sum()
    spent = bets_results.loc[group.index, 'kelly_bet'].sum()
    capital += earnings
    
    print(f'Date {date} - Earnings:{earnings:.2f} - Spent:{spent:.2f} - ROI:{(earnings/spent)*100:.2f}% - Number of Bets: {len(group)} - Capital: {capital:.2f}')
    full_model_capital = pd.concat([full_model_capital, pd.DataFrame({'date':pd.to_datetime(date), f'capital':capital}, index=[0])], axis = 0).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(full_model_capital['date'], full_model_capital['capital'], marker='o')

plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.title(f'Capital Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()

#plt.yscale('log')

plt.tight_layout()
plt.show()

