import pandas as pd
import numpy as np
from tqdm import tqdm
from Scripts.functions import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
pd.set_option('display.max_rows',200)

df = pd.read_csv("data.csv")
df.drop(columns=['favored_avg_odds.1', 'underdog_avg_odds.1'], inplace=True)

df['favored_odds'] = np.where(df['favored_pinnacle_odds'].notnull(), df['favored_pinnacle_odds'], np.where(df['favored_avg_odds'].notnull(), df['favored_avg_odds'], df['favored_bet365_odds']))
df['underdog_odds'] = np.where(df['underdog_pinnacle_odds'].notnull(), df['underdog_pinnacle_odds'], np.where(df['underdog_avg_odds'].notnull(), df['underdog_avg_odds'], df['underdog_bet365_odds']))

df = df.dropna(subset = ['favored_odds', 'underdog_odds', 'favored_win']).reset_index(drop = True)

enc = OneHotEncoder()
dummies = pd.DataFrame(enc.fit_transform(df[['tourney_series', 'surface', 'round', 'favored_rank_group', 'underdog_rank_group', 'favored_entry', 'underdog_entry']]).toarray(),
                    columns = enc.get_feature_names_out(['tourney_series', 'surface', 'round', 'favored_rank_group', 'underdog_rank_group', 'favored_entry', 'underdog_entry']))
dummies.drop(columns = [dummies.filter(like='series_').columns[-1]] +  [dummies.filter(like='surface_').columns[-1]] + [dummies.filter(like='round_').columns[-1]] +
            [dummies.filter(like='favored_rank_group_').columns[-1]] + [dummies.filter(like='underdog_rank_group_').columns[-1]], axis = 1, inplace = True)
df = pd.concat([df, dummies], axis=1)

# Variables
dummies_names = dummies.columns.tolist()
numerical_features_names = [col for col in df.columns if col not in ['favored', 'underdog', 'favored_id', 'underdog_id', 'game_id', 'season', 'tourney_id', 'tourney_name', 'tourney_full_name',
                            'tourney_series', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num', 'round', 'favored_entry', 'underdog_entry', 'favored_rank', 'underdog_rank',
                            'favored_rank_group', 'underdog_rank_group', 'favored_rank_pts', 'underdog_rank_pts', 'favored_elo', 'underdog_elo', 'favored_elo_surface', 'underdog_elo_surface',
                            'underdog_elo_diff', 'underdog_elo_surface_diff', 'favored_win', 'favored_max_odds', 'underdog_max_odds', 'favored_avg_odds', 'underdog_avg_odds',
                            'favored_pinnacle_odds', 'underdog_pinnacle_odds', 'favored_bet365_odds', 'underdog_bet365_odds', 'favored_odds', 'underdog_odds',
                            'favored_avg_service_rating_h2h', 'favored_avg_return_rating_h2h', 'favored_avg_performance_h2h', 'underdog_avg_service_rating_h2h',
                            'underdog_avg_return_rating_h2h', 'underdog_avg_performance_h2h',
                            'favored_avg_return_rating_rank', 'underdog_avg_return_rating_rank', 'favored_avg_return_rating_surface', 'underdog_avg_return_rating_surface',
                            'favored_avg_return_rating_series', 'underdog_avg_return_rating_series', 'favored_avg_return_rating_round', 'underdog_avg_return_rating_round'] + dummies_names]

print(df[numerical_features_names].isna().sum())
df = df.dropna(subset = numerical_features_names).reset_index(drop = True)
df = df[(df['underdog_pinnacle_odds'] <= 30) & (df['favored_pinnacle_odds'] <= 2)].reset_index(drop=True)

X = df[dummies_names + numerical_features_names]
y = df['favored_win']
odds = df[['favored', 'underdog', 'season', 'tourney_full_name', 'tourney_date', 'round', 'favored_elo', 'underdog_elo', 'favored_elo_surface', 'underdog_elo_surface', 'favored_win',
        'favored_odds', 'underdog_odds', 'favored_max_odds', 'underdog_max_odds']].copy()

implied_probs = np.array([get_no_vig_odds(odds1=favored_odds_row, odds2=underdog_odds_row, odds0=None) for favored_odds_row, underdog_odds_row in zip(odds['favored_odds'], odds['underdog_odds'])])
odds['implied_favored_prob'], odds['implied_underdog_prob'] = 1 / implied_probs[:, 0], 1 / implied_probs[:, 1]


train_index = df[df['season'] < 2018].index
test_index = df[df['season'] >= 2018].index
X_train, X_test = X.loc[train_index], X.loc[test_index]
#df.loc[train_index].to_csv("train.csv", index=False)

X_train_clean = X_train.copy()
for col in X_train.columns:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    X_train_clean = X_train[(X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)]

y_train, y_test = y.loc[X_train_clean.index], y.loc[X_test.index]
odds_train, odds_test = odds.loc[X_train_clean.index], odds.loc[X_test.index]

# Scaling
scaler = StandardScaler()
scaler.fit(X_train_clean[numerical_features_names])

X_train_scaled = pd.concat([X_train_clean[dummies_names], pd.DataFrame(scaler.transform(X_train_clean[numerical_features_names]), columns = X_train_clean[numerical_features_names].columns, index = X_train_clean.index)], axis = 1)
X_test_scaled = pd.concat([X_test[dummies_names], pd.DataFrame(scaler.transform(X_test[numerical_features_names]), columns = X_train[numerical_features_names].columns, index = X_test.index)], axis = 1)

# Elastic Net #
elastic_net = LogisticRegressionCV(
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 1],
    Cs=[0.01, 0.05, 0.1, 0.5, 1],
    cv=10,
    scoring='neg_log_loss',
    random_state=42, max_iter=10_000
)
elastic_net.fit(X_train_scaled, y_train)

print("Best l1_ratio:", elastic_net.l1_ratio_[0])
print("Best C:", elastic_net.C_[0])

odds_train['favored_prob'] = elastic_net.predict_proba(X_train_scaled)[:, 1]
odds_test['favored_prob'] = elastic_net.predict_proba(X_test_scaled)[:, 1]
odds_train['underdog_prob'] = elastic_net.predict_proba(X_train_scaled)[:, 0]
odds_test['underdog_prob'] = elastic_net.predict_proba(X_test_scaled)[:, 0]
print(f"Train Log Loss: {log_loss(odds_train['favored_win'], odds_train['favored_prob']):.2f}")
print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob']):.2f}")

coefficients = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': elastic_net.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print(coefficients)


# Neural Network
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)


# Predict probabilities
odds_train['favored_prob'] = model.predict(X_train_scaled).flatten()
odds_train['underdog_prob'] = 1 - odds_train['favored_prob']
odds_test['favored_prob'] = model.predict(X_test_scaled).flatten()
odds_test['underdog_prob'] = 1 - odds_test['favored_prob']

# Log loss
# Log loss
log_loss_test_results = pd.DataFrame()
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Odds', log_loss(odds_test["favored_win"].astype(int), odds_test["implied_favored_prob"])]
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Logit', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob"])]
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Neural Network', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob"])]


# Pinnacle Odds and ML Model
cl_fit = fit_second_stage(odds_train['favored_win'], odds_train['underdog_prob'], odds_train['favored_prob'], odds_train['implied_underdog_prob'], odds_train['implied_favored_prob'])
print(cl_fit.summary())

train_second_stage_probs = predict_second_stage(cl_fit.params, odds_train['underdog_prob'], odds_train['favored_prob'], odds_train['implied_underdog_prob'], odds_train['implied_favored_prob'])
odds_train['2st_underdog_prob'], odds_train['2st_favored_prob'] = train_second_stage_probs[:,0], train_second_stage_probs[:,1]

test_second_stage_probs = predict_second_stage(cl_fit.params, odds_test['underdog_prob'], odds_test['favored_prob'], odds_test['implied_underdog_prob'], odds_test['implied_favored_prob'])
odds_test['2st_underdog_prob'], odds_test['2st_favored_prob'] = test_second_stage_probs[:,0], test_second_stage_probs[:,1]

log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Neural Network & Odds', log_loss(odds_test["favored_win"].astype(int), odds_test["2st_favored_prob"])]



# Betting
odds_test = odds_test.dropna(subset=['favored_max_odds', 'underdog_max_odds', 'favored_pinnacle_odds', 'underdog_pinnacle_odds']).reset_index(drop=True)
odds_test = odds_test[(odds_test['underdog_max_odds'] < 3) & (odds_test['underdog_max_odds'] > 1.5)].reset_index(drop=True)

expected_return_favored = (odds_test['favored_max_odds']-1)*odds_test['2st_favored_prob'] - 1*(1-odds_test['2st_favored_prob'])
expected_return_underdog = (odds_test['underdog_max_odds']-1)*odds_test['2st_underdog_prob'] - 1*(1-odds_test['2st_underdog_prob'])
odds_test['bet'] = np.where(np.maximum(expected_return_favored, expected_return_underdog) < 0, 'no_bet', np.where(expected_return_favored > expected_return_underdog, 'favored', 'underdog'))
odds_test['expected_return'] = np.where(odds_test['bet'] == 'no_bet', 0, np.where(expected_return_favored > expected_return_underdog, expected_return_favored, expected_return_underdog))

odds_test['probs'] = np.where(odds_test['bet'] == 'favored', odds_test['2st_favored_prob'], odds_test['2st_underdog_prob'])
odds_test['odds'] = np.where(odds_test['bet'] == 'favored', odds_test['favored_max_odds'], odds_test['underdog_max_odds'])

odds_test['kelly_fraction_fixed'] = kelly_fraction_calc(odds_test['odds'], odds_test['probs']).apply(lambda x: max(x,0))

odds_test['result_kelly_fixed'] = np.where(odds_test['bet'] == 'no_bet', 0,
                                        np.where((odds_test['bet'] == 'favored') & (odds_test['favored_win']== 1), (odds_test['favored_max_odds']-1)*odds_test['kelly_fraction_fixed'],
                                                np.where((odds_test['bet'] == 'underdog') & (odds_test['favored_win']== 0), (odds_test['underdog_max_odds']-1)*odds_test['kelly_fraction_fixed'],
                                                        -odds_test['kelly_fraction_fixed'] )))	

total_bets_kelly = len(odds_test)
bets_placed_kelly = (odds_test['bet'] != 'no_bet').sum()
wins_kelly = (pd.to_numeric(odds_test['result_kelly_fixed'], errors='coerce') > 0).sum()
hit_rate_kelly = wins_kelly/bets_placed_kelly
earnings_kelly = pd.to_numeric(odds_test['result_kelly_fixed'], errors='coerce').fillna(0).sum()
spent_kelly = (odds_test['kelly_fraction_fixed']).sum()
return_kelly = earnings_kelly/spent_kelly 
earnings_per_bet_kelly = earnings_kelly/bets_placed_kelly
earnings_per_win_kelly = earnings_kelly/wins_kelly

predictions = {'bets':odds_test,
            'stats':{'earnings':earnings_kelly, 'spent':spent_kelly, 'return':return_kelly,'per_bet':earnings_per_bet_kelly, 'per_win':earnings_per_win_kelly}, 
            'other_stats':{'total':total_bets_kelly, 'placed':bets_placed_kelly, "wins":wins_kelly, 'hit_rate':hit_rate_kelly}}
print(predictions)