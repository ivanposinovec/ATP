import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from Scripts.functions import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from xgboost import XGBClassifier
from skopt import gp_minimize
from skopt.space import Integer, Real
from sklearn.metrics import log_loss
pd.set_option('display.max_rows',600)

df = pd.read_csv('games_merged.csv')
df['tourney_date'] = pd.to_datetime(df['tourney_date'])
df['round'] = pd.Categorical(df['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)

with open('odds_data.json', 'r') as json_file:
    odds_data = json.load(json_file)
odds = pd.DataFrame(odds_data)

odds = odds[(~odds['comment'].isin(['canc.', 'w.o.', 'award.', 'ret.']))].reset_index(drop=True)

odds = odds.explode('odds').reset_index(drop=True)
odds = pd.concat([odds.drop(columns='odds').reset_index(drop=True), pd.json_normalize(odds['odds']).reset_index(drop=True)], axis=1)
odds.rename(columns={'clossing_odds1':'closing_odds1', 'clossing_odds2':'closing_odds2', 'odds1':'mean_closing_odds1', 'odds2':'mean_closing_odds2'}, inplace=True)
odds['opening_time'] = pd.to_datetime(odds['opening_time'])
odds['closing_time'] = pd.to_datetime(odds['closing_time'])

bookmaker_counts = odds['bookmaker'].value_counts()
selected_bookmakers = bookmaker_counts[bookmaker_counts >= 55000].index.tolist()
odds = odds[odds['bookmaker'].isin(selected_bookmakers)].reset_index(drop=True)


odds_wide = odds.pivot_table(
    index=['game_url', 'player1', 'player2', 'mean_closing_odds1', 'mean_closing_odds2'],
    columns='bookmaker',
    values=['opening_odds1', 'opening_odds2', 'closing_odds1', 'closing_odds2']
)
odds_wide.columns = ['_'.join([str(col[1]), col[0]]) for col in odds_wide.columns]
odds_wide = odds_wide.reset_index()

odds_wide.dropna(subset=['Betsafe_closing_odds1', 'Betsson_closing_odds1', 'Betway_closing_odds1', 'NordicBet_closing_odds1', 'Pinnacle_closing_odds1', 'bet-at-home_closing_odds1',
                        'bet365_closing_odds1', 'Betsafe_closing_odds2', 'Betsson_closing_odds2', 'Betway_closing_odds2', 'NordicBet_closing_odds2', 'Pinnacle_closing_odds2',
                        'bet-at-home_closing_odds2', 'bet365_closing_odds2', 'Betsafe_opening_odds1', 'Betsson_opening_odds1', 'Betway_opening_odds1', 'NordicBet_opening_odds1',
                        'Pinnacle_opening_odds1', 'bet-at-home_opening_odds1', 'bet365_opening_odds1', 'Betsafe_opening_odds2', 'Betsson_opening_odds2', 'Betway_opening_odds2',
                        'NordicBet_opening_odds2', 'Pinnacle_opening_odds2',  'bet-at-home_opening_odds2', 'bet365_opening_odds2'])

games = pd.merge(df[['game_url', 'season', 'tourney_date', 'tourney_name', 'round', 'winner', 'winner_name', 'loser', 'loser_name']], odds_wide, on = ['game_url'], how = 'right', indicator = False)

for bookmaker in selected_bookmakers:
    games[f'w_{bookmaker}_opening_odds'] = games.apply(lambda row: row[f'{bookmaker}_opening_odds1'] if row['player1'] == row['winner'] else row[f'{bookmaker}_opening_odds2'], axis=1)
    games[f'l_{bookmaker}_opening_odds'] = games.apply(lambda row: row[f'{bookmaker}_opening_odds2'] if row['player1'] == row['winner'] else row[f'{bookmaker}_opening_odds1'], axis=1)
    
    games[f'w_{bookmaker}_closing_odds'] = games.apply(lambda row: row[f'{bookmaker}_closing_odds1'] if row['player1'] == row['winner'] else row[f'{bookmaker}_closing_odds2'], axis=1)
    games[f'l_{bookmaker}_closing_odds'] = games.apply(lambda row: row[f'{bookmaker}_closing_odds2'] if row['player1'] == row['winner'] else row[f'{bookmaker}_closing_odds1'], axis=1)
    
    games = games[(games[f'w_{bookmaker}_opening_odds'] >= 1) & (games[f'w_{bookmaker}_closing_odds'] >= 1) & (games[f'l_{bookmaker}_opening_odds'] >= 1) & (games[f'l_{bookmaker}_closing_odds'] >= 1)].reset_index(drop=True)

games['w_opening_odds'] = games['w_Pinnacle_opening_odds']
games['l_opening_odds'] = games['l_Pinnacle_opening_odds']

games['w_closing_odds'] = games['w_Pinnacle_closing_odds']
games['l_closing_odds'] = games['l_Pinnacle_closing_odds']

games[f'w_mean_closing_odds'] = games.apply(lambda row: row[f'mean_closing_odds1'] if row['player1'] == row['winner'] else row[f'mean_closing_odds2'], axis=1)
games[f'l_mean_closing_odds'] = games.apply(lambda row: row[f'mean_closing_odds2'] if row['player1'] == row['winner'] else row[f'mean_closing_odds1'], axis=1)

# Win-Loser to Favored-Underdog
games['favored_win'] = games.apply(lambda row: 1 if row['w_opening_odds'] < row['l_opening_odds'] else 0, axis=1)

games['favored'] = games.apply(lambda row: row['winner_name'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_name'], axis=1)
games['underdog'] = games.apply(lambda row: row['loser_name'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_name'], axis=1)

games[f'favored_odds'] = games.apply(lambda row: row[f'w_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row[f'l_opening_odds'], axis=1)
games[f'underdog_odds'] = games.apply(lambda row: row[f'l_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row[f'w_opening_odds'], axis=1)

games = games[(games['underdog_odds'] <= 29) & (games['underdog_odds'] >= 1.85) & (games['favored_odds'] <= 2) & (games['favored_odds'] >= 1)].reset_index(drop=True)

games[f'favored_mean_closing_odds'] = games.apply(lambda row: row[f'w_mean_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row[f'l_mean_closing_odds'], axis=1)
games[f'underdog_mean_closing_odds'] = games.apply(lambda row: row[f'l_mean_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row[f'w_mean_closing_odds'], axis=1)

games = games[(games['underdog_mean_closing_odds'] <= 29) & (games['underdog_mean_closing_odds'] >= 1.85) & (games['favored_mean_closing_odds'] <= 2) & (games['favored_mean_closing_odds'] >= 1)].reset_index(drop=True)

for bookmaker in selected_bookmakers:
    games[f'favored_{bookmaker}_opening_odds'] = games.apply(lambda row: row[f'w_{bookmaker}_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row[f'l_{bookmaker}_opening_odds'], axis=1)
    games[f'underdog_{bookmaker}_opening_odds'] = games.apply(lambda row: row[f'l_{bookmaker}_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row[f'w_{bookmaker}_opening_odds'], axis=1)
    
    games[f'favored_{bookmaker}_closing_odds'] = games.apply(lambda row: row[f'w_{bookmaker}_closing_odds'] if row['w_closing_odds'] < row['l_closing_odds'] else row[f'l_{bookmaker}_closing_odds'], axis=1)
    games[f'underdog_{bookmaker}_closing_odds'] = games.apply(lambda row: row[f'l_{bookmaker}_closing_odds'] if row['w_closing_odds'] < row['l_closing_odds'] else row[f'w_{bookmaker}_closing_odds'], axis=1)
    
    games = games[(games[f'underdog_{bookmaker}_opening_odds'] <= 29) & (games[f'underdog_{bookmaker}_opening_odds'] >= 1.85) & (games[f'favored_{bookmaker}_opening_odds'] <= 2) & (games[f'favored_{bookmaker}_opening_odds'] >= 1)].reset_index(drop=True)
    
    implied_probs = np.array([get_no_vig_odds(odds1=favored_odds_row, odds2=underdog_odds_row, odds0=None) for favored_odds_row, underdog_odds_row in zip(games[f'favored_{bookmaker}_opening_odds'], games[f'underdog_{bookmaker}_opening_odds'])])
    games[f'favored_opening_prob_{bookmaker}'], games[f'underdog_opening_prob_{bookmaker}'] = 1 / implied_probs[:, 0], 1 / implied_probs[:, 1]
    
    implied_probs = np.array([get_no_vig_odds(odds1=favored_odds_row, odds2=underdog_odds_row, odds0=None) for favored_odds_row, underdog_odds_row in zip(games[f'favored_{bookmaker}_closing_odds'], games[f'underdog_{bookmaker}_closing_odds'])])
    games[f'favored_closing_prob_{bookmaker}'], games[f'underdog_closing_prob_{bookmaker}'] = 1 / implied_probs[:, 0], 1 / implied_probs[:, 1]


features = [col for col in games.columns if 'favored' in col and 'prob' in col and 'opening' in col]
X = games[features]
y = games['favored_win']
odds = games.copy()

models = {}
model = 'xgb'
for fold_number in range(0, 6):
    train_lower = 2010 + fold_number
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
    
    scaler = StandardScaler()
    scaler.fit(X_train_val[features])
    
    X_train_scaled = pd.DataFrame(scaler.transform(X_train[features]), columns=features)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val[features]), columns=features)
    X_train_val_scaled = pd.DataFrame(scaler.transform(X_train_val[features]), columns=features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[features]), columns=features)
    
    models[f'{test_season}'] = {}
    models[f'{test_season}']['data'] = {'X_train_scaled':X_train_scaled, 'X_val_scaled':X_val_scaled, 'X_train_val':X_train_val_scaled, 'X_test_scaled':X_test_scaled,
                                        'y_train':y_train, 'y_val':y_val, 'y_train_val':y_train_val, 'y_test':y_test,
                                        'odds_train':odds_train, 'odds_val':odds_val, 'odds_train_val':odds_train_val, 'odds_test':odds_test}
    
    if model == 'enet':
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
        
        coefficients = pd.DataFrame({
            'Feature': X_train_scaled.columns,
            'Coefficient': best_logit.coef_[0]
        }).sort_values(by='Coefficient', ascending=False)
        print(coefficients)
        selected_features = list(coefficients[coefficients['Coefficient'] != 0]['Feature'])
        
        odds_train_val[f'favored_prob_{model}'] = best_logit.predict_proba(X_train_val_scaled)[:, 1]
        odds_test[f'favored_prob_{model}'] = best_logit.predict_proba(X_test_scaled)[:, 1]
        odds_train_val[f'underdog_prob_{model}'] = best_logit.predict_proba(X_train_val_scaled)[:, 0]
        odds_test[f'underdog_prob_{model}'] = best_logit.predict_proba(X_test_scaled)[:, 0]
    
    elif model == 'xgb':
        search_space = [
            Integer(50, 1000, name='n_estimators'),
            Integer(2, 8, name='max_depth'),
            Real(0.001, 0.1, name= 'learning_rate'),
            Real(0.7, 1, name= 'subsample'),
            Real(0.7, 1, name= 'colsample_bytree'),
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
    
    
    log_loss_df = pd.DataFrame({'Bookmaker':'Model',
                                    'Train Log Loss':log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_prob_{model}']),
                                    'Test Log Loss':log_loss(odds_test['favored_win'], odds_test[f'favored_prob_{model}'])}, index=[0])
    for bookmaker in selected_bookmakers:
        log_loss_df = pd.concat([log_loss_df, pd.DataFrame({
            'Bookmaker': bookmaker,
            'Opening/Closing': 'Opening',
            'Train Log Loss': log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_opening_prob_{bookmaker}']),
            'Test Log Loss': log_loss(odds_test['favored_win'], odds_test[f'favored_opening_prob_{bookmaker}'])
        }, index=[0])], axis = 0, ignore_index=True)
        log_loss_df = pd.concat([log_loss_df, pd.DataFrame({
            'Bookmaker': bookmaker,
            'Opening/Closing': 'Closing',
            'Train Log Loss': log_loss(odds_train_val['favored_win'], odds_train_val[f'favored_closing_prob_{bookmaker}']),
            'Test Log Loss': log_loss(odds_test['favored_win'], odds_test[f'favored_closing_prob_{bookmaker}'])
        }, index=[0])], axis = 0, ignore_index=True)
    print(f'\n======== Train Log Loss for {model} model (Train: {train_lower} - {val_upper}) (Test: {test_season}) ========')
    print(log_loss_df.set_index('Bookmaker').sort_values(by='Test Log Loss', ascending=True))
    
    models[f'{(test_season)}']['predictions'] = {'log_loss':log_loss_df.set_index('Bookmaker').sort_values(by='Test Log Loss', ascending=True),
                                                'train_predictions':odds_train_val, 'test_predictions':odds_test}

# Betting
bets = pd.concat([models[str(year)]['predictions']['test_predictions'] for year in range(2020, 2026)], ignore_index=True)

bets['favored_max_opening_odds'] = bets[[f'favored_{bookmaker}_opening_odds' for bookmaker in selected_bookmakers]].max(axis=1)
bets['favored_mean_opening_odds'] = bets[[f'favored_{bookmaker}_opening_odds' for bookmaker in selected_bookmakers]].mean(axis=1)
bets['underdog_max_opening_odds'] = bets[[f'underdog_{bookmaker}_opening_odds' for bookmaker in selected_bookmakers]].max(axis=1)
bets['underdog_mean_opening_odds'] = bets[[f'underdog_{bookmaker}_opening_odds' for bookmaker in selected_bookmakers]].mean(axis=1)

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

bets[bets['season'] == 2025].sort_values('expected_return', ascending=False).head(10)[['tourney_name', 'favored', 'underdog', 'favored_odds', 'underdog_odds', 'favored_mean_closing_odds', 'underdog_mean_closing_odds', 'bet', 'expected_return', 'kelly_fraction_fixed', 'result_kelly_fixed']]

bets.loc[12559]


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


bets_results = bets[bets['expected_return'] > 0.0].sort_values(['tourney_date', 'tourney_name', 'round']).copy()
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

plt.yscale('log')

plt.tight_layout()
plt.show()


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