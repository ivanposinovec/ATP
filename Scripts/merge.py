import pandas as pd
import numpy as np
from tqdm import tqdm
import janitor
from Scripts.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
pd.set_option('display.max_rows',600)
import json

with open('odds_data.json', 'r') as json_file:
    odds_data = json.load(json_file)

odds = pd.DataFrame(odds_data)

games_odds = pd.read_csv('games_oddsportal2.csv')
games_odds.rename(columns={'tournament_stats':'tourney_name'}, inplace=True)
games_odds = games_odds[(~games_odds['comment'].isin(['canc.', 'w.o.', 'award.']))].reset_index(drop=True)
games_odds[games_odds['player1'].isna()]


odds = pd.merge(games_odds[['tourney_name', 'season', 'game_url', 'player1', 'player2', 'comment', 'odds1', 'odds2']], odds, on = ['game_url', 'player1', 'player2', 'comment', 'odds1', 'odds2'], how = 'left')
odds['season'] = odds['season'].astype(int)

# Stats data
seasons = list(range(1978, 2026))
games = pd.DataFrame()
for season in seasons:
    if season != 2025:
        df = pd.concat([pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv'),pd.read_csv(f'tennis_atp-master/atp_matches_qual_chall_{season}.csv')],axis = 0).reset_index(drop=True)
    else:
        df = pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv')
    df.insert(0, 'season', season)
    df.loc[df['tourney_name'].str.contains(' Olympics', na=False), 'tourney_level'] = 'O'
    
    games = pd.concat([games, df], axis = 0).reset_index(drop=True)

different_names_dict = {'Edouard Roger-Vasselin':'Edouard Roger Vasselin'}
games['winner_name'].replace(different_names_dict, inplace=True)
games['loser_name'].replace(different_names_dict, inplace=True)

games['tourney_date'] = pd.to_datetime(games['tourney_date'], format = '%Y%m%d')
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games = games[(~games['tourney_name'].isin(['Laver Cup']))].sort_values(by=['tourney_date', 'tourney_name', 'round']).reset_index(drop=True)

#games.drop(columns = ['winner', 'loser', 'game_id'],inplace=True) 
games.insert(games.columns.get_loc('winner_name'), 'winner', games['winner_name'].apply(lambda x: ' '.join(x.strip().split(' ')[1:]) + ' ' + x.strip().split(' ')[0][0] + '.' if len(x.strip().split(' ')) > 1 else x))
games.insert(games.columns.get_loc('loser_name'), 'loser', games['loser_name'].apply(lambda x: ' '.join(x.strip().split(' ')[1:]) + ' ' + x.strip().split(' ')[0][0] + '.' if len(x.strip().split(' ')) > 1 else x))
games.insert(games.columns.get_loc('match_num'), 'game_id', games.groupby(['winner', 'loser', 'tourney_name', 'tourney_date', 'match_num', 'season']).ngroup() + 1)
games.insert(games.columns.get_loc('tourney_id'), 'week', games['tourney_date'].dt.isocalendar().week)
print(games[games.duplicated(subset=['game_id'], keep=False)][['winner', 'loser', 'tourney_name', 'season']].sort_values(by=['tourney_name', 'season']))

for index, row in tqdm(games.iterrows(), total = len(games)):
    if row['winner_name'] == 'Alex Kuznetsov':
        games.loc[index, 'winner'] = 'Kuznetsov Al.'
    elif row['winner_name'] == 'Andrey Kuznetsov':
        games.loc[index, 'winner'] = 'Kuznetsov An.'
    elif row['winner_name'] == 'Ze Zhang':
        games.loc[index, 'winner'] = 'Zhang Ze.'
    elif row['winner_name'] == 'Zhizhen Zhang':
        games.loc[index, 'winner'] = 'Zhang Zh.'
        
    if row['loser_name'] == 'Alex Kuznetsov':
        games.loc[index, 'loser'] = 'Kuznetsov Al.'
    elif row['loser_name'] == 'Andrey Kuznetsov':
        games.loc[index, 'loser'] = 'Kuznetsov An.'
    elif row['loser_name'] == 'Ze Zhang':
        games.loc[index, 'loser'] = 'Zhang Ze.'
    elif row['loser_name'] == 'Zhizhen Zhang':
        games.loc[index, 'loser'] = 'Zhang Zh.'

odds[['player1', 'player2', 'tourney_name', 'season']]
games[['winner', 'loser', 'tourney_name', 'season']]

merged1 = odds.merge(
    games[['winner', 'loser', 'tourney_name', 'season']],
    left_on=['player1', 'player2', 'tourney_name', 'season'],
    right_on=['winner', 'loser', 'tourney_name', 'season'],
    how='inner'
)

# Second merge: player1 == loser, player2 == winner
merged2 = odds.merge(
    games[['winner', 'loser', 'tourney_name', 'season']],
    left_on=['player1', 'player2', 'tourney_name', 'season'],
    right_on=['loser', 'winner', 'tourney_name', 'season'],
    how='inner'
)
merged1['match_order'] = 'normal'
merged2['match_order'] = 'reversed'

combined = pd.concat([merged1, merged2], ignore_index=True)
odds = odds.merge(combined[['game_url', 'winner', 'loser', 'match_order']], on = 'game_url', how = 'left')

error_df = odds[odds['winner'].isna()].sort_values(by=['player1', 'season']).tail(500)

odds.to_csv('test.csv', index=False)




df = df.explode('odds').reset_index(drop=True)
odds_expanded = pd.json_normalize(df['odds'])
odds = pd.concat([df.drop(columns='odds').reset_index(drop=True), odds_expanded.reset_index(drop=True)], axis=1)
bookmakers = list(odds['bookmaker'].unique())

wide_df = odds.pivot_table(
    index=['game_url',],
    columns='bookmaker',
    values = ['opening_time', 'opening_odds1',  'clossing_odds1',  'opening_odds2',  'clossing_odds2'],
    aggfunc='first'  # or use list if multiple entries per bookmaker/game
)

wide_df.columns = ['_'.join(col).strip('_') for col in wide_df.columns.values]
odds = pd.merge(pd.DataFrame(odds_data), wide_df, on = 'game_url')


