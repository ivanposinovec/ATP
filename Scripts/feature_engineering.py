import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from Scripts.functions import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from skopt import BayesSearchCV
from bayes_opt import BayesianOptimization
from datetime import datetime
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
odds.rename(columns={'clossing_odds1':'closing_odds1', 'clossing_odds2':'closing_odds2'}, inplace=True)
odds['opening_time'] = pd.to_datetime(odds['opening_time'])
odds['closing_time'] = pd.to_datetime(odds['closing_time'])

mean_odds = odds.groupby('game_url').agg({'opening_odds1':'mean', 'opening_odds2':'mean', 'closing_odds1':'mean', 'closing_odds2':'mean'}).rename(columns= {'opening_odds1':'mean_opening_odds1', 'opening_odds2':'mean_opening_odds2', 'closing_odds1':'mean_closing_odds1', 'closing_odds2':'mean_closing_odds2'})
max_odds = odds.groupby('game_url').agg({'opening_odds1':'max', 'opening_odds2':'max', 'closing_odds1':'max', 'closing_odds2':'max'}).rename(columns= {'opening_odds1':'max_opening_odds1', 'opening_odds2':'max_opening_odds2', 'closing_odds1':'max_closing_odds1', 'closing_odds2':'max_closing_odds2'})
min_odds = odds.groupby('game_url').agg({'opening_odds1':'min', 'opening_odds2':'min', 'closing_odds1':'min', 'closing_odds2':'min'}).rename(columns= {'opening_odds1':'min_opening_odds1', 'opening_odds2':'min_opening_odds2', 'closing_odds1':'min_closing_odds1', 'closing_odds2':'min_closing_odds2'})
pinnacle_odds = odds[odds['bookmaker'] == 'Pinnacle'].set_index('game_url')[['opening_odds1', 'opening_odds2', 'closing_odds1', 'closing_odds2']].rename(columns= {'opening_odds1':'pinnacle_opening_odds1', 'opening_odds2':'pinnacle_opening_odds2', 'closing_odds1':'pinnacle_closing_odds1', 'closing_odds2':'pinnacle_closing_odds2'})

odds = odds.sort_values(['game_url', 'opening_time']).drop_duplicates(subset = 'game_url').set_index('game_url').drop(columns=['player1', 'player2', 'odds1', 'odds2', 'comment', 'bookmaker', 'opening_odds1', 'opening_odds2', 'closing_odds1', 'closing_odds2'])
odds = pd.concat([odds, mean_odds, max_odds, min_odds, pinnacle_odds], axis = 1)

games = pd.merge(df, odds, on = ['game_url'], how = 'left', indicator = False)

print(games[games.duplicated(subset=['game_id'], keep=False)][['player1', 'player2', 'game_id', 'tourney_name', 'season', 'round', 'mean_closing_odds1', 'mean_closing_odds2']].sort_values(by=['tourney_name', 'season']))
games = games.drop([324836, 324868, 329750, 329758, 353046, 353058, 368609, 368630, 306571, 306600, 251350, 251362, 373499, 373507, 341340, 341345, 354865, 354875, 308961, 308984,
                    335763, 335772, 379184, 379195, 298301, 298317, 350717, 350730, 249235, 249267, 357004, 357011, 310232, 310246, 235821, 235832, 298528, 298537, 378726, 378741,
                    225882, 225895, 273253, 273269, 297665, 297675, 310596, 310623], axis=0).reset_index(drop=True)

# Elo rankings
games['surface'].replace({'Carpet':'Hard'},inplace=True)
class CalculateElo:
    def __init__(self, surface, games):
        # Initialize player Elo and match count dictionaries
        self.surface = surface
        self.playersToElo = {}
        self.matchesCount = {}
        self.games = games
        
        self.elo_df = pd.DataFrame()
    
    # Update matches count
    def updateMatchesCount(self,playerA, playerB):
        self.matchesCount[playerA] = self.matchesCount.get(playerA, 0) + 1
        self.matchesCount[playerB] = self.matchesCount.get(playerB, 0) + 1

    # Update Elo ratings
    def updateElo(self, playerA, playerB, winner, level, match_date, match_num):
        rA = self.playersToElo.get(playerA, [(1500, datetime(1900, 1, 1), 0)])[-1][0]
        rB = self.playersToElo.get(playerB, [(1500, datetime(1900, 1, 1), 0)])[-1][0]

        eA = 1 / (1 + 10 ** ((rB - rA) / 400))
        eB = 1 / (1 + 10 ** ((rA - rB) / 400))
        sA, sB = (1, 0) if winner == playerA else (0, 1)

        kA = 250 / ((self.matchesCount[playerA] + 5) ** 0.4)
        kB = 250 / ((self.matchesCount[playerB] + 5) ** 0.4)
        k = 1.1 if level == "G" else 1.0

        rA_new = rA + (k * kA) * (sA - eA)
        rB_new = rB + (k * kB) * (sB - eB)

        self.playersToElo.setdefault(playerA, []).append((rA_new, match_date, match_num))
        self.playersToElo.setdefault(playerB, []).append((rB_new, match_date, match_num))
        return rA, rB

    def calculate_elo(self):
        if self.surface == None:
            for index, row in tqdm(self.games.iterrows(), total = len(self.games), desc = f'Getting Elo'):
                playerA, playerB = row["winner_name"], row["loser_name"]
                self.updateMatchesCount(playerA, playerB)
                self.elo_df.loc[index, ['winner_elo', 'loser_elo']] = self.updateElo(playerA, playerB, row["winner_name"], row["tourney_level"], row["tourney_date"], row["match_num"])
        else:
            self.games = self.games[self.games['surface'] == self.surface]
            for index, row in tqdm(self.games.iterrows(), total = len(self.games), desc = f'Getting Elo {self.surface}'):
                playerA, playerB = row["winner_name"], row["loser_name"]
                self.updateMatchesCount(playerA, playerB)
                self.elo_df.loc[index, [f'winner_elo_{self.surface}', f'loser_elo_{self.surface}']] = self.updateElo(playerA, playerB, row["winner_name"], row["tourney_level"], row["tourney_date"], row["match_num"])

ranking_elo = {}
for surface in [None, 'Hard', 'Grass', 'Clay']:
    EloCalculator = CalculateElo(surface=surface, games = games)
    EloCalculator.calculate_elo()
    if surface is None:
        ranking_elo[f'Elo'] = EloCalculator.elo_df
    else:
        ranking_elo[f'Elo {surface}'] = EloCalculator.elo_df

elo = ranking_elo['Elo'][['winner_elo', 'loser_elo']]
elo_surface = pd.concat([ranking_elo['Elo Hard'].rename(columns={'winner_elo_Hard':'winner_elo_surface', 'loser_elo_Hard':'loser_elo_surface'})[['winner_elo_surface', 'loser_elo_surface']],
                        ranking_elo['Elo Grass'].rename(columns={'winner_elo_Grass':'winner_elo_surface', 'loser_elo_Grass':'loser_elo_surface'})[['winner_elo_surface', 'loser_elo_surface']],
                        ranking_elo['Elo Clay'].rename(columns={'winner_elo_Clay':'winner_elo_surface', 'loser_elo_Clay':'loser_elo_surface'})[['winner_elo_surface', 'loser_elo_surface']]], axis = 0)
games = pd.concat([games, elo, elo_surface],axis=1)
#games[(games['winner_name'] == 'Novak Djokovic') | (games['loser_name'] == 'Novak Djokovic')][['winner_name', 'loser_name', 'tourney_date', 'season', 'tourney_name', 'round', 'winner_elo', 'winner_elo_surface', 'loser_elo', 'loser_elo_surface']]

games = games[(~games['tourney_level'].isin(['D'])) & (games['season'] >= 1990)].reset_index(drop=True)
sorted(list(games[games['tourney_level'] != 'C']['tourney_name'].unique()))
games.rename(columns = {'ATP Rio de Janeiro':'Rio de Janeiro', 'Rio De Janeiro':'Rio de Janeiro', 'Adelaide 1':'Adelaide', 'Adelaide 2':'Adelaide',
                    'Belgrade ':'Belgrade', 'Belgrade 2':'Belgrade', 'Us Open':'US Open', 'Next Gen Finals':'NextGen Finals', 'Kuala Lumpur-1':'Kuala Lumpur', 'Kuala Lumpur-2':'Kuala Lumpur',
                    'Cologne 1':'Cologne', 'Cologne 2':'Cologne', 'Masters Cup':'Tour Finals', 'Doha Aus Open Qualies':'Australian Open'}, inplace=True)


# Match features # 
#print(games[games['score'].str.contains('[a-zA-Z]', na=False)]['score'].str.extract(r'([a-zA-Z]+)').drop_duplicates())
games.drop(columns=['comment'], inplace=True)
games.insert(games.columns.get_loc('score'), 'comment', np.where(games['score'].isna(), 'No Data',
                                                            np.where(games['score'].str.contains(r'\b(RET|RE|Ret)\b', na=False), 'Retired',
                                                                np.where(games['score'].str.contains(r'\bW/O\b', na=False), 'Walkover',
                                                                    np.where(games['score'].str.contains(r'\b(DEF|Def)\b', na=False), 'Default', 'Completed')))))

games[['w_set1', 'l_set1', 'w_set2', 'l_set2', 
        'w_set3', 'l_set3', 'w_set4', 'l_set4', 
        'w_set5', 'l_set5']] = games['score'].apply(lambda x: pd.Series(parse_score(x)))

games['w_sets'] = games[['w_set1', 'w_set2', 'w_set3', 'w_set4', 'w_set5']].gt(
    games[['l_set1', 'l_set2', 'l_set3', 'l_set4', 'l_set5']].values).sum(axis=1)
games['l_sets'] = games[['l_set1', 'l_set2', 'l_set3', 'l_set4', 'l_set5']].gt(
    games[['w_set1', 'w_set2', 'w_set3', 'w_set4', 'w_set5']].values).sum(axis=1)

games['w_games'] = games[['w_set1', 'w_set2', 'w_set3', 'w_set4', 'w_set5']].fillna(0).sum(axis=1)
games['l_games'] = games[['l_set1', 'l_set2', 'l_set3', 'l_set4', 'l_set5']].fillna(0).sum(axis=1)

games['w_bpWon'] = games['l_bpFaced'] - games['l_bpSaved']
games['l_bpWon'] = games['w_bpFaced'] - games['w_bpSaved']

games['w_svpt_won'] = games['w_1stWon']+games['w_2ndWon']
games['l_svpt_won'] = games['l_1stWon']+games['l_2ndWon']

games['w_rtpt_won'] = games['l_svpt']-games['l_1stWon']-games['l_2ndWon']
games['l_rtpt_won'] = games['w_svpt']-games['w_1stWon']-games['w_2ndWon']

games["winner_rank"]=games["winner_rank"].replace(np.nan,2500).astype(int)
games["loser_rank"]=games["loser_rank"].replace(np.nan,2500).astype(int)

games['winner_rank_group'] = games['winner_rank'].apply(rank_group)
games['loser_rank_group'] = games['loser_rank'].apply(rank_group)

games['w_mean_opening_odds'] = games.apply(lambda row: row['mean_opening_odds1'] if row['player1'] == row['winner'] else row['mean_opening_odds2'], axis=1)
games['l_mean_opening_odds'] = games.apply(lambda row: row['mean_opening_odds2'] if row['player1'] == row['winner'] else row['mean_opening_odds1'], axis=1)

games['w_mean_closing_odds'] = games.apply(lambda row: row['mean_closing_odds1'] if row['player1'] == row['winner'] else row['mean_closing_odds2'], axis=1)
games['l_mean_closing_odds'] = games.apply(lambda row: row['mean_closing_odds2'] if row['player1'] == row['winner'] else row['mean_closing_odds1'], axis=1)

games['w_max_opening_odds'] = games.apply(lambda row: row['max_opening_odds1'] if row['player1'] == row['winner'] else row['max_opening_odds2'], axis=1)
games['l_max_opening_odds'] = games.apply(lambda row: row['max_opening_odds2'] if row['player1'] == row['winner'] else row['max_opening_odds1'], axis=1)

games['w_max_closing_odds'] = games.apply(lambda row: row['max_closing_odds1'] if row['player1'] == row['winner'] else row['max_closing_odds2'], axis=1)
games['l_max_closing_odds'] = games.apply(lambda row: row['max_closing_odds2'] if row['player1'] == row['winner'] else row['max_closing_odds1'], axis=1)

games['w_min_opening_odds'] = games.apply(lambda row: row['min_opening_odds1'] if row['player1'] == row['winner'] else row['min_opening_odds2'], axis=1)
games['l_min_opening_odds'] = games.apply(lambda row: row['min_opening_odds2'] if row['player1'] == row['winner'] else row['min_opening_odds1'], axis=1)

games['w_min_closing_odds'] = games.apply(lambda row: row['min_closing_odds1'] if row['player1'] == row['winner'] else row['min_closing_odds2'], axis=1)
games['l_min_closing_odds'] = games.apply(lambda row: row['min_closing_odds2'] if row['player1'] == row['winner'] else row['min_closing_odds1'], axis=1)

games['w_pinnacle_opening_odds'] = games.apply(lambda row: row['pinnacle_opening_odds1'] if row['player1'] == row['winner'] else row['pinnacle_opening_odds2'], axis=1)
games['l_pinnacle_opening_odds'] = games.apply(lambda row: row['pinnacle_opening_odds2'] if row['player1'] == row['winner'] else row['pinnacle_opening_odds1'], axis=1)

games['w_pinnacle_closing_odds'] = games.apply(lambda row: row['pinnacle_closing_odds1'] if row['player1'] == row['winner'] else row['pinnacle_closing_odds2'], axis=1)
games['l_pinnacle_closing_odds'] = games.apply(lambda row: row['pinnacle_closing_odds2'] if row['player1'] == row['winner'] else row['pinnacle_closing_odds1'], axis=1)

games['w_opening_odds'] = np.where(games['w_pinnacle_opening_odds'].notnull(), games['w_pinnacle_opening_odds'], games['w_mean_opening_odds'])
games['l_opening_odds'] = np.where(games['l_pinnacle_opening_odds'].notnull(), games['l_pinnacle_opening_odds'], games['l_mean_opening_odds'])

games['w_closing_odds'] = np.where(games['w_pinnacle_closing_odds'].notnull(), games['w_pinnacle_closing_odds'], games['w_mean_closing_odds'])
games['l_closing_odds'] = np.where(games['l_pinnacle_closing_odds'].notnull(), games['l_pinnacle_closing_odds'], games['l_mean_closing_odds'])

# Winner-Loser to Favored-Underdog
games['favored_win'] = games.apply(lambda row: 1 if row['w_opening_odds'] < row['l_opening_odds'] else 0, axis=1)

games['favored'] = games.apply(lambda row: row['winner_name'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_name'], axis=1)
games['underdog'] = games.apply(lambda row: row['loser_name'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_name'], axis=1)

games['favored_id'] = games.apply(lambda row: row['winner_id'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_id'], axis=1)
games['underdog_id'] = games.apply(lambda row: row['loser_id'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_id'], axis=1)

games['favored_entry'] = games.apply(lambda row: row['winner_entry'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_entry'], axis=1)
games['underdog_entry'] = games.apply(lambda row: row['loser_entry'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_entry'], axis=1)

games['favored_seed'] = games.apply(lambda row: row['winner_seed'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_seed'], axis=1)
games['underdog_seed'] = games.apply(lambda row: row['loser_seed'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_seed'], axis=1)

games['favored_hand'] = games.apply(lambda row: row['winner_hand'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_hand'], axis=1)
games['underdog_hand'] = games.apply(lambda row: row['loser_hand'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_hand'], axis=1)

games['favored_ht'] = games.apply(lambda row: row['winner_ht'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_ht'], axis=1)
games['underdog_ht'] = games.apply(lambda row: row['loser_ht'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_ht'], axis=1)

games['favored_ioc'] = games.apply(lambda row: row['winner_ioc'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_ioc'], axis=1)
games['underdog_ioc'] = games.apply(lambda row: row['loser_ioc'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_ioc'], axis=1)

games['favored_age'] = games.apply(lambda row: row['winner_age'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_age'], axis=1)
games['underdog_age'] = games.apply(lambda row: row['loser_age'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_age'], axis=1)

games['favored_rank'] = games.apply(lambda row: row['winner_rank'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_rank'], axis=1)
games['underdog_rank'] = games.apply(lambda row: row['loser_rank'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_rank'], axis=1)

games['favored_rank_group'] = games.apply(lambda row: row['winner_rank_group'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_rank_group'], axis=1)
games['underdog_rank_group'] = games.apply(lambda row: row['loser_rank_group'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_rank_group'], axis=1)

games['favored_elo'] = games.apply(lambda row: row['winner_elo'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_elo'], axis=1)
games['underdog_elo'] = games.apply(lambda row: row['loser_elo'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_elo'], axis=1)

games['favored_elo_surface'] = games.apply(lambda row: row['winner_elo_surface'] if row['w_opening_odds'] < row['l_opening_odds'] else row['loser_elo_surface'], axis=1)
games['underdog_elo_surface'] = games.apply(lambda row: row['loser_elo_surface'] if row['w_opening_odds'] < row['l_opening_odds'] else row['winner_elo_surface'], axis=1)

games['favored_sets'] = games.apply(lambda row: row['w_sets'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_sets'], axis=1)
games['underdog_sets'] = games.apply(lambda row: row['l_sets'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_sets'], axis=1)

games['favored_games'] = games.apply(lambda row: row['w_games'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_games'], axis=1)
games['underdog_games'] = games.apply(lambda row: row['l_games'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_games'], axis=1)

games['favored_ace'] = games.apply(lambda row: row['w_ace'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_ace'], axis=1)
games['underdog_ace'] = games.apply(lambda row: row['l_ace'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_ace'], axis=1)

games['favored_df'] = games.apply(lambda row: row['w_df'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_df'], axis=1)
games['underdog_df'] = games.apply(lambda row: row['l_df'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_df'], axis=1)

games['favored_svpt'] = games.apply(lambda row: row['w_svpt'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_svpt'], axis=1)
games['underdog_svpt'] = games.apply(lambda row: row['l_svpt'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_svpt'], axis=1)

games['favored_1st_in'] = games.apply(lambda row: row['w_1stIn'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_1stIn'], axis=1)
games['underdog_1st_in'] = games.apply(lambda row: row['l_1stIn'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_1stIn'], axis=1)

games['favored_1st_won'] = games.apply(lambda row: row['w_1stWon'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_1stWon'], axis=1)
games['underdog_1st_won'] = games.apply(lambda row: row['l_1stWon'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_1stWon'], axis=1)

games['favored_2nd_won'] = games.apply(lambda row: row['w_2ndWon'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_2ndWon'], axis=1)
games['underdog_2nd_won'] = games.apply(lambda row: row['l_2ndWon'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_2ndWon'], axis=1)

games['favored_serve_games'] = games.apply(lambda row: row['w_SvGms'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_SvGms'], axis=1)
games['underdog_serve_games'] = games.apply(lambda row: row['l_SvGms'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_SvGms'], axis=1)

games['favored_svpt_won'] = games.apply(lambda row: row['w_svpt_won'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_svpt_won'], axis=1)
games['underdog_svpt_won'] = games.apply(lambda row: row['l_svpt_won'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_svpt_won'], axis=1)

games['favored_rtpt_won'] = games.apply(lambda row: row['w_rtpt_won'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_rtpt_won'], axis=1)
games['underdog_rtpt_won'] = games.apply(lambda row: row['l_rtpt_won'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_rtpt_won'], axis=1)

games['favored_bp_saved'] = games.apply(lambda row: row['w_bpSaved'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_bpSaved'], axis=1)
games['underdog_bp_saved'] = games.apply(lambda row: row['l_bpSaved'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_bpSaved'], axis=1)

games['favored_bp_faced'] = games.apply(lambda row: row['w_bpFaced'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_bpFaced'], axis=1)
games['underdog_bp_faced'] = games.apply(lambda row: row['l_bpFaced'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_bpFaced'], axis=1)

games['favored_bp_won'] = games.apply(lambda row: row['w_bpWon'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_bpWon'], axis=1)
games['underdog_bp_won'] = games.apply(lambda row: row['l_bpWon'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_bpWon'], axis=1)

games['favored_dominance_ratio'] = (games['favored_rtpt_won'] / games['underdog_svpt']) / (games['underdog_rtpt_won'] / games['favored_svpt'])
games['underdog_dominance_ratio'] = (games['underdog_rtpt_won'] / games['favored_svpt']) / (games['favored_rtpt_won'] / games['underdog_svpt'])

games['favored_odds'] = np.where(games['w_opening_odds'] < games['l_opening_odds'], games['w_opening_odds'], games['l_opening_odds'])
games['underdog_odds'] = np.where(games['w_opening_odds'] >= games['l_opening_odds'], games['w_opening_odds'], games['l_opening_odds'])

games['favored_mean_opening_odds'] = games.apply(lambda row: row['w_mean_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_mean_opening_odds'], axis=1)
games['underdog_mean_opening_odds'] = games.apply(lambda row: row['l_mean_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_mean_opening_odds'], axis=1)

games['favored_mean_closing_odds'] = games.apply(lambda row: row['w_mean_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_mean_closing_odds'], axis=1)
games['underdog_mean_closing_odds'] = games.apply(lambda row: row['l_mean_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_mean_closing_odds'], axis=1)

games['favored_max_opening_odds'] = games.apply(lambda row: row['w_max_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_max_opening_odds'], axis=1)
games['underdog_max_opening_odds'] = games.apply(lambda row: row['l_max_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_max_opening_odds'], axis=1)

games['favored_max_closing_odds'] = games.apply(lambda row: row['w_max_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_max_closing_odds'], axis=1)
games['underdog_max_closing_odds'] = games.apply(lambda row: row['l_max_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_max_closing_odds'], axis=1)

games['favored_min_opening_odds'] = games.apply(lambda row: row['w_min_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_min_opening_odds'], axis=1)
games['underdog_min_opening_odds'] = games.apply(lambda row: row['l_min_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_min_opening_odds'], axis=1)

games['favored_min_closing_odds'] = games.apply(lambda row: row['w_min_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_min_closing_odds'], axis=1)
games['underdog_min_closing_odds'] = games.apply(lambda row: row['l_min_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_min_closing_odds'], axis=1)

games['favored_pinnacle_opening_odds'] = games.apply(lambda row: row['w_pinnacle_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_pinnacle_opening_odds'], axis=1)
games['underdog_pinnacle_opening_odds'] = games.apply(lambda row: row['l_pinnacle_opening_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_pinnacle_opening_odds'], axis=1)

games['favored_pinnacle_closing_odds'] = games.apply(lambda row: row['w_pinnacle_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['l_pinnacle_closing_odds'], axis=1)
games['underdog_pinnacle_closing_odds'] = games.apply(lambda row: row['l_pinnacle_closing_odds'] if row['w_opening_odds'] < row['l_opening_odds'] else row['w_pinnacle_closing_odds'], axis=1)

games.drop(columns = ['winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
        'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age',
        'player1', 'player2',  'mean_opening_odds1', 'mean_opening_odds2', 'mean_closing_odds1', 'mean_closing_odds2', 'max_opening_odds1', 'max_opening_odds2', 'max_closing_odds1', 'max_closing_odds2',
        'min_opening_odds1', 'min_opening_odds2', 'min_closing_odds1', 'min_closing_odds2', 'pinnacle_opening_odds1', 'pinnacle_opening_odds2', 'pinnacle_closing_odds1', 'pinnacle_closing_odds2',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
        'w_set1', 'l_set1', 'w_set2', 'l_set2', 'w_set3', 'l_set3', 'w_set4', 'l_set4', 'w_set5', 'l_set5', 'w_sets', 'l_sets', 'w_games', 'l_games', 'w_bpWon', 'l_bpWon', 'w_svpt_won', 'l_svpt_won', 'w_rtpt_won', 'l_rtpt_won',
        'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points', 'winner_elo', 'loser_elo', 'winner_elo_surface', 'loser_elo_surface', 'winner_rank_group', 'loser_rank_group', 
        'w_mean_opening_odds', 'l_mean_opening_odds', 'w_mean_closing_odds', 'l_mean_closing_odds', 'w_max_opening_odds', 'l_max_opening_odds', 'w_max_closing_odds', 'l_max_closing_odds',
        'w_min_opening_odds', 'l_min_opening_odds', 'w_min_closing_odds', 'l_min_closing_odds', 'w_pinnacle_opening_odds', 'l_pinnacle_opening_odds', 'w_pinnacle_closing_odds', 'l_pinnacle_closing_odds',
        'w_opening_odds', 'l_opening_odds', 'w_closing_odds', 'l_closing_odds'], inplace=True)

# Rank points diff
games['elo_diff'] = games['favored_elo'] - games['underdog_elo']
games['elo_surface_diff'] = games['favored_elo_surface'] - games['underdog_elo_surface']

games['log_elo_diff'] = np.log(games['favored_elo']) - np.log(games['underdog_elo'])
games['log_elo_surface_diff'] = np.log(games['favored_elo_surface']) - np.log(games['underdog_elo_surface'])

games.to_csv('games.csv')

games = pd.read_csv('games.csv')
games['tourney_date'] = pd.to_datetime(games['tourney_date'])
games.set_index('Unnamed: 0', inplace=True)
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
#games = games[games['tourney_level'].isin(['G', 'M', 'O', 'A', 'F'])].reset_index(drop=True)

# Surface weighting
surfaces = ['Hard', 'Grass', 'Clay']
win_pct_by_surface = {}
for surface in surfaces:
    surface_data_favorites = games[games['surface'] == surface].copy()
    surface_data_favorites.rename(columns={'favored':'player', 'underdog':'rival'}, inplace=True)
    surface_data_favorites['win'] = surface_data_favorites['favored_win'] 
    
    surface_data_underdogs = games[games['surface'] == surface].copy()
    surface_data_underdogs.rename(columns={'favored':'rival', 'underdog':'player'}, inplace=True)
    surface_data_underdogs['win'] = np.where(surface_data_underdogs['favored_win'] == 1, 0, 1)
    
    surface_data = pd.concat([surface_data_favorites, surface_data_underdogs], axis=0).sort_values(by = ['tourney_date', 'round'])
    player_counts = surface_data['player'].value_counts()
    
    win_pct_by_surface[surface] = surface_data.groupby('player')['win'].mean()

win_pct_df = pd.DataFrame(win_pct_by_surface).dropna()
surface_correlation_matrix = win_pct_df.corr()
print(surface_correlation_matrix)


# Mean previous stats feature engineering
games_player_rival = games.copy()
games_player_rival.columns = games_player_rival.columns.str.replace('favored', 'player', regex=False)
games_player_rival.columns = games_player_rival.columns.str.replace('underdog', 'rival', regex=False)
games_player_rival['win'] = games_player_rival['player_win'] 
games_player_rival.drop(columns='player_win',inplace=True)

games_rival_player = games.copy()
games_rival_player.columns = games_rival_player.columns.str.replace('favored', 'rival', regex=False)
games_rival_player.columns = games_rival_player.columns.str.replace('underdog', 'player', regex=False)
games_rival_player['win'] = np.where(games_rival_player['rival_win'] == 1, 0, 1)
games_rival_player.drop(columns='rival_win',inplace=True)

full_games = pd.concat([games_player_rival, games_rival_player], ignore_index=True).sort_values(by=['tourney_date', 'tourney_name', 'round', 'game_id']).reset_index(drop=True)
games_to_process = games[(games['season'] >= 2009) & (games['favored_odds'].notna()) & (games['underdog_odds'].notna())].copy()
features = pd.DataFrame()
for index, row in tqdm(games_to_process.iterrows(), total=len(games_to_process)):
    favored_df = full_games[(full_games['player'] == row['favored']) & (full_games['tourney_date'] < row['tourney_date']) & (full_games['comment'] == 'Completed')].copy()
    underdog_df = full_games[(full_games['player'] == row['underdog']) & (full_games['tourney_date'] < row['tourney_date']) & (full_games['comment'] == 'Completed')].copy()
    
    # Time weights
    delta = 1.5
    n = len(favored_df)
    times = (row['tourney_date'] - favored_df['tourney_date']).dt.days / 365.25
    weights = np.minimum(np.exp(-delta*times), 0.8)
    favored_df['time_weight'] = weights
    
    n = len(underdog_df)
    times = (row['tourney_date'] - underdog_df['tourney_date']).dt.days / 365.25
    weights = np.minimum(np.exp(-delta*times), 0.8)
    underdog_df['time_weight'] = weights
    
    # Surface weights
    surface_correlation = surface_correlation_matrix.loc[row['surface']]
    favored_df['surface_weight'] = favored_df['surface'].map(surface_correlation).fillna(0)
    underdog_df['surface_weight'] = underdog_df['surface'].map(surface_correlation).fillna(0)
    
    # get weights based on tourney_se
    #favored_df['series_weight'] = np.where(favored_df['tourney_series'] == 'Grand Slam', 0.275, np.where(favored_df['tourney_series'] == 'Masters 1000', 0.25, np.where(favored_df['tourney_series'] == 'ATP500', 0.225, np.where(favored_df['tourney_series'] == 'ATP250', 0.2, 0.05))))
    #underdog_df['series_weight'] =  np.where(underdog_df['tourney_series'] == 'Grand Slam', 0.275, np.where(underdog_df['tourney_series'] == 'Masters 1000', 0.25, np.where(underdog_df['tourney_series'] == 'ATP500', 0.225, np.where(underdog_df['tourney_series'] == 'ATP250', 0.2, 0.05))))
    
    # Overall weights
    favored_df['weight'] = (favored_df['time_weight']/sum(favored_df['time_weight'])) * (favored_df['surface_weight']/sum(favored_df['surface_weight'])) #* (favored_df['series_weight']/sum(favored_df['series_weight']))
    underdog_df['weight'] = (underdog_df['time_weight']/sum(underdog_df['time_weight'])) * (underdog_df['surface_weight']/sum(underdog_df['surface_weight'])) #* (underdog_df['series_weight']/sum(underdog_df['series_weight']))
    
    # Match uncertainty
    #weights_per_favored_opp = favored_df.groupby('rival').agg({'time_weight':'sum'})
    #weights_per_underdog_opp = underdog_df.groupby('rival').agg({'time_weight':'sum'})
    weights_per_match_favored = favored_df['time_weight'].sum()
    weights_per_match_underdog = underdog_df['time_weight'].sum()
    features.loc[index, 'uncertainty'] = 1/(weights_per_match_favored*weights_per_match_underdog) 
    
    # Fatigue
    filtered_favored_df = favored_df[(favored_df['tourney_date'] >= row['tourney_date']-timedelta(days=21))]
    features.loc[index, 'favored_games_fatigue'] = len(filtered_favored_df)
    features.loc[index, 'favored_minutes_fatigue'] = filtered_favored_df['minutes'].sum()
    
    filtered_underdog_df = underdog_df[(underdog_df['tourney_date'] >= row['tourney_date']-timedelta(days=21))]
    features.loc[index, 'underdog_games_fatigue'] = len(filtered_underdog_df)
    features.loc[index, 'underdog_minutes_fatigue'] = filtered_underdog_df['minutes'].sum()
    
    # Inactivity
    filtered_favored_df = favored_df[(favored_df['tourney_date'] < row['tourney_date'])]
    features.loc[index, 'favored_inactivity'] = (row['tourney_date'] - filtered_favored_df['tourney_date'].iloc[-1]).days / 7 if len(filtered_favored_df) > 0 else np.inf
    
    filtered_underdog_df = underdog_df[(underdog_df['tourney_date'] < row['tourney_date'])]
    features.loc[index, 'underdog_inactivity'] = (row['tourney_date'] - filtered_underdog_df['tourney_date'].iloc[-1]).days / 7 if len(filtered_underdog_df) > 0 else np.inf
    
    # H2H
    filtered_favored_df = favored_df[favored_df['rival'] == row['underdog']]
    features.loc[index, 'favored_win_pct_h2h'] = np.dot(filtered_favored_df['win'], filtered_favored_df['weight']/sum(filtered_favored_df['weight']))
    
    filtered_underdog_df = underdog_df[underdog_df['rival'] == row['favored']]
    features.loc[index, 'underdog_win_pct_h2h'] = np.dot(filtered_underdog_df['win'], filtered_underdog_df['weight']/sum(filtered_underdog_df['weight']))
    
    # Tourney record
    filtered_favored_df = favored_df[(favored_df['tourney_name'] == row['tourney_name']) & (favored_df['tourney_date'] < row['tourney_date'])]
    features.loc[index, 'favored_win_pct_tourney'] = np.dot(filtered_favored_df['win'], filtered_favored_df['weight']/sum(filtered_favored_df['weight']))
    
    filtered_underdog_df = underdog_df[(underdog_df['tourney_name'] == row['tourney_name']) & (underdog_df['tourney_date'] < row['tourney_date'])]
    features.loc[index, 'underdog_win_pct_tourney'] = np.dot(filtered_underdog_df['win'], filtered_underdog_df['weight']/sum(filtered_underdog_df['weight']))
    
    # Distance to max-min-avg elo
    features.loc[index, 'favored_distance_max_elo'] = row['favored_elo'] - favored_df['player_elo'].max()
    features.loc[index, 'favored_distance_min_elo'] = row['favored_elo'] - favored_df['player_elo'].min()
    features.loc[index, 'favored_distance_avg_elo'] = row['favored_elo'] - favored_df['player_elo'].mean()
    
    features.loc[index, 'favored_log_distance_max_elo'] = np.log(row['favored_elo']) - np.log(favored_df['player_elo'].max())
    features.loc[index, 'favored_log_distance_min_elo'] = np.log(row['favored_elo']) - np.log(favored_df['player_elo'].min())
    features.loc[index, 'favored_log_distance_avg_elo'] = np.log(row['favored_elo']) - np.log(favored_df['player_elo'].mean())
    
    features.loc[index, 'underdog_distance_max_elo'] = row['underdog_elo'] - underdog_df['player_elo'].max()
    features.loc[index, 'underdog_distance_min_elo'] = row['underdog_elo'] - underdog_df['player_elo'].min()
    features.loc[index, 'underdog_distance_avg_elo'] = row['underdog_elo'] - underdog_df['player_elo'].mean()
    
    features.loc[index, 'underdog_log_distance_max_elo'] = np.log(row['underdog_elo']) - np.log(underdog_df['player_elo'].max())
    features.loc[index, 'underdog_log_distance_min_elo'] = np.log(row['underdog_elo']) - np.log(underdog_df['player_elo'].min())
    features.loc[index, 'underdog_log_distance_avg_elo'] = np.log(row['underdog_elo']) - np.log(underdog_df['player_elo'].mean())
    
    # Favored features
    features.loc[index, 'favored_win_pct'] = weighted_average(favored_df['win'], favored_df['weight'])
    features.loc[index, 'favored_avg_tpt_won_pct'] = weighted_average((favored_df['player_svpt_won']+favored_df['player_rtpt_won'])/(favored_df['player_svpt']+favored_df['rival_svpt']), favored_df['weight'])
    features.loc[index, 'favored_avg_svpt_won_pct'] = weighted_average((favored_df['player_svpt_won'])/(favored_df['player_svpt']), favored_df['weight'])
    features.loc[index, 'favored_avg_1st_in_pct'] = weighted_average((favored_df['player_1st_in'])/(favored_df['player_svpt']), favored_df['weight'])
    features.loc[index, 'favored_avg_1st_won_pct'] = weighted_average((favored_df['player_1st_won'])/(favored_df['player_1st_in']), favored_df['weight'])
    features.loc[index, 'favored_avg_2nd_won_pct'] = weighted_average((favored_df['player_2nd_won'])/(favored_df['player_svpt']-favored_df['player_1st_in']), favored_df['weight'])
    features.loc[index, 'favored_avg_ace'] = weighted_average((favored_df['player_ace'])/(favored_df['player_svpt']), favored_df['weight'])
    features.loc[index, 'favored_avg_df'] = weighted_average((favored_df['player_df'])/(favored_df['player_svpt']), favored_df['weight'])
    features.loc[index, 'favored_avg_bp_saved_pct'] = weighted_average(((favored_df['player_bp_saved'])/(favored_df['player_bp_faced'])).fillna(1), favored_df['weight'])
    
    features.loc[index, 'favored_avg_rtpt_won_pct'] = weighted_average((favored_df['player_rtpt_won'])/(favored_df['rival_svpt']), favored_df['weight'])
    features.loc[index, 'favored_avg_1st_return_won_pct'] = weighted_average((favored_df['rival_1st_in']-favored_df['rival_1st_won'])/(favored_df['rival_1st_in']), favored_df['weight'])
    features.loc[index, 'favored_avg_2nd_return_won_pct'] = weighted_average((favored_df['rival_svpt']-favored_df['rival_1st_in']-favored_df['rival_2nd_won'])/(favored_df['rival_svpt']-favored_df['rival_1st_in']), favored_df['weight'])
    features.loc[index, 'favored_avg_bp_won_pct'] = weighted_average(((favored_df['player_bp_won'])/(favored_df['rival_bp_faced'])).fillna(0), favored_df['weight'])
    features.loc[index, 'favored_avg_dominance_ratio'] = weighted_average(favored_df['player_dominance_ratio'], favored_df['weight'])
    
    
    # Underdog features
    features.loc[index, 'underdog_win_pct'] = weighted_average(underdog_df['win'], underdog_df['weight'])
    features.loc[index, 'underdog_avg_tpt_won_pct'] = weighted_average((underdog_df['player_svpt_won']+underdog_df['player_rtpt_won'])/(underdog_df['player_svpt']+underdog_df['rival_svpt']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_svpt_won_pct'] = weighted_average((underdog_df['player_svpt_won'])/(underdog_df['player_svpt']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_1st_in_pct'] = weighted_average((underdog_df['player_1st_in'])/(underdog_df['player_svpt']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_1st_won_pct'] = weighted_average((underdog_df['player_1st_won'])/(underdog_df['player_1st_in']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_2nd_won_pct'] = weighted_average((underdog_df['player_2nd_won'])/(underdog_df['player_svpt']-underdog_df['player_1st_in']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_ace'] = weighted_average((underdog_df['player_ace'])/(underdog_df['player_svpt']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_df'] = weighted_average((underdog_df['player_df'])/(underdog_df['player_svpt']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_bp_saved_pct'] = weighted_average(((underdog_df['player_bp_saved'])/(underdog_df['player_bp_faced'])).fillna(1), underdog_df['weight'])
    
    features.loc[index, 'underdog_avg_rtpt_won_pct'] = weighted_average((underdog_df['player_rtpt_won'])/(underdog_df['rival_svpt']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_1st_return_won_pct'] = weighted_average((underdog_df['rival_1st_in']-underdog_df['rival_1st_won'])/(underdog_df['rival_1st_in']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_2nd_return_won_pct'] = weighted_average((underdog_df['rival_svpt']-underdog_df['rival_1st_in']-underdog_df['rival_2nd_won'])/(underdog_df['rival_svpt']-underdog_df['rival_1st_in']), underdog_df['weight'])
    features.loc[index, 'underdog_avg_bp_won_pct'] = weighted_average(((underdog_df['player_bp_won'])/(underdog_df['rival_bp_faced'])).fillna(0), underdog_df['weight'])
    features.loc[index, 'underdog_avg_dominance_ratio'] = weighted_average(underdog_df['player_dominance_ratio'], underdog_df['weight'])


# Serve advantage
features['favored_serve_adv'] = (features['favored_avg_svpt_won_pct'] - features['underdog_avg_rtpt_won_pct'])
features['underdog_serve_adv'] = (features['underdog_avg_svpt_won_pct'] - features['favored_avg_rtpt_won_pct'])

# Features differences
features['games_fatigue_diff'] = features['favored_games_fatigue'] - features['underdog_games_fatigue']
features['minutes_fatigue_diff'] = features['favored_minutes_fatigue'] - features['underdog_minutes_fatigue']
features['inactivity_diff'] = features['favored_inactivity'] - features['underdog_inactivity']
features['h2h_diff'] = features['favored_win_pct_h2h'] - features['underdog_win_pct_h2h']
features['win_pct_tourney_diff'] = features['favored_win_pct_tourney'] - features['underdog_win_pct_tourney']
features['distance_max_elo_diff'] = features['favored_distance_max_elo'] - features['underdog_distance_max_elo'] 
features['distance_min_elo_diff'] = features['favored_distance_min_elo'] - features['underdog_distance_min_elo'] 
features['distance_avg_elo_diff'] = features['favored_distance_avg_elo'] - features['underdog_distance_avg_elo'] 
features['log_distance_max_elo_diff'] = features['favored_log_distance_max_elo'] - features['underdog_log_distance_max_elo'] 
features['log_distance_min_elo_diff'] = features['favored_log_distance_min_elo'] - features['underdog_log_distance_min_elo'] 
features['log_distance_avg_elo_diff'] = features['favored_log_distance_avg_elo'] - features['underdog_log_distance_avg_elo'] 
features['win_pct_diff'] = features['favored_win_pct'] - features['underdog_win_pct']
features['tpt_won_pct_diff'] = features['favored_avg_tpt_won_pct'] - features['underdog_avg_tpt_won_pct']
features['svpt_won_pct_diff'] = features['favored_avg_svpt_won_pct'] - features['underdog_avg_svpt_won_pct']
features['1st_in_pct_diff'] = features['favored_avg_1st_in_pct'] - features['underdog_avg_1st_in_pct']
features['1st_won_pct_diff'] = features['favored_avg_1st_won_pct'] - features['underdog_avg_1st_won_pct']
features['2nd_won_pct_diff'] = features['favored_avg_2nd_won_pct'] - features['underdog_avg_2nd_won_pct']
features['ace_diff'] = features['favored_avg_ace'] - features['underdog_avg_ace']
features['df_diff'] = features['favored_avg_df'] - features['underdog_avg_df']
features['bp_saved_pct_diff'] = features['favored_avg_bp_saved_pct'] - features['underdog_avg_bp_saved_pct']
features['rtpt_won_pct_diff'] = features['favored_avg_rtpt_won_pct'] - features['underdog_avg_rtpt_won_pct']
features['1st_return_won_pct_diff'] = features['favored_avg_1st_return_won_pct'] - features['underdog_avg_1st_return_won_pct']
features['2nd_return_won_pct_diff'] = features['favored_avg_2nd_return_won_pct'] - features['underdog_avg_2nd_return_won_pct']
features['bp_won_pct_diff'] = features['favored_avg_bp_won_pct'] - features['underdog_avg_bp_won_pct']
features['dominance_ratio_diff'] = features['favored_avg_dominance_ratio']-features['underdog_avg_dominance_ratio']
features['serve_adv_diff'] = features['favored_serve_adv'] - features['underdog_serve_adv']

# Inactivity como variable binaria
features['favored_inactive'] = np.where((features['favored_inactivity'] > 60) | (features['favored_inactivity'].isna()), 1, 0)
features['underdog_inactive'] = np.where((features['underdog_inactivity'] > 60) | (features['underdog_inactivity'].isna()), 1, 0)
features['inactive_match'] = np.where((features['favored_inactive'] == 1) & (features['underdog_inactive'] == 1), 3,
                            np.where((features['favored_inactive'] == 1) & (features['underdog_inactive'] == 0), 2,
                                    np.where((features['favored_inactive'] == 0) & (features['underdog_inactive'] == 1), 1, 0))).astype(int)

features.to_csv('features.csv')
