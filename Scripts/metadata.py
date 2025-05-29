import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from Scripts.functions import *
from bs4 import BeautifulSoup
from io import StringIO
pd.set_option('display.max_rows',600)

"""
# Odds data
seasons = list(range(2001, 2026))
data = pd.DataFrame()
for season in seasons:
    try:
        df = pd.read_excel(f'Odds/{season}.xlsx')
    except:
        df = pd.read_excel(f'Odds/{season}.xls')
    df.insert(0, 'Season', season)
    data = pd.concat([data, df], axis = 0).reset_index(drop=True)
data.to_csv('Odds/odds.csv',index=False)

data['Series'].replace({'International':'ATP250', 'International Gold':'ATP500', 'Masters':'Masters 1000', 'Masters Cup':'Masters 1000'},inplace=True)

tournaments = pd.read_csv('tournaments.csv')
tournaments_by_season = data[['Tournament', 'Series', 'Season']].drop_duplicates(subset=['Tournament', 'Series', 'Season'])
tournaments_by_season = janitor.clean_names(tournaments_by_season).rename(columns={'tournament':'tournament_odds'})

tournaments_by_season = pd.merge(tournaments_by_season, tournaments[['tournament_odds', 'tournament_stats']], how='left', on=['tournament_odds'])

tournaments_by_season = tournaments_by_season[['tournament_odds', 'tournament_stats', 'series', 'season']].sort_values(by = ['series', 'tournament_stats', 'season']).reset_index(drop=True)
tournaments_by_season.to_csv('tournaments_by_season.csv', index=False)
"""

#<------------------------------------------------------------------------------------------------------------- 
# Add challenger tournaments
tournaments_by_season = pd.read_csv('tournaments_by_season.csv')
tournaments_by_season = tournaments_by_season.sort_values(by = ['series', 'tournament_stats', 'season']).reset_index(drop=True)

seasons = list(range(1978, 2025))
games = pd.DataFrame()
for season in seasons:
    df = pd.concat([pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv'),pd.read_csv(f'tennis_atp-master/atp_matches_qual_chall_{season}.csv')],axis = 0).reset_index(drop=True)
    df.insert(0, 'season', season)
    df.loc[df['tourney_name'].str.contains(' Olympics', na=False), 'tourney_level'] = 'O'    
    
    games = pd.concat([games, df], axis = 0).reset_index(drop=True)

games['tourney_id'] = games['tourney_id'].apply(lambda x: x.split('-')[1])
games['best_of'] = games['best_of'].astype(int)

tournaments_stats_by_season = games[['tourney_date', 'tourney_name', 'tourney_level', 'season', 'tourney_id', 'draw_size', 'best_of']].drop_duplicates(subset = ['tourney_name', 'season']).reset_index(drop=True)
for index, row in tqdm(tournaments_stats_by_season.iterrows(), total = len(tournaments_stats_by_season)):
    if row['tourney_name'] in ['NextGen Finals', 'Next Gen Finals']:
        tournaments_stats_by_season.at[index, 'tourney_level'] = 'F'
tournaments_stats_by_season = tournaments_stats_by_season[(tournaments_stats_by_season['season'] >= 2001) & (~tournaments_stats_by_season['tourney_level'].isin(['D']))].reset_index(drop=True)
tournaments_stats_by_season.rename(columns={'tourney_name':'tournament_stats'},inplace=True)
tournaments_stats_by_season['series'] = tournaments_stats_by_season['tournament_stats'].apply(lambda x: 'Challenger' if 'CH' in x else ('Masters 1000' if 'Masters' in x else None))
for index, row in tqdm(tournaments_stats_by_season.iterrows(), total = len(tournaments_stats_by_season)):
    if row['tourney_level'] == 'O':
        tournaments_stats_by_season.at[index, 'series'] = 'Masters 1000'
    if row['tournament_stats'] in ['NextGen Finals', 'Next Gen Finals']:
        tournaments_stats_by_season.at[index, 'series'] = 'ATP250'
    if row['tournament_stats'] in ['Doha Aus Open Qualies']:
        tournaments_stats_by_season.at[index, 'series'] = 'Grand Slam'
    if row['tournament_stats'] in ['Challenger Tour Finals']:
        tournaments_stats_by_season.at[index, 'series'] = 'Challenger'


full_tournaments_by_season = pd.concat([tournaments_by_season, tournaments_stats_by_season],axis = 0).drop_duplicates(subset=['tournament_stats', 'season'])
full_tournaments_by_season['tourney_date'] = pd.to_datetime(full_tournaments_by_season['tourney_date'].astype('Int64'), format='%Y%m%d')
full_tournaments_by_season.sort_values(by = ['series', 'tournament_stats', 'season']).to_csv('tournaments_by_season.csv', index=False)

#<-------------------------------------------------------------------------------------------------------------
# 2025 Challengers
challengers_2025 = pd.read_csv('tennis_atp-master/atp_matches_2025.csv')
challengers_2025 = challengers_2025[challengers_2025['tourney_name'].str.contains(' CH', na=False)].reset_index(drop=True)
challengers_2025['season'] = 2025
challengers_2025['series'] = 'Challenger'
challengers_2025['tourney_level'] = 'C'
challengers_2025['best_of'] = '3'
challengers_2025['tourney_date'] = pd.to_datetime(challengers_2025['tourney_date'].astype('Int64'), format='%Y%m%d')
challengers_2025 = challengers_2025[['tourney_date', 'tourney_name', 'series', 'tourney_level', 'season', 'tourney_id', 'draw_size', 'best_of']].drop_duplicates(subset = ['tourney_name', 'season']).reset_index(drop=True)
challengers_2025.rename(columns={'tourney_name':'tournament_stats'},inplace=True)

pd.concat([pd.read_csv('tournaments_by_season.csv'), challengers_2025], axis = 0).sort_values(by = ['series', 'tournament_stats', 'season']).to_csv('tournaments_by_season.csv', index=False)

#<------------------------------------------------------------------------------------------------------------- 
# Add extra tournament info
tournaments_by_season = pd.read_csv('tournaments_by_season.csv')
tournaments_by_season = tournaments_by_season.sort_values(by = ['series', 'tournament_stats', 'season']).reset_index(drop=True)

seasons = list(range(1978, 2025))
games = pd.DataFrame()
for season in seasons:
    df = pd.concat([pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv'),pd.read_csv(f'tennis_atp-master/atp_matches_qual_chall_{season}.csv')],axis = 0).reset_index(drop=True)
    df.insert(0, 'season', season)
    df.loc[df['tourney_name'].str.contains(' Olympics', na=False), 'tourney_level'] = 'O'
    
    games = pd.concat([games, df], axis = 0).reset_index(drop=True)

games['tourney_id'] = games['tourney_id'].apply(lambda x: x.split('-')[1])
games['best_of'] = games['best_of'].astype(int)

tournaments_stats_by_season = games[['tourney_date', 'tourney_name', 'tourney_level', 'season', 'tourney_id', 'draw_size', 'best_of']].drop_duplicates(subset = ['tourney_name', 'season'])
tournaments_stats_by_season.rename(columns={'tourney_name':'tournament_stats'},inplace=True)
pd.merge(tournaments_by_season, tournaments_stats_by_season, on = ['tournament_stats', 'season'], how = 'left').sort_values(by = ['tournament_stats', 'season']).to_csv('tournaments_by_season2.csv', index=False)


#<------------------------------------------------------------------------------------------------------------- 
# Players database
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
games['tourney_date'] = pd.to_datetime(games['tourney_date'].astype('Int64'), format='%Y%m%d')

players = games.sort_values(by='tourney_date').drop_duplicates(subset=['loser_name'], keep='last')[['loser_name', 'loser_id', 'loser_hand', 'loser_ht', 'loser_ioc', 'tourney_date']]
players.columns = ['player', 'id', 'hand', 'height', 'ioc', 'last_seen']
players['id'] = players['id'].astype('Int64')

# Current players 
players = pd.read_csv('players.csv')

response = requests.get('https://tennisabstract.com/reports/atpRankings.html')
soup = BeautifulSoup(response.content, 'html.parser')

tables = soup.find_all('table', id='reportable')
players_new = pd.read_html(StringIO(str(tables[0])))[0]
players_new.rename(columns = {'Player':'player', 'Birthdate':'birthdate'}, inplace=True)
players_new['player'] = players_new['player'].str.strip().str.replace('\xa0', ' ')

players = pd.merge(players, players_new[['player', 'birthdate']], on = 'player', how = 'left')

players.to_csv('players.csv', index=False)


