import pandas as pd
import numpy as np
from tqdm import tqdm
import janitor
from Scripts.functions import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows',200)

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
tournaments_by_season[['tournament_odds', 'tournament_stats', 'series', 'season']].sort_values(by = ['series', 'tournament_stats', 'season']).to_csv('tournaments_by_season.csv', index=False)

# Stats data
seasons = list(range(1978, 2025))
games = pd.DataFrame()
for season in seasons:
    df = pd.concat([pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv'),pd.read_csv(f'tennis_atp-master/atp_matches_qual_chall_{season}.csv')],axis = 0).reset_index(drop=True)
    df.insert(0, 'season', season)
    df.loc[df['tourney_name'].str.contains(' Olympics', na=False), 'tourney_level'] = 'O'
    
    games = pd.concat([games, df], axis = 0).reset_index(drop=True)

different_names_dict = {'Edouard Roger-Vasselin':'Edouard Roger Vasselin'}
games['winner_name'].replace(different_names_dict, inplace=True)
games['loser_name'].replace(different_names_dict, inplace=True)

#games.drop(columns = ['winner', 'loser', 'game_id'],inplace=True) 
games.insert(games.columns.get_loc('winner_name'), 'winner', games['winner_name'].apply(lambda x: ' '.join(x.split(' ')[1:])) + ' ' + games['winner_name'].apply(lambda x: x.split(' ')[0][0]) + '.')
games.insert(games.columns.get_loc('loser_name'), 'loser', games['loser_name'].apply(lambda x: ' '.join(x.split(' ')[1:])) + ' ' + games['loser_name'].apply(lambda x: x.split(' ')[0][0]) + '.')
games.insert(games.columns.get_loc('winner'), 'game_id', games.groupby(['winner', 'loser', 'tourney_name', 'tourney_date', 'match_num', 'season']).ngroup() + 1)
for index, row in games.iterrows():
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

tournaments = pd.read_csv('tournaments_by_season.csv')
tournaments.drop_duplicates(subset=['tournament_stats', 'series', 'season'], inplace=True)

#games.drop(columns=['tourney_full_name', 'tourney_series'], inplace=True)
games = pd.merge(games, tournaments.rename(columns={'tournament_stats':'tourney_name', 'tournament_odds':'tourney_full_name', 'series':'tourney_series'}), on=['tourney_name', 'season'], how='left')
tourney_full_name_col = games.pop('tourney_full_name')
games.insert(games.columns.get_loc('surface'), 'tourney_full_name', tourney_full_name_col)
tourney_series_col = games.pop('tourney_series')
games.insert(games.columns.get_loc('surface'), 'tourney_series', tourney_series_col)

for index, row in tqdm(games.iterrows(), total = len(games)):
    if row['tourney_level'] == 'O':
        games.at[index, 'tourney_series'] = 'Masters 1000'
    elif row['tourney_level'] == 'C':
        games.at[index, 'tourney_series'] = 'Challenger'

games['tourney_date'] = pd.to_datetime(games['tourney_date'], format = '%Y%m%d')
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games = games[(~games['tourney_name'].isin(['Laver Cup']))].sort_values(by=['tourney_date', 'tourney_name', 'round']).reset_index(drop=True)

games['surface'].replace({'Carpet':'Hard'},inplace=True)

def elo_rankings(games, surface=None):
    if surface == None:
        players=list(pd.Series(list(games['winner_name'])+list(games['loser_name'])).value_counts().index)
        elo=pd.Series(np.ones(len(players))*1500,index=players)
        games_played = pd.Series(0,index=players)
        ranking_elo=[(1500,1500)]
        for i in tqdm(range(1,len(games)), desc='Computing Elo'):
            w=games.iloc[i-1,:]['winner_name']
            l=games.iloc[i-1,:]['loser_name']
            elow=elo[w]
            elol=elo[l]
            m_win = games_played[w]
            m_los = games_played[l]
            pwin=1 / (1 + 10 ** ((elol - elow) / 400))
            K_win=20000/((5+m_win)**4)
            K_los=20000/((5+m_los)**4)
            new_elow=elow+K_win*(1-pwin)
            new_elol=elol-K_los*(1-pwin)
            games_played[w] += 1
            games_played[l] += 1
            elo[w]=new_elow
            elo[l]=new_elol
            ranking_elo.append((elo[games.iloc[i,:]['winner_name']],elo[games.iloc[i,:]['loser_name']]))
        ranking_elo=pd.DataFrame(ranking_elo,columns=["winner_elo","loser_elo"])
    else:
        games = games[games['surface'] == surface]
        players=list(pd.Series(list(games['winner_name'])+list(games['loser_name'])).value_counts().index)
        elo=pd.Series(np.ones(len(players))*1500,index=players)
        ranking_elo=[(1500,1500)]
        for i in tqdm(range(1,len(games)), desc=f'Computing {surface} Elo'):
            w=games.iloc[i-1,:]['winner_name']
            l=games.iloc[i-1,:]['loser_name']
            elow=elo[w]
            elol=elo[l]
            pwin=1 / (1 + 10 ** ((elol - elow) / 400))    
            K_win=32
            K_los=32
            new_elow=elow+K_win*(1-pwin)
            new_elol=elol-K_los*(1-pwin)
            elo[w]=new_elow
            elo[l]=new_elol
            ranking_elo.append((elo[games.iloc[i,:]['winner_name']],elo[games.iloc[i,:]['loser_name']])) 
        ranking_elo=pd.DataFrame(ranking_elo,columns=["winner_elo_surface","loser_elo_surface"], index=games.index)
    return ranking_elo

ranking_elo = elo_rankings(games)
ranking_elo_surface = pd.concat([elo_rankings(games, surface='Hard'), elo_rankings(games, surface='Grass'), elo_rankings(games, surface='Clay')], axis = 0) # Grass, Clay, Hard
games = pd.concat([games, ranking_elo, ranking_elo_surface],axis=1)

games = games[(~games['tourney_level'].isin(['D'])) & (games['season'] >= 2001)].reset_index(drop=True)

# Match features # 
#print(games[games['score'].str.contains('[a-zA-Z]', na=False)]['score'].str.extract(r'([a-zA-Z]+)').drop_duplicates())
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

games.to_csv('stats.csv', index=False)


games = pd.read_csv('stats.csv')
games['tourney_date'] = pd.to_datetime(games['tourney_date'], format = 'mixed')

games["winner_rank"]=games["winner_rank"].replace(np.nan,2500).astype(int)
games["loser_rank"]=games["loser_rank"].replace(np.nan,2500).astype(int)
games["winner_rank_points"]=games["winner_rank_points"].replace(np.nan,0).astype(int)
games["loser_rank_points"]=games["loser_rank_points"].replace(np.nan,0).astype(int)

games['winner_rank_group'] = games['winner_rank'].apply(rank_group)
games['loser_rank_group'] = games['loser_rank'].apply(rank_group)

# Imput missing games
games[(games['w_SvGms'] == 0) & (games['comment'] == 'Completed')][['winner_name', 'loser_name', 'tourney_name', 'season', 'score', 'comment', 'w_SvGms', 'l_SvGms']]
for index, row in tqdm(games.iterrows(), total = len(games)):
    if np.ceil((row['w_games'] + row['l_games']) / 2) != row['w_SvGms'] or np.floor((row['w_games'] + row['l_games']) / 2) != row['w_SvGms']:
        games.at[index, 'w_SvGms'] = np.ceil((row['w_games'] + row['l_games']) / 2)
    if np.floor((row['w_games'] + row['l_games']) / 2) != row['l_SvGms'] or np.ceil((row['w_games'] + row['l_games']) / 2) != row['l_SvGms']:
        games.at[index, 'l_SvGms'] = np.floor((row['w_games'] + row['l_games']) / 2)

# Winner-Loser to Favored-Underdog
games['favored_win'] = games.apply(lambda row: 1 if row['winner_rank'] < row['loser_rank'] else 0, axis=1)

games['favored'] = games.apply(lambda row: row['winner_name'] if row['winner_rank'] < row['loser_rank'] else row['loser_name'], axis=1)
games['underdog'] = games.apply(lambda row: row['loser_name'] if row['winner_rank'] < row['loser_rank'] else row['winner_name'], axis=1)

games['favored_id'] = games.apply(lambda row: row['winner_id'] if row['winner_rank'] < row['loser_rank'] else row['loser_id'], axis=1)
games['underdog_id'] = games.apply(lambda row: row['loser_id'] if row['winner_rank'] < row['loser_rank'] else row['winner_id'], axis=1)

games['favored_entry'] = games.apply(lambda row: row['winner_entry'] if row['winner_rank'] < row['loser_rank'] else row['loser_entry'], axis=1)
games['underdog_entry'] = games.apply(lambda row: row['loser_entry'] if row['winner_rank'] < row['loser_rank'] else row['winner_entry'], axis=1)

games['favored_rank'] = games.apply(lambda row: row['winner_rank'] if row['winner_rank'] < row['loser_rank'] else row['loser_rank'], axis=1)
games['underdog_rank'] = games.apply(lambda row: row['loser_rank'] if row['winner_rank'] < row['loser_rank'] else row['winner_rank'], axis=1)

games['favored_rank_group'] = games.apply(lambda row: row['winner_rank_group'] if row['winner_rank'] < row['loser_rank'] else row['loser_rank_group'], axis=1)
games['underdog_rank_group'] = games.apply(lambda row: row['loser_rank_group'] if row['winner_rank'] < row['loser_rank'] else row['winner_rank_group'], axis=1)

games['favored_rank_pts'] = games.apply(lambda row: row['winner_rank_points'] if row['winner_rank'] < row['loser_rank'] else row['loser_rank_points'], axis=1)
games['underdog_rank_pts'] = games.apply(lambda row: row['loser_rank_points'] if row['winner_rank'] < row['loser_rank'] else row['winner_rank_points'], axis=1)

games['favored_elo'] = games.apply(lambda row: row['winner_elo'] if row['winner_rank'] < row['loser_rank'] else row['loser_elo'], axis=1)
games['underdog_elo'] = games.apply(lambda row: row['loser_elo'] if row['winner_rank'] < row['loser_rank'] else row['winner_elo'], axis=1)

games['favored_elo_surface'] = games.apply(lambda row: row['winner_elo_surface'] if row['winner_rank'] < row['loser_rank'] else row['loser_elo_surface'], axis=1)
games['underdog_elo_surface'] = games.apply(lambda row: row['loser_elo_surface'] if row['winner_rank'] < row['loser_rank'] else row['winner_elo_surface'], axis=1)

games['favored_sets'] = games.apply(lambda row: row['w_sets'] if row['winner_rank'] < row['loser_rank'] else row['l_sets'], axis=1)
games['underdog_sets'] = games.apply(lambda row: row['l_sets'] if row['winner_rank'] < row['loser_rank'] else row['w_sets'], axis=1)

games['favored_games'] = games.apply(lambda row: row['w_games'] if row['winner_rank'] < row['loser_rank'] else row['l_games'], axis=1)
games['underdog_games'] = games.apply(lambda row: row['l_games'] if row['winner_rank'] < row['loser_rank'] else row['w_games'], axis=1)

games['favored_ace'] = games.apply(lambda row: row['w_ace'] if row['winner_rank'] < row['loser_rank'] else row['l_ace'], axis=1)
games['underdog_ace'] = games.apply(lambda row: row['l_ace'] if row['winner_rank'] < row['loser_rank'] else row['w_ace'], axis=1)

games['favored_df'] = games.apply(lambda row: row['w_df'] if row['winner_rank'] < row['loser_rank'] else row['l_df'], axis=1)
games['underdog_df'] = games.apply(lambda row: row['l_df'] if row['winner_rank'] < row['loser_rank'] else row['w_df'], axis=1)

games['favored_svpt'] = games.apply(lambda row: row['w_svpt'] if row['winner_rank'] < row['loser_rank'] else row['l_svpt'], axis=1)
games['underdog_svpt'] = games.apply(lambda row: row['l_svpt'] if row['winner_rank'] < row['loser_rank'] else row['w_svpt'], axis=1)

games['favored_1st_in'] = games.apply(lambda row: row['w_1stIn'] if row['winner_rank'] < row['loser_rank'] else row['l_1stIn'], axis=1)
games['underdog_1st_in'] = games.apply(lambda row: row['l_1stIn'] if row['winner_rank'] < row['loser_rank'] else row['w_1stIn'], axis=1)

games['favored_1st_won'] = games.apply(lambda row: row['w_1stWon'] if row['winner_rank'] < row['loser_rank'] else row['l_1stWon'], axis=1)
games['underdog_1st_won'] = games.apply(lambda row: row['l_1stWon'] if row['winner_rank'] < row['loser_rank'] else row['w_1stWon'], axis=1)

games['favored_2nd_won'] = games.apply(lambda row: row['w_2ndWon'] if row['winner_rank'] < row['loser_rank'] else row['l_2ndWon'], axis=1)
games['underdog_2nd_won'] = games.apply(lambda row: row['l_2ndWon'] if row['winner_rank'] < row['loser_rank'] else row['w_2ndWon'], axis=1)

games['favored_serve_games'] = games.apply(lambda row: row['w_SvGms'] if row['winner_rank'] < row['loser_rank'] else row['l_SvGms'], axis=1)
games['underdog_serve_games'] = games.apply(lambda row: row['l_SvGms'] if row['winner_rank'] < row['loser_rank'] else row['w_SvGms'], axis=1)

games['favored_bp_saved'] = games.apply(lambda row: row['w_bpSaved'] if row['winner_rank'] < row['loser_rank'] else row['l_bpSaved'], axis=1)
games['underdog_bp_saved'] = games.apply(lambda row: row['l_bpSaved'] if row['winner_rank'] < row['loser_rank'] else row['w_bpSaved'], axis=1)

games['favored_bp_faced'] = games.apply(lambda row: row['w_bpFaced'] if row['winner_rank'] < row['loser_rank'] else row['l_bpFaced'], axis=1)
games['underdog_bp_faced'] = games.apply(lambda row: row['l_bpFaced'] if row['winner_rank'] < row['loser_rank'] else row['w_bpFaced'], axis=1)

games['favored_bp_won'] = games.apply(lambda row: row['w_bpWon'] if row['winner_rank'] < row['loser_rank'] else row['l_bpWon'], axis=1)
games['underdog_bp_won'] = games.apply(lambda row: row['l_bpWon'] if row['winner_rank'] < row['loser_rank'] else row['w_bpWon'], axis=1)

# Service Rating #
# Favored
favored_stats = games[['favored', 'underdog', 'tourney_name', 'season', 'comment', 'game_id']].copy()
favored_stats['player'] = favored_stats['favored']
favored_stats['player_type'] = 'favored'
favored_stats['ace_pct'] = games['favored_ace']/games['favored_svpt']
favored_stats['1st_in_pct'] = games['favored_1st_in']/games['favored_svpt']
favored_stats['1st_won_pct'] = games['favored_1st_won']/games['favored_1st_in']
favored_stats['2nd_won_pct'] = games['favored_2nd_won']/(games['favored_svpt']-games['favored_1st_in'])
favored_stats['serve_pts_won_pct'] = (games['favored_1st_won']+games['favored_2nd_won'])/games['favored_svpt']
favored_stats['serve_games_won_pct'] = (games['favored_games']-games['favored_bp_won'])/games['favored_serve_games']
favored_stats['bp_saved_pct'] = (games['favored_bp_saved']/games['favored_bp_faced']).fillna(1)

# Underdog 
underdog_stats = games[['favored', 'underdog', 'tourney_name', 'season', 'comment', 'game_id']].copy()
underdog_stats['player'] = underdog_stats['underdog']
underdog_stats['player_type'] = 'underdog'
underdog_stats['ace_pct'] = games['underdog_ace']/games['underdog_svpt']
underdog_stats['1st_in_pct'] = games['underdog_1st_in']/games['underdog_svpt']
underdog_stats['1st_won_pct'] = games['underdog_1st_won']/games['underdog_1st_in']
underdog_stats['2nd_won_pct'] = games['underdog_2nd_won']/(games['underdog_svpt']-games['underdog_1st_in'])
underdog_stats['serve_pts_won_pct'] = (games['underdog_1st_won']+games['underdog_2nd_won'])/games['underdog_svpt']
underdog_stats['serve_games_won_pct'] = (games['underdog_games']-games['underdog_bp_won'])/games['underdog_serve_games']
underdog_stats['bp_saved_pct'] = (games['underdog_bp_saved']/games['underdog_bp_faced']).fillna(1)

service_rating_long = pd.concat([favored_stats, underdog_stats], axis=0).reset_index(drop=True)

features = ['ace_pct', '1st_in_pct', '1st_won_pct', '2nd_won_pct', 'serve_pts_won_pct', 'serve_games_won_pct', 'bp_saved_pct']
service_rating_long = service_rating_long.replace([np.inf, -np.inf], np.nan).dropna(subset=features)[(service_rating_long['comment'] == 'Completed') & (service_rating_long['serve_games_won_pct'] <= 1)]

pca = PCA()
service_rating_long['service_rating'] = pd.Series(pca.fit_transform(service_rating_long[features])[:, 0], index = service_rating_long.index)

plt.figure(figsize=(10, 6))
sns.histplot(service_rating_long['service_rating'].dropna(), kde=True, bins=30, color='blue')
plt.title('Distribution of Service Ratings', fontsize=16)
plt.xlabel('Service Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()

service_rating_long[['player', 'ace_pct', '1st_in_pct', '1st_won_pct', '2nd_won_pct', 'serve_pts_won_pct', 'serve_games_won_pct', 'bp_saved_pct', 'service_rating']].sort_values(by = ['service_rating'], ascending=True).head(10)

service_rating_wide = service_rating_long.pivot(index='game_id', columns='player_type', values='service_rating').reset_index() 
service_rating_wide.columns = ['game_id', 'favored_service_rating', 'underdog_service_rating']

games = pd.merge(games, service_rating_wide, on='game_id', how='left')


# Return Rating #
# Favored
favored_stats = games[['favored', 'underdog', 'tourney_name', 'season', 'comment', 'game_id']].copy()
favored_stats['player'] = favored_stats['favored']
favored_stats['player_type'] = 'favored'
favored_stats['1st_return_won_pct'] = (games['underdog_1st_in']-games['underdog_1st_won'])/games['underdog_1st_in']
favored_stats['2nd_return_won_pct'] = (((games['underdog_svpt']-games['underdog_1st_in'])-games['underdog_2nd_won']) - games['underdog_df'])/(games['underdog_svpt']-games['underdog_1st_in'])
favored_stats['return_pts_won_pct'] = ((games['underdog_svpt']-games['underdog_1st_won']-games['underdog_2nd_won'])-games['underdog_df'])/games['underdog_svpt']
favored_stats['return_games_won_pct'] = games['favored_bp_won']/games['underdog_serve_games']
favored_stats['bp_won_pct'] = (games['favored_bp_won']/games['underdog_bp_faced']).fillna(0)

# Underdog 
underdog_stats = games[['favored', 'underdog', 'tourney_name', 'season', 'comment', 'game_id']].copy()
underdog_stats['player'] = underdog_stats['underdog']
underdog_stats['player_type'] = 'underdog'
underdog_stats['1st_return_won_pct'] = (games['favored_1st_in']-games['favored_1st_won'])/games['favored_1st_in']
underdog_stats['2nd_return_won_pct'] = (((games['favored_svpt']-games['favored_1st_in'])-games['favored_2nd_won']) - games['favored_df'])/(games['favored_svpt']-games['favored_1st_in'])
underdog_stats['return_pts_won_pct'] = ((games['favored_svpt']-games['favored_1st_won']-games['favored_2nd_won'])-games['favored_df'])/games['favored_svpt']
underdog_stats['return_games_won_pct'] = games['underdog_bp_won']/games['favored_serve_games']
underdog_stats['bp_won_pct'] = (games['underdog_bp_won']/games['favored_bp_faced']).fillna(0)

return_rating_long = pd.concat([favored_stats, underdog_stats], axis=0).reset_index(drop=True)

features = ['1st_return_won_pct', '2nd_return_won_pct', 'return_pts_won_pct', 'return_games_won_pct', 'bp_won_pct']
return_rating_long = return_rating_long.replace([np.inf, -np.inf], np.nan).dropna(subset=features)[(return_rating_long['comment'] == 'Completed') & (return_rating_long['return_games_won_pct'] <= 1)]

pca = PCA()
return_rating_long['return_rating'] = pd.Series(pca.fit_transform(return_rating_long[features])[:, 0], index = return_rating_long.index)

plt.figure(figsize=(10, 6))
sns.histplot(return_rating_long['return_rating'].dropna(), kde=True, bins=30, color='blue')
plt.title('Distribution of Return Ratings', fontsize=16)
plt.xlabel('Return Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
#plt.show()

return_rating_long[['player', '1st_return_won_pct', '2nd_return_won_pct', 'return_pts_won_pct', 'return_games_won_pct', 'bp_won_pct', 'return_rating']].sort_values(by = ['return_rating'], ascending=False).head(10)

return_rating_wide = return_rating_long.pivot(index='game_id', columns='player_type', values='return_rating').reset_index() 
return_rating_wide.columns = ['game_id', 'favored_return_rating', 'underdog_return_rating']

games = pd.merge(games, return_rating_wide, on='game_id', how='left')

# Performance
games['favored_performance'] = (games['favored_sets']-games['underdog_sets']) + 1/6*(games['favored_games']-games['underdog_games']) + 1/24*(games['favored_svpt']-games['underdog_svpt']) + 1/24*((games['underdog_svpt']-games['underdog_1st_won']-games['underdog_2nd_won'])-(games['favored_svpt']-games['favored_1st_won']-games['favored_2nd_won']))
games['underdog_performance'] = -games['favored_performance']

games['points_diff'] = games['favored_rank_pts'] - games['favored_rank_pts'] 
games['favored_elo_diff'] = games['favored_elo'] - games['underdog_elo']
games['underdog_elo_diff'] = -games['favored_elo_diff']
games['favored_elo_surface_diff'] = games['favored_elo_surface'] - games['underdog_elo_surface']
games['underdog_elo_surface_diff'] = -games['favored_elo_surface_diff']

scaler = MinMaxScaler(feature_range=(-1, 1))
games[['favored_performance', 'underdog_performance', 'favored_elo_diff', 'underdog_elo_diff', 'favored_elo_surface_diff', 'underdog_elo_surface_diff']] = scaler.fit_transform(games[['favored_performance', 'underdog_performance', 'favored_elo_diff', 'underdog_elo_diff', 'favored_elo_surface_diff', 'underdog_elo_surface_diff']])

# Hay que tratar de hacer una especie de performance condicional al elo diff

games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
# Rolling Averages
features = pd.DataFrame()
for index, row in tqdm(games[games['season'] >= 2002].iterrows(), total=games[games['season'] >= 2002].shape[0]):    
    favored_as_favored = games[(games['favored'] == row['favored']) & ((games['tourney_date'] < row['tourney_date']) | ((games['tourney_date'] == row['tourney_date']) & (games['round'] < row['round']))) & (games['tourney_date'] >= row['tourney_date']-timedelta(days=365)) & (games['comment'] == 'Completed')].copy()
    favored_as_favored.rename(columns={'favored':'player', 'underdog':'rival', 'favored_rank_group':'player_rank_group', 'underdog_rank_group':'rival_rank_group',
        'favored_sets':'player_sets','underdog_sets':'rival_sets', 'favored_games':'player_games', 'underdog_games':'rival_games', 'favored_service_rating':'player_service_rating',
        'underdog_service_rating':'rival_service_rating', 'favored_return_rating':'player_return_rating', 'underdog_return_rating':'rival_return_rating', 'favored_performance':'player_performance',
        'underdog_performance':'rival_performance'}, inplace=True)
    favored_as_favored['win'] = favored_as_favored['favored_win'] 
    
    favored_as_underdog = games[(games['underdog'] == row['favored']) & ((games['tourney_date'] < row['tourney_date']) | ((games['tourney_date'] == row['tourney_date']) & (games['round'] < row['round']))) & (games['tourney_date'] >= row['tourney_date']-timedelta(days=365)) & (games['comment'] == 'Completed')].copy()
    favored_as_underdog.rename(columns={'favored':'rival', 'underdog':'player', 'favored_rank_group':'rival_rank_group', 'underdog_rank_group':'player_rank_group', 
        'favored_sets':'rival_sets','underdog_sets':'player_sets', 'favored_games':'rival_games', 'underdog_games':'player_games', 'favored_service_rating':'rival_service_rating',
        'underdog_service_rating':'player_service_rating', 'favored_return_rating':'rival_return_rating', 'player_return_rating':'rival_return_rating', 'favored_performance':'rival_performance',
        'underdog_performance':'player_performance'}, inplace=True)
    favored_as_underdog['win'] = np.where(favored_as_underdog['favored_win'] == 1, 0, 1)
    favored_df = pd.concat([favored_as_favored, favored_as_underdog], axis=0)
    
    underdog_as_favored = games[(games['favored'] == row['underdog']) & ((games['tourney_date'] < row['tourney_date']) | ((games['tourney_date'] == row['tourney_date']) & (games['round'] < row['round']))) & (games['tourney_date'] >= row['tourney_date']-timedelta(days=365)) & (games['comment'] == 'Completed')].copy()
    underdog_as_favored.rename(columns={'favored':'player', 'underdog':'rival', 'favored_rank_group':'player_rank_group', 'underdog_rank_group':'rival_rank_group',
        'favored_sets':'player_sets', 'underdog_sets':'rival_sets', 'favored_games':'player_games', 'underdog_games':'rival_games', 'favored_service_rating':'player_service_rating',
        'underdog_service_rating':'rival_service_rating', 'favored_return_rating':'player_return_rating', 'underdog_return_rating':'rival_return_rating', 'favored_performance':'player_performance',
        'underdog_performance':'rival_performance'}, inplace=True)
    underdog_as_favored['win'] = underdog_as_favored['favored_win'] 
    
    underdog_as_underdog = games[(games['underdog'] == row['underdog']) & ((games['tourney_date'] < row['tourney_date']) | ((games['tourney_date'] == row['tourney_date']) & (games['round'] < row['round']))) & (games['tourney_date'] >= row['tourney_date']-timedelta(days=365)) & (games['comment'] == 'Completed')].copy()
    underdog_as_underdog.rename(columns={'favored':'rival', 'underdog':'player', 'favored_rank_group':'rival_rank_group', 'underdog_rank_group':'player_rank_group',
        'favored_sets':'rival_sets', 'underdog_sets':'player_sets', 'favored_games':'rival_games', 'underdog_games':'player_games', 'favored_service_rating':'rival_service_rating',
        'underdog_service_rating':'player_service_rating', 'favored_return_rating':'rival_return_rating', 'player_return_rating':'rival_return_rating', 'favored_performance':'rival_performance',
        'underdog_performance':'player_performance'}, inplace=True)
    underdog_as_underdog['win'] = np.where(underdog_as_underdog['favored_win'] == 1, 0, 1)
    underdog_df = pd.concat([underdog_as_favored, underdog_as_underdog], axis=0)
    
    #underdog_df['win'].mean()
    
    #alpha = 0.01
    #n = len(underdog_df)
    #weights = np.array([(1 - alpha)**(n - i - 1) for i in range(n)])
    #weights /= weights.sum()    
    #print(np.dot(underdog_df['win'], weights))
    
    # Overall
    # Favored 
    features.loc[index, 'favored_games_played'] = len(favored_df)
    features.loc[index, 'favored_win_pct'] = favored_df['win'].mean() if len(favored_df) > 0 else 0
    features.loc[index, 'favored_avg_set_spread'] = favored_df['player_sets'].mean()-favored_df['rival_sets'].mean() if len(favored_df) > 0 else 0
    features.loc[index, 'favored_avg_games_spread'] = favored_df['player_games'].mean()-favored_df['rival_games'].mean() if len(favored_df) > 0 else 0
    features.loc[index, 'favored_avg_service_rating'] = favored_df['player_service_rating'].mean()if len(favored_df) > 0 else 0
    features.loc[index, 'favored_avg_return_rating'] = favored_df['player_return_rating'].mean()if len(favored_df) > 0 else 0
    features.loc[index, 'favored_avg_performance'] = favored_df['player_performance'].mean()if len(favored_df) > 0 else 0
    
    
    # Underdog
    features.loc[index, 'underdog_games_played'] = len(underdog_df)
    features.loc[index, 'underdog_win_pct'] = underdog_df['win'].mean() if len(underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_set_spread'] = underdog_df['player_sets'].mean()-underdog_df['rival_sets'].mean() if len(underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_games_spread'] = underdog_df['player_games'].mean()-underdog_df['rival_games'].mean() if len(underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_service_rating'] = underdog_df['player_service_rating'].mean()if len(underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_performance'] = underdog_df['player_performance'].mean()if len(underdog_df) > 0 else 0
    
    
    # Record against rank
    # Favored
    filtered_favored_df = favored_df[favored_df['rival_rank_group'] == row['underdog_rank_group']]
    features.loc[index, 'favored_games_played_rank'] = len(filtered_favored_df)
    features.loc[index, 'favored_win_pct_rank'] = filtered_favored_df['win'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_set_spread_rank'] = filtered_favored_df['player_sets'].mean()-filtered_favored_df['rival_sets'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_games_spread_rank'] = filtered_favored_df['player_games'].mean()-filtered_favored_df['rival_games'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_service_rating_rank'] = filtered_favored_df['player_service_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_return_rating_rank'] = filtered_favored_df['player_return_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_performance_rank'] = filtered_favored_df['player_performance'].mean()if len(filtered_favored_df) > 0 else 0
    
    # Underdog
    filtered_underdog_df = underdog_df[underdog_df['rival_rank_group'] == row['favored_rank_group']]
    features.loc[index, 'underdog_games_played_rank'] = len(filtered_underdog_df)
    features.loc[index, 'underdog_win_pct_rank'] = filtered_underdog_df['win'].mean() if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_set_spread_rank'] = filtered_underdog_df['player_sets'].mean()-filtered_underdog_df['rival_sets'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_games_spread_rank'] = filtered_underdog_df['player_games'].mean()-filtered_underdog_df['rival_games'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_service_rating_rank'] = filtered_underdog_df['player_service_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_return_rating_rank'] = filtered_underdog_df['player_return_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_performance_rank'] = filtered_underdog_df['player_performance'].mean()if len(filtered_underdog_df) > 0 else 0
    
    
    # Record in this surface
    # Favored
    filtered_favored_df = favored_df[favored_df['surface'] == row['surface']]
    features.loc[index, 'favored_games_played_surface'] = len(filtered_favored_df)
    features.loc[index, 'favored_win_pct_surface'] = filtered_favored_df['win'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_set_spread_surface'] = filtered_favored_df['player_sets'].mean()-filtered_favored_df['rival_sets'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_games_spread_surface'] = filtered_favored_df['player_games'].mean()-filtered_favored_df['rival_games'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_service_rating_surface'] = filtered_favored_df['player_service_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_return_rating_surface'] = filtered_favored_df['player_return_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_performance_surface'] = filtered_favored_df['player_performance'].mean()if len(filtered_favored_df) > 0 else 0
    
    # Underdog
    filtered_underdog_df = underdog_df[underdog_df['surface'] == row['surface']]
    features.loc[index, 'underdog_games_played_surface'] = len(filtered_underdog_df)
    features.loc[index, 'underdog_win_pct_surface'] = filtered_underdog_df['win'].mean() if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_set_spread_surface'] = filtered_underdog_df['player_sets'].mean()-filtered_underdog_df['rival_sets'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_games_spread_surface'] = filtered_underdog_df['player_games'].mean()-filtered_underdog_df['rival_games'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_service_rating_surface'] = filtered_underdog_df['player_service_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_return_rating_surface'] = filtered_underdog_df['player_return_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_performance_surface'] = filtered_underdog_df['player_performance'].mean()if len(filtered_underdog_df) > 0 else 0
    
    
    # Record in this series
    # Favored
    filtered_favored_df = favored_df[favored_df['tourney_series'] == row['tourney_series']]
    features.loc[index, 'favored_games_played_series'] = len(filtered_favored_df)
    features.loc[index, 'favored_win_pct_series'] = filtered_favored_df['win'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_set_spread_series'] = filtered_favored_df['player_sets'].mean()-filtered_favored_df['rival_sets'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_games_spread_series'] = filtered_favored_df['player_games'].mean()-filtered_favored_df['rival_games'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_service_rating_series'] = filtered_favored_df['player_service_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_return_rating_series'] = filtered_favored_df['player_return_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_performance_series'] = filtered_favored_df['player_performance'].mean()if len(filtered_favored_df) > 0 else 0
    
    # Underdog
    filtered_underdog_df = underdog_df[underdog_df['tourney_series'] == row['tourney_series']]
    features.loc[index, 'underdog_games_played_series'] = len(filtered_underdog_df)
    features.loc[index, 'underdog_win_pct_series'] = filtered_underdog_df['win'].mean() if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_set_spread_series'] = filtered_underdog_df['player_sets'].mean()-filtered_underdog_df['rival_sets'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_games_spread_series'] = filtered_underdog_df['player_games'].mean()-filtered_underdog_df['rival_games'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_service_rating_series'] = filtered_underdog_df['player_service_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_return_rating_series'] = filtered_underdog_df['player_return_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_performance_series'] = filtered_underdog_df['player_performance'].mean()if len(filtered_underdog_df) > 0 else 0
    
    
    # Record in this round
    # Favored
    filtered_favored_df = favored_df[favored_df['round'] == row['round']]
    features.loc[index, 'favored_games_played_round'] = len(filtered_favored_df)
    features.loc[index, 'favored_win_pct_round'] = filtered_favored_df['win'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_set_spread_round'] = filtered_favored_df['player_sets'].mean()-filtered_favored_df['rival_sets'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_games_spread_round'] = filtered_favored_df['player_games'].mean()-filtered_favored_df['rival_games'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_service_rating_round'] = filtered_favored_df['player_service_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_return_rating_round'] = filtered_favored_df['player_return_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_performance_round'] = filtered_favored_df['player_performance'].mean()if len(filtered_favored_df) > 0 else 0
    
    # Underdog
    filtered_underdog_df = underdog_df[underdog_df['round'] == row['round']]
    features.loc[index, 'underdog_games_played_round'] = len(filtered_underdog_df)
    features.loc[index, 'underdog_win_pct_round'] = filtered_underdog_df['win'].mean() if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_set_spread_round'] = filtered_underdog_df['player_sets'].mean()-filtered_underdog_df['rival_sets'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_games_spread_round'] = filtered_underdog_df['player_games'].mean()-filtered_underdog_df['rival_games'].mean()  if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_service_rating_round'] = filtered_underdog_df['player_service_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_return_rating_round'] = filtered_underdog_df['player_return_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_performance_round'] = filtered_underdog_df['player_performance'].mean()if len(filtered_underdog_df) > 0 else 0
    
    
    # H2H (Record between players)
    # Favored
    filtered_favored_df = favored_df[favored_df['rival'] == row['underdog']]
    features.loc[index, 'favored_games_played_h2h'] = len(filtered_favored_df)
    features.loc[index, 'favored_win_pct_h2h'] = filtered_favored_df['win'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_set_spread_h2h'] = filtered_favored_df['player_sets'].mean()-filtered_favored_df['rival_sets'].mean() if len(filtered_favored_df) > 0 else 0    
    features.loc[index, 'favored_avg_games_spread_h2h'] = filtered_favored_df['player_games'].mean()-filtered_favored_df['rival_games'].mean() if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_service_rating_h2h'] = filtered_favored_df['player_service_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_return_rating_h2h'] = filtered_favored_df['player_return_rating'].mean()if len(filtered_favored_df) > 0 else 0
    features.loc[index, 'favored_avg_performance_h2h'] = filtered_favored_df['player_performance'].mean()if len(filtered_favored_df) > 0 else 0
    
    # Favored
    filtered_underdog_df = underdog_df[underdog_df['rival'] == row['favored']]
    features.loc[index, 'underdog_avg_service_rating_h2h'] = filtered_underdog_df['player_service_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_return_rating_h2h'] = filtered_underdog_df['player_return_rating'].mean()if len(filtered_underdog_df) > 0 else 0
    features.loc[index, 'underdog_avg_performance_h2h'] = filtered_underdog_df['player_performance'].mean()if len(filtered_underdog_df) > 0 else 0

features = pd.read_csv('features.csv')
features = features.set_index(games[games['season'] >= 2002].index).drop(columns = ['Unnamed: 0'])
features.rename(columns = {'favored_avg_win_pct_rank':'favored_win_pct_rank', 'underdog_avg_win_pct_rank':'underdog_win_pct_rank',
                        'favored_avg_win_pct_surface':'favored_win_pct_surface', 'underdog_avg_games_played_surface':'underdog_games_played_surface', 'underdog_avg_win_pct_surface':'underdog_win_pct_surface', 
                        'favored_avg_games_played_series':'favored_games_played_series', 'favored_avg_win_pct_series':'favored_win_pct_series', 'underdog_avg_games_played_series':'underdog_games_played_series', 'underdog_avg_win_pct_series':'underdog_win_pct_series',
                        'favored_avg_games_played_round':'favored_games_played_round', 'favored_avg_win_pct_round':'favored_win_pct_round', 'underdog_avg_games_played_round':'underdog_games_played_round', 'underdog_avg_win_pct_round':'underdog_win_pct_round',
                        'favored_avg_games_played_h2h':'favored_games_played_h2h', 'favored_avg_win_pct_h2h':'favored_win_pct_h2h'}, inplace=True)
features = features[[col for col in features.columns if '_avg_' in col or 'games_played' in col or 'win_pct' in col]]
features.to_csv('features.csv')

games = pd.concat([games, features], axis=1)
games = games[(games['season'] >= 2002) & (games['season'] <= 2024)].reset_index(drop=True)



odds = pd.read_csv('Odds/odds.csv')
odds = janitor.clean_names(odds).rename(columns={'tournament':'tourney_full_name'})
odds = odds[(odds['season'] >= 2002) & (odds['season'] <= 2024)].reset_index(drop=True)

odds=odds.sort_values("date").reset_index(drop=True)
odds["wrank"]=odds["wrank"].replace(np.nan,2500).replace("NR",2500).astype(float).astype(int)
odds["lrank"]=odds["lrank"].replace(np.nan,2500).replace("NR",2500).astype(float).astype(int)
odds['winner'] = odds['winner'].str.strip()
odds['loser'] = odds['loser'].str.strip()

sorted([player for player in odds['loser'].unique() if player not in list(games['loser'])])
players_dict = {' Hajek J.':'Hajek J.', 'Al Ghareeb M.':'Ghareeb M.', 'Alvarez E.':'Benfele Alvarez E.', 'Ancic I.':'Ancic M.', 'Andersen J.F.':'Frode Andersen J.', 'Ascione A.':'Ascione T.',
'Auger-Aliassime F.':'Auger Aliassime F.', 'Bachelot J.F':'Francois Bachelot J.', 'Barrios M.':'Barrios Vera T.', 'Barrios Vera M.T.':'Barrios Vera T.', 'Bautista R.':'Bautista Agut R.',
'Berdych T. ':'Berdych T.', 'Berrettini M. ':'Berrettini M.', 'Bogomolov A.':'Bogomolov Jr A.', 'Bogomolov Jr. A.':'Bogomolov Jr A.', 'Bogomolov Jr.A.':'Bogomolov Jr A.',
'Brugues-Davi A.':'Brugues Davi A.', 'Brzezicki J.P.':'Pablo Brzezicki J.', 'Bu Y.':'Yunchaokete B.', 'Burruchaga R.':'Andres Burruchaga R.', 'Carreno-Busta P.':'Carreno Busta P.',
'Cerundolo J.M.':'Manuel Cerundolo J.', 'Cervantes I.':'Cervantes Huegun I.', 'Chardy J. ':'Chardy J.', 'Chela J.':'Ignacio Chela J.', 'Chela J.I.':'Ignacio Chela J.', 'Chiudinelli M. ':'Chiudinelli M.',
'Cilic M. ':'Cilic M.', 'Cuevas P. ':'Cuevas P.', 'Dasnieres de Veigy J.':'Dasnieres De Veigy J.', 'Davydenko N. ':'Davydenko N.', 'De Heart R.':'Deheart R.', 'Del Bonis F.':'Delbonis F.',
'Del Potro J.':'Martin del Potro J.', 'Del Potro J. M.':'Martin del Potro J.', 'Del Potro J.M.':'Martin del Potro J.', "Dell'Acqua M.":'Dellacqua M.', 'Djokovic N. ':'Djokovic N.',
'Dolgopolov O.':'Dolgopolov A.', 'Dosedel S.':'Dosedel S.', 'Duclos P.L.':'Ludovic Duclos P.', 'Dutra Da Silva R.':'Dutra Silva R.', 'Estrella Burgos V.':'Estrella V.', 'Etcheverry T.':'Martin Etcheverry T.',
'Faurel J.C.':'Christophe Faurel J.', 'Federer R. ':'Federer R.', 'Ferrer D. ':'Ferrer D.', 'Ferrero J.':'Carlos Ferrero J.', 'Ferrero J.C.':'Carlos Ferrero J.', 'Ficovich J.P.':'Pablo Ficovich J.',
'Fromberg R.':'Fromberg R.', 'Galan D.':'Elahi Galan D.', 'Galan D.E.':'Elahi Galan D.', 'Gambill J. M.':'Michael Gambill J.', 'Gambill J.M.':'Michael Gambill J.', 'Garcia-Lopez G.':'Garcia Lopez G.', 
'Garcia-Lopez G. ':'Garcia Lopez G.', 'Gasquet R. ':'Gasquet R.', 'Gimeno-Traver D.':'Gimeno Traver D.', 'Goellner M.K.':'Kevin Goellner M.', 'Gomez A.':'Gomez Gb42 A.', 'Gomez F.':'Agustin Gomez F.',
'Guccione A.':'Guccione C.', 'Gutierrez-Ferrol S.':'Gutierrez Ferrol S.', 'Guzman J.P.':'Pablo Guzman J.', 'Hantschek M.':'Hantschk M.', 'Haider-Mauer A.':'Haider Maurer A.', 'Haider-Maurer A.':'Haider Maurer A.',
'Herbert P.':'Hugues Herbert P.', 'Herbert P.H':'Hugues Herbert P.', 'Herbert P.H.':'Hugues Herbert P.', 'Hernych J. ':'Hernych J.', 'Hong S.':'Chan Hong S.', 'Hsu Y.':'Hsiou Hsu Y.',
'Huesler M.A.':'Andrea Huesler M.', 'Isner J. ':'Isner J.', 'Jun W.S.':'Sun Jun W.', 'Kohlschreiber P..':'Kohlschreiber P.', 'Korolev E. ':'Korolev E.', 'Kucera V.':'Kucera K.', 'Kwon S.W.':'Woo Kwon S.',
'Lammer M. ':'Lammer M.', 'Lee D.H.':'Hee Lee D.', 'Lee H.T.':'Taik Lee H.', 'Lisnard J.':'Rene Lisnard J.', 'Lisnard J.R.':'Rene Lisnard J.', 'Londero J.I.':'Ignacio Londero J.',
'Lopez F. ':'Lopez F.', 'Lopez-Jaen M.A.':'Angel Lopez Jaen M.', 'Lu Y.':'Hsun Lu Y.', 'Lu Y.H.':'Hsun Lu Y.', 'Marin J.A':'Antonio Marin J.', 'Marin J.A.':'Antonio Marin J.', 'Mathieu P.H.':'Henri Mathieu P.',
'Mayer L. ':'Mayer L.', 'McDonald M.':'Mcdonald M.', 'Mecir M.':'Mecir Jr M.', 'Menendez-Maceiras A.':'Menendez Maceiras A.', 'Monaco J. ':'Monaco J.', 'Monfils G. ':'Monfils G.', 'Montanes A. ':'Montanes A.',
'Moroni G.':'Marco Moroni G.', 'Mpetshi G.':'Mpetshi Perricard G.', 'Munoz De La Nava D.':'Munoz de la Nava D.', 'Munoz-De La Nava D.':'Munoz de la Nava D.', 'Murray A. ':'Murray A.', 'Nadal R. ':'Nadal R.',
'Nadal-Parera R.':'Nadal R.', 'Navarro-Pastor I.':'Navarro I.', 'Nieminen J. ':'Nieminen J.', 'O Connell C.':'Oconnell C.', 'Olivieri G.':'Alberto Olivieri G.', 'Pucinelli de Almeida M.':'Pucinelli De Almeida M.',
'Querry S.':'Querrey S.', 'Qureshi A.':'Ul Haq Qureshi A.', 'Qureshi A.U.H.':'Ul Haq Qureshi A.', 'Ramirez-Hidalgo R.':'Ramirez Hidalgo R.', 'Ramos-Vinolas A.':'Ramos A.', 'Roger-Vasselin E.':'Roger Vasselin E.',
'Robredo R.':'Robredo T.', 'Robredo T. ':'Robredo T.', 'Scherrer J.C.':'Claude Scherrer J.', 'Schuettler P.':'Claude Scherrer J.', 'Seppi A. ':'Seppi A.', 'Serra F. ':'Serra F.',
'Silva F.F.':'Ferreira Silva F.', 'Simon G. ':'Simon G.', 'Smith J.P.':'Patrick Smith J.', 'Srichaphan N.':'Srichaphan P.', 'Statham J.':'Rubin Statham J.', 'Stebe C-M.':'Marcel Stebe C.',
'Stebe C.M.':'Marcel Stebe C.', 'Stepanek R. ':'Stepanek R.', 'Struff J.L.':'Lennard Struff J.', 'Thiem D. ':'Thiem D.', 'Tipsarevic J. ':'Tipsarevic J.', 'Tirante T.A.':'Agustin Tirante T.',
'Troicki V. ':'Troicki V.', 'Trujillo G.':'Trujillo Soler G.', 'Tseng C. H.':'Hsin Tseng C.', 'Tseng C.H.':'Hsin Tseng C.', 'Tsitsipas S. ':'Tsitsipas S.', 'Tsonga J.W.':'Tsonga J.',
'Van der Merwe I.':'Van Der Merwe I.', 'Varillas J. P.':'Pablo Varillas J.', 'Varillas J.P.':'Pablo Varillas J.', 'Vassallo-Arguello M.':'Vassallo Arguello M.', 'Verdasco F. ':'Verdasco F.',
'Verdasco M.':'Verdasco F.', 'Viloca J.A.':'Albert Viloca Puig J.', 'Wang Y. Jr':'Jr Wang Y.', 'Wang Y.T.':'Wang J.', 'Wawrinka S. ':'Wawrinka S.', 'Wolf J.J.':'J Wolf J.', 'Wu T.L.':'Lin Wu T.',
'Yoon Y.':'Il Yoon Y.', 'Youzhny A.':'Youzhny M.', 'Youzhny M. ':'Youzhny M.', 'Zeng S.X.':'Xuan Zeng S.', 'Zhang Ze':'Zhang Ze.', 'Zhang Z.':'Zhang Ze.', 'Zhu B.Q.':'Qiang Zhu B.', 'Zverev A. ':'Zverev A.',
'de Chaunac S.':'De Chaunac S.', 'de Voest R.':'De Voest R.', 'di Mauro A.':'Di Mauro A.', 'di Pasquale A.':'Di Pasquale A.', 'van Gemerden M.':'Van Gemerden M.', 'van Lottum J.':'Van Lottum J.',
'van Scheppingen D.':'Van Scheppingen D.', 
'Al Khulaifi N.G.':'Ghanim Al Khulaifi N.', 'Al-Ghareeb M.':'Ghareeb M.', 'Alawadhi O.':'Awadhy O.', 'Albert M.':'Albert Ferrando M.', 'Ali Mutawa J.M.':'Al Mutawa J.', 'Aragone J.C.':'Aragone J.', 'Aragone JC':'Aragone J.',
'Aranguren J.M.':'Martin Aranguren J.', 'Bahrouzyan O.':'Awadhy O.', 'Bailly G.':'Arnaud Bailly G.', 'Cabal J.S.':'Sebastian Cabal J.', 'Chekov P.':'Chekhov P.', 'Deen Heshaam A.':'Elyaas Deen Heshaam A.',
'Dev Varman S.':'Devvarman S.', 'Do M.Q.':'Quan Do M.', 'Fish A.':'Fish M.', 'Fornell M.':'Fornell Mestres M.', 'Fruttero J.P.':'Paul Fruttero J.', 'Gallardo M.':'Gallardo Valles M.', 'Gimeno D.':'Gimeno Traver D.',
'Gomez-Herrera C.':'Gomez Herrera C.', 'Gong M.X.':'Xin Gong M.', 'Granollers Pujol G.':'Granollers G.', 'Granollers-Pujol G.':'Granollers G.', 'Granollers-Pujol M.':'Granollers G.',
'Gruber K.':'Don Gruber K.', 'Guzman J.':'Pablo Guzman J.', 'Haji A.':'Hajji A.', 'Herbert P-H.':'Hugues Herbert P.', 'Hernandez-Fernandez J':'Hernandez J.', 'Hernandez-Fernandez J.':'Hernandez J.',
'Im K.T.':'Tae Im K.', 'Ivanov-Smolensky K.':'Ivanov Smolensky K.', 'Jeong S.Y.':'Young Jeong S.', 'Jones G.D.':'D Jones G.', 'Jun W.':'Sun Jun W.', 'Kim K':'Kim K.', 'King-Turner D.':'King Turner D.',
'Kunitcin I.':'Kunitsyn I.', 'Kwiatkowski T.S.':'Son Kwiatkowski T.', 'Li Zh.':'Li Z.', 'Lin J.M.':'Mingjie Lin J.', 'Lopez-Perez E.':'Lopez Perez E.', 'Luncanu P.A.':'Alexandru Luncanu P.',
'Luque D.':'Luque Velasco D.', 'MacLagan M.':'Maclagan M.', 'Madaras D.':'Nicolae Madaras D.', 'March O.':'Marach O.', 'Mathieu P.':'Henri Mathieu P.', 'Matos-Gil I.':'Matos Gil I.', 'McClune M.':'Mcclune M.',
'Meligeni Rodrigues F':'Meligeni F.', 'Mo Y.':'Cong Mo Y.', 'Moroni G.M.':'Marco Moroni G.', 'Mukund S.':'Kumar Mukund S.', 'Munoz de La Nava D.':'Munoz de la Nava D.', 'Nader M.':'Nader Al Baloushi M.',
'Nam H.W.':'Woo Nam H.', 'Nam J.S.':'Woo Nam H.', 'Nedovyesov O.':'Nedovyesov A.', "O'Connell C.":'Oconnell C.', "O'Neal J.":'Oneal J.', 'Ortega-Olmedo R.':'Ortega Olmedo R.', 'Podlipnik H.':'Podlipnik Castillo H.',
'Prashanth V.':'Vijay Sundar Prashanth N.', 'Prpic A.':'Prpic F.', 'Rascon T.':'Luis Tati Rascon J.', 'Rehberg M.':'Hans Rehberg M.', 'Reyes-Varela M.A.':'Angel Reyes Varela M.',
'Riba-Madrid P.':'Riba P.', 'Rodriguez Taverna S.':'Fa Rodriguez Taverna S.', 'Ruevski P.':'Rusevski P.', 'Salva B.':'Salva Vidal B.', 'Samper-Montana J.':'Samper Montana J.',
'Sanchez De Luna J.':'Antonio Sanchez De Luna J.', 'Sanchez de Luna J.A.':'Antonio Sanchez De Luna J.', 'Scherrer J.':'Claude Scherrer J.', 'Schuttler P.':'Schuettler R.', 'Si Y.M.':'Ming Si Y.',
'Silva D.':'Dutra Da Silva D.', 'Silva F.':'Ferreira Silva F.', 'Statham R.':'Rubin Statham J.', 'Struff J-L.':'Lennard Struff J.', 'Sultan-Khalfan A.':'Khalfan S.', 'Tyurnev E.':'Tiurnev E.',
'Van D. Merwe I.':'Van Der Merwe I.', 'Van der Dium A.':'Van Der Duim A.', 'Vicente M.':'Vicente F.', 'Viola Mat.':'Viola M.', 'Wang Y.':'Wang J.', 'Wang Y.Jr.':'Jr Wang Y.', 'Xu J.C.':'Chao Xu J.',
'Yang T.H.':'Hua Yang T.', 'Yu X.Y.':'Yuan Yu X.', 'Zayed M. S.':'Shanan Zayed M.', 'Zayed M.S.':'Shanan Zayed M.', 'Zayid M.':'Shannan Zayid M.', 'Zayid M. S.':'Shannan Zayid M.', 'Zayid M.S.':'Shannan Zayid M.',
'van der Meer N.':'Van Der Meer N.'}
odds['winner'].replace(players_dict, inplace=True)
odds['loser'].replace(players_dict, inplace=True)


# Merge (falta corregir un par de partidos en odds)
data = pd.merge(games, odds[['winner', 'loser', 'tourney_full_name', 'season', 'psw', 'psl', 'b365w', 'b365l', 'maxw', 'maxl', 'avgw', 'avgl']], how = 'left', on = ['winner', 'loser', 'tourney_full_name', 'season'])

data['favored_max_odds'] = data.apply(lambda row: row['maxw'] if row['winner_rank'] < row['loser_rank'] else row['maxl'], axis=1)
data['underdog_max_odds'] = data.apply(lambda row: row['maxl'] if row['winner_rank'] < row['loser_rank'] else row['maxw'], axis=1)

data['favored_avg_odds'] = data.apply(lambda row: row['avgw'] if row['winner_rank'] < row['loser_rank'] else row['avgl'], axis=1)
data['underdog_avg_odds'] = data.apply(lambda row: row['avgl'] if row['winner_rank'] < row['loser_rank'] else row['avgw'], axis=1)

data['favored_pinnacle_odds'] = data.apply(lambda row: row['psw'] if row['winner_rank'] < row['loser_rank'] else row['psl'], axis=1)
data['underdog_pinnacle_odds'] = data.apply(lambda row: row['psl'] if row['winner_rank'] < row['loser_rank'] else row['psw'], axis=1)

data['favored_bet365_odds'] = data.apply(lambda row: row['b365w'] if row['winner_rank'] < row['loser_rank'] else row['b365l'], axis=1)
data['underdog_bet365_odds'] = data.apply(lambda row: row['b365l'] if row['winner_rank'] < row['loser_rank'] else row['b365w'], axis=1)

# Drop rows by index
print(data[data.duplicated(subset=['game_id'], keep=False)][['winner', 'loser', 'tourney_full_name', 'season', 'b365w', 'b365l', 'psw', 'psl', 'maxw', 'maxl', 'avgw', 'avgl']])
data = data.drop([7173, 13406, 14370, 14384, 22097, 22110, 58551, 58564, 86745, 86754, 147700, 147707, 226546, 226559], axis=0).reset_index(drop=True)

# Columns to keep
columns_to_keep = ['favored', 'underdog', 'favored_id', 'underdog_id', 'game_id', 'season', 'tourney_id', 'tourney_name', 'tourney_full_name', 'tourney_series', 'surface', 'draw_size', 'tourney_level',
                'tourney_date', 'match_num', 'round', 'favored_entry', 'underdog_entry', 'favored_rank', 'underdog_rank', 'favored_rank_group', 'underdog_rank_group', 'favored_rank_pts', 'underdog_rank_pts',
                'favored_elo', 'underdog_elo', 'favored_elo_surface', 'underdog_elo_surface', 'points_diff', 'favored_elo_diff', 'underdog_elo_diff',
                'favored_elo_surface_diff', 'underdog_elo_surface_diff'] + ['favored_win', 'favored_max_odds', 'underdog_max_odds', 'favored_avg_odds', 'underdog_avg_odds',
                'favored_pinnacle_odds', 'underdog_pinnacle_odds', 'favored_bet365_odds', 'underdog_bet365_odds'] + [col for col in data.columns if '_avg_' in col or 'games_played' in col or 'win_pct' in col]
data[columns_to_keep].to_csv('data.csv', index=False)

data = pd.read_csv('data_backup.csv')
