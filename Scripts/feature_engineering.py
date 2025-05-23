import pandas as pd
import numpy as np
from tqdm import tqdm
import janitor
from Scripts.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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
tournaments_by_season = pd.read_csv('tournaments_by_season.csv')


tournaments_by_season = tournaments_by_season[['tournament_odds', 'tournament_stats', 'series', 'season']].sort_values(by = ['series', 'tournament_stats', 'season']).reset_index(drop=True)
tournaments_by_season.to_csv('tournaments_by_season.csv', index=False)

tournaments_stats_by_season = games[['tourney_name', 'tourney_level', 'season']].drop_duplicates(subset = ['tourney_name', 'season'])
tournaments_stats_by_season = tournaments_stats_by_season[(tournaments_stats_by_season['season'] >= 1990) & (~tournaments_stats_by_season['tourney_level'].isin(['D']))].reset_index(drop=True)
tournaments_stats_by_season.rename(columns={'tourney_name':'tournament_stats'},inplace=True)
tournaments_stats_by_season['series'] = tournaments_stats_by_season['tournament_stats'].apply(lambda x: 'Challenger' if 'CH' in x else ('Masters 1000' if 'Masters' in x else None))
list(tournaments_stats_by_season['tournament_stats'].unique())

tournaments_by_season = pd.concat([tournaments_by_season, tournaments_stats_by_season[['tournament_stats', 'series', 'season']]],axis = 0).drop_duplicates(subset=['tournament_stats', 'season'])
tournaments_by_season.to_csv('tournaments_by_season2.csv', index=False)
"""


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

tournaments = pd.read_csv('tournaments_by_season.csv')

#games.drop(columns=['tourney_full_name', 'tourney_series'], inplace=True)
games['tourney_name'] = games['tourney_name'].str.strip()

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

# Merge odds data
odds = pd.read_csv('Odds/odds.csv')
odds = janitor.clean_names(odds).rename(columns={'tournament':'tourney_full_name'})
odds = odds[(odds['season'] >= 2002) & (odds['season'] <= 2024)].reset_index(drop=True)

odds=odds.sort_values("date").reset_index(drop=True)
odds['winner'] = odds['winner'].str.strip()
odds['loser'] = odds['loser'].str.strip()

#sorted([player for player in odds['loser'].unique() if player not in list(games['loser'])])
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

missing_tournaments = odds[['season', 'tourney_full_name']].drop_duplicates()
missing_tournaments = missing_tournaments[~missing_tournaments.isin(games[['season', 'tourney_full_name']].drop_duplicates().to_dict(orient='list')).all(axis=1)]
missing_tournaments

# Merge (falta corregir un par de partidos en odds)
games = pd.merge(games, odds[['winner', 'loser', 'tourney_full_name', 'season', 'psw', 'psl', 'b365w', 'b365l', 'maxw', 'maxl', 'avgw', 'avgl']], how = 'left', on = ['winner', 'loser', 'tourney_full_name', 'season'])

# Drop rows by index
#games[games['_merge'] == 'right_only'][['winner', 'loser', 'round', 'tourney_full_name', 'season']].sort_values(by = ['tourney_full_name', 'season'])
print(games[games.duplicated(subset=['game_id'], keep=False)][['winner', 'loser', 'game_id', 'tourney_full_name', 'season', 'b365w', 'b365l', 'psw', 'psl', 'maxw', 'maxl', 'avgw', 'avgl']].sort_values(by=['tourney_full_name', 'season']))
games = games.drop([153325, 159886, 160850, 160864, 168897, 168910, 206667, 206680, 235828, 235837, 298525, 298532, 378695, 378708], axis=0).reset_index(drop=True)

# Elo rankings
games['surface'].replace({'Carpet':'Hard'},inplace=True)
def elo_rankings(games, surface=None):
    if surface:
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
    ranking_elo=pd.DataFrame(ranking_elo,columns=["winner_elo","loser_elo"], index = games.index)
    if surface:
        ranking_elo.columns = ["winner_elo_surface","loser_elo_surface"]
    return ranking_elo

ranking_elo = elo_rankings(games)
ranking_elo_surface = pd.concat([elo_rankings(games, surface='Hard'), elo_rankings(games, surface='Grass'), elo_rankings(games, surface='Clay')], axis = 0) # Grass, Clay, Hard
games = pd.concat([games, ranking_elo, ranking_elo_surface],axis=1)
games[['winner', 'loser', 'winner_elo_surface', 'loser_elo_surface']]


"""
# Glicko rankings
from glicko2 import Player
def glicko2_rankings(games, surface=None):
    players = {}
    rankings = []
    
    if surface:
        games = games[games['surface'] == surface]
    
    for index, row in tqdm(games.iterrows(), total=len(games), desc=f'Computing {surface or "Overall"} Glicko-2'):
        winner = row['winner_name']
        loser = row['loser_name']
        
        if winner not in players:
            players[winner] = Player()
        if loser not in players:
            players[loser] = Player()
        
        winner_player = players[winner]
        loser_player = players[loser]
        
        winner_player.update_player(
            rating_list=[loser_player.getRating()],
            RD_list=[loser_player.getRd()],
            outcome_list=[1]
        )
        loser_player.update_player(
            rating_list=[winner_player.getRating()],
            RD_list=[winner_player.getRd()],
            outcome_list=[1]
        )
        
        rankings.append((winner_player.getRating(), loser_player.getRating()))
    
    rankings_df = pd.DataFrame(rankings, columns=["winner_glicko2", "loser_glicko2"], index=games.index)
    if surface:
        rankings_df.columns = [f"winner_glicko2_{surface.lower()}", f"loser_glicko2_{surface.lower()}"]
    
    return rankings_df

glicko2_rankings_overall = glicko2_rankings(games)
glicko2_rankings_surface = pd.concat([glicko2_rankings(games, surface='Hard'), glicko2_rankings(games, surface='Grass'), glicko2_rankings(games, surface='Clay')],axis=1)
games = pd.concat([games, glicko2_rankings_overall, glicko2_rankings_surface], axis=1)
"""


games = games[(~games['tourney_level'].isin(['D'])) & (games['season'] >= 1990)].reset_index(drop=True)

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

games['w_svpt_won'] = games['w_1stWon']+games['w_2ndWon']
games['l_svpt_won'] = games['l_1stWon']+games['l_2ndWon']

games['w_rtpt_won'] = games['l_svpt']-games['l_1stWon']-games['l_2ndWon']
games['l_rtpt_won'] = games['w_svpt']-games['w_1stWon']-games['w_2ndWon']

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

games['favored_seed'] = games.apply(lambda row: row['winner_seed'] if row['winner_rank'] < row['loser_rank'] else row['loser_seed'], axis=1)
games['underdog_seed'] = games.apply(lambda row: row['loser_seed'] if row['winner_rank'] < row['loser_rank'] else row['winner_seed'], axis=1)

games['favored_hand'] = games.apply(lambda row: row['winner_hand'] if row['winner_rank'] < row['loser_rank'] else row['loser_hand'], axis=1)
games['underdog_hand'] = games.apply(lambda row: row['loser_hand'] if row['winner_rank'] < row['loser_rank'] else row['winner_hand'], axis=1)

games['favored_ht'] = games.apply(lambda row: row['winner_ht'] if row['winner_rank'] < row['loser_rank'] else row['loser_ht'], axis=1)
games['underdog_ht'] = games.apply(lambda row: row['loser_ht'] if row['winner_rank'] < row['loser_rank'] else row['winner_ht'], axis=1)

games['favored_ioc'] = games.apply(lambda row: row['winner_ioc'] if row['winner_rank'] < row['loser_rank'] else row['loser_ioc'], axis=1)
games['underdog_ioc'] = games.apply(lambda row: row['loser_ioc'] if row['winner_rank'] < row['loser_rank'] else row['winner_ioc'], axis=1)

games['favored_age'] = games.apply(lambda row: row['winner_age'] if row['winner_rank'] < row['loser_rank'] else row['loser_age'], axis=1)
games['underdog_age'] = games.apply(lambda row: row['loser_age'] if row['winner_rank'] < row['loser_rank'] else row['winner_age'], axis=1)

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

games['favored_svpt_won'] = games.apply(lambda row: row['w_svpt_won'] if row['winner_rank'] < row['loser_rank'] else row['l_svpt_won'], axis=1)
games['underdog_svpt_won'] = games.apply(lambda row: row['l_svpt_won'] if row['winner_rank'] < row['loser_rank'] else row['w_svpt_won'], axis=1)

games['favored_rtpt_won'] = games.apply(lambda row: row['w_rtpt_won'] if row['winner_rank'] < row['loser_rank'] else row['l_rtpt_won'], axis=1)
games['underdog_rtpt_won'] = games.apply(lambda row: row['l_rtpt_won'] if row['winner_rank'] < row['loser_rank'] else row['w_rtpt_won'], axis=1)

games['favored_bp_saved'] = games.apply(lambda row: row['w_bpSaved'] if row['winner_rank'] < row['loser_rank'] else row['l_bpSaved'], axis=1)
games['underdog_bp_saved'] = games.apply(lambda row: row['l_bpSaved'] if row['winner_rank'] < row['loser_rank'] else row['w_bpSaved'], axis=1)

games['favored_bp_faced'] = games.apply(lambda row: row['w_bpFaced'] if row['winner_rank'] < row['loser_rank'] else row['l_bpFaced'], axis=1)
games['underdog_bp_faced'] = games.apply(lambda row: row['l_bpFaced'] if row['winner_rank'] < row['loser_rank'] else row['w_bpFaced'], axis=1)

games['favored_bp_won'] = games.apply(lambda row: row['w_bpWon'] if row['winner_rank'] < row['loser_rank'] else row['l_bpWon'], axis=1)
games['underdog_bp_won'] = games.apply(lambda row: row['l_bpWon'] if row['winner_rank'] < row['loser_rank'] else row['w_bpWon'], axis=1)

games['favored_dominance_ratio'] = (games['favored_rtpt_won'] / games['underdog_svpt']) / (games['underdog_rtpt_won'] / games['favored_svpt'])
games['underdog_dominance_ratio'] = (games['underdog_rtpt_won'] / games['favored_svpt']) / (games['favored_rtpt_won'] / games['underdog_svpt'])

games['favored_max_odds'] = games.apply(lambda row: row['maxw'] if row['winner_rank'] < row['loser_rank'] else row['maxl'], axis=1)
games['underdog_max_odds'] = games.apply(lambda row: row['maxl'] if row['winner_rank'] < row['loser_rank'] else row['maxw'], axis=1)

games['favored_avg_odds'] = games.apply(lambda row: row['avgw'] if row['winner_rank'] < row['loser_rank'] else row['avgl'], axis=1)
games['underdog_avg_odds'] = games.apply(lambda row: row['avgl'] if row['winner_rank'] < row['loser_rank'] else row['avgw'], axis=1)

games['favored_pinnacle_odds'] = games.apply(lambda row: row['psw'] if row['winner_rank'] < row['loser_rank'] else row['psl'], axis=1)
games['underdog_pinnacle_odds'] = games.apply(lambda row: row['psl'] if row['winner_rank'] < row['loser_rank'] else row['psw'], axis=1)

games['favored_bet365_odds'] = games.apply(lambda row: row['b365w'] if row['winner_rank'] < row['loser_rank'] else row['b365l'], axis=1)
games['underdog_bet365_odds'] = games.apply(lambda row: row['b365l'] if row['winner_rank'] < row['loser_rank'] else row['b365w'], axis=1)

games['favored_odds'] = np.where(games['favored_pinnacle_odds'].notnull(), games['favored_pinnacle_odds'], np.where(games['favored_avg_odds'].notnull(), games['favored_avg_odds'], games['favored_bet365_odds']))
games['underdog_odds'] = np.where(games['underdog_pinnacle_odds'].notnull(), games['underdog_pinnacle_odds'], np.where(games['underdog_avg_odds'].notnull(), games['underdog_avg_odds'], games['underdog_bet365_odds']))

games.drop(columns = ['winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
        'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
        'w_set1', 'l_set1', 'w_set2', 'l_set2', 'w_set3', 'l_set3', 'w_set4', 'l_set4', 'w_set5', 'l_set5', 'w_sets', 'l_sets', 'w_games', 'l_games', 'w_bpWon', 'l_bpWon', 'w_svpt_won', 'l_svpt_won', 'w_rtpt_won', 'l_rtpt_won',
        'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points', 'winner_elo', 'loser_elo', 'winner_elo_surface', 'loser_elo_surface', 'winner_rank_group', 'loser_rank_group'], inplace=True)

# Rank points diff
games['points_diff'] = games['favored_rank_pts'] - games['underdog_rank_pts']
games['elo_diff'] = games['favored_elo'] - games['underdog_elo']
games['elo_surface_diff'] = games['favored_elo_surface'] - games['underdog_elo_surface']

games['log_points_diff'] = np.log(games['favored_rank_pts']) - np.log(games['underdog_rank_pts'])
games['log_elo_diff'] = np.log(games['favored_elo']) - np.log(games['underdog_elo'])
games['log_elo_surface_diff'] = np.log(games['favored_elo_surface']) - np.log(games['underdog_elo_surface'])


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

games.to_csv('games.csv')

games = pd.read_csv('games.csv')
games['tourney_date'] = pd.to_datetime(games['tourney_date'])
games.set_index('Unnamed: 0', inplace=True)
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games = games[games['tourney_level'].isin(['G', 'M', 'O', 'A', 'F'])].reset_index(drop=True)


# Create a DataFrame with games based on player and rival
games_player_rival = games.copy()
games_player_rival.columns = games_player_rival.columns.str.replace('favored', 'player', regex=False)
games_player_rival.columns = games_player_rival.columns.str.replace('underdog', 'rival', regex=False)
games_player_rival['win'] = games_player_rival['player_win'] 
games_player_rival.drop(columns='player_win',inplace=True)

# Duplicate rows and swap player and rival columns
games_rival_player = games.copy()
games_rival_player.columns = games_rival_player.columns.str.replace('favored', 'rival', regex=False)
games_rival_player.columns = games_rival_player.columns.str.replace('underdog', 'player', regex=False)
games_rival_player['win'] = np.where(games_rival_player['rival_win'] == 1, 0, 1)
games_rival_player.drop(columns='rival_win',inplace=True)

# Combine the original and swapped DataFrames
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
    
    # Common opponents
    #common_opponents = list(set(favored_df['rival']).intersection(set(underdog_df['rival'])))
    #favored_df = favored_df[favored_df['rival'].isin(common_opponents)].reset_index(drop=True)
    #underdog_df = underdog_df[underdog_df['rival'].isin(common_opponents)].reset_index(drop=True)
    
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


games = pd.read_csv('games.csv')
games.set_index('Unnamed: 0', inplace=True)
games['tourney_date'] = pd.to_datetime(games['tourney_date'])
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games = games[games['tourney_level'].isin(['G', 'M', 'O', 'A', 'F'])].reset_index(drop=True)

features = pd.read_csv('features.csv')
features = features.set_index('Unnamed: 0')

features_to_keep = [feature for feature in features.columns if '_diff' in feature] + ['inactive_match', 'uncertainty']
games = pd.concat([games, features[features_to_keep]], axis=1)

games = games[(games['underdog_odds'] <= 30) & (games['favored_odds'] <= 2)].reset_index(drop=True)
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

# Feature selection
features_names = ['season', 'week',  'elo_diff', 'elo_surface_diff',  'log_elo_diff', 'log_elo_surface_diff', 'age_diff', 'height_diff'] + features_to_keep
features_names = [feature for feature in features_names if feature not in ['inactivity_diff', 'inactive_match']]
games[features_names] = games[features_names].replace([np.inf, -np.inf], np.nan)
games = games.dropna(subset = ['favored_odds', 'underdog_odds', 'hand_match', 'entry_match'] + features_names).reset_index(drop = True)

# One Hot Encoding
enc = OneHotEncoder()
dummies = pd.DataFrame(enc.fit_transform(games[['tourney_series', 'surface', 'round', 'best_of', 'hand_match', 'entry_match', 'inactive_match']]).toarray(),
                    columns = enc.get_feature_names_out(['tourney_series', 'surface', 'round', 'best_of', 'hand_match', 'entry_match', 'inactive_match']))
dummies.drop(columns = [dummies.filter(like='series_').columns[-1]] + [dummies.filter(like='surface_').columns[-1]] + [dummies.filter(like='round_').columns[-1]] + [dummies.filter(like='best_of_').columns[-1]] +
            [dummies.filter(like='hand_match_').columns[-1]] + [dummies.filter(like='entry_match_').columns[-1]] + [dummies.filter(like='inactive_match_').columns[-1]], axis = 1, inplace = True)
dummies_features = dummies.columns.tolist()
games = pd.concat([games, dummies], axis=1)


"""
# Randomly shuffle who is favored and underdog
np.random.seed(42) 
shuffle_mask = np.random.rand(len(games)) < 0.5
games['shuffle_mask'] = shuffle_mask

for col in games.columns:
    if 'favored' in col:
        counterpart_col = col.replace('favored', 'underdog')
        if counterpart_col in games.columns:
            print((col, counterpart_col))
            games.loc[shuffle_mask, [col, counterpart_col]] = games.loc[shuffle_mask, [counterpart_col, col]].values
        elif col == 'favored_win':
            games[col] = np.where(games['shuffle_mask'] == True, np.where(games[col] == 1, 0, 1), games[col])
    elif col.endswith('_diff'):
        games[col] = np.where(games['shuffle_mask'] == True, -games[col], games[col])
"""

X = games[features_names+dummies_features]
y = games['favored_win']
odds = games[['favored', 'underdog', 'season', 'tourney_full_name', 'tourney_date', 'round', 'comment', 'favored_win', 'favored_odds', 'underdog_odds', 'favored_max_odds', 'underdog_max_odds', 'favored_prob_implied', 'underdog_prob_implied']].copy()

train_index = games[(games['season'] >= 2009) & (games['season'] < 2018)].index
val_index = games[(games['season'] >= 2018) & (games['season'] < 2020)].index
train_val_index = games[(games['season'] >= 2009) & (games['season'] < 2020)].index
test_index = games[games['season'] >= 2020].index
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

#uncertainty_threshold = X_train_val['uncertainty'].quantile(0.9)
#X_train_val[(X_train_val['uncertainty'] <= uncertainty_threshold) & (odds_train_val['comment'] == 'Completed')].index

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

X_train_val = X_train_val.loc[indexes]
y_train_val = y_train_val.loc[indexes]
odds_train_val = odds_train_val.loc[indexes]

"""
def create_interaction_features(X):
    # Interaction features
    interaction_features = pd.DataFrame(index=X.index)
    for i, feature1 in enumerate(X.columns):
        for feature2 in X.columns[i+1:]:
            interaction_features[f"{feature1}_x_{feature2}"] = X[feature1] * X[feature2]
    X = pd.concat([X, interaction_features], axis=1)
    
    # Squared features
    #squared_features = pd.DataFrame(index=X.index)
    #for feature in X.columns:
    #    squared_features[f"{feature}_squared"] = X[feature] ** 2
    #X = pd.concat([X, squared_features], axis=1)
    return X

X_train = create_interaction_features(X_train)
X_val = create_interaction_features(X_val)
X_train_val = create_interaction_features(X_train_val)
X_test = create_interaction_features(X_test)
"""

# Scaling
features_to_scale = [feature for feature in X_train_val.columns if X_train_val[feature].nunique() > 2]
features_not_to_scale = [feature for feature in X_train_val.columns if feature not in features_to_scale]

scaler = StandardScaler()
scaler.fit(X_train_val[features_to_scale])

X_train_scaled = pd.concat([X_train[features_not_to_scale], pd.DataFrame(scaler.transform(X_train[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_train.index)], axis = 1)
X_val_scaled = pd.concat([X_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_val[features_to_scale]), columns = X_val[features_to_scale].columns, index = X_val.index)], axis = 1)
X_train_val_scaled = pd.concat([X_train_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_train_val[features_to_scale]), columns = X_train_val[features_to_scale].columns, index = X_train_val.index)], axis = 1)
X_test_scaled = pd.concat([X_test[features_not_to_scale], pd.DataFrame(scaler.transform(X_test[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_test.index)], axis = 1)


#--------------------------------------------------------------------------- Elastic Net ---------------------------------------------------------------------------#
# Define the search space for Bayesian optimization
search_space = {
    'l1_ratio': (0.1, 1.0, 'uniform'),
    'C': (0.1, 5, 'log-uniform')
}


opt = BayesSearchCV(
    estimator=LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        random_state=42,
        fit_intercept=True,
        max_iter=1000
    ),
    search_spaces=search_space,
    n_iter=100,
    scoring='neg_log_loss',
    cv=5,
    random_state=42,
    n_jobs=-1
)
opt.fit(X_train_val_scaled, y_train_val)

# Extract the best parameters
best_params = opt.best_params_
print(f"Best parameters: {best_params}")
elastic_net = opt.best_estimator_

odds_train_val['favored_prob_enet'] = elastic_net.predict_proba(X_train_val_scaled)[:, 1]
odds_test['favored_prob_enet'] = elastic_net.predict_proba(X_test_scaled)[:, 1]
odds_train_val['underdog_prob_enet'] = elastic_net.predict_proba(X_train_val_scaled)[:, 0]
odds_test['underdog_prob_enet'] = elastic_net.predict_proba(X_test_scaled)[:, 0]
print(f"Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_enet']):.5f}")
print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_enet']):.5f}")

coefficients = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': elastic_net.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print(coefficients)
selected_features = list(coefficients[coefficients['Coefficient'] != 0]['Feature'])


# Log loss
log_loss_test_results = pd.DataFrame()
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Odds', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_implied"])]
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Logit', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_enet"])]


#--------------------------------------------------------------------------- Random Forest ---------------------------------------------------------------------------#
# Define the search space for Bayesian optimization
search_space = {
    'n_estimators': (50, 500),
    'max_depth': (1, 40),
    'min_samples_split': (2, 40),
    'min_samples_leaf': (1, 40)
}

# Perform Bayesian optimization
opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    search_spaces=search_space,
    n_iter=30,
    scoring='neg_log_loss',
    cv=3,
    random_state=42,
    n_jobs=-1
)
opt.fit(X_train_val_scaled[selected_features], y_train_val)

# Extract the best parameters
best_params = opt.best_params_
print(f"Best parameters: {best_params}")
rf_model = opt.best_estimator

# Predict probabilities
odds_train_val['favored_prob_rf'] = rf_model.predict_proba(X_train_val_scaled[selected_features])[:, 1]
odds_train_val['underdog_prob_rf'] = 1 - odds_train_val['favored_prob_rf']
odds_test['favored_prob_rf'] = rf_model.predict_proba(X_test_scaled[selected_features])[:, 1]
odds_test['underdog_prob_rf'] = 1 - odds_test['favored_prob_rf']

# Evaluate the model
print(f"Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_rf']):.5f}")
print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_rf']):.5f}")

# Log loss
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Random Forest', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_rf"])]

#<-------------------------------------- XGBoost ------------------------------------------------
from xgboost import XGBClassifier
search_space = {
    'n_estimators': (50,1000),    # Number of trees
    'max_depth': (2,10),            # Maximum depth of trees
    'learning_rate': (0.001, 0.1), # Step size shrinkage
    'subsample': (0.7, 1.0),           # Fraction of samples to grow trees
    'colsample_bytree': (0.7, 1.0)    
}

# Perform Bayesian optimization
opt = BayesSearchCV(
    estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    search_spaces=search_space,
    n_iter=30,
    scoring='neg_log_loss',
    cv=3,
    random_state=42,
    n_jobs=-1
)
opt.fit(X_train_val_scaled[selected_features], y_train_val)


# Extract the best parameters
best_params = opt.best_params_
print(f"Best parameters: {best_params}")
xgb_model = opt.best_estimator_

odds_train_val['favored_prob_xgb'] = xgb_model.predict_proba(X_train_val_scaled[selected_features])[:, 1]
odds_train_val['underdog_prob_xgb'] = 1 - odds_train_val['favored_prob_xgb']
odds_test['favored_prob_xgb'] = xgb_model.predict_proba(X_test_scaled[selected_features])[:, 1]
odds_test['underdog_prob_xgb'] = 1 - odds_test['favored_prob_xgb']

print(f"Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['favored_prob_xgb']):.5f}")
print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_xgb']):.5f}")

log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['XGBoost', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_xgb"])]


#<--------------------------------------------------------------------------------------------------------------------------------------------
# Stacking
from mlxtend.classifier import StackingCVClassifier

clf1 = LogisticRegression(penalty='elasticnet',
                        l1_ratio=elastic_net.l1_ratio,
                        C=elastic_net.C,
                        solver='saga',
                        random_state=42,
                        max_iter=10_0000)
clf2 = RandomForestClassifier(
        n_estimators=rf_model.n_estimators,
        max_depth=rf_model.max_depth,
        min_samples_split=rf_model.min_samples_split,
        min_samples_leaf=rf_model.min_samples_leaf,
        random_state=42)
clf3 = XGBClassifier(n_estimators=xgb_model.n_estimators,   
    max_depth=xgb_model.max_depth,           
    learning_rate=xgb_model.learning_rate, 
    subsample=xgb_model.subsample,           
    colsample_bytree=xgb_model.colsample_bytree,
    random_state=42)
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                            meta_classifier=lr,
                            use_probas=True,
                            cv=5,
                            random_state=42)
sclf.fit(X_train_val[selected_features], y_train_val)

odds_train_val['favored_prob_meta'] = sclf.predict_proba(X_train_val[selected_features])[:,1]
odds_train_val['underdog_prob_meta'] = 1-odds_train_val['favored_prob_meta']

odds_test['favored_prob_meta'] = sclf.predict_proba(X_test[selected_features])[:,1]
odds_test['underdog_prob_meta'] = 1-odds_test['favored_prob_meta']

print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_enet']):.5f}")
print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_rf']):.5f}")
print(f"Test Log Loss (Meta Model): {log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_meta"]):.5f}")

#<--------------------------------------------------------------------------------------------------------------------------------------------
# Stacking
from scipy.optimize import minimize

selected_models = ['enet', 'rf', 'nn', 'svm', 'implied']
X_train_val_meta = odds_train_val[['favored_prob_' + model for model in selected_models]].values
y_train_val_meta = odds_train_val['favored_win'].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store out-of-fold predictions
oof_preds = np.zeros_like(y_train_val_meta, dtype=float)
for train_idx, val_idx in kf.split(X_train_val_meta):
    X_train_meta, X_val_meta = X_train_val_meta[train_idx], X_train_val_meta[val_idx]
    y_train_meta = y_train_val_meta[train_idx]
    
    # Objective: minimize log loss on validation
    def objective(weights):
        preds = X_train_meta @ weights
        preds = 1 / (1 + np.exp(-preds))
        return log_loss(y_train_meta, preds)
    
    # Initial weights and constraints
    init_weights = np.zeros(X_train_val_meta.shape[1])
    result = minimize(objective, init_weights, method='BFGS')
    fold_weights = result.x
    
    # Predict on validation fold
    val_logits = X_val_meta @ fold_weights
    val_preds = 1 / (1 + np.exp(-val_logits))
    oof_preds[val_idx] = val_preds
    print(oof_preds)

# Store stacked predictions
odds_train_val['favored_prob_meta'] = oof_preds

print(f"Test Log Loss (Meta Model): {log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_meta"]):.5f}")

#<--------------------------------------------------------------------------------------------------------------------------------------------
# Stacking
selected_models = ['enet', 'rf', 'nn', 'svm', 'implied']
weight_model = LogisticRegression(fit_intercept=False, solver='lbfgs', max_iter=10_000)
weight_model.fit(odds_train_val[[f'favored_prob_{model}' for model in selected_models]], y_train_val)
odds_test['favored_prob_meta'] = weight_model.predict_proba(odds_test[[f'favored_prob_{model}' for model in selected_models]])[:, 1]


weights = weight_model.coef_[0]
weights /= weights.sum()
weights = dict(zip(selected_models, weights))
print(f"Model Weights: {weights}")

odds_test['favored_prob_meta'] = weight_model.predict_proba(odds_test[[f'favored_prob_{model}' for model in selected_models]])[:, 1]

# Calculate log loss for the test set
print(f"Test Log Loss (Meta Model): {log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_meta"]):.5f}")

# Add the result to the log loss test results
log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Meta Model', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_meta"])]

#<--------------------------------------------------------------------------------------------------------------------------------------------
# Stacking
over_probs = np.column_stack([
    odds_train_val['favored_prob_enet'],
    odds_train_val['favored_prob_rf'],
    odds_train_val['favored_prob_nn'],
    odds_train_val['favored_prob_svm'],
    odds_train_val['favored_prob_implied']
])

under_probs = np.column_stack([
    odds_train_val['underdog_prob_enet'],
    odds_train_val['underdog_prob_rf'],
    odds_train_val['underdog_prob_nn'],
    odds_train_val['underdog_prob_svm'],
    odds_train_val['underdog_prob_implied']
])

df_favored = pd.DataFrame({'prop_id':list(range(1,len(y_train)+1)), 'bet_type':'favored', 'result':y_train})
df_underdog = pd.DataFrame({'prop_id':list(range(1,len(y_train)+1)), 'bet_type':'underdog', 'result':y_train})
selected_models = ['enet', 'rf', 'nn', 'svm', 'implied']
for model_name in selected_models:
    df_favored['probs_'+model_name] = odds_train_val['favored_prob_'+model_name]
    df_underdog['probs_'+model_name] = odds_train_val['underdog_prob_'+model_name]

df = pd.concat([df_favored, df_underdog], ignore_index=True, axis = 0).sort_values(by = ['prop_id', 'bet_type']).reset_index(drop=True)
df['y'] = np.where(df['bet_type'] == 'favored', np.where(df['result'] == 1, 1,0), np.where(df['result'] == 0, 1,0))

cl_model = ConditionalLogit(endog = df['y'], exog = df[['probs_'+model_name for model_name in selected_models]], groups = df['prop_id'])
cl_fit = cl_model.fit()
print(cl_fit.summary())
beta = cl_fit.params

over_probs = np.column_stack([
    odds_test['favored_prob_enet'],
    odds_test['favored_prob_rf'],
    odds_test['favored_prob_nn'],
    odds_test['favored_prob_svm'],
    odds_test['favored_prob_implied']
])

under_probs = np.column_stack([
    odds_test['underdog_prob_enet'],
    odds_test['underdog_prob_rf'],
    odds_test['underdog_prob_nn'],
    odds_test['underdog_prob_svm'],
    odds_test['underdog_prob_implied']
])

norm_beta = beta/beta.sum()

num = np.exp(np.dot(over_probs, beta))
den = np.exp(np.dot(over_probs, beta)) + np.exp(np.dot(under_probs, beta)) 
predicted_probs_over = num/den

odds_test[['underdog_prob_implied', 'favored_prob_meta']] = np.vstack((1-predicted_probs_over, predicted_probs_over)).T

log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = ['Logit & Odds', log_loss(odds_test["favored_win"].astype(int), odds_test["favored_prob_meta"])]




#<--------------------------------------------------------------------------------------------------------------------------------------------
# Pinnacle Odds and ML Model
model = 'enet'
cl_fit = fit_second_stage(odds_train_val['favored_win'], odds_train_val[f'underdog_prob_{model}'], odds_train_val[f'favored_prob_{model}'], odds_train_val['underdog_prob_implied'], odds_train_val['favored_prob_implied'])
print(cl_fit.summary())

train_second_stage_probs = predict_second_stage(cl_fit.params, odds_train_val[f'underdog_prob_{model}'], odds_train_val[f'favored_prob_{model}'], odds_train_val['underdog_prob_implied'], odds_train_val['favored_prob_implied'])
odds_train_val['2st_underdog_prob'], odds_train_val['2st_favored_prob'] = train_second_stage_probs[:,0], train_second_stage_probs[:,1]

test_second_stage_probs = predict_second_stage(cl_fit.params, odds_test[f'underdog_prob_{model}'], odds_test[f'favored_prob_{model}'], odds_test['underdog_prob_implied'], odds_test['favored_prob_implied'])
odds_test['2st_underdog_prob'], odds_test['2st_favored_prob'] = test_second_stage_probs[:,0], test_second_stage_probs[:,1]

print(f"Train Log Loss: {log_loss(odds_train_val['favored_win'], odds_train_val['2st_favored_prob']):.5f}")
print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['2st_favored_prob']):.5f}")

log_loss_test_results.loc[len(log_loss_test_results), ['model', 'log_loss']] = [f'{model} & Odds', log_loss(odds_test["favored_win"].astype(int), odds_test["2st_favored_prob"])]


# Betting
model = 'enet'
#odds_test = full_odds_test.copy()
expected_return_favored = (odds_test['favored_max_odds']-1)*odds_test[f'favored_prob_{model}'] - 1*(1-odds_test[f'favored_prob_{model}'])
expected_return_underdog = (odds_test['underdog_max_odds']-1)*odds_test[f'underdog_prob_{model}'] - 1*(1-odds_test[f'underdog_prob_{model}'])
odds_test['bet'] = np.where(np.maximum(expected_return_favored, expected_return_underdog) < 0, 'no_bet', np.where(expected_return_favored > expected_return_underdog, 'favored', 'underdog'))
odds_test['expected_return'] = np.where(odds_test['bet'] == 'no_bet', 0, np.where(expected_return_favored > expected_return_underdog, expected_return_favored, expected_return_underdog))

odds_test['probs'] = np.where(odds_test['bet'] == 'favored', odds_test[f'favored_prob_{model}'], odds_test[f'underdog_prob_{model}'])
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

thresholds = list(np.arange(0, 0.2, 0.01))
results_by_threshold = []
for index, threshold in enumerate(thresholds):
    full_bets_threshold = odds_test[odds_test['expected_return'] > threshold].copy()
    
    results = {'threshold':f'{threshold*100:.1f}%', 'bets_placed':(full_bets_threshold['bet'] != 'No Bet').sum(), 'wins':(pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce') > 0).sum(),
            'hit_rate':(pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce') > 0).sum()/(full_bets_threshold['bet'] != 'No Bet').sum(),
            'earnings':pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce').fillna(0).sum(), 'spent':(full_bets_threshold['kelly_fraction_fixed']).sum(),
            'expected_return':f'{pd.to_numeric(full_bets_threshold['expected_return'], errors='coerce').mean()*100:.2f}%',
            'return':f'{pd.to_numeric(full_bets_threshold['result_kelly_fixed'], errors='coerce').fillna(0).sum()/(full_bets_threshold['kelly_fraction_fixed']).sum()*100:.2f}%'}
    results_by_threshold.append(results)
print(pd.DataFrame(results_by_threshold))


bets_results = odds_test[odds_test['expected_return'] > 0.1].copy()
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

