import pandas as pd
from tqdm import tqdm
import re
pd.set_option('display.max_rows',600)

# Stats data
seasons = list(range(1978, 2026))
games = pd.DataFrame()
for season in seasons:
    if season != 2025:
        df = pd.concat([pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv'),pd.read_csv(f'tennis_atp-master/atp_matches_qual_chall_{season}.csv')],axis = 0).reset_index(drop=True)
    else:
        df = pd.read_csv(f'tennis_atp-master/atp_matches_{season}.csv')
        df.insert(4, 'match_num', 301-(df.groupby('tourney_name').cumcount() + 1))
        
    df.insert(0, 'season', season)
    df.loc[df['tourney_name'].str.contains(' Olympics', na=False), 'tourney_level'] = 'O'
    
    games = pd.concat([games, df], axis = 0).reset_index(drop=True)

different_names_dict = {'Edouard Roger-Vasselin':'Edouard Roger Vasselin'}
games['winner_name'].replace(different_names_dict, inplace=True)
games['loser_name'].replace(different_names_dict, inplace=True)

games['tourney_date'] = pd.to_datetime(games['tourney_date'], format = '%Y%m%d')
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games = games[(~games['tourney_name'].isin(['Laver Cup']))].sort_values(by=['tourney_date', 'tourney_name', 'round']).reset_index(drop=True)

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
    elif row['winner_name'] == 'Sonchat Ratiwatana':
        games.loc[index, 'winner'] = 'Ratiwatana So.'
    elif row['winner_name'] == 'Sanchai Ratiwatana':
        games.loc[index, 'winner'] = 'Ratiwatana Sa.'
    elif row['winner_name'] == 'Petros Tsitsipas':
        games.loc[index, 'winner'] = 'Tsitsipas Pe.'
    elif row['winner_name'] == 'Pavlos Tsitsipas':
        games.loc[index, 'winner'] = 'Tsitsipas Pa.'
    elif row['winner_name'] == 'Darwin Blanch':
        games.loc[index, 'winner'] = 'Blanch Dar.'
    elif row['winner_name'] == 'Dali Blanch':
        games.loc[index, 'winner'] = 'Blanch Dal.'
    elif row['winner_name'] == 'Martin Redlicki':
        games.loc[index, 'winner'] = 'Redlicki Ma.'
    elif row['winner_name'] == 'Michael Redlicki':
        games.loc[index, 'winner'] = 'Redlicki Mi.'
        
    if row['loser_name'] == 'Alex Kuznetsov':
        games.loc[index, 'loser'] = 'Kuznetsov Al.'
    elif row['loser_name'] == 'Andrey Kuznetsov':
        games.loc[index, 'loser'] = 'Kuznetsov An.'
    elif row['loser_name'] == 'Ze Zhang':
        games.loc[index, 'loser'] = 'Zhang Ze.'
    elif row['loser_name'] == 'Zhizhen Zhang':
        games.loc[index, 'loser'] = 'Zhang Zh.'
    elif row['loser_name'] == 'Sonchat Ratiwatana':
        games.loc[index, 'loser'] = 'Ratiwatana So.'
    elif row['loser_name'] == 'Sanchai Ratiwatana':
        games.loc[index, 'loser'] = 'Ratiwatana Sa.'
    elif row['loser_name'] == 'Petros Tsitsipas':
        games.loc[index, 'loser'] = 'Tsitsipas Pe.'
    elif row['loser_name'] == 'Pavlos Tsitsipas':
        games.loc[index, 'loser'] = 'Tsitsipas Pa.'
    elif row['loser_name'] == 'Darwin Blanch':
        games.loc[index, 'loser'] = 'Blanch Dar.'
    elif row['loser_name'] == 'Dali Blanch':
        games.loc[index, 'loser'] = 'Blanch Dal.'
    elif row['loser_name'] == 'Martin Redlicki':
        games.loc[index, 'loser'] = 'Redlicki Ma.'
    elif row['loser_name'] == 'Michael Redlicki':
        games.loc[index, 'loser'] = 'Redlicki Mi.'

tournaments = pd.read_csv('tournaments_by_season_oddsportal.csv')
tournaments.rename(columns={'tournament_stats':'tourney_name', 'series':'tourney_series', 'country_code':'tourney_ioc'}, inplace=True)
tournaments.loc[tournaments['tourney_name'] == 'Tokyo Olympics', 'season'] = 2021


games = pd.merge(games, tournaments[['tourney_name', 'season', 'tourney_series', 'tourney_ioc', 'surface_speed']], on=['tourney_name', 'season'], how='left')
tourney_series_col = games.pop('tourney_series')
games.insert(games.columns.get_loc('surface'), 'tourney_series', tourney_series_col)
tourney_ioc_col = games.pop('tourney_ioc')
games.insert(games.columns.get_loc('surface'), 'tourney_ioc', tourney_ioc_col)
surface_speed_col = games.pop('surface_speed')
games.insert(games.columns.get_loc('surface'), 'surface_speed', surface_speed_col)


# Odds data
games_odds = pd.read_csv('games_oddsportal.csv')

games_odds.rename(columns={'tournament_stats':'tourney_name'}, inplace=True)
games_odds = games_odds[(~games_odds['comment'].isin(['canc.', 'w.o.', 'award.', 'ret.']))].reset_index(drop=True)

games_odds[(games_odds['tourney_name'] == 'Australian Open') & (games_odds['season'] == 2021)][['winner', 'loser', 'season', 'tourney_name']]
games_odds[(games_odds['tourney_name'] == 'Doha Aus Open Qualies') & (games_odds['season'] == 2021)][['winner', 'loser', 'season', 'tourney_name']]
indexes_to_drop = list(games_odds[(games_odds['tourney_name'] == 'Doha Aus Open Qualies') & (games_odds.index < 16961)].index) + list(games_odds[(games_odds['tourney_name'] == 'Australian Open') & (games_odds['season'] == 2021) & (games_odds.index >= 5529)].index)
games_odds.drop(index = indexes_to_drop, inplace=True)

games_odds.loc[games_odds['tourney_name'] == 'Tokyo Olympics', 'season'] = 2021

def extract_zhang_full_name(url):
    match = re.search(r'zhang-([a-z]+)', url)
    if match:
        # Capitalize both parts
        return f"Zhang {match.group(1).capitalize()}"
    return None
games_odds.loc[games_odds[games_odds['winner'] == 'Zhang Z.'].index, 'winner'] = games_odds[games_odds['winner'] == 'Zhang Z.']['game_url'].apply(extract_zhang_full_name).replace({'Zhang Zhizhen':'Zhang Zh.', 'Zhang Ze':'Zhang Ze.'})
games_odds.loc[games_odds[games_odds['loser'] == 'Zhang Z.'].index, 'loser'] = games_odds[games_odds['loser'] == 'Zhang Z.']['game_url'].apply(extract_zhang_full_name).replace({'Zhang Zhizhen':'Zhang Zh.', 'Zhang Ze':'Zhang Ze.'})
games_odds.loc[games_odds[games_odds['player1'] == 'Zhang Z.'].index, 'player1'] = games_odds[games_odds['player1'] == 'Zhang Z.']['game_url'].apply(extract_zhang_full_name).replace({'Zhang Zhizhen':'Zhang Zh.', 'Zhang Ze':'Zhang Ze.'})
games_odds.loc[games_odds[games_odds['player2'] == 'Zhang Z.'].index, 'player2'] = games_odds[games_odds['player2'] == 'Zhang Z.']['game_url'].apply(extract_zhang_full_name).replace({'Zhang Zhizhen':'Zhang Zh.', 'Zhang Ze':'Zhang Ze.'})


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
'Yoon Y.':'Il Yoon Y.', 'Youzhny A.':'Youzhny M.', 'Youzhny M. ':'Youzhny M.', 'Zeng S.X.':'Xuan Zeng S.', 'Zhu B.Q.':'Qiang Zhu B.', 'Zverev A. ':'Zverev A.',
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
'van der Meer N.':'Van Der Meer N.',
'Londero J. I.':'Ignacio Londero J.', 'Chela J. I.':'Ignacio Chela J.', 'Hernandez Perez O.':'Hernandez O.', 'Ferrero J. C.':'Carlos Ferrero J.', 'Ruiz-Rosales A.':'Ruiz Rosales A.',
'Cabal J. S.':'Sebastian Cabal J.', 'Duclos P.':'Ludovic Duclos P.', 'Nunez M. A.':'Aurelio Nunez M.', 'Diaz-Barriga L.':'Diaz Barriga L.', 'Gallardo-Valles M.':'Gallardo Valles M.', 'Diaz-Figueroa C.':'Diaz Figueroa C.',
'Reyes-Varela M. A.':'Angel Reyes Varela M.', 'Peralta-Tello E.':'Angel Reyes Varela M.', 'Smith J. P.': 'Patrick Smith J.', 'Carpenter K. J.':'Jack Carpenter K.', 'Rubio Fierros A. F.':'Fernando Rubio Fierros A.',
'Cid Subervi R.':'Cid R.', 'Galan D. E.':'Elahi Galan D.', 'Kwon S.':'Woo Kwon S.', 'Stebe C.':'Marcel Stebe C.', 'Justo G. I.':'Ivan Justo G.', 'Statham J. R.':'Rubin Statham J.', 'Gomez F. A.':'Agustin Gomez F.',
'Romios M. C.':'Christopher Romios M.', 'Fancutt T. J.':'Fancutt T.', 'McCabe J.':'Mccabe J.', 'Etcheverry T. M.': 'Martin Etcheverry T.', 'Bautista-Agut R.':'Bautista Agut R.',
'Cerundolo J. M.':'Manuel Cerundolo J.', 'Huesler M.':'Andrea Huesler M.', 'Silva D. R.':'Dutra Silva R.', 'Iakovlev A.':'Yakovlev A.', 'Lee D. H.':'Hee Lee D.', 'Sasikumar M.':'Kumar Mukund S.',
'Turker M. N.':'Naci Turker M.', 'Tsonga J-W.':'Tsonga J.', 'Ionel N. D.':'David Ionel N.', 'Bailly G. A.': 'Arnaud Bailly G.', 'van Assche L.':'Van Assche L.', 'van Rijthoven T.':'Van Rijthoven T.',
'Jianu F. C.':'Cristian Jianu F.', 'Schuttler R.':'Schuettler R.', 'Bogomolov A. Jr':'Bogomolov Jr A.', "O'Brien D.":'Obrien D.', 'Spir J.':'Carlos Spir J.', 'Aubone J.':'Yves Aubone J.',
'Huey T.':'Conrad Huey T.', 'Wattanakul P.':'Wattanakul S.', 'Statham M.':'Oliver Statham M.', 'MacKenzie L.':'Mackenzie L.', 'Yang T.':'Hua Yang T.', 'McLachlan B.':'Mclachlan B.',
'Huang L. C.':'Chi Huang L.', 'Woo C. H.':'Hyo Woo C.', 'Lee H.':'Taik Lee H.', 'Im K.':'Tae Im K.', 'Clar-Rossello P.':'Clar Rossello P.', 'Checa-Calvo J.':'Checa Calvo J.', 'Poch-Gradin C.':'Poch Gradin C.', 'Huta-Galung J.':'Huta Galung J.',
'Brezac C.':'Antoine Brezac C.', 'Alcaide-Justell G.':'Alcaide G.', 'Matsukevich D.':'Matsukevitch D.', 'Brzezicki J. P.':'Pablo Brzezicki J.', 'Podlipnik-Castillo H.':'Podlipnik Castillo H.',
'Campozano J.':'Cesar Campozano J.', 'McGee J.':'Mcgee J.', 'Ojeda L. R.':'Ojeda Lara R.', 'Moroni M.':'Marco Moroni G.', 'Vilella M. M.':'Vilella Martinez M.', 'Wu Tung-Lin':'Lin Wu T.', 
'Kwiatkowski T. S.':'Son Kwiatkowski T.', 'Ficovich J. P.':'Pablo Ficovich J.', 'Nam Ji S.':'Sung Nam J.', 'Tirante T. A.':'Agustin Tirante T.', 'Hsu Y. H.':'Hsiou Hsu Y.', 'Olivieri G. A.':'Alberto Olivieri G.',
'Burruchaga R. A.':'Andres Burruchaga R.', 'Trotter J. K.':'Trotter J.', 'Dellien Velasco M. A.':'Alejandro Dellien Velasco M.', 'Lertchai K.-P.':'Pop Lertchai K.', 'Isaro P.':'Isarow P.',
'Rojer J.':'Julien Rojer J.', 'Kaewsuto Ch.':'Kaewsuto C.', 'Jeong S.':'Young Jeong S.', 'Hung J.':'Chen Hung J.', 'Iam La-Or W.':'Iam La Or W.', 'Coll-Riudavets I.':'Coll Riudavets I.', 'Sabate-Bretos O.':'Sabate Bretos O.',
'Toledo B. P.':'Toledo Bague P.', 'Lindell Ch.':'Lindell C.', 'Boe-Wiegaard W.':'Boe Wiegaard W.', 'Photos K.':'Kallias P.', 'Maqdes A.':'Maqdas A.', 'Madaras N.':'Nicolae Madaras D.',
'Galarza J. I.':'Ignacio Galarza J.', 'Gutierrez O. J.':'Jose Gutierrez O.', 'Gong M.':'Xin Gong M.', 'Luncanu P.':'Alexandru Luncanu P.', 'Ruiz P. P.':'Pablo Ruiz P.', 'Ruiz Naranjo M. A.':'Andres Ruiz Naranjo M.',
'Carvajal Torres J. F.':'Fernando Carvajal Torres J.', 'Gomez J. S.':'Sebastian Gomez J.', 'Mateus Rubiano J.':'Jose Mateus Rubiano J.', 'Pabon Cuesta R.':'Pabon R.', 'Felix J. F.':'Francisco Felix J.',
'Bendeck J. D.':'Daniel Bendeck J.', 'Erazo Rodriguez C.':'Andres Erazo C.', 'Craciun T.':'Dacian Craciun T.', 'Damian M. D.':'Daniel Damian M.', 'Dancescu A.':'Marin Dancescu A.', 'Marasin A. C.':'Catalin Marasin A.',
'Gavrila L.':'Ady Gavrila L.', 'Anagnastopol V.':'Mugurel Anagnastopol V.', 'Ghilea V. A.':'Alexandru Ghilea V.', 'Marin T. N.':'Nicolae Marin T.', 'Apostol A. S.':'Stefan Apostol A.',
'Carpen A.':'Daniel Carpen A.', 'Marian Ch.':'Marian C.', 'Cornea V. V.':'Victor Cornea V.', 'Apostol B. I.':'Ionut Apostol B.', 'Tatomir L. G.':'George Tatomir L.', 'Andreescu S. A.':'Adrian Andreescu S.',
'Amado J.':'Pablo Amado J.', 'Aranguren J.':'Martin Aranguren J.', 'Paz J. P.':'Pablo Paz J.', 'Iliev J. I.':'Ignacio Iliev J.', 'Kim Y. S.':'Seok Kim Y.', 'Matute J. M.':'Manuel Matute J.',
'Otegui J. B.':'Bautista Otegui J.', 'Torres J. B.':'Bautista Torres J.', 'El M. A.':'El Mihdawy A.', 'Lin J. M.':'Mingjie Lin J.', 'Aguilar J. C. M.':'Carlos Manuel Aguilar J.', 'Mazon-Hernandez R.':'Mazon Hernandez R.',
'Tamimount Y.':'Taimimount Y.', 'Kraimi M. A.':'Ali Kraimi M.', 'Zhong S. H.':'Hao Zhong S.', 'Hong S. C.':'Chan Hong S.', 'Jayaprakash M. M.':'Mayur Jayaprakash M.', 'Balaji N. S.':'Sriram Balaji N.',
'Virali-Murugesan R.':'Virali Murugesan R.', 'Kaza V. S.':'Vinayak Sharma K.', 'Manoah R.':'Robinson Manoah R.', 'Sood Ch.':'Sood C.', 'Ravi R.':'Raswant R.', 'Prabodh S. R.':'R Prabodh S.',
'McCarthy D.':'Mccarthy D.', 'McNally J.':'Mcnally J.', 'Angele J. F.':'Floyd Angele J.', 'Descotte M. F.':'Franco Descotte M.', 'Trombetta T. D.':'Trombetta T.', 'Nunez A. A.':'Alan Nunez Aguilera',
'El Harib A.':'Al Harib A.', 'Al-Saygh A.':'Al Saygh A.', 'Al-Jufairi A.':'Al Jufairi A.', 'Van der Duim A.':'Van Der Duim A.', 'Janahi H. A.':'Abbas Janahi H.', 'De Valk R.':'Sarut De Valk R.',
'Maamoun K.M.':'Mohamed Maamoun K.', 'Al Allaf K.':'Allaf K.', "O'Mara J.": 'Omara J.', 'Thornton-Brown A.':'Thornton Brown A.', 'Sterland-Markovic S.':'Sterland Markovic S.', 'Cunha-Silva F.':'Cunha Silva F.',
'Pereira R.':'Periera R.', 'Munoz-Abreu J.':'Munoz Abreu J.', 'Olguin N. D.':'Daniel Olguin N.', 'Esteve L. E.':'Esteve Lobato E.', 'Severino C. E.':'Eduardo Severino C.', 'Zaitcev A.':'Zaitsev A.',
'Brunken J. F.':'Frederik Brunken J.', 'Rehberg M. H.':'Hans Rehberg M.', 'Hildebrandt J. J.':'Jeremy Hildebrandt J.', 'Dedura D.':'Dedura Palomero D.', 'Wong H. K.':'Kit Jack Wong H.',
'Ng K. L.':'Lung Ng K.', 'Warren . D.':'Warren D.', 'Fruttero J.':'Paul Fruttero J.', 'De Armas Jo.':'De Armas J.', 'Weissborn S.':'Samuel Weissborn T.', 'Vibert F.':'Arthur Vibert F.',
'Hsieh C.':'Peng Hsieh C.', 'Schwaerzler J. J.':'Schwaerzler J.', 'Si Y.':'Ming Si Y.', 'Deen Heshaam A. E.':'Elyaas Deen Heshaam A.', 'Merzuki M. A.':'Assri Merzuki M.', 'Bin Zainal-Abidin M.':'Ashaari Bin Zinal Abidin M.',
'Syed Naguib A.':'Mohd Agil Syed Naguib S.', 'Koay H. S.':'Sheng Koay H.', 'Abdul Razak A. D.':'Deedat Abdul Razak A.', 'Kim C.':'Cheong Eui Kim', 'Lee Ch. O.':'Oliver Lee C.', 'Perez-Perez L.':'Antonio Perez Perez L.',
'Batalla Diez J. I.':'Ignacio Batalla Diez J.', 'Alvarez Valdes L. C.':'Carlos Alvarez L.', 'Panta J.':'Brian Panta Herreros J.',
'Cornut-Chauvinc A.':'Cornut Chauvinc A.', 'Ktiri M.':'Othman Ktiri M.', 'Nadal J.':'Nadal Vives J.', 'Wang T.':'Khunn Wang T.', 'Wang Ch.-F.':'Fu Wang C.', 'Grimal-Ferrer J.':'Marc Grimal Ferrer J.',
'Vernet J. P.':'Pascal Vernet J.', 'Sadlowski J. O.':'Sadlowski', 'Ghedjemis R.':'Roumane R.', 'Kodat T.':'Alex Kodat T.',
'Arenas-Gualda I.':'Arenas Gualda I.', 'Hu N. S.':'S Hu N.', 'Ayeni A.':'Alafia Damina Ayeni O.', 'Storrie R. J.':'James Storrie R.', 'Ivanov S. A.':'Anthony Ivanov S.',
'Reis Da Silva J.':'Lucas Reis Da Silva J.', 'Sorgi J. P.':'Pedro Sorgi J.', 'Tatlot J. S.':'Sebastien Tatlot J.', 'van Der Linden L.':'Van Der Linden L.', 'van der Lans F.':'Van Der Lans F.',
'Bakalov B.':'Nicola Bakalov B.', 'Saavedra-C.C.':'Saavedra Corvalan C.', 'Rivera-Aranguiz G.':'Rivera Aranguiz G.', 'Kauer N. G.':'Gustavo Kauer N.', 'Saez J. C.':'Carlos Saez J.',
'Nunez D. A.':'Antonio Nunez D.', 'Torres Fernandez B. I.':'Ignacio Torres Fernandez B.', 'Dias Marcos V.':'Vinicius Dias M.', 'Pereira V. H.':'Hugo Portugal Pereira V.', 'Lee J. M.':'Moon Lee J.',
'Zeng S.':'Xuan Zeng S.', 'Ma Y.':'Nan Ma Y.', 'Li Y. C.':'Cheng Li Y.', 'Ly Nam H.':'Hoang Ly N.', 'Zhu J. C.':'Cheng Zhu J.', 'Reyes N. A.':'Alexander Reyes N.', 'Dzhanashiya S.':'Dzhanashia S.',
'Uzhylovskyi V.':'Uzhylovsky V.', 'Shchaya-Zubrov P.':'Shchaya Zubrov P.', 'Elrich M. J.':'Jupiter Elrich M.', "D'Agord L.":'Dagord L.', 'van Overbeek J. R.':'Robert Van Overbeek J.',
'Salazar Martin J. A.':'Antonio Salazar Martin J.', 'Kocevar-Desman T.':'Kocevar Desman T.',
'Arce Garcia A.':'Arce A.', 'Walterscheid-Tukic N.':'Walterscheid Tukic N.', 'Zgombic F. Z.':'Zvonimir Zgombic F.', 'Garcia-Villanueva C.':'Garcia Villanueva C.',
'Haider-Maurer M.':'Haider Maurer M.', 'Shane J.':'S Shane J,', 'McHugh A.':'Mchugh A.', 'Crowley K.':'Patrick Crowley K.', 'Grant M.':'Grant Gd13 M.', 'Bloom S.': 'D Bloom S.',
'Montes-De La Torre I.':'Montes De La Torre I.', 'Mo Y. C.':'Cong Mo Y.', 'Gard C.':'Ionut Gard C.', 'Hernandez A.':'Alejandro Hernandez Serrano J.', 'Nunes J.': 'Ricardo Nunes J.'}

games_odds['winner'].replace(players_dict, inplace=True)
games_odds['loser'].replace(players_dict, inplace=True)
games_odds['player1'].replace(players_dict, inplace=True)
games_odds['player2'].replace(players_dict, inplace=True)

#print(odds[odds['loser'].isna()]) # ver para retiros
# Imput missing winners and losers
missing_winner_odds = games_odds[(games_odds['comment'] == 'ret.') | ((games_odds['tourney_name'] == 'Indian Wells Masters') & (games_odds['season'] == 2009)) | ((games_odds['tourney_name'] == 'Wimbledon') & (games_odds['season'] == 2009))].drop(columns = ['winner', 'loser'])
merged1 = missing_winner_odds.merge(
    games[['winner', 'loser', 'tourney_name', 'season']],
    left_on=['player1', 'player2', 'tourney_name', 'season'],
    right_on=['winner', 'loser', 'tourney_name', 'season'],
    how='inner'
)

# Second merge: player1 == loser, player2 == winner
merged2 = missing_winner_odds.merge(
    games[['winner', 'loser', 'tourney_name', 'season']],
    left_on=['player1', 'player2', 'tourney_name', 'season'],
    right_on=['loser', 'winner', 'tourney_name', 'season'],
    how='inner'
)
merged1['match_order'] = 'normal'
merged2['match_order'] = 'reversed'

combined = pd.concat([merged1, merged2], ignore_index=True)
for index, row in tqdm(combined.iterrows(), total = len(combined)):
    mask = (
        (games_odds['game_url'] == row['game_url']) & ((games_odds['comment'] == 'ret.') |
        ((games_odds['tourney_name'] == 'Indian Wells Masters') & (games_odds['season'] == 2009)) |
        ((games_odds['tourney_name'] == 'Wimbledon') & (games_odds['season'] == 2009)))
    )
    games_odds.loc[mask, ['winner', 'loser']] = row[['winner', 'loser']].values


#[player for player in list(games_odds['loser'].unique()) if player not in list(games['loser'].unique())]
full_df = pd.merge(games, games_odds, on = ['winner', 'loser', 'tourney_name', 'season'], how = 'left', indicator=False)
full_df.drop(columns=['odds1','odds2'], inplace=True)
#full_df[full_df['_merge'] == 'right_only'].sort_values(['tourney_name', 'season'])[['winner', 'loser', 'tourney_name', 'season']]

full_df.to_csv('games_merged.csv', index=False)




