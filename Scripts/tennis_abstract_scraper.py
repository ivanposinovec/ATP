import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from time import sleep
from random import random
from tqdm import tqdm
import re
import janitor
import numpy as np
from Scripts.functions import *

# Players
url = 'https://tennisabstract.com/reports/atpRankings.html'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tables = soup.find_all('table', id='reportable')
players = pd.read_html(StringIO(str(tables[0])))[0]

# Player games #
# Missing players
missing_players = ['Ki Lung Ng', 'Adam Taylor', 'Gillian Osmont', 'Adrian Arcon', 'Heremana Courte', 'Yuttana Charoenphon', 'Natthayut Nithithananont', 'Tim Van Rijthoven', 'Juan Ignacio Centurion Delvalle', 'Gonzalo Ariel Karakachian', 'Yuttana Charoenphon', 'Mick Veldheer', 'Dino Molokova Ferreira', 'Luca Sanchez', 'George Goldhoff', 'Gabriel Roveri Sidney', 'Keerthivassan Suresh', 'Ivan Liutarevich', 'Noah Lopez Cherubino', 'Dhruva Mulye', 'Yusuf Ebrahim Ahmed Abdulla Qaed', 'Elyas Abduljalil', 'Ivan Liutarevich', 'Marco Bortolotti', 'Cheik Pandzou Ekoume', 'Kryce Didier Momo Kassa', 'Nicholas Alan Van Aken', 'Yassine Smiej', 'Guelfo Borghini Baldovinetti', 'Paterne Mamata', 'Mubarak Shannan Zayid', 'Rafael Alfonso De Alba Valdes', 'Niki Kaliyanda Poonacha', 'Abdulrahman Al Janahi', 'Etienne Niyigena', 'Claude Ishimwe', 'Kelsey Stevenson', 'Joshua Muhire', 'Alexander Georg Mandma', 'Christos Glavas', 'Evangelos Kypriotis', 'Denis Istomin', 'Christos Glavas', 'Hendrik Jebens', 'Mark Wallner', 'Mathis Bondaz', 'Federico Gaston Gonzalez Benitez', 'Alex Santino Nunez Vera', 'Tennyson Whiting', 'Diego Bustamante', 'Sergio Ingles Garre', 'Eneko Rios Perez', 'Frane Nincevic', 'Noah Regas Luis', 'Breno Braga', 'Gabriel Roveri Sidney', 'Gustavo Albieri', 'Diego Eloy Mendez Montiel', 'Amine Jamji', 'Izan Corretja', 'Sergio Martos Gornes', 'Ivan Lopez Martos', 'Abdoulaziz Bationo', 'Dino Molokova Ferreira', 'Andre Rodeia', 'Azariah Rusher', 'Abdoulaziz Bationo', 'Zijiang Yang', 'Manuel Lazic', 'Pedro Pinto', 'Luis Guto Miguel', 'Vicente Freda', 'Ivan Liutarevich', 'Bruno Malacarne', 'Juan Esteban Trujillo Hernandez', 'Marcelo Demoliner', 'Dino Molokova Ferreira', 'Alaa Trifi', 'Omar Knani', 'Adam Nagoudi', 'Roko Horvat', 'Matei Todoran']
players = pd.concat([players, pd.DataFrame({'Player': missing_players})], axis = 0).reset_index(drop=True)
games_df = pd.DataFrame()
games_df = pd.read_csv('atp_matches_2025.csv')

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
scraped_players = list(games_df['Player'].unique()) if len(games_df) > 0 else []
for index, row in tqdm(players.iterrows(), total=len(players), desc='Getting games'):
    if row['Player'] not in scraped_players:
        try:
            player_url = f'https://www.tennisabstract.com/cgi-bin/player.cgi?p={"".join(row["Player"].split())}&f=A2025qq'
            driver.get(player_url)
            sleep(2 + random())
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            table = soup.find('table', id='matches')
            if table is not None:
                if len(table) > 0:
                    player_games_df = pd.read_html(StringIO(str(table)))[0]
                    player_games_df = player_games_df.iloc[:-1]
                    
                    for text in soup.stripped_strings:
                        if str(text).startswith('Plays:'):
                            hand = text
                            break
                    
                    player_games_df.insert(0, 'Player', row['Player'])
                    player_games_df.insert(1, 'Hand', hand.split(':')[1].split('(')[0].strip()[0])
                    player_games_df.insert(2, 'IOC', row['Country'])
                    player_games_df.insert(3, 'Birthdate', row['Birthdate'])
                    
                    # Return stats
                    wait = WebDriverWait(driver, 2)
                    wait.until(expected_conditions.element_to_be_clickable((By.XPATH, ".//*[@id='abovestats']/span[5]"))).click()
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    table = soup.find('table', id='matches')
                    player_games_return_df = pd.read_html(StringIO(str(table)))[0]
                    player_games_return_df = player_games_return_df.iloc[:-1]
                    player_games_df = pd.concat([player_games_df, player_games_return_df[['TPW', 'RPW', 'vA%', 'v1st%', 'v2nd%', 'BPCnv']]], axis = 1)
                    
                    # Raw stats
                    wait.until(expected_conditions.element_to_be_clickable((By.XPATH, ".//*[@id='abovestats']/span[6]"))).click()
                    
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    table = soup.find('table', id='matches')
                    player_games_raw_df = pd.read_html(StringIO(str(table)))[0]
                    player_games_raw_df = player_games_raw_df.iloc[:-1]
                    player_games_df = pd.concat([player_games_df, player_games_raw_df[['TP', 'Aces', 'DFs', 'SP', '1SP', '2SP', 'vA']]], axis = 1)
                games_df = pd.concat([games_df, player_games_df], ignore_index=True)
        except Exception as e:
            print(f'Error processing player {row["Player"]}: {e}')
        sleep(2 + random())
driver.quit()

games_df.to_csv('atp_matches_2025.csv', index=False)

games_df = pd.read_csv('atp_matches_2025.csv')
games_df['Date'] = pd.to_datetime(games_df['Date'].str.replace('‑', '-', regex=False), format='%d-%b-%Y')
games_df['Rd'] = pd.Categorical(games_df['Rd'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games_df = games_df[(games_df['Date'] < '2025-05-18') & (games_df['Score'] != 'Live Scores') & (~games_df['Tournament'].str.contains('Davis Cup|M15|M25', na=False))].sort_values(['Date', 'Tournament', 'Rd']).reset_index(drop=True)
games_df.drop_duplicates(subset = ['Tournament', 'Rd', 'Date', 'Unnamed: 6', 'Player'], inplace=True)
games_df['Tournament'].replace({'Buenos Aires Nautico Hacoaj CH':'Buenos Aires Tigre CH', 'Cuernavaca (Morelos) CH':'Cuernavaca CH'}, inplace=True)

games_df['Player'] = games_df['Player'].str.replace('\xa0', ' ', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
games_df['Unnamed: 6'] = games_df['Unnamed: 6'].str.replace('\xa0', ' ', regex=False)
games_df['Matchup'] = games_df['Unnamed: 6'].str.replace(r'\s*\[.*?\]', '', regex=True).str.strip()


games_df['Matchup'] = games_df.apply(lambda row: replace_with_full_name(row['Matchup'], row['Player']), axis=1)
games_df['Matchup'] = games_df['Matchup'].str.replace(r'\s+', ' ', regex=True).str.strip()
games_df['Matchup'] = games_df['Matchup'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

games_df[['winner_name', 'loser_name']] = games_df['Matchup'].str.extract(r'^(.*?)\s*d\.\s*(.*)$')
games_df[['winner_seed', 'winner_entry', 'loser_seed', 'loser_entry']] = games_df['Unnamed: 6'].str.extract(
    r'(?:\((?P<winner_seed>\d+)\))?\s*(?:\((?P<winner_entry>[A-Z]{1,3})\))?.*d\.\s*(?:\((?P<loser_seed>\d+)\))?\s*(?:\((?P<loser_entry>[A-Z]{1,3})\))?'
)

games_df['Minutes'] = games_df['Time'].apply(time_to_minutes)
games_df['Birthdate'] = pd.to_datetime(games_df['Birthdate'], format='%Y-%m-%d')
games_df['Age'] = round((games_df['Date'] - games_df['Birthdate']).dt.days / 365, 1)


winners_df = games_df[games_df['Player'] == games_df['winner_name']].copy()
players_columns = ['Hand', 'IOC', 'DR', 'Rk', 'A%', 'DF%', '1stIn', '1st%', '2nd%', 'BPSvd', 'TPW', 'RPW', 'BPCnv', 'TP', 'Aces', 'DFs', 'SP', '1SP', '2SP', 'vA', 'Age']
winner_columns = ['winner_'+col for col in players_columns]
winners_df.rename(columns=dict(zip(players_columns, winner_columns)), inplace=True)
winners_df = winners_df[~winners_df['Tournament'].str.contains('Davis Cup|M15|M25', na=False)]

losers_df = games_df[games_df['Player'] == games_df['loser_name']].copy()
players_columns = ['Hand', 'IOC', 'DR', 'Rk', 'A%', 'DF%', '1stIn', '1st%', '2nd%', 'BPSvd', 'TPW', 'RPW', 'BPCnv', 'TP', 'Aces', 'DFs', 'SP', '1SP', '2SP', 'vA', 'Age']
loser_columns = ['loser_'+col for col in players_columns]
losers_df.rename(columns=dict(zip(players_columns, loser_columns)), inplace=True)
losers_df = losers_df[~losers_df['Tournament'].str.contains('Davis Cup|M15|M25', na=False)]

#[player for player in games_df['loser_name'] if player not in games_df['Player'].unique()]

error_df = games_df[(games_df['Player'] != games_df['winner_name']) & (games_df['Player'] != games_df['loser_name'])]

game_columns = ['Tournament', 'Rd', 'Surface', 'Date', 'Matchup', 'winner_seed', 'winner_entry', 'winner_name', 'loser_seed', 'loser_entry', 'loser_name', 'Score', 'Time', 'Minutes']
games_merged = pd.merge(winners_df[game_columns+winner_columns], losers_df[['Matchup', 'Tournament', 'Rd']+loser_columns], on=['Matchup', 'Tournament', 'Rd'], how='outer', indicator=False)

games_merged = janitor.clean_names(games_merged)
games_merged = games_merged.sort_values(by=['date', 'tournament', 'rd', 'matchup']).reset_index(drop=True)

players = pd.read_csv('players.csv')
games_merged['winner_id'] = games_merged['winner_name'].map(players.set_index('player')['id']).astype(int, errors='ignore')
games_merged['winner_ht'] = games_merged['winner_name'].map(players.set_index('player')['height']).astype(int, errors='ignore')
games_merged['winner_ioc'] = games_merged['winner_name'].map(players.set_index('player')['ioc'])

games_merged['loser_id'] = games_merged['loser_name'].map(players.set_index('player')['id']).astype(int, errors='ignore')
games_merged['loser_ht'] = games_merged['loser_name'].map(players.set_index('player')['height']).astype(int, errors='ignore')
games_merged['loser_ioc'] = games_merged['loser_name'].map(players.set_index('player')['ioc'])

season = 2025 
tournaments = pd.read_csv('tournaments_by_season.csv')
tournaments = tournaments[tournaments['season'] == season].reset_index(drop=True)
games_merged['tournament_id'] = games_merged['tournament'].map(tournaments.set_index('tournament_stats')['tourney_id']).apply(lambda x: f'{str(season)}-{int(x)}' if pd.notna(x) else np.nan)
games_merged['draw_size'] = games_merged['tournament'].map(tournaments.set_index('tournament_stats')['draw_size']).astype(int, errors='ignore')
games_merged['tournament_level'] = games_merged['tournament'].map(tournaments.set_index('tournament_stats')['tourney_level'])
games_merged['best_of'] = games_merged['tournament'].map(tournaments.set_index('tournament_stats')['best_of']).astype(int, errors='ignore')

games_merged[['winner_bpsvd', 'winner_bpfcd']] = games_merged['winner_bpsvd'].str.split('/', expand=True)
games_merged[['loser_bpsvd', 'loser_bpfcd']] = games_merged['loser_bpsvd'].str.split('/', expand=True)

games_merged[['w_set1', 'l_set1', 'w_set2', 'l_set2', 'w_set3', 'l_set3', 'w_set4', 'l_set4', 'w_set5', 'l_set5']] = games_merged['score'].apply(lambda x: pd.Series(parse_score(x)))
games_merged['w_games'] = games_merged[['w_set1', 'w_set2', 'w_set3', 'w_set4', 'w_set5']].fillna(0).sum(axis=1)
games_merged['l_games'] = games_merged[['l_set1', 'l_set2', 'l_set3', 'l_set4', 'l_set5']].fillna(0).sum(axis=1)

for index, row in tqdm(games_merged.iterrows(), total = len(games_merged)):
    games_merged.at[index, 'w_SvGms'] = np.ceil((row['w_games'] + row['l_games']) / 2)
    games_merged.at[index, 'l_SvGms'] = np.floor((row['w_games'] + row['l_games']) / 2)

games_merged['winner_1stin'] = games_merged['winner_1stin'].str.rstrip('%').astype(float) / 100
games_merged['loser_1stin'] = games_merged['loser_1stin'].str.rstrip('%').astype(float) / 100

games_merged['winner_1st%'] = games_merged['winner_1st%'].str.rstrip('%').astype(float) / 100
games_merged['loser_1st%'] = games_merged['loser_1st%'].str.rstrip('%').astype(float) / 100

games_merged['winner_2nd%'] = games_merged['winner_2nd%'].replace('-', np.nan).str.rstrip('%').astype(float) / 100
games_merged['loser_2nd%'] = games_merged['loser_2nd%'].replace('-', np.nan).str.rstrip('%').astype(float) / 100

games_merged['winner_svpt_1stin'] = round(games_merged['winner_sp']*games_merged['winner_1stin'], 0).astype('Int64')
games_merged['loser_svpt_1stin'] = round(games_merged['loser_sp']*games_merged['loser_1stin'], 0).astype('Int64')

games_merged['winner_1stwon'] = round(games_merged['winner_svpt_1stin']*games_merged['winner_1st%'], 0).astype('Int64')
games_merged['loser_1stwon'] = round(games_merged['loser_svpt_1stin']*games_merged['loser_1st%'], 0).astype('Int64')

games_merged['winner_2ndwon'] = round((games_merged['winner_sp']-games_merged['winner_svpt_1stin'])*games_merged['winner_2nd%'], 0).astype('Int64')
games_merged['loser_2ndwon'] = round((games_merged['loser_sp']-games_merged['loser_svpt_1stin'])*games_merged['loser_2nd%'], 0).astype('Int64')

games_merged.drop(columns = ['matchup', 'time',  'winner_dr', 'winner_a%', 'winner_df%', 'winner_1stin', 'winner_1st%' ,'winner_2nd%', 'winner_tpw', 'winner_rpw', 'winner_tp', 'winner_1sp', 'winner_2sp', 'winner_bpcnv', 'winner_va',
                            'loser_dr', 'loser_a%', 'loser_df%', 'loser_1stin', 'loser_1st%', 'loser_2nd%', 'loser_tpw', 'loser_rpw', 'loser_tp', 'loser_1sp', 'loser_2sp', 'loser_bpcnv', 'loser_va',
                            'w_set1', 'l_set1', 'w_set2', 'l_set2', 'w_set3', 'l_set3', 'w_set4', 'l_set4', 'w_set5', 'l_set5', 'w_games', 'l_games'], inplace=True)
games_merged.rename(columns={
    'tournament':'tourney_name', 'rd':'round', 'date':'tourney_date', 'tournament_id':'tourney_id', 'tournament_level':'tourney_level', 
    'winner_rk':'winner_rank','winner_aces':'w_ace', 'winner_dfs':'w_df', 'winner_sp':'w_svpt', 'winner_svpt_1stin':'w_1stIn', 'winner_1stwon':'w_1stWon', 'winner_2ndwon':'w_2ndWon', 'winner_bpsvd':'w_bpSaved', 'winner_bpfcd':'w_bpFaced',
    'loser_rk':'loser_rank','loser_aces':'l_ace', 'loser_dfs':'l_df', 'loser_sp':'l_svpt', 'loser_svpt_1stin':'l_1stIn', 'loser_1stwon':'l_1stWon', 'loser_2ndwon':'l_2ndWon', 'loser_bpsvd':'l_bpSaved', 'loser_bpfcd':'l_bpFaced',
}, inplace=True)

# dtypes combine
games_merged['tourney_date'] = games_merged['tourney_date'].apply(lambda x: x.strftime('%Y%m%d'))
games_merged['winner_id'] = games_merged['winner_id'].astype('Int64')
games_merged['loser_id'] = games_merged['loser_id'].astype('Int64')


games_merged.to_csv('tennis_atp-master/atp_matches_2025.csv', index=False)

test = pd.read_csv('tennis_atp-master/atp_matches_2024.csv')

pd.concat([test, games_merged], axis = 0).to_csv('test.csv', index=False)