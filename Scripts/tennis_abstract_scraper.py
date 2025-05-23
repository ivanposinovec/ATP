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

# Players
url = 'https://tennisabstract.com/reports/atpRankings.html'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tables = soup.find_all('table', id='reportable')
players = pd.read_html(StringIO(str(tables[0])))[0]


# Tournaments
tournaments = []
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)
for index, row in tqdm(players.head(500).iterrows(), total=len(500), desc='Getting games'):
    player_url = f'https://www.tennisabstract.com/cgi-bin/player.cgi?p={"".join(row["Player"].split())}&f=A2025qq'
    driver.get(player_url)
    sleep(2 + random())
    response_content = driver.page_source
    soup = BeautifulSoup(response_content, 'html.parser')
    
    table = soup.find('table', id='matches')
    
    if table is not None:
        if len(table) > 0:
            player_games_df = pd.read_html(StringIO(str(table)))[0]
            player_games_df = player_games_df.iloc[:-1]
            tournament_list = list(player_games_df['Tournament'].unique())
            for tournament in tournament_list:
                if tournament not in tournaments:
                    tournaments.append(tournament)
                    print(f'Added tournament: {tournament}')
    sleep(2 + random())
driver.quit()
tournaments = [t for t in tournaments if 'Davis Cup' not in t and 'M15' not in t and 'M25' not in t]

# Games 
def extract_player_name(s):
    return re.sub(r'\s*\(.*?\)|\s*\[.*?\]', '', str(s)).strip().split('|')[0].strip()
for tournament in tournaments:
    url = f'https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025{"_".join(tournament.split())}'
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find_all('table', id='singles-results')
    
    df = pd.read_html(StringIO(str(table)))[0]
    df.rename(columns={'Unnamed: 2':'Winner', 'Unnamed: 5':'Loser'}, inplace=True)
    df.drop(columns=['Unnamed: 3'], inplace=True)
    
    # Clean Winner and Loser columns to keep only player names
    df['Winner'] = df['Winner'].apply(extract_player_name)
    df['Loser'] = df['Loser'].apply(extract_player_name)

list(df.columns)

# Missing players
missing_players = ['Ki Lung Ng', 'Adam Taylor', 'Gillian Osmont', 'Adrian Arcon', 'Heremana Courte', 'Yuttana Charoenphon', 'Natthayut Nithithananont', 'Tim Van Rijthoven', 'Juan Ignacio Centurion Delvalle', 'Gonzalo Ariel Karakachian', 'Yuttana Charoenphon', 'Mick Veldheer', 'Dino Molokova Ferreira', 'Luca Sanchez', 'George Goldhoff', 'Gabriel Roveri Sidney', 'Keerthivassan Suresh', 'Ivan Liutarevich', 'Noah Lopez Cherubino', 'Dhruva Mulye', 'Yusuf Ebrahim Ahmed Abdulla Qaed', 'Elyas Abduljalil', 'Ivan Liutarevich', 'Marco Bortolotti', 'Cheik Pandzou Ekoume', 'Kryce Didier Momo Kassa', 'Nicholas Alan Van Aken', 'Yassine Smiej', 'Guelfo Borghini Baldovinetti', 'Paterne Mamata', 'Mubarak Shannan Zayid', 'Rafael Alfonso De Alba Valdes', 'Niki Kaliyanda Poonacha', 'Abdulrahman Al Janahi', 'Etienne Niyigena', 'Claude Ishimwe', 'Kelsey Stevenson', 'Joshua Muhire', 'Alexander Georg Mandma', 'Christos Glavas', 'Evangelos Kypriotis', 'Denis Istomin', 'Christos Glavas', 'Hendrik Jebens', 'Mark Wallner', 'Mathis Bondaz', 'Federico Gaston Gonzalez Benitez', 'Alex Santino Nunez Vera', 'Tennyson Whiting', 'Diego Bustamante', 'Sergio Ingles Garre', 'Eneko Rios Perez', 'Frane Nincevic', 'Noah Regas Luis', 'Breno Braga', 'Gabriel Roveri Sidney', 'Gustavo Albieri', 'Diego Eloy Mendez Montiel', 'Amine Jamji', 'Izan Corretja', 'Sergio Martos Gornes', 'Ivan Lopez Martos', 'Abdoulaziz Bationo', 'Dino Molokova Ferreira', 'Andre Rodeia', 'Azariah Rusher', 'Abdoulaziz Bationo', 'Zijiang Yang', 'Manuel Lazic', 'Pedro Pinto', 'Luis Guto Miguel', 'Vicente Freda', 'Ivan Liutarevich', 'Bruno Malacarne', 'Juan Esteban Trujillo Hernandez', 'Marcelo Demoliner', 'Dino Molokova Ferreira', 'Alaa Trifi', 'Omar Knani', 'Adam Nagoudi', 'Roko Horvat', 'Matei Todoran']
players = pd.concat([players, pd.DataFrame({'Player': missing_players})], axis = 0).reset_index(drop=True)
games_df = pd.DataFrame()
games_df = pd.read_csv('atp_matches_2025.csv')

# Player games
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
games_df = games_df[(games_df['Date'] < '2025-05-18') & (games_df['Score'] != 'Live Scores') & (~games_df['Tournament'].str.contains('Davis Cup|M15|M25', na=False))].sort_values(['Date', 'Tournament', 'Rd']).reset_index(drop=True)
games_df['Rd'] = pd.Categorical(games_df['Rd'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)
games_df['Date'] = pd.to_datetime(games_df['Date'].str.replace('‑', '-', regex=False), format='%d-%b-%Y')
games_df.drop_duplicates(subset = ['Tournament', 'Rd', 'Date', 'Unnamed: 6', 'Player'], inplace=True)



games_df['Player'] = games_df['Player'].str.replace('\xa0', ' ', regex=False).str.replace(r'\s+', ' ', regex=True).str.strip()
games_df['Unnamed: 6'] = games_df['Unnamed: 6'].str.replace('\xa0', ' ', regex=False)
games_df['Matchup'] = games_df['Unnamed: 6'].str.replace(r'\s*\[.*?\]', '', regex=True).str.strip()

games_df[games_df['Unnamed: 6'] == '(5)Cerundolo d. (Q)Juan Manuel Cerundolo [ARG]'][['Player', 'Unnamed: 6','Matchup']]

def replace_with_full_name(match_string, full_name_string):
    full_name = full_name_string.split()[1:]
    words_in_match = re.findall(r'[A-Za-z]+', match_string)
    name_in_match = list()
    for word in words_in_match:
        if word in full_name:
            name_in_match.append(word)
    name_in_match = ' '.join(name_in_match)
    return match_string.replace(name_in_match, full_name_string)

def replace_with_full_name(match_string, full_name_string):
    last_surname = full_name_string.split()[-1].strip()
    players_in_match = match_string.split('d.')
    players_in_match = [word.strip() for word in players_in_match]
    
    surname_in_player = list()
    for player in players_in_match:
        words_in_player = re.findall(r'[A-Za-z]+', player)
        if last_surname in words_in_player:
            surname_in_player.append(player)
    
    name_in_match = min(surname_in_player, key=lambda x: len(x.split()))
    return match_string.replace(name_in_match, full_name_string)

games_df['Matchup'] = games_df.apply(lambda row: replace_with_full_name(row['Matchup'], row['Player']), axis=1)
games_df['Matchup'] = games_df['Matchup'].str.replace(r'\s+', ' ', regex=True).str.strip()
games_df['Matchup'] = games_df['Matchup'].str.replace(r'\[.*?\]', '', regex=True).str.strip()

pattern = r'(?:\((?P<winner_seed>\d+)\))?\s*(?:\((?P<winner_entry>[A-Z]{1,3})\))?\s*(?P<winner_name>[^\d()]+?)\s*d\.\s*(?:\((?P<loser_seed>\d+)\))?\s*(?:\((?P<loser_entry>[A-Z]{1,3})\))?\s*(?P<loser_name>.+)'
games_df[['winner_seed', 'winner_entry', 'winner_name', 'loser_seed', 'loser_entry', 'loser_name']] = games_df['Matchup'].str.extract(pattern)
games_df['winner_name'] = games_df['winner_name'].str.replace(r'\s*\[.*?\]', '', regex=True)
games_df['loser_name'] = games_df['loser_name'].str.replace(r'\s*\[.*?\]', '', regex=True)
games_df['Matchup'] = games_df['Matchup'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
games_df['winner_name'] = games_df['winner_name'].str.replace(r'\(.*?\)', '', regex=True).str.strip()
games_df['loser_name'] = games_df['loser_name'].str.replace(r'\(.*?\)', '', regex=True).str.strip()



def time_to_minutes(t):
        if pd.isna(t):
            return None
        parts = str(t).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return None
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
test = pd.merge(winners_df[game_columns+winner_columns], losers_df[['Matchup', 'Tournament', 'Rd']+loser_columns], on=['Matchup', 'Tournament', 'Rd'], how='outer', indicator=True)



test[(test['_merge']== 'left_only')][['Matchup', 'winner_name', 'loser_name']].sort_values(by='loser_name')

test.to_csv('test.csv', index=False)





games_df = games_df[~games_df['Tournament'].str.contains('Davis Cup|M15|M25', na=False)]