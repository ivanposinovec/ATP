from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from Scripts.functions import *
from random import random
from tqdm import tqdm
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import json

# Games
tournaments = pd.read_csv('tournaments_by_season_oddsportal2.csv')
tournaments = tournaments[(tournaments['season'] >= 2009) & (tournaments['tourney_level'] != 'C')].reset_index(drop=True)

#games = pd.read_csv('games_oddsportal.csv')
#scraped_tournaments = list((games['game_url'].str.split('/').str[:4].str.join('/') + '/').unique())

data = []
for index, row in tqdm(tournaments.iterrows(), total = len(tournaments)):
    game_rows = []
    for page in range(1, 8):
        options = Options()
        driver = webdriver.Chrome(options=options)
        
        url = f'https://www.oddsportal.com{row["url"]}' if row['last_edition'] == True else f'https://www.oddsportal.com{row["url"].replace('/results/', '')}-{row["season"]}/results/'
        driver.get(url+'#/page/'+str(page)+'/')
        
        # Scroll down to fully load the page
        sleep(2+random())
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(3)  # Wait for the page to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        html = driver.page_source
        
        driver.quit()
        
        soup = BeautifulSoup(html, 'html.parser')
        game_rows += soup.select('div.group.flex')
        sleep(0.5+random())
        
        if page == 1:
            number_of_pages = len(soup.find_all('a', class_='pagination-link'))-1 if len(soup.find_all('a', class_='pagination-link')) != 0 else 1
        if page == number_of_pages:
            break
    
    for row_html in game_rows:
        try:
            link = row_html.find('a')['href'] if row_html.find('a') else None
            
            players = row_html.find_all('p', class_='participant-name truncate')
            
            comment = row_html.find('div',attrs={'data-testid':'game-status-box'}).find('p').text if row_html.find('div',attrs={'data-testid':'game-status-box'}).find('p') else None
            
            if comment is None: 
                sets1 = int(row_html.find('div', class_= 'text-gray-dark relative flex').find('div').find_all('div')[0].text.strip())
                sets2 = int(row_html.find('div', class_= 'text-gray-dark relative flex').find('div').find_all('div')[1].text.strip())
                
                odds1 = row_html.find('div', attrs={'data-testid':"odd-container-winning"}).find('p').text.strip() if sets1 > sets2 else row_html.find('div', attrs={'data-testid':"odd-container-default"}).find('p').text.strip()
                odds2 = row_html.find('div', attrs={'data-testid':"odd-container-default"}).find('p').text.strip() if sets1 > sets2 else row_html.find('div', attrs={'data-testid':"odd-container-winning"}).find('p').text.strip()
                
                winner = players[0].text.strip() if sets1 > sets2 else players[1].text.strip()
                loser = players[1].text.strip() if sets1 > sets2 else players[0].text.strip()
            else:
                odds1 = row_html.find_all('div', attrs={'data-testid':"odd-container-default"})[0].find('p').text.strip() if row_html.find('div', attrs={'data-testid':"odd-container-default"}) else None
                odds2 = row_html.find_all('div', attrs={'data-testid':"odd-container-default"})[1].find('p').text.strip() if row_html.find('div', attrs={'data-testid':"odd-container-default"}) else None
                
                winner = None
                loser = None
                
            odds1 = american_odds_conversor(float(odds1)) if odds1 and (float(odds1) > 100 or float(odds1) < -100) else odds1
            odds2 = american_odds_conversor(float(odds2)) if odds1 and (float(odds2) > 100 or float(odds2) < -100) else odds2
            
            data.append({'tournament_stats':row['tournament_stats'], 'season':row['season'], 'tournament_url':row['url'], 'game_url':link,
                        'player1': players[0].text.strip(), 'player2': players[1].text.strip(), 'comment':comment, 'odds1': odds1, 'odds2':odds2, 'winner':winner, 'loser':loser})
            
        except Exception as e:
            link = row_html.find('a')['href'] if row_html.find('a') else None
            data.append({'tournament_stats':row['tournament_stats'], 'season':row['season'], 'tournament_url':row['url'], 'game_url':link})
            
            print(f'Could not extract data from row: {link} - Error message: {e}')

games = pd.DataFrame(data)
games.to_csv('games_oddsportal3.csv', index=False)
with open('games_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4, default=str)


tournaments_with_missing_games = games[(games['winner'].isna()) & (games['comment'].isna())][['tournament_url', 'season']].drop_duplicates().reset_index(drop=True).rename(columns = {'tournament_url':'url'})
tournaments_with_missing_games = tournaments.merge(tournaments_with_missing_games, on=['url', 'season'], how ='right')

new_data = []
for index, row in tqdm(tournaments_with_missing_games.iterrows(), total = len(tournaments_with_missing_games)):
    if (row['tournament_stats'], row['season']) not in list({(item['tournament_stats'], item['season']) for item in new_data}):
        game_rows = []
        for page in range(1, 8):
            options = Options()
            driver = webdriver.Chrome(options=options)
            
            url = f'https://www.oddsportal.com{row["url"]}' if row['last_edition'] == True else f'https://www.oddsportal.com{row["url"].replace('/results/', '')}-{row["season"]}/results/'
            driver.get(url+'#/page/'+str(page)+'/')
            
            # Scroll down to fully load the page
            sleep(2+random())
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                sleep(3)  # Wait for the page to load
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            
            html = driver.page_source
            
            driver.quit()
            
            soup = BeautifulSoup(html, 'html.parser')
            game_rows += soup.select('div.group.flex')
            sleep(0.5+random())
            
            if page == 1:
                number_of_pages = len(soup.find_all('a', class_='pagination-link'))-1 if len(soup.find_all('a', class_='pagination-link')) != 0 else 1
            if page == number_of_pages:
                break
        
        for row_html in game_rows:
            link = row_html.find('a')['href'] if row_html.find('a') else None
            
            players = row_html.find_all('p', class_='participant-name truncate')
            
            comment = row_html.find('div',attrs={'data-testid':'game-status-box'}).find('p').text if row_html.find('div',attrs={'data-testid':'game-status-box'}).find('p') else None
            if comment is None: 
                sets1 = int(row_html.find('div', class_= 'text-gray-dark relative flex').find('div').find_all('div')[0].text.strip())
                sets2 = int(row_html.find('div', class_= 'text-gray-dark relative flex').find('div').find_all('div')[1].text.strip())
                
                # Safely extract odds1 and odds2 depending on which containers exist
                odd_default = row_html.find_all('div', attrs={'data-testid':"odd-container-default"})
                odd_winning = row_html.find_all('div', attrs={'data-testid':"odd-container-winning"})
                
                if len(odd_default) == 1:
                    if odd_default[0] is not None and odd_winning[0] is not None:
                        odds1 = odd_winning[0].find('p').text.strip() if sets1 > sets2 else odd_default[0].find('p').text.strip()
                        odds2 = odd_default[0].find('p').text.strip() if sets1 > sets2 else odd_winning[0].find('p').text.strip()
                    elif odd_default[0] is not None:
                        odds1 = None if sets1 > sets2 else odd_default.find('p').text.strip()
                        odds2 = odd_default[0].find('p').text.strip() if sets1 < sets2 else None
                    elif odd_winning[0] is not None:
                        # Only winning odds found, assign both to winning
                        odds1 = odd_winning[0].find('p').text.strip() if sets1 > sets2 else None
                        odds2 = None if sets1 > sets2 else odd_winning.find('p').text.strip()
                    else:
                        odds1 = None
                        odds2 = None
                else:
                    odds1 = odd_default[0].find('p').text.strip() if row_html.find('div', attrs={'data-testid':"odd-container-default"}) else None
                    odds2 = odd_default[1].find('p').text.strip() if row_html.find('div', attrs={'data-testid':"odd-container-default"}) else None
                
                winner = players[0].text.strip() if sets1 > sets2 else players[1].text.strip()
                loser = players[1].text.strip() if sets1 > sets2 else players[0].text.strip()
            
            else:
                odds_list = row_html.find_all('div', attrs={'data-testid': ["odd-container-winning", "odd-container-default"]})
                if len(odds_list) >= 2:            
                    odds1 = odds_list[0].find('p').text.strip() if odds_list[0].find('p') else None
                    odds2 = odds_list[1].find('p').text.strip() if odds_list[1].find('p') else None
                else:
                    odds1 = None
                    odds2 = None
                    
                winner = None
                loser = None
                
            odds1 = american_odds_conversor(float(odds1)) if odds1 and (float(odds1) > 100 or float(odds1) < -100) else odds1
            odds2 = american_odds_conversor(float(odds2)) if odds1 and (float(odds2) > 100 or float(odds2) < -100) else odds2
            
            new_data.append({'tournament_stats':row['tournament_stats'], 'season':row['season'], 'tournament_url':row['url'], 'game_url':link,
                        'player1': players[0].text.strip(), 'player2': players[1].text.strip(), 'comment':comment, 'odds1': odds1, 'odds2':odds2, 'winner':winner, 'loser':loser})
#new_data = [item for item in new_data if not (item.get('tournament_stats') == 'Rome' and item.get('season') == 2024)]

new_games = pd.DataFrame(new_data)
fixed_games = list(new_games['game_url'])
data = [item for item in data if item.get('game_url') not in fixed_games]
data.extend(new_data)

games = pd.DataFrame(data).sort_values(['tournament_stats', 'season']).reset_index(drop=True)
games.to_csv('games_oddsportal3.csv', index=False)
with open('games_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4, default=str)





"""
games_old = pd.read_csv('games_oddsportal2.csv')
len(games_old.drop_duplicates(subset=['tournament_url', 'season']))


[game_url for game_url in list(games['game_url'].unique()) if game_url not in list(games_old['game_url'].unique())]



games[~games['game_url'].isin(list(games_old['game_url'].unique()))]
games_old[~games_old['game_url'].isin(list(games['game_url'].unique()))].head(40)

games[(games['tournament_stats'] == 'Adelaide') & (games['season'] == 2024)]


tournaments_final = pd.read_csv('tournaments_by_season_oddsportal2.csv')
tournaments_final.rename(columns={'url':'tournament_url'}, inplace=True)
games = pd.read_csv('games_oddsportal.csv')

games.insert(0, 'tournament_url', (games['game_url'].str.split('/').str[:4].apply(lambda parts: '/'.join(parts[:3] + [re.sub(r'-\d{4}$', '', parts[3])]) + '/results/')))
games.insert(1, 'season', games['game_url'].str.extract(r'-(\d{4})/').astype('float'))

last_found_edition = games.dropna(subset='season').drop_duplicates(subset='tournament_url', keep='last').set_index('tournament_url')['season']
games['season'] = games.apply(lambda row: last_found_edition[row['tournament_url']] + 1 if pd.isna(row['season']) and row['tournament_url'] in last_found_edition else row['season'], axis=1)

games.loc[(games['tournament_url'] == '/tennis/kazakhstan/atp-almaty/results/'), 'season'] = 2024
games.loc[(games['tournament_url'] == '/tennis/turkey/atp-antalya/results/') & (games['season'] == 2020), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/bosnia-and-herzegovina/atp-banja-luka/results/'), 'season'] = 2023
games.loc[(games['tournament_url'] == '/tennis/italy/atp-cagliari/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/germany/atp-cologne/results/'), 'season'] = 2020
games.loc[(games['tournament_url'] == '/tennis/germany/atp-cologne-2/results/'), 'season'] = 2020
games.loc[(games['tournament_url'] == '/tennis/italy/atp-florence/results/'), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/spain/atp-gijon/results/'), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/china/atp-hangzhou/results/'), 'season'] = 2024
games.loc[(games['tournament_url'] == '/tennis/usa/atp-indianapolis/results/'), 'season'] = 2009
games.loc[(games['tournament_url'] == '/tennis/spain/atp-marbella/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/australia/atp-melbourne-great-ocean-road-open/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/australia/atp-melbourne-murray-river-open/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/australia/atp-melbourne-summer-set/results/'), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/canada/atp-montreal/results/'), 'season'] = 2024
games.loc[(games['tournament_url'] == '/tennis/russia/atp-moscow/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/italy/atp-napoli/results/'), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/italy/atp-parma/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/italy/atp-sardinia/results/'), 'season'] = 2020
games.loc[(games['tournament_url'] == '/tennis/south-korea/atp-seoul/results/'), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/singapore/atp-singapore/results/'), 'season'] = 2021
games.loc[(games['tournament_url'] == '/tennis/australia/atp-sydney/results/') & (games['season'] == 2020), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/israel/atp-tel-aviv/results/'), 'season'] = 2022
games.loc[(games['tournament_url'] == '/tennis/canada/atp-toronto/results/') & (games['season'] == 2022), 'season'] = 2023
games.loc[(games['tournament_url'] == '/tennis/china/atp-zhuhai/results/') & (games['season'] == 2020), 'season'] = 2023
games.loc[(games['tournament_url'] == '/tennis/serbia/atp-belgrade-2/results/') & (games['season'] == 2022), 'season'] = 2024
games.loc[(games['tournament_url'] == '/tennis/france/atp-lyon-2/results/'), 'season'] = 2009
games.loc[(games['tournament_url'] == '/tennis/world/atp-olympic-games/results/') & (games['season'] == 2021), 'season'] = 2020

df = pd.merge(tournaments_final, games, on = ['tournament_url', 'season'], how = 'right')
print(df[(df['tournament_stats'] == 'Doha Aus Open Qualies') & (df['player1'] == 'Barrere G.')])
print(df[(df['tournament_stats'] == 'Australian Open') & (df['season'] == 2021) & (df['player1'] == 'Barrere G.')])

indexes_to_drop = list(df[(df['tournament_stats'] == 'Doha Aus Open Qualies') & (df.index < 6013)].index) + list(df[(df['tournament_stats'] == 'Australian Open') & (df['season'] == 2021) & (df.index >= 6012)].index)
df.drop(index = indexes_to_drop, inplace=True)

df.to_csv('games_oddsportal2.csv', index=False)
"""