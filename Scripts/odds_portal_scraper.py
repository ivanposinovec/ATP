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

# Tournaments
url = 'https://www.oddsportal.com/tennis/results/'
options = Options()
options.add_argument("--log-level=3")
options.add_argument("--headless") 
driver = webdriver.Chrome(options=options)

driver.get(url)
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(2)  # Wait for the page to load
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

html = driver.page_source
driver.quit()

soup = BeautifulSoup(html, 'html.parser')
countries = soup.select('div.flex')

tournaments = []
for country in countries:
    country_tourneys = country.find_all('a', class_='text-xs font-normal underline font-main text-black-main')
    
    for a_object in country_tourneys:
        try:
            tournament = a_object.text.strip()
            url = a_object['href']
            
            tournaments.append({'tournament_odds_portal':tournament, 'url':url})
        except:
            print(f'Could not extract data from row: {a_object}')
tournaments_df = pd.DataFrame(tournaments)
tournaments_df = tournaments_df.drop_duplicates(subset=['tournament_odds_portal'])

tournaments_df_old = pd.read_csv('tournaments_by_season.csv')
tournaments_df_old = tournaments_df_old.drop_duplicates(subset=['tournament_stats', 'season'])
tournaments_df_old['tournament_odds_portal'] = tournaments_df_old['tournament_stats'].apply(lambda x: x.replace('Masters','').strip()).replace({'Adelaide 1': 'Adelaide', 'Cologne 1': 'Cologne',
                                                                                                                                'Great Ocean Road Open':'Melbourne (Great Ocean Road Open)',
                                                                                                                                'Melbourne':'Melbourne (Summer Set)', 'Nur-Sultan':'Astana',
                                                                                                                                'Murray River Open':'Melbourne (Murray River Open)', 'Naples':'Napoli', "Queen's Club":'London',
                                                                                                                                'ATP Rio de Janeiro':'Rio de Janeiro', 'Rio De Janeiro':'Rio de Janeiro',
                                                                                                                                'Roland Garros':'French Open', 's Hertogenbosch':'Hertogenbosch',
                                                                                                                                'St Petersburg':'St. Petersburg', 'Us Open': 'US Open',
                                                                                                                                'Tour Finals':'Finals - Turin','Doha Aus Open Qualies':'Australian Open',
                                                                                                                                'Next Gen Finals':'Next Gen Finals - Jeddah',
                                                                                                                                'NextGen Finals':'Next Gen Finals - Jeddah',
                                                                                                                                'London Olympics':'Olympic Games', 'Paris Olympics':'Olympic Games',
                                                                                                                                'Rio Olympics':'Olympic Games', 'Tokyo Olympics':'Olympic Games'})
tournaments_df_old['tournament_odds_portal'] = tournaments_df_old.apply(
lambda x: x['tournament_odds_portal'].replace('Canada', 'Montreal') if x['season'] in [2009, 2011, 2013, 2015, 2017, 2019, 2022, 2024] 
else x['tournament_odds_portal'].replace('Canada', 'Toronto'), axis=1
)
tournaments_df_old['tournament_odds_portal'] = tournaments_df_old.apply(
lambda x: x['tournament_odds_portal'].replace('Belgrade', 'Belgrade 2') if x['season'] in [2024] 
else x['tournament_odds_portal'], axis=1
)
tournaments_df_old['tournament_odds_portal'] = tournaments_df_old.apply(
lambda x: x['tournament_odds_portal'].replace('Lyon', 'Lyon 2') if x['season'] in [2008, 2009] 
else x['tournament_odds_portal'], axis=1
)
tournaments_df_old['tournament_odds_portal'] = tournaments_df_old.apply(
lambda x: x['tournament_odds_portal'].replace('Santiago', 'Vina del Mar') if x['season'] in [2012, 2013] 
else x['tournament_odds_portal'], axis=1
)
tournaments_df_old['tournament_odds_portal'] = np.where(tournaments_df_old['tourney_level'] != 'C', 'ATP ' + tournaments_df_old['tournament_odds_portal'],
                                                                                                    'Challenger Men ' + tournaments_df_old['tournament_odds_portal'].str.strip(' CH'))

tournaments_final = pd.merge(tournaments_df_old, tournaments_df, how='left', on='tournament_odds_portal').sort_values(by=['tournament_odds_portal', 'season']).reset_index(drop=True)
tournaments_final['last_edition'] = ~tournaments_final.duplicated(subset=['tournament_odds_portal'], keep='last')

tournaments_final.loc[(tournaments_final['season'].isin([2013, 2014])) & (tournaments_final['tournament_odds_portal'] == 'ATP Estoril'), 'url'] = '/tennis/portugal/atp-oeiras/results/'
tournaments_final.loc[(tournaments_final['season'].isin([2020, 2021])) & (tournaments_final['tournament_odds_portal'] == 'ATP Astana'), 'url'] = '/tennis/kazakhstan/atp-nur-sultan/results/'
tournaments_final.loc[(tournaments_final['season'].isin([2024])) & (tournaments_final['tournament_odds_portal'] == 'ATP French Open'), 'last_edition'] = False
tournaments_final.loc[(tournaments_final['season'].isin([2017, 2018, 2019, 2020])) & (tournaments_final['tournament_odds_portal'] == 'ATP Finals - Turin'), 'url'] = '/tennis/world/atp-finals-london/results/'
tournaments_final.loc[(tournaments_final['season'] <= 2016) & (tournaments_final['tournament_odds_portal'] == 'ATP Finals - Turin'), 'url'] = '/tennis/world/atp-world-tour-finals-london/results/'
tournaments_final.loc[(tournaments_final['season'] <= 2022) & (tournaments_final['tournament_odds_portal'] == 'ATP Next Gen Finals - Jeddah'), 'url'] = '/tennis/world/atp-next-gen-finals-milan/results/'
tournaments_final.loc[(tournaments_final['season'] == 2021) & (tournaments_final['tournament_odds_portal'] == 'ATP Olympic Games'), 'season'] = 2020

tournaments_final.to_csv('tournaments_by_season_oddsportal2.csv', index=False)


# Games
tournaments_final = pd.read_csv('tournaments_by_season_oddsportal2.csv')
tournaments_final = tournaments_final[tournaments_final['season'] >= 2009].reset_index(drop=True)
games = pd.read_csv('games_oddsportal.csv')


scraped_tournaments = list((games['game_url'].str.split('/').str[:4].str.join('/') + '/').unique())
missing_tournaments = []
for index, row in tqdm(tournaments_final.iterrows(), total = len(tournaments_final)):
    if row["url"] is not np.nan:
        url = f'https://www.oddsportal.com{row["url"]}' if row['last_edition'] == True else f'https://www.oddsportal.com{row["url"].replace('/results/', '')}-{row["season"]}/results/'
        if url.replace(f'https://www.oddsportal.com','').replace('results/','') not in scraped_tournaments:
            print(url)
            missing_tournaments.append(url)

data = []
for url in tqdm(missing_tournaments):
    game_rows = []
    for page in range(1, 8):
        options = Options()
        driver = webdriver.Chrome(options=options)
        
        driver.get(url+'#/page/'+str(page)+'/')
        
        sleep(4+random())
        # Scroll down to fully load the page
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(2)  # Wait for the page to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        html = driver.page_source
        
        driver.quit()
        
        soup = BeautifulSoup(html, 'html.parser')
        game_rows += soup.select('div.group.flex')
        sleep(3+random())
        
        if page == 1:
            number_of_pages = len(soup.find_all('a', class_='pagination-link'))-1 if len(soup.find_all('a', class_='pagination-link')) != 0 else 1
        if page == number_of_pages:
            break
    
    for row in game_rows:
        try:
            link = row.find('a')['href'] if row.find('a') else None
            
            players = row.find_all('p', class_='participant-name truncate')
            
            comment = row.find('div',attrs={'data-testid':'game-status-box'}).find('p').text if row.find('div',attrs={'data-testid':'game-status-box'}).find('p') else None
            
            if comment is None: 
                sets1 = int(row.find('div', class_= 'text-gray-dark relative flex').find('div').find_all('div')[0].text.strip())
                sets2 = int(row.find('div', class_= 'text-gray-dark relative flex').find('div').find_all('div')[1].text.strip())
                odds1 = row.find('div', attrs={'data-testid':"odd-container-winning"}).find('p').text.strip() if sets1 > sets2 else row.find('div', attrs={'data-testid':"odd-container-default"}).find('p').text.strip()
                odds2 = row.find('div', attrs={'data-testid':"odd-container-default"}).find('p').text.strip() if sets1 > sets2 else row.find('div', attrs={'data-testid':"odd-container-winning"}).find('p').text.strip()
            else:
                odds1 = row.find_all('div', attrs={'data-testid':"odd-container-default"})[0].find('p').text.strip() if row.find('div', attrs={'data-testid':"odd-container-default"}) else None
                odds2 = row.find_all('div', attrs={'data-testid':"odd-container-default"})[1].find('p').text.strip() if row.find('div', attrs={'data-testid':"odd-container-default"}) else None
            
            odds1 = american_odds_conversor(float(odds1)) if odds1 and (float(odds1) > 100 or float(odds1) < -100) else odds1
            odds2 = american_odds_conversor(float(odds2)) if odds1 and (float(odds2) > 100 or float(odds2) < -100) else odds2
            
            data.append({'game_url':link, 'player1': players[0].text.strip(), 'player2': players[1].text.strip(), 'comment':comment, 'odds1': odds1, 'odds2':odds2})
        except:
            link = row.find('a')['href'] if row.find('a') else None
            data.append({'game_url':link})
            
            print(f'Could not extract data from row: {link}')

new_games = pd.DataFrame(data)

games = pd.concat([games, new_games], axis = 0).reset_index(drop=True)
games.to_csv('games_oddsportal.csv', index=False)




tournaments_final = pd.read_csv('tournaments_by_season_oddsportal2.csv')
tournaments_final.rename(columns={'url':'tournament_url'}, inplace=True)
games = pd.read_csv('games_oddsportal.csv')

games.insert(0, 'tournament_url', (games['game_url'].str.split('/').str[:4].apply(lambda parts: '/'.join(parts[:3] + [re.sub(r'-\d{4}$', '', parts[3])]) + '/results/')))
games.insert(1, 'season', games['game_url'].str.extract(r'-(\d{4})/').astype('float'))

missing_mask = games['game_url'].str.extract(r'-(\d{4})/')[0].isna()
games['season'] = games['season'].fillna(method='bfill').astype(int)

pd.merge(tournaments_final, games, on = ['tournament_url', 'season'], how = 'right').to_csv('games_oddsportal2.csv', index=False)