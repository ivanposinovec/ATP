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
tournaments_final = tournaments_final[(tournaments_final['season'] >= 2009) & (tournaments_final['tourney_level'] != 'C')].reset_index(drop=True)

#games = pd.read_csv('games_oddsportal.csv')
#scraped_tournaments = list((games['game_url'].str.split('/').str[:4].str.join('/') + '/').unique())

data = []
for index, row in tqdm(tournaments_final.iterrows(), total = len(tournaments_final)):
    game_rows = []
    for page in range(1, 8):
        options = Options()
        driver = webdriver.Chrome(options=options)
        
        url = f'https://www.oddsportal.com{row["url"]}' if row['last_edition'] == True else f'https://www.oddsportal.com{row["url"].replace('/results/', '')}-{row["season"]}/results/'
        driver.get(url+'#/page/'+str(page)+'/')
        
        sleep(3+random())
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
        sleep(1+random())
        
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
games.to_csv('games_oddsportal.csv', index=False)




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

