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
def adelaide_swap(name, season):
    if season == 2023:
        if name == 'Adelaide 2':
            return 'Adelaide'
        elif name == 'Adelaide':
            return 'Adelaide 2'
    return name
tournaments_df_old['tournament_odds_portal'] = tournaments_df_old.apply(
    lambda x: adelaide_swap(x['tournament_odds_portal'], x['season']), axis=1
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







