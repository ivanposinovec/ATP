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
            
            tournaments.append({'tournament':tournament, 'url':url})
        except:
            print(f'Could not extract data from row: {a_object}')
tournaments_df = pd.DataFrame(tournaments)
tournaments_df = tournaments_df.drop_duplicates(subset=['tournament'])


tournaments_df_old = pd.read_csv('tournaments_by_season.csv')
tournaments_df_old = tournaments_df_old[tournaments_df_old['season'] >= 2009].reset_index(drop=True)
tournaments_df_old = tournaments_df_old.drop_duplicates(subset=['tournament_stats', 'season'])
tournaments_df_old['tournament_stats'] = tournaments_df_old['tournament_stats'].apply(lambda x: x.replace('Masters','').strip()).replace({'Adelaide 1': 'Adelaide', 'Cologne 1': 'Cologne',
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
tournaments_df_old['tournament_stats'] = tournaments_df_old.apply(
lambda x: x['tournament_stats'].replace('Canada', 'Montreal') if x['season'] in [2009, 2011, 2013, 2015, 2017, 2019, 2022, 2024] 
else x['tournament_stats'].replace('Canada', 'Toronto'), axis=1
)
tournaments_df_old['tournament_stats'] = tournaments_df_old.apply(
lambda x: x['tournament_stats'].replace('Belgrade', 'Belgrade 2') if x['season'] in [2024] 
else x['tournament_stats'], axis=1
)
tournaments_df_old['tournament_stats'] = tournaments_df_old.apply(
lambda x: x['tournament_stats'].replace('Lyon', 'Lyon 2') if x['season'] in [2008, 2009] 
else x['tournament_stats'], axis=1
)
tournaments_df_old['tournament_stats'] = tournaments_df_old.apply(
lambda x: x['tournament_stats'].replace('Santiago', 'Vina del Mar') if x['season'] in [2012, 2013] 
else x['tournament_stats'], axis=1
)
tournaments_df_old['tournament'] = 'ATP ' + tournaments_df_old['tournament_stats']

tournaments_final = pd.merge(tournaments_df_old[(tournaments_df_old['season'] >= 2009) & (tournaments_df_old['series'] != 'Challenger')], tournaments_df, how='left', on='tournament').sort_values(by=['tournament', 'season']).reset_index(drop=True)
tournaments_final['last_edition'] = ~tournaments_final.duplicated(subset=['tournament'], keep='last')

tournaments_final.loc[(tournaments_final['season'].isin([2013, 2014])) & (tournaments_final['tournament_stats'] == 'Estoril'), 'url'] = '/tennis/portugal/atp-oeiras/results/'
tournaments_final.loc[(tournaments_final['season'].isin([2020, 2021])) & (tournaments_final['tournament_stats'] == 'Astana'), 'url'] = '/tennis/kazakhstan/atp-nur-sultan/results/'
tournaments_final.loc[(tournaments_final['season'].isin([2024])) & (tournaments_final['tournament_stats'] == 'French Open'), 'last_edition'] = False
tournaments_final.loc[(tournaments_final['season'].isin([2017, 2018, 2019, 2020])) & (tournaments_final['tournament_stats'] == 'Finals - Turin'), 'url'] = '/tennis/world/atp-finals-london/results/'
tournaments_final.loc[(tournaments_final['season'].isin([2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])) & (tournaments_final['tournament_stats'] == 'Finals - Turin'), 'url'] = '/tennis/world/atp-world-tour-finals-london/results/'
tournaments_final.loc[(tournaments_final['season'] <= 2022) & (tournaments_final['tournament_stats'] == 'Next Gen Finals - Jeddah'), 'url'] = '/tennis/world/atp-next-gen-finals-milan/results/'
tournaments_final.loc[(tournaments_final['season'] == 2021) & (tournaments_final['tournament_stats'] == 'Olympic Games'), 'season'] = 2020

tournaments_final.to_csv('tournaments_by_season_oddsportal.csv', index=False)


# Games
tournaments_final = pd.read_csv('tournaments_by_season_oddsportal.csv')
games = pd.read_csv('games_oddsportal2.csv')

scraped_tournaments = list((games['game_url'].str.split('/').str[:4].str.join('/') + '/').unique())
missing_tournaments = []
for index, row in tqdm(tournaments_final.iterrows(), total = len(tournaments_final)):
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
games.to_csv('games_oddsportal2.csv', index=False)


#<------------------------------------------------------------
# Games odds
games = pd.read_csv('games_oddsportal.csv')

with open('odds_data2.json', 'r') as json_file:
    odds_data = json.load(json_file)

chrome_options = webdriver.ChromeOptions()
#user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
#chrome_options.add_argument(f"--user-agent={user_agent}")
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
count = 0
for index, row in tqdm(games.iterrows(), total = len(games)):
    if row['comment'] not in ['canc.', 'w.o.', 'award.'] and row['game_url'] is not None and row['game_url'] not in [game['game_url'] for game in odds_data]:
        try:
            row_dict = row.to_dict()
            url = f'https://www.oddsportal.com{row["game_url"]}'
            driver.get(url)
            
            wait = WebDriverWait(driver, 1)
            try:
                cookie_button = wait.until(expected_conditions.presence_of_element_located((By.XPATH, ".//button[@id='onetrust-reject-all-handler']")))
                if cookie_button.is_displayed():
                    cookie_button.click()
            except:
                pass
            try:
                survey_button = wait.until(expected_conditions.presence_of_element_located((By.XPATH, ".//a[@class='sg-js-d']")))
                if survey_button.is_displayed():
                    survey_button.click()
            except:
                pass
            
            closing_time_items = wait.until(expected_conditions.presence_of_all_elements_located((By.XPATH, "//div[@data-testid='game-time-item']")))[0].find_elements(By.XPATH, "//p/following-sibling::p")
            closing_time = pd.to_datetime(closing_time_items[0].text + closing_time_items[1].text, format = '%d %b %Y,%H:%M')
            
            bookmakers = wait.until(expected_conditions.presence_of_all_elements_located((By.XPATH, "//div[@data-testid='over-under-expanded-row']")))
            bookmakers_odds = []
            for bookmaker in bookmakers:
                bookmaker_name = bookmaker.find_element(By.TAG_NAME, "p").text
                odds_dict = {'bookmaker': bookmaker_name}
                
                odd_containers = bookmaker.find_elements(By.XPATH, ".//div[@data-testid='odd-container']")
                for i,odd_container in enumerate(odd_containers, 1):
                    odds = odd_container.find_element(By.TAG_NAME, "p")
                    closing_odd = float(odds.text) if odds.text != '-' else None
                    
                    driver.execute_script("""arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});""", odds)
                    sleep(0.03)  
                    
                    wait.until(expected_conditions.element_to_be_clickable(odds)).click() 
                    
                    try:
                        opening_odds_element = wait.until(expected_conditions.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Opening odds')]"))).find_elements(By.XPATH, "./following-sibling::*")[0].text
                        opening_time = pd.to_datetime(str(closing_time.year) + " " + opening_odds_element.split("\n")[0], format='%Y %d %b, %H:%M')
                        opening_odd = float(opening_odds_element.split("\n")[1])
                    except:
                        opening_time = None
                        opening_odd = None
                    
                    if opening_time is not None:
                        odds_dict[f'opening_time'] = opening_time 
                    odds_dict[f'closing_time'] = closing_time
                    odds_dict[f'opening_odds{i}'] = american_odds_conversor(float(opening_odd)) if opening_odd and (float(opening_odd) > 100 or float(opening_odd) < -100) and opening_odd is not None else opening_odd
                    odds_dict[f'clossing_odds{i}'] = american_odds_conversor(float(closing_odd)) if closing_odd and (float(closing_odd) > 100 or float(closing_odd) < -100) and closing_odd is not None else closing_odd
                bookmakers_odds.append(odds_dict)
            row_dict['odds'] = bookmakers_odds
            odds_data.append(row_dict)
        except Exception as e:
            print(f"Error on {url}: {e}")
        sleep(1+random())
        
        count += 1
        if count % 500 == 0:
            with open('odds_data2.json', 'w') as json_file:
                json.dump(odds_data, json_file, indent=4, default=str)
            print(f"Saved {count} games to odds_data2.json")
driver.quit()

with open('odds_data.json', 'w') as json_file:
    json.dump(odds_data, json_file, indent=4, default=str)

expanded_data = []
for game in odds_data:
    base_data = {key: value for key, value in game.items() if key != 'odds'}
    for odds_entry in game.get('odds', []):
        expanded_row = {**base_data, **odds_entry}
        expanded_data.append(expanded_row)

print(pd.DataFrame(expanded_data))

