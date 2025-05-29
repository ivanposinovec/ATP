from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import random
from tqdm import tqdm
from functions import *
import json

games = pd.read_csv('games_oddsportal2.csv')

with open('odds_data.json', 'r') as json_file:
    odds_data = json.load(json_file)
print(f'--Lenght odds_data {len(odds_data)}--')

games_to_scrape = games[(~games['game_url'].isin([game['game_url'] for game in odds_data])) & (games['game_url'].notnull()) & (~games['comment'].isin(['canc.', 'w.o.', 'award.']))]

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
count = 0
odds_data_current_session = []
for index, row in tqdm(games_to_scrape.iterrows(), total = len(games_to_scrape)):
    if row['comment'] not in ['canc.', 'w.o.', 'award.'] and row['game_url'] is not None:
        try:
            if count % 50 == 0:
                if 'driver' in locals():
                    driver.quit()
                driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
            
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
                    sleep(0.02)  
                    
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
            odds_data_current_session.append(row_dict)
        except Exception as e:
            print(f"Error on {url}: {e}")
        sleep(0.5+random())
        
        count += 1
        if count % 100 == 0:
            with open('odds_data.json', 'r') as json_file:
                odds_data = json.load(json_file)
            
            odds_data = odds_data+odds_data_current_session
            print(f'--Lenght odds_data{len(odds_data)}--')
            
            with open('odds_data.json', 'w') as json_file:
                json.dump(odds_data, json_file, indent=4, default=str)
            odds_data_current_session = []
            print(f"Saved {count} games to odds_data2.json")
driver.quit()


odds_data = odds_data+odds_data_current_session
with open('odds_data.json', 'w') as json_file:
    json.dump(odds_data, json_file, indent=4, default=str)


"""
#games = games[(~games['game_url'].str.startswith('/tennis/switzerland/atp-geneva/')) & (~games['game_url'].str.startswith('/tennis/italy/atp-rome/')) & (~games['game_url'].str.startswith('/tennis/germany/atp-hamburg/'))
        # & (~games['game_url'].str.startswith('/tennis/france/atp-french-open/')) & (~games['game_url'].str.startswith('/tennis/serbia/atp-belgrade-2/')) & (~games['game_url'].str.startswith('/tennis/serbia/atp-belgrade/'))].reset_index(drop=True)

with open('odds_data.json', 'r') as json_file:
    odds_data = json.load(json_file)

df = pd.DataFrame(odds_data)
df = df.explode('odds').reset_index(drop=True)
odds_expanded = pd.json_normalize(df['odds'])
expanded_df = pd.concat([df.drop(columns='odds').reset_index(drop=True), odds_expanded.reset_index(drop=True)], axis=1)

with open('odds_data3.json', 'r') as json_file:
    odds_data3 = json.load(json_file)

df3 = pd.DataFrame(odds_data3)
df3 = df3.explode('odds').reset_index(drop=True)
odds_expanded = pd.json_normalize(df3['odds'])
expanded_df3 = pd.concat([df3.drop(columns='odds').reset_index(drop=True), odds_expanded.reset_index(drop=True)], axis=1)


full_df = pd.concat([pd.DataFrame(odds_data), pd.DataFrame(odds_data3)], ignore_index=True)
full_df.drop_duplicates(subset=['game_url'])

full_df = pd.concat([expanded_df, expanded_df3], ignore_index=True)
full_df.drop_duplicates(subset=['game_url', 'bookmaker'])

combined_odds_data = odds_data + odds_data3
combined_odds_data = list({item['game_url']: item for item in combined_odds_data if 'game_url' in item}.values())
with open('odds_data.json', 'w') as json_file:
    json.dump(combined_odds_data, json_file, indent=4, default=str)
"""
