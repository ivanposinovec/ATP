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
games_to_scrape = games[(~games['game_url'].isin([game['game_url'] for game in odds_data]) )& (games['game_url'].notnull()) & (~games['comment'].isin(['canc.', 'w.o.', 'award.']))]
for index, row in tqdm(games_to_scrape.iterrows(), total = len(games_to_scrape)):
    if row['comment'] not in ['canc.', 'w.o.', 'award.'] and row['game_url'] is not None:
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
            odds_data.append(row_dict)
        except Exception as e:
            print(f"Error on {url}: {e}")
        sleep(1+random())
        
        count += 1
        print(count)
        if count % 100 == 0:
            with open('odds_data2.json', 'w') as json_file:
                json.dump(odds_data, json_file, indent=4, default=str)
            print(f"Saved {count} games to odds_data2.json")
driver.quit()

with open('odds_data2.json', 'w') as json_file:
    json.dump(odds_data, json_file, indent=4, default=str)

