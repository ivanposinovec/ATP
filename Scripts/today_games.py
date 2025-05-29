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
import json
import re
from Scripts.functions import american_odds_conversor

date = (pd.Timestamp.today() + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
current_tournaments = ['French Open']
tournaments = pd.read_csv('tournaments_by_season_oddsportal.csv')
season = 2025

options = Options()
options.add_argument("--log-level=3")
driver = webdriver.Chrome(options=options)
for tournament in current_tournaments:
    game_rows = []
    url = 'https://www.oddsportal.com' + tournaments.loc[(tournaments['tournament_odds'] == tournament) & (tournaments['season'] == season), 'url'].values[0].replace('results/', '')
    
    driver.get(url)
    sleep(2+random())
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(2)  # Wait for the page to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    
    html = driver.page_source
    
    soup = BeautifulSoup(html, 'html.parser')
    game_rows += soup.select('div.group.flex')
    
    data = []
    date_row = None
    for row in game_rows:
        date_html = row.find_parent('div').find_parent('div').find('div', attrs = {'data-testid':"date-header"}).text if row.find_parent('div').find_parent('div').find('div', attrs = {'data-testid':"date-header"}) is not None else None
        if date_html is not None:
            date_row = pd.to_datetime(f'{date_html.split(', ')[1].strip()} {season}', format='%d %b %Y').strftime('%Y-%m-%d') if len(date_html.split(', '))>1 else pd.to_datetime(f'{date_html.strip()}', format='%d %b %Y').strftime('%Y-%m-%d')    
        
        if date == date_row:
            try:
                link = row.find('a')['href'] if row.find('a') else None
                
                players = row.find_all('p', class_='participant-name truncate')
                
                comment = row.find('div',attrs={'data-testid':'game-status-box'}).find('p').text if row.find('div',attrs={'data-testid':'game-status-box'}).find('p') else None
                
                try:
                    odds = row.find_all('div', attrs={'data-testid':"odd-container-default"})+row.find_all('div', attrs={'data-testid':"odd-container-winning"})
                    odds1 = odds[0].find('p').text.strip() if row.find('div', attrs={'data-testid':"odd-container-default"}) else None
                    odds2 = odds[1].find('p').text.strip() if row.find('div', attrs={'data-testid':"odd-container-default"}) else None
                except:
                    odds = row.find_all('p', attrs={'class':"height-content"})
                    odds1 = odds[0]
                    odds2 = odds[1]
                
                
                data.append({'game_url':link, 'player1': players[0].text.strip(), 'player2': players[1].text.strip(), 'comment':comment, 'odds1': odds1, 'odds2':odds2})
            except:
                link = row.find('a')['href'] if row.find('a') else None
                data.append({'game_url':link})
                
                print(f'Could not extract data from row: {link}')
    pd.DataFrame(data)

driver.quit()
