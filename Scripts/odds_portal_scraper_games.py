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
from Scripts.functions import *


url = 'https://www.oddsportal.com/tennis/argentina/atp-buenos-aires/cerundolo-juan-manuel-faria-jaime-QspPAFQF/'
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.get(url)

wait = WebDriverWait(driver, 10)
wait.until(expected_conditions.element_to_be_clickable((By.XPATH, ".//button[@id='onetrust-reject-all-handler']"))).click()

bookmakers = wait.until(expected_conditions.presence_of_all_elements_located((By.XPATH, "//div[@data-testid='over-under-expanded-row']")))
data = []
for bookmaker in bookmakers:
    bookmaker_name = bookmaker.find_element(By.TAG_NAME, "p").text
    odds_dict = {'bookmaker': bookmaker_name}
    
    odd_containers = bookmaker.find_elements(By.XPATH, ".//div[@data-testid='odd-container']")
    for i,odd_container in enumerate(odd_containers, 1):
        odds = odd_container.find_element(By.TAG_NAME, "p")
        closing_odd = float(odds.text)
        
        driver.execute_script("""arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});""", odds)
        sleep(0.1)  
        
        wait = WebDriverWait(driver, 10)
        wait.until(expected_conditions.element_to_be_clickable(odds)).click() 
        
        opening_odds_element = wait.until(expected_conditions.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Opening odds')]"))).find_elements(By.XPATH, "./following-sibling::*")[0].text
        opening_odd = float(opening_odds_element.split("\n")[1])
        
        odds_dict[f'opening_odds_{i}'] = american_odds_conversor(int(opening_odd))
        odds_dict[f'clossing_odds_{i}'] = american_odds_conversor(int(closing_odd))
    data.append(odds_dict)

df = pd.DataFrame(data)
driver.quit()


# Game odds