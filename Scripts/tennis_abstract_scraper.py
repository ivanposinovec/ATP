import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO


list_ = ['Banana', 'Apple', 'Orange', 'Grapes', 'Pineapple']
list_[0]

url = 'https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0416/Rome-Masters'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

tables = soup.find_all('table', id='singles-results')
tables[0]


df = pd.read_html(StringIO(str(tables[0])))[0]
df.rename(columns={'Unnamed: 0': 'Winner', 'Unnamed: 4':'Loser'}, inplace=True)