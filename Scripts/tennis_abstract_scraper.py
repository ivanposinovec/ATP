import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

url = 'https://www.tennisabstract.com/cgi-bin/tourney.cgi?t=2025-0416/Rome-Masters'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

table = soup.find('table', id='singles-results')

df = pd.read_html(StringIO(str(table)))
print(df)


table = soup.find('table', id='singles-results')
if table is not None:
    tables = pd.read_html(StringIO(str(table)))
    df = tables[0]
    print(df)
else:
    print("Table not found.")
