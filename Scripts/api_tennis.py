import requests
import pandas as pd

API_KEY = '3996a076cc53f593b4219e585fb4f15b525a6e813720529d37c3e706a5633fbc'
url = "https://api.api-tennis.com/tennis"

# Circuits
params = {'method':"get_events", 'APIkey':API_KEY}
response = requests.get(url,params=params)
circuits_df = pd.json_normalize(response.json()['result'])


# Tournaments
params = {'method':"get_tournaments", 'APIkey':API_KEY}
response = requests.get(url,params=params)
tournaments_df = pd.json_normalize(response.json()['result'])
tournaments_df[tournaments_df['event_type_key'] == 265][tournaments_df['tournament_name'] == 'ATP Australian Open'].to_list()

# Matches
params = {'method':"get_fixtures",'date_start':'2021-01-01', 'date_stop':'2023-12-31', 'APIkey':API_KEY}
response = requests.get(url,params=params)
games_df = pd.json_normalize(response.json()['result'])

# Odds
