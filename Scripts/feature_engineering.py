import pandas as pd
import numpy as np
from tqdm import tqdm
import janitor
from Scripts.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
pd.set_option('display.max_rows',600)
import json


# Odds data
with open('odds_data.json', 'r') as json_file:
    odds_data = json.load(json_file)

odds = pd.DataFrame(odds_data)
odds.head(10)

df = pd.read_csv('games_merged.csv')
df['odds'] = df['odds'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
df = df[~df['odds'].isna()]
df = df.explode('odds').reset_index(drop=True)


odds_expanded = pd.json_normalize(df['odds'])
odds = pd.concat([df.drop(columns='odds').reset_index(drop=True), odds_expanded.reset_index(drop=True)], axis=1)
bookmakers = list(odds['bookmaker'].unique())

wide_df = odds.pivot_table(
    index=['game_url'],
    columns='bookmaker',
    values = ['opening_time', 'opening_odds1',  'clossing_odds1',  'opening_odds2',  'clossing_odds2'],
    aggfunc='first'  # or use list if multiple entries per bookmaker/game
)

wide_df.columns = ['_'.join(col).strip('_') for col in wide_df.columns.values]
odds = pd.merge(pd.DataFrame(odds_data), wide_df, on = 'game_url')