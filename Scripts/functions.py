import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import math
from statsmodels.discrete.conditional_models import ConditionalLogit

def dump(obj,name):
	pickle.dump(obj,open(name+'.p',"wb")) 
def load(name):
	obj=pickle.load( open( name+".p", "rb" ) ) 
	return obj

def compute_elo_rankings(data):
    """
    Given the list on matches in chronological order, for each match, computes 
    the elo ranking of the 2 players at the beginning of the match
    
    """
    print("Elo rankings computing...")
    players=list(pd.Series(list(data['Winner'])+list(data['Loser'])).value_counts().index)
    elo=pd.Series(np.ones(len(players))*1500,index=players)
    ranking_elo=[(1500,1500)]
    for i in range(1,len(data)):
        w=data.iloc[i-1,:]['Winner']
        l=data.iloc[i-1,:]['Loser']
        elow=elo[w]
        elol=elo[l]
        pwin=1 / (1 + 10 ** ((elol - elow) / 400))    
        K_win=32
        K_los=32
        new_elow=elow+K_win*(1-pwin)
        new_elol=elol-K_los*(1-pwin)
        elo[w]=new_elow
        elo[l]=new_elol
        ranking_elo.append((elo[data.iloc[i,:]['Winner']],elo[data.iloc[i,:]['Loser']])) 
        if i%5000==0:
            print(str(i)+" matches computed...")
    ranking_elo=pd.DataFrame(ranking_elo,columns=["Elo_winner","Elo_loser"])    
    ranking_elo["proba_elo"]=1 / (1 + 10 ** ((ranking_elo["Elo_loser"] - ranking_elo["Elo_winner"]) / 400))   
    return ranking_elo


def parse_score(score):
    if pd.notna(score):
        sets = score.split()
        sets = [s.strip('?') for s in sets if not any(keyword in s for keyword in ['RET', 'W/O', 'DEF', 'Default', 'ABD', 'Walkover', 'Played', 'and', 'unfinished', 'Def.', 'RE', '>', "Ret'd", 'UNK'])]
        
        set_details = []
        
        for set_score in sets:
            if '(' in set_score:
                set_score = set_score.split('(')[0]
            games = list(map(int, set_score.split('-')))
            set_details.append(games)
        
        # Ensure we have exactly 5 sets, pad with [None, None] if fewer
        while len(set_details) < 5:
            set_details.append([None, None])
        
        return [game for set_detail in set_details for game in set_detail]
    else:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


def rank_group(rank):
    if rank <= 10:
        return 'top_10'
    elif rank <= 20:
        return 'top_20'
    elif rank <= 32:
        return 'top_32'
    elif rank <= 64:
        return 'top_64'
    elif rank <= 100:
        return 'top_100'
    elif rank <= 200:
        return 'top_200'
    else:
        return 'over_200'


def weighted_average(values, weights):
    valid_data = values.dropna()
    weights = weights.loc[valid_data.index]
    return np.dot(valid_data, weights / sum(weights))


def get_no_vig_odds(odds1: float, odds0: tuple[float, None], odds2: float):

    """

    take 1X2 odds and remove overround/set to target level
    accuracy can be set, in terms of number of decimal places of odds
    uses Newton-Raphson iteration, namely, increment size

    :param odds1: float
    :param odds0: float
    :param odds2: float
    :return:

    """

    c, accuracy, current_error = 1, 3, 1000
    max_error = (10 ** (-accuracy)) / 2

    if odds0 is not None:

        while current_error > max_error:
            f = (1 / odds1) ** c + (1 / odds0) ** c + (1 / odds2) ** c - 1
            h = -f / (((1 / odds1) ** c) * (-math.log(odds1)) + ((1 / odds2) ** c) * (-math.log(odds2)) + ((1 / odds0) ** c) * (-math.log(odds0)))
            c += h
            current_error = abs((1 / odds1) ** c + (1 / odds2) ** c + (1 / odds0) ** c - 1)
        return odds1 ** c, odds0 ** c, odds2 ** c

    else:
        while current_error > max_error:
            f = (1 / odds1) ** c + (1 / odds2) ** c - 1
            h = -f / (((1 / odds1) ** c) * (-math.log(odds1)) + ((1 / odds2) ** c) * (-math.log(odds2)))
            c += h
            current_error = abs((1 / odds1) ** c + (1 / odds2) ** c - 1)
        return odds1 ** c, odds2 ** c

def fit_second_stage(y, under_probs, over_probs, implied_under_probs, implied_over_probs):
    df_over = pd.DataFrame({'prop_id':list(range(1,len(y)+1)), 'bet_type':'over',
                            'probs':over_probs, 'implied_probs':implied_over_probs, 'result':y})
    df_under = pd.DataFrame({'prop_id':list(range(1,len(y)+1)), 'bet_type':'under',
                            'probs':under_probs, 'implied_probs':implied_under_probs, 'result':y})
    
    df = pd.concat([df_over, df_under], ignore_index=True, axis = 0).sort_values(by = ['prop_id', 'bet_type']).reset_index(drop=True)
    df['y'] = np.where(df['bet_type'] == 'over', np.where(df['result'] == 1, 1,0), np.where(df['result'] == 0, 1,0))
    
    cl_model = ConditionalLogit(endog = df['y'], exog = df[['probs', 'implied_probs']], groups = df['prop_id'])
    cl_fit = cl_model.fit()
    return cl_fit

def predict_second_stage(beta, under_probs, over_probs, implied_under_probs, implied_over_probs):
    num = np.exp(np.dot(np.array([over_probs, implied_over_probs]).T, beta))
    den = np.exp(np.dot(np.array([over_probs, implied_over_probs]).T, beta)) + np.exp(np.dot(np.array([under_probs, implied_under_probs]).T, beta)) 
    predicted_probs_over = num/den
    
    return np.vstack((1-predicted_probs_over, predicted_probs_over)).T


def kelly_fraction_calc(odds, probs):
    return ((odds-1)*probs-(1-probs))/(odds-1).apply(lambda x: max(x,0))


def american_odds_conversor(value):
    if value >= 0:
        return value/100 + 1
    elif value < 0:
        return 100/(-value) + 1 