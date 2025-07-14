import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from Scripts.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.spatial.distance import euclidean, correlation
from sklearn.decomposition import TruncatedSVD
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
pd.set_option('display.max_rows',600)

games = pd.read_csv('games.csv')
games.drop(columns=['Unnamed: 0'], inplace=True)
games['tourney_date'] = pd.to_datetime(games['tourney_date'])
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)

games['favored_win_prob'] = elo_probability(games['favored_elo_surface'], games['underdog_elo_surface'])
games['underdog_win_prob'] = 1 - games['favored_win_prob']
games['favored_game_score'] = np.where(games['favored_win'] == 1, games['underdog_win_prob'], -games['favored_win_prob'])
games['underdog_game_score'] = np.where(games['favored_win'] == 1, -games['underdog_win_prob'], games['favored_win_prob'])


# Mean previous stats feature engineering
games_player_rival = games.copy()
games_player_rival.columns = games_player_rival.columns.str.replace('favored', 'player', regex=False)
games_player_rival.columns = games_player_rival.columns.str.replace('underdog', 'rival', regex=False)
games_player_rival['win'] = games_player_rival['player_win'] 
games_player_rival.drop(columns='player_win',inplace=True)

games_rival_player = games.copy()
games_rival_player.columns = games_rival_player.columns.str.replace('favored', 'rival', regex=False)
games_rival_player.columns = games_rival_player.columns.str.replace('underdog', 'player', regex=False)
games_rival_player['win'] = np.where(games_rival_player['rival_win'] == 1, 0, 1)
games_rival_player.drop(columns='rival_win',inplace=True)

full_games = pd.concat([games_player_rival, games_rival_player], ignore_index=True).sort_values(by=['tourney_date', 'tourney_name', 'round', 'game_id']).reset_index(drop=True)
full_games = full_games[(full_games['season'] >= 2006) & (full_games['season'] <= 2025) & (full_games['tourney_level'] != 'C') & (full_games['round'] > 'Q3') & (full_games['comment'] == 'Completed')].reset_index(drop=True)

seasons = list(range(2009, 2026))
for season in tqdm(seasons):
    full_games_train = full_games[(full_games['season'] >= season-3) & (full_games['season'] < season)].copy()
    
    # Latent Features Extraction
    players_df = full_games_train['player'].value_counts().reset_index()
    players_df.columns = ['player', 'num_obs']
    players_df = players_df[players_df['num_obs'] >= 20].reset_index(drop=True)
    players = players_df['player'].tolist()
    full_games_train = full_games_train[(full_games_train['player'].isin(players)) & (full_games_train['rival'].isin(players))]
    
    score_df = pd.pivot_table(
        full_games_train,
        values='player_game_score',
        index='player',
        columns='rival',
        aggfunc='mean'
    ).reindex(index=players, columns=players)
    #score_df = score_df.iloc[:60, :60]
    games_mask = ~np.isnan(score_df.values)
    #print(score_df)
    
    def style_vector(player, exclude):
        return [score_df.loc[player, other] for other in score_df.columns if other != player and other != exclude]
    
    distance_matrix = pd.DataFrame(index=players, columns=players, dtype=float)
    for player_a in tqdm(players):
        for player_b in players:
            if player_a == player_b:
                distance_matrix.loc[player_a, player_b] = 0.0
            else:
                vec_a = np.array(style_vector(player_a, player_b))
                vec_b = np.array(style_vector(player_b, player_a))
                mask = ~np.isnan(vec_a) & ~np.isnan(vec_b)
                vec_a = vec_a[mask]
                vec_b = vec_b[mask]
                #distance = euclidean(vec_a, vec_b) / len(vec_a) if len(vec_a) > 0 else np.nan
                distance = correlation(vec_a, vec_b)
                distance_matrix.loc[player_a, player_b] = distance
    #print(distance_matrix)
    distance_matrix_values = np.nan_to_num(distance_matrix)

    linked = sch.linkage(squareform(distance_matrix_values), method='ward')  # También puedes usar 'ward', 'complete', etc.
    plt.figure(figsize=(8, 5))
    dendro = sch.dendrogram(linked, labels=distance_matrix.index.tolist())
    plt.title("Clustering jerárquico de jugadores por estilo")
    plt.ylabel("Distancia")
    plt.tight_layout()
    #plt.show()
    
    hier_num_clusters = fcluster(linked, t = 1, criterion='distance')
    hier_num_clusters = len(set(dendro['color_list']))-1 # Unique colors of dendogram - 1 (blue)
    print(f'Number of Clusters ({season} Season): {hier_num_clusters}')
    hier_clusters = fcluster(linked, hier_num_clusters, criterion='maxclust')
    hier_cluster_df = pd.DataFrame({'player': distance_matrix.index, 'hier_cluster': hier_clusters}).set_index('player').sort_values('hier_cluster')
    hier_cluster_df['hier_cluster'] -= 1
    print(hier_cluster_df)
    
    score_df = score_df.fillna(0)
    cluster_matchups = np.zeros((hier_num_clusters, hier_num_clusters))
    counts = np.zeros((hier_num_clusters, hier_num_clusters))
    for i, player_a in enumerate(players):
        for j, player_b in enumerate(players):
            if games_mask[i, j]:
                c_p = hier_cluster_df.loc[player_a].values[0]
                c_q = hier_cluster_df.loc[player_b].values[0]
                cluster_matchups[c_p, c_q] += score_df.loc[player_a, player_b]
                counts[c_p, c_q] += 1
    
    mean_cluster_effect = cluster_matchups / np.maximum(counts, 1)
    sns.heatmap(mean_cluster_effect, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Cluster-vs-Cluster Style Exploitation Matrix")
    plt.xlabel("Defender Style Cluster")
    plt.ylabel("Attacker Style Cluster")
    #plt.show()

    games.loc[(games['season'] == season) & (games['favored'].isin(hier_cluster_df.index)), 'favored_cluster'] = games.loc[(games['season'] == season) & (games['favored'].isin(hier_cluster_df.index)), 'favored'].map(hier_cluster_df['hier_cluster']).astype('Int64')
    games.loc[(games['season'] == season) & (games['underdog'].isin(hier_cluster_df.index)), 'underdog_cluster'] = games.loc[(games['season'] == season) & (games['underdog'].isin(hier_cluster_df.index)), 'underdog'].map(hier_cluster_df['hier_cluster']).astype('Int64')

    games.loc[(games['season'] == season) & (games['favored'].isin(hier_cluster_df.index)), 'cluster_score'] = games.loc[(games['season'] == season) & (games['favored'].isin(hier_cluster_df.index))].apply(
    lambda row: mean_cluster_effect[int(row['favored_cluster']), int(row['underdog_cluster'])]
    if pd.notnull(row['favored_cluster']) and pd.notnull(row['underdog_cluster'])
    else np.nan,
    axis=1)

# Predict favored_win using cluster interactions and elo_diff
games_pred = games[(~games['favored_cluster'].isna()) & 
                (~games['underdog_cluster'].isna()) & 
                (~games['elo_surface_diff'].isna())].copy()

games_pred['elo_surface_diff_x_cluster_score_diff'] = games_pred['elo_surface_diff'] * games_pred['cluster_score']


# Combine features with dummies
feature_cols = [
    'elo_surface_diff', 
    'cluster_score', 
    'elo_surface_diff_x_cluster_score_diff'
]
X_pred = pd.concat([
    games_pred[feature_cols]
], axis=1)
X_pred = games_pred[feature_cols]
y_pred = games_pred['favored_win']

scaler = StandardScaler()
X_pred[feature_cols] = scaler.fit_transform(X_pred[feature_cols])

# Fit logistic regression
lr_model = LogisticRegression(random_state=42, max_iter=100_000)
lr_model.fit(X_pred, y_pred)
y_pred_proba = lr_model.predict_proba(X_pred)[:, 1]

# Print logistic regression coefficients
coef_df = pd.DataFrame({'feature': X_pred.columns, 'coefficient': lr_model.coef_[0]})
print(coef_df)
print(f'Intercept: {lr_model.intercept_[0]:.4f}')

# Evaluate
print(f'Log Loss (cluster+elo): {log_loss(y_pred, y_pred_proba):.4f}')
print(f'Log Loss (elo only): {log_loss(y_pred, elo_probability(games_pred["favored_elo_surface"], games_pred["underdog_elo_surface"])):.4f}')

games.to_csv('games_with_clusters.csv')


"""
for season in tqdm(seasons):
    full_games_train = full_games[(full_games['season'] >= season-3) & (full_games['season'] < season)].copy()
    
    # Latent Features Extraction
    players_df = full_games_train['player'].value_counts().reset_index()
    players_df.columns = ['player', 'num_obs']
    players_df = players_df[players_df['num_obs'] >= 20].reset_index(drop=True)
    players = players_df['player'].tolist()
    full_games_train = full_games_train[(full_games_train['player'].isin(players)) & (full_games_train['rival'].isin(players))]
    
    score_df = pd.pivot_table(
        full_games_train,
        values='player_game_score',
        index='player',
        columns='rival',
        aggfunc='mean'
    ).reindex(index=players, columns=players)
    #score_df = score_df.iloc[:60, :60]
    games_mask = ~np.isnan(score_matrix)
    print(score_df)
    
    score_df = score_df.fillna(0)
    score_matrix = score_df.values
    num_users, num_items = score_matrix.shape
    num_factors = 4
    
    svd = TruncatedSVD(n_components=num_factors, random_state=42)
    P = svd.fit_transform(score_matrix)
    Q = svd.components_.T
    
    
    # Clusters
    P_scaled = StandardScaler().fit_transform(P)
    Q_scaled = StandardScaler().fit_transform(Q)
    P_players_df = pd.DataFrame({'player': score_df.index}).set_index('player')
    Q_players_df = pd.DataFrame({'player': score_df.columns}).set_index('player')
    P_players_df[[f'off_latent_factor{i}' for i in range(0, num_factors)]] = P_scaled
    Q_players_df[[f'def_latent_factor{i}' for i in range(0, num_factors)]] = Q_scaled
    
    players_df.drop(columns=[col for col in players_df.columns if col.startswith('off_latent') or col.startswith('def_latent') or col.startswith('attacker_cluster')], inplace=True)
    players_df.set_index('player', inplace=True)
    players_df = pd.concat([players_df, P_players_df, Q_players_df], axis=1)
    
    K_range = range(2,20)
    off_factors = [f'off_latent_factor{i}' for i in range(0, num_factors)]
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(players_df[off_factors].values)
        score = silhouette_score(players_df[off_factors].values, labels)
        
        silhouette_scores.append(float(score))
    silhouette_scores_df = pd.DataFrame({'k': K_range, 'silhouette_score': silhouette_scores})
    print(silhouette_scores_df)
    
    num_off_clusters = silhouette_scores_df.loc[np.argmax(silhouette_scores_df['silhouette_score']), 'k']
    kmeans = KMeans(n_clusters=num_off_clusters, random_state=42)
    players_df['attacker_cluster'] = kmeans.fit_predict(players_df[off_factors].values) 
    print(players_df.sort_values('attacker_cluster'))
    
    value_combinations = list(combinations(range(0, num_factors), 2))
    for i,j in value_combinations:
        plt.figure(figsize=(8, 6))
        plt.scatter(players_df[off_factors].values[:, i], players_df[off_factors].values[:, j], c=players_df['attacker_cluster'], cmap='tab10', s=100)
        for m, name in enumerate(players_df.index):
            plt.text(players_df[off_factors].values[m, i]+0.02, players_df[off_factors].values[m, j]+0.02, name, fontsize=9)
        
        plt.title("Offensive Player Style Clusters")
        plt.xlabel(f"Latent Style Factor {i+1}")
        plt.ylabel(f"Latent Style Factor {j+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def_factors = [f'def_latent_factor{i}' for i in range(0, num_factors)]
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(players_df[def_factors].values)
        score = silhouette_score(players_df[def_factors].values, labels)
        
        silhouette_scores.append(float(score))
    silhouette_scores_df = pd.DataFrame({'k': K_range, 'silhouette_score': silhouette_scores})
    print(silhouette_scores_df)
    
    num_def_clusters = silhouette_scores_df.loc[np.argmax(silhouette_scores_df['silhouette_score']), 'k']
    kmeans = KMeans(n_clusters=num_def_clusters, random_state=42)
    players_df['defender_cluster'] = kmeans.fit_predict(players_df[def_factors].values) 
    print(players_df.sort_values('defender_cluster'))
    
    value_combinations = list(combinations(range(0, num_factors), 2))
    for i,j in value_combinations:
        plt.figure(figsize=(8, 6))
        plt.scatter(players_df[def_factors].values[:, i], players_df[def_factors].values[:, j], c=players_df['defender_cluster'], cmap='tab10', s=100)
        for m, name in enumerate(players_df.index):
            plt.text(players_df[def_factors].values[m, i]+0.02, players_df[def_factors].values[m, j]+0.02, name, fontsize=9)
        
        plt.title("Defensive Player Style Clusters")
        plt.xlabel(f"Latent Style Factor {i+1}")
        plt.ylabel(f"Latent Style Factor {j+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    cluster_matchups = np.zeros((num_off_clusters, num_def_clusters))
    counts = np.zeros((num_off_clusters, num_def_clusters))
    for i in range(num_users):
        for j in range(num_items):
            if games_mask[i, j]:
                c_p = players_df['attacker_cluster'][i]
                c_q = players_df['defender_cluster'][j]
                cluster_matchups[c_p, c_q] += score_matrix[i, j]
                counts[c_p, c_q] += 1
    
    mean_cluster_effect = cluster_matchups / np.maximum(counts, 1)
    sns.heatmap(mean_cluster_effect, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Cluster-vs-Cluster Style Exploitation Matrix")
    plt.xlabel("Defender Style Cluster")
    plt.ylabel("Attacker Style Cluster")
    plt.show()
    
    games.loc[(games['season'] == season) & (games['favored'].isin(players_df.index)), 'favored_att_cluster'] = games.loc[(games['season'] == season) & (games['favored'].isin(players_df.index)), 'favored'].map(players_df['attacker_cluster']).astype('Int64')
    games.loc[(games['season'] == season) & (games['favored'].isin(players_df.index)), 'favored_def_cluster'] = games.loc[(games['season'] == season) & (games['favored'].isin(players_df.index)), 'favored'].map(players_df['defender_cluster']).astype('Int64')
    games.loc[(games['season'] == season) & (games['underdog'].isin(players_df.index)), 'underdog_att_cluster'] = games.loc[(games['season'] == season) & (games['underdog'].isin(players_df.index)), 'underdog'].map(players_df['attacker_cluster']).astype('Int64')
    games.loc[(games['season'] == season) & (games['underdog'].isin(players_df.index)), 'underdog_def_cluster'] = games.loc[(games['season'] == season) & (games['underdog'].isin(players_df.index)), 'underdog'].map(players_df['defender_cluster']).astype('Int64')

    # Predict using cluster scores against other clusters and using cluster dummies
    games_pred = games[(~games['favored_cluster'].isna()) & 
                       (~games['underdog_cluster'].isna()) & 
                       (~games['elo_surface_diff'].isna())].copy()

    # Cluster-vs-cluster score as feature
    games_pred['favored_cluster_score'] = games_pred.apply(
        lambda row: mean_cluster_effect[int(row['favored_att_cluster']), int(row['underdog_def_cluster'])], axis=1)
    games_pred['underdog_cluster_score'] = games_pred.apply(
        lambda row: mean_cluster_effect[int(row['underdog_att_cluster']), int(row['favored_def_cluster'])], axis=1)
    games_pred['cluster_score_diff'] = games_pred['favored_cluster_score'] - games_pred['underdog_cluster_score']

    # Create cluster dummy variables (one-hot encoding)
    favored_cluster_dummies = pd.get_dummies(games_pred['favored_cluster'], prefix='fav_clust')
    underdog_cluster_dummies = pd.get_dummies(games_pred['underdog_cluster'], prefix='und_clust')

    # Interactions
    cluster_interactions = pd.DataFrame(
        np.outer(favored_cluster_dummies.values, underdog_cluster_dummies.values).reshape(
            favored_cluster_dummies.shape[0], -1
        ),
        columns=[
            f"{fc}_{uc}" for fc in favored_cluster_dummies.columns for uc in underdog_cluster_dummies.columns
        ],
        index=favored_cluster_dummies.index
    )
    games_pred['elo_cluster_interaction'] = games_pred['elo_surface_diff'] * games_pred['cluster_score_diff']
    
    # Combine features
    X = pd.concat([
        games_pred[['elo_surface_diff', 'cluster_score_diff', 'elo_cluster_interaction']],
        favored_cluster_dummies,
        underdog_cluster_dummies
    ], axis=1)
    X = games_pred[['elo_surface_diff', 'cluster_score_diff', 'elo_cluster_interaction']]
    y = games_pred['favored_win'].values

    # Fit logistic regression with cluster dummies
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    y_pred_proba = lr.predict_proba(X)[:, 1]

    # Print logistic regression coefficients
    coef_df = pd.DataFrame({'feature': X.columns, 'coefficient': lr.coef_[0]})
    print(coef_df)
    print(f'Intercept: {lr.intercept_[0]:.4f}')
    
    # Evaluate
    print(f'Log Loss (elo+cluster_score+dummies): {log_loss(y, y_pred_proba):.4f}')
    print(f'Log Loss (elo only): {log_loss(y, elo_probability(games_pred["favored_elo_surface"], games_pred["underdog_elo_surface"])):.4f}')
"""