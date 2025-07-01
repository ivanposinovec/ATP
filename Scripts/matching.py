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
pd.set_option('display.max_rows',600)

games = pd.read_csv('games.csv')
games.drop(columns=['Unnamed: 0'], inplace=True)
games['tourney_date'] = pd.to_datetime(games['tourney_date'])
games['round'] = pd.Categorical(games['round'], categories=['Q1', 'Q2', 'Q3', 'ER', 'RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'BR', 'F'], ordered=True)

# Surface weighting
surfaces = ['Hard', 'Grass', 'Clay']
win_pct_by_surface = {}
for surface in surfaces:
    surface_data_favorites = games[games['surface'] == surface].copy()
    surface_data_favorites.rename(columns={'favored':'player', 'underdog':'rival'}, inplace=True)
    surface_data_favorites['win'] = surface_data_favorites['favored_win'] 
    
    surface_data_underdogs = games[games['surface'] == surface].copy()
    surface_data_underdogs.rename(columns={'favored':'rival', 'underdog':'player'}, inplace=True)
    surface_data_underdogs['win'] = np.where(surface_data_underdogs['favored_win'] == 1, 0, 1)
    
    surface_data = pd.concat([surface_data_favorites, surface_data_underdogs], axis=0).sort_values(by = ['tourney_date', 'round'])
    
    win_pct_by_surface[surface] = surface_data.groupby('player')['win'].mean()

win_pct_df = pd.DataFrame(win_pct_by_surface).dropna()
surface_correlation_matrix = win_pct_df.corr()
print(surface_correlation_matrix)

games['favored_win_prob'] = elo_probability(games['favored_elo_surface'], games['underdog_elo_surface'])
games['underdog_win_prob'] = 1 - games['favored_win_prob']
games['favored_game_score'] = np.where(games['favored_win'] == 1, games['underdog_win_prob'], -games['favored_win_prob'])
games['underdog_game_score'] = np.where(games['favored_win'] == 1, -games['underdog_win_prob'], games['favored_win_prob'])


games['favored_tpt_won_pct'] = (games['favored_svpt_won']+games['favored_rtpt_won'])/(games['favored_svpt']+games['underdog_svpt'])
games['favored_svpt_won_pct'] = (games['favored_svpt_won'])/(games['favored_svpt'])
games['favored_1st_in_pct'] = (games['favored_1st_in'])/(games['favored_svpt'])
games['favored_1st_won_pct'] = (games['favored_1st_won'])/(games['favored_1st_in'])
games['favored_2nd_won_pct'] = (games['favored_2nd_won'])/(games['favored_svpt']-games['favored_1st_in'])
games['favored_ace_pct'] = (games['favored_ace'])/(games['favored_svpt'])
games['favored_df_pct'] = (games['favored_df'])/(games['favored_svpt'])
games['favored_bp_saved_pct'] = ((games['favored_bp_saved'])/(games['favored_bp_faced'])).fillna(1)
games['favored_rtpt_won_pct'] = (games['favored_rtpt_won'])/(games['underdog_svpt'])
games['favored_1st_return_won_pct'] = (games['underdog_1st_in']-games['underdog_1st_won'])/(games['underdog_1st_in'])
games['favored_2nd_return_won_pct'] = (games['underdog_svpt']-games['underdog_1st_in']-games['underdog_2nd_won'])/(games['underdog_svpt']-games['underdog_1st_in'])
games['favored_avg_bp_won_pct'] = ((games['favored_bp_won'])/(games['underdog_bp_faced'])).fillna(0)

games['underdog_tpt_won_pct'] = (games['underdog_svpt_won']+games['underdog_rtpt_won'])/(games['underdog_svpt']+games['favored_svpt'])
games['underdog_svpt_won_pct'] = (games['underdog_svpt_won'])/(games['underdog_svpt'])
games['underdog_1st_in_pct'] = (games['underdog_1st_in'])/(games['underdog_svpt'])
games['underdog_1st_won_pct'] = (games['underdog_1st_won'])/(games['underdog_1st_in'])
games['underdog_2nd_won_pct'] = (games['underdog_2nd_won'])/(games['underdog_svpt']-games['underdog_1st_in'])
games['underdog_ace_pct'] = (games['underdog_ace'])/(games['underdog_svpt'])
games['underdog_df_pct'] = (games['underdog_df'])/(games['underdog_svpt'])
games['underdog_bp_saved_pct'] = ((games['underdog_bp_saved'])/(games['underdog_bp_faced'])).fillna(1)
games['underdog_rtpt_won_pct'] = (games['underdog_rtpt_won'])/(games['favored_svpt'])
games['underdog_1st_return_won_pct'] = (games['favored_1st_in']-games['favored_1st_won'])/(games['favored_1st_in'])
games['underdog_2nd_return_won_pct'] = (games['favored_svpt']-games['favored_1st_in']-games['favored_2nd_won'])/(games['favored_svpt']-games['favored_1st_in'])
games['underdog_avg_bp_won_pct'] = ((games['underdog_bp_won'])/(games['favored_bp_faced'])).fillna(0)

surface_dummies = pd.get_dummies(games[['surface']], prefix='surface')
hand_dummies = pd.get_dummies(games[['favored_hand', 'underdog_hand']])
games = pd.concat([games, surface_dummies, hand_dummies], axis=1)


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


# PCA
# One-hot encoding of surface
features = ['player_game_score', 'player_ace_pct', 'player_df_pct', 'player_tpt_won_pct', 'player_svpt_won_pct', 'player_1st_in_pct', 'player_1st_won_pct', 'player_2nd_won_pct', 'player_bp_saved_pct', 'player_rtpt_won_pct',
        'player_1st_return_won_pct', 'player_2nd_return_won_pct', 'player_avg_bp_won_pct', 'player_dominance_ratio', 'surface_Hard', 'surface_Grass', 'surface_Clay']
features = ['player_ace_pct', 'player_df_pct', 'player_1st_in_pct', 'player_1st_won_pct', 'player_2nd_won_pct',
        'player_1st_return_won_pct', 'player_2nd_return_won_pct', 'player_avg_bp_won_pct', 'surface_Hard', 'surface_Grass', 'surface_Clay', 'player_ht']
full_games = full_games.replace(np.inf, np.nan).dropna(subset=features)

seasons = list(range(2009, 2026))
for season in tqdm(seasons):
    full_games_train = full_games[(full_games['season'] >= season-3) & (full_games['season'] < season)].copy()
    
    scaler = StandardScaler()
    X = full_games_train[features]
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=len(features))
    pca.fit(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = explained_var.cumsum()
    print(cumulative_var)
    
    n_components = 7 #int(input("Enter the number of components to keep based on cumulative variance: "))
    full_games_train[[f'PCA{i}' for i in range(0, n_components)]] = pca.transform(scaler.transform(full_games_train[features]))[:, :n_components]
    
    mask = (
        (games['season'] == season) &
        (games[[feature.replace('player', 'favored') for feature in features]].notna().all(axis=1)) &
        (~np.isinf(games[[feature.replace('player', 'favored') for feature in features]]).any(axis=1)) &
        (~np.isinf(games[[feature.replace('player', 'underdog') for feature in features]]).any(axis=1))
    )
    games.loc[mask, [f'favored_PCA{i}' for i in range(0, n_components)]] = pca.transform(scaler.transform(games.loc[mask, [feature.replace('player', 'favored') for feature in features]].values))[:, :n_components]
    games.loc[mask, [f'underdog_PCA{i}' for i in range(0, n_components)]] = pca.transform(scaler.transform(games.loc[mask, [feature.replace('player', 'underdog') for feature in features]].values))[:, :n_components]
    
    # Count number of observations for each player
    players_df = full_games_train['player'].value_counts().reset_index()
    players_df.columns = ['player', 'num_obs']
    players_df = players_df[players_df['num_obs'] >= 20].reset_index(drop=True)
    features = [feature for feature in features if feature not in ['surface_Hard', 'surface_Grass', 'surface_Clay', 'player_hand_L', 'player_hand_R', 'player_ht']]
    for index, row in tqdm(players_df.iterrows(), total=len(players_df)):
        player_log = full_games_train[(full_games_train['player'] == row['player']) & (full_games_train['comment'] == 'Completed')].copy()
        
        # Time weights
        delta = 1.3
        n = len(player_log)
        times = (datetime(season, 12, 31) - player_log['tourney_date']).dt.days / 365.25
        weights = np.minimum(np.exp(-delta*times), 0.8)
        player_log['time_weight'] = weights
        
        player_log['time_weight'] /= player_log['time_weight'].sum()
        
        # Overall weights
        player_log['weight'] = player_log['time_weight'] 
        
        for feature in features:
            players_df.loc[index, feature] = weighted_average(player_log[feature], player_log['time_weight'])
        for i in range(0, n_components):
            players_df.loc[index, f'player_avg_PCA{i}'] = weighted_average(player_log[f'PCA{i}'], player_log['time_weight'])
    
    PCA_avg = players_df[['player'] + [f'player_avg_PCA{i}' for i in range(0, n_components)]].set_index('player')
    K_range = range(2, 20)
    
    silhouette_scores = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(PCA_avg)
        if k > 1:
            score = silhouette_score(PCA_avg, labels)
        else:
            score = np.nan
        silhouette_scores.append(score)
    silhouette_scores_df = pd.DataFrame({'k': K_range, 'silhouette_score': silhouette_scores})
    
    n_clusters = int(silhouette_scores_df.loc[np.argmax(silhouette_scores)]['k'])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(PCA_avg)
    players_df['player_cluster'] = clusters
    
    """
    value_combinations = list(combinations(range(0, n_components), 2))
    for i,j in value_combinations:
        plt.figure(figsize=(8, 6))
        plt.scatter(PCA_avg[f'player_avg_PCA{i}'], PCA_avg[f'player_avg_PCA{j}'], c=clusters, cmap='tab10', s=100)
        for name in PCA_avg.index:
            plt.text(PCA_avg.loc[name, f'player_avg_PCA{i}']+0.02, PCA_avg.loc[name, f'player_avg_PCA{j}']+0.02, name, fontsize=9)
        
        plt.title("Player Style Clusters")
        plt.xlabel(f"Component {i+1}")
        plt.ylabel(f"Component {j+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    """
    
    for i in range(0, n_components):
        games.loc[games['season'] == season, f'favored_avg_PC{i}'] = games['favored'].map(players_df.set_index('player')[f'player_avg_PCA{i}'])
        games.loc[games['season'] == season, f'underdog_avg_PC{i}'] = games['underdog'].map(players_df.set_index('player')[f'player_avg_PCA{i}'])
        
        full_games.loc[full_games['season'] == season, f'player_avg_PC{i}'] = full_games['player'].map(players_df.set_index('player')[f'player_avg_PCA{i}'])
        full_games.loc[full_games['season'] == season, f'rival_avg_PC{i}'] = full_games['rival'].map(players_df.set_index('player')[f'player_avg_PCA{i}'])
    
    games.loc[games['season'] == season, 'favored_PCA_cluster'] = games['favored'].map(players_df.set_index('player')['player_cluster']).astype('Int64')
    games.loc[games['season'] == season, 'underdog_PCA_cluster'] = games['underdog'].map(players_df.set_index('player')['player_cluster']).astype('Int64')
    
    # Latent Features Extraction
    players = players_df['player'].tolist()
    full_games_train = full_games_train[(full_games_train['player'].isin(players)) & (full_games_train['rival'].isin(players))]
    
    score_df = pd.pivot_table(
        full_games_train,
        values='player_game_score',
        index='player',
        columns='rival',
        aggfunc='mean'
    ).reindex(index=players, columns=players)
    score_df = score_df.iloc[:10, :10]
    score_matrix = score_df.values
    games_mask = ~np.isnan(score_matrix)
    print(score_df)
    
    from scipy.spatial.distance import pdist, squareform
    score_df = score_df.fillna(0)
    score_matrix = score_df.values
    dist_matrix = squareform(pdist(score_matrix, metric='correlation'))
    similarity_df = pd.DataFrame(dist_matrix, index=score_df.index, columns=score_df.index)
    print(similarity_df)
    
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
                distance = normalized_distance(vec_a, vec_b, len(vec_a)) if len(vec_a) > 0 else np.nan
                #distance = correlation(vec_a, vec_b)
                distance_matrix.loc[player_a, player_b] = distance
    print(distance_matrix)
    
    # Impute NaNs in score_df with the most frequent value in each row
    for idx in distance_matrix.index:
        row = distance_matrix.loc[idx]
        if row.notna().any():
            most_freq = row.mean()
            distance_matrix.loc[idx] = row.fillna(most_freq)
    
    
    score_df = score_df.fillna(0)
    score_matrix = score_df.values
    num_users, num_items = score_matrix.shape
    num_factors = 4
    
    svd = TruncatedSVD(n_components=num_factors, random_state=42)
    P = svd.fit_transform(score_matrix)
    Q = svd.components_.T
    
    
    """
    records = []
    for player in distance_matrix.index:
        for rival in distance_matrix.columns:
            if player != rival:
                score = score_df.loc[player, rival]
                records.append({'player': player, 'rival': rival, 'score': score})

    ratings = pd.DataFrame(records)
    
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate
    ratings['player_id'] = ratings['player'].astype("category").cat.codes
    ratings['rival_id'] = ratings['rival'].astype("category").cat.codes

    reader = Reader(rating_scale=(ratings['score'].min(), ratings['score'].max()))
    data = Dataset.load_from_df(ratings[['player_id', 'rival_id', 'score']], reader)

    algo = SVD(n_factors=5, random_state=42)
    cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=True)

    # Fit on full data
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    """
    
    """
    score_matrix = score_df.values
    num_users, num_items = score_matrix.shape
    num_factors = 4
    
    # Random initialization
    np.random.seed(42)
    P = np.random.rand(num_users, num_factors)
    Q = np.random.rand(num_items, num_factors)

    # Mask for known ratings
    mask = score_matrix != 0

    # Training
    alpha = 0.01
    lambda_reg = 0.1
    epochs = 5000
    for epoch in tqdm(range(epochs)):
        for i in range(num_users):
            for j in range(num_items):
                if mask[i, j]:
                    error = score_matrix[i, j] - np.dot(P[i, :], Q[j, :])
                    # Update rules
                    P[i, :] += alpha * (error * Q[j, :] - lambda_reg * P[i, :])
                    Q[j, :] += alpha * (error * P[i, :] - lambda_reg * Q[j, :])
    
    # Testing
    R_hat = np.dot(P, Q.T)
    score_df_hat = pd.DataFrame(R_hat, index=score_df.index, columns=score_df.columns)
    print(score_df_hat)
    """
    
    """
    # PMF
    def matrix_to_triplets(M_df):
        triplets = []
        for i, row in enumerate(M_df.index):
            for j, col in enumerate(M_df.columns):
                value = M_df.iloc[i, j]
                if not np.isnan(value):
                    triplets.append((row, col, value))
        return triplets
    
    from cornac.data import Reader, Dataset
    from cornac.models import PMF
    triplets = matrix_to_triplets(score_df)
    reader = Reader()
    data = Dataset.from_uir(triplets)
    
    num_factors = 4
    pmf = PMF(k=num_factors,  # número de factores latentes (ajustalo)
            learning_rate=0.005,
            lambda_reg=0.02,
            verbose=True,
            seed=42)
    pmf.fit(data)
    
    P = pmf.get_user_vectors()  # matriz de jugadores fila
    Q = pmf.get_item_vectors()
    """

    score_df = score_df.fillna(0)
    score_matrix = score_df.values
    
    M_min = np.nanmin(score_matrix)
    M_max = np.nanmax(score_matrix)
    M_scaled = (score_matrix - M_min) / (M_max - M_min)
    
    from sklearn.decomposition import NMF
    num_factors=10
    nmf = NMF(n_components=num_factors, init='nndsvda', max_iter=1000, random_state=42)
    P = nmf.fit_transform(M_scaled)
    Q = nmf.components_.T
    
    
    # Clusters
    P_scaled = StandardScaler().fit_transform(P)
    Q_scaled = StandardScaler().fit_transform(Q)
    P_players_df = pd.DataFrame({'player': score_df.index}).set_index('player')
    Q_players_df = pd.DataFrame({'rival': score_df.columns}).set_index('rival')
    P_players_df[[f'off_latent_factor{i}' for i in range(0, num_factors)]] = P_scaled
    Q_players_df[[f'def_latent_factor{i}' for i in range(0, num_factors)]] = Q_scaled
    
    players_df.drop(columns=[col for col in players_df.columns if col.startswith('off_latent') or col.startswith('def_latent') or col.startswith('attacker_cluster')], inplace=True)
    players_df = pd.concat([players_df, P_players_df, Q_players_df], axis=1)
    
    scaler = StandardScaler()
    #X_scaled = pd.concat([pd.DataFrame(scaler.fit_transform(players_df[features]),columns=features, index = players_df.index), players_df[[f'off_latent_factor{i}' for i in range(0, num_factors)]]], axis = 1)
    #X_scaled = pd.DataFrame(scaler.fit_transform(players_df[features]),columns=features)
    
    pca = PCA(n_components=len(X_scaled.columns))
    pca.fit(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = explained_var.cumsum()
    print(cumulative_var)
    
    players_df = players_df.loc[:, ~players_df.columns.str.startswith('PCA')]
    n_components = 2 #int(input("Enter the number of components to keep based on cumulative variance: "))
    players_df[[f'PCA{i}' for i in range(0, n_components)]] = pca.transform(X_scaled)[:, :n_components]
    
    silhouette_scores = []
    #factors = [f'latent_factor{i}' for i in range(0, num_factors)] #+ [f'player_avg_PC{i}' for i in range(0, 6)]
    #factors = [f'PCA{i}' for i in range(0, n_components)]
    factors = [f'off_latent_factor{i}' for i in range(0, num_factors)]
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(players_df[factors].values)
        score = silhouette_score(players_df[factors].values, labels)
        
        silhouette_scores.append(score)
    silhouette_scores_df = pd.DataFrame({'k': K_range, 'silhouette_score': silhouette_scores})
    
    num_clusters = silhouette_scores_df.loc[np.argmax(silhouette_scores_df['silhouette_score']), 'k']
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    players_df['attacker_cluster'] = kmeans.fit_predict(players_df[factors].values) 
    print(players_df.sort_values('attacker_cluster'))
    
    value_combinations = list(combinations(range(0, num_factors), 2))
    for i,j in value_combinations:
        plt.figure(figsize=(8, 6))
        plt.scatter(players_df[factors].values[:, i], players_df[factors].values[:, j], c=players_df['attacker_cluster'], cmap='tab10', s=100)
        for m, name in enumerate(players_df.index):
            plt.text(players_df[factors].values[m, i]+0.02, players_df[factors].values[m, j]+0.02, name, fontsize=9)

        plt.title("Offensive Player Style Clusters")
        plt.xlabel(f"Latent Style Factor {i+1}")
        plt.ylabel(f"Latent Style Factor {j+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    value_combinations = list(combinations(range(0, num_factors), 2))
    for i,j in value_combinations:
        plt.figure(figsize=(8, 6))
        plt.scatter(Q_scaled[:, i], Q_scaled[:, j], c=Q_clusters, cmap='tab10', s=100)
        for m, name in enumerate(score_df.index):
            plt.text(Q_scaled[m, i]+0.02, Q_scaled[m, j]+0.02, name, fontsize=9)
        
        plt.title("Defensive Player Style Clusters")
        plt.xlabel(f"Latent Style Factor {i+1}")
        plt.ylabel(f"Latent Style Factor {j+1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    cluster_matchups = np.zeros((num_clusters, num_clusters))
    counts = np.zeros((num_clusters, num_clusters))
    for i in range(num_users):
        for j in range(num_items):
            if mask[i, j]:
                c_p = P_clusters[i]
                c_q = Q_clusters[j]
                cluster_matchups[c_p, c_q] += score_matrix[i, j]
                counts[c_p, c_q] += 1

    mean_cluster_effect = cluster_matchups / np.maximum(counts, 1)
    sns.heatmap(mean_cluster_effect, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Cluster-vs-Cluster Style Exploitation Matrix")
    plt.xlabel("Defender Style Cluster")
    plt.ylabel("Attacker Style Cluster")
    plt.show()


# Undirected Graph
import networkx as nx
from community import community_louvain

# Matchup Matrix
G = nx.Graph()
players = score_df.index
for i, p1 in enumerate(score_df):
    for j, p2 in enumerate(score_df):
        if j <= i:
            continue  # para no duplicar aristas (solo parte superior de la matriz)
        
        if p2 in score_df.columns:
            val = score_df.loc[p1, p2]
            if pd.notna(val):
                weight = abs(val)
                if weight > 0:
                    G.add_edge(p1, p2, weight=weight)

# Distance Matrix
G = nx.Graph()
players = distance_matrix.index
for i, p1 in enumerate(players):
    for j, p2 in enumerate(players):
        if j <= i:
            continue  # ensure undirected (i < j)
        
        val1 = distance_matrix.loc[p1, p2] if p2 in distance_matrix.columns else np.nan
        val2 = distance_matrix.loc[p2, p1] if p1 in distance_matrix.columns else np.nan
        
        if pd.notna(val1) and pd.notna(val2):
            weight = (val1 + val2) / 2
        elif pd.notna(val1):
            weight = val1
        elif pd.notna(val2):
            weight = val2
        else:
            continue  # no edge if both are NaN
        
        G.add_edge(p1, p2, weight=weight)


# Step 3: Louvain community detection
partition = community_louvain.best_partition(G, weight='weight')

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=[partition[n] for n in G.nodes()], cmap=plt.cm.Set3, node_size=600)
plt.title("Louvain Communities")
plt.show()

edges = []
for u, v, d in G.edges(data=True):
    edges.append({
        'player': u,
        'rival': v,
        'score': d['weight'],
        'player_cluster': partition[u],
        'rival_cluster': partition[v]
    })

df_edges = pd.DataFrame(edges)
community_matrix = df_edges.groupby(['player_cluster', 'rival_cluster'])['score'].mean().unstack()
print(community_matrix.round(3))
df_edges = df_edges.drop_duplicates(subset=['player']).drop(columns = ['rival', 'score', 'rival_cluster']).sort_values('player_cluster').reset_index(drop=True)
print(df_edges)


# Directed Graph
import networkx as nx
G = nx.DiGraph()

for player in score_df.index:
    for rival in score_df.columns:
        value = score_df.loc[player, rival]
        if pd.notna(value) and value > 0:
            G.add_edge(player, rival, weight=value)

pos = nx.spring_layout(G, seed=42)
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

nx.draw(
    G, pos,
    with_labels=True,
    node_size=600,
    edge_color=edge_weights,
    edge_cmap=plt.cm.viridis,
    arrows=True,
    width=2,
    connectionstyle='arc3,rad=0.1'
)
plt.title("Directed Graph from Anti-symmetric Score Matrix")
plt.show()


import igraph as ig
import leidenalg

edges = []
weights = []
for player in score_df.index:
    for rival in score_df.columns:
        value = score_df.loc[player, rival]
        if pd.notna(value):
            edges.append((player, rival))
            weights.append(value)

# Crear el grafo en igraph
players = list(score_df.index)
g = ig.Graph(directed=True)
g.add_vertices(players)
g.add_edges(edges)
g.es['weight'] = weights

# Detección de comunidades
partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, weights=g.es['weight'], resolution_parameter=0.1)

# Mostrar resultados
for vertex, cluster in zip(g.vs['name'], partition.membership):
    print(f"{vertex}: Cluster {cluster}")

# Dibujar el grafo con clusters
layout = g.layout('fr')  # Fruchterman-Reingold layout (bastante estético)

# Colores para cada cluster
import random
random.seed(42)
num_clusters = len(set(partition.membership))
palette = [plt.cm.tab10(i) for i in range(num_clusters)]

colors = [palette[cluster] for cluster in partition.membership]

fig, ax = plt.subplots(figsize=(8, 6))
ig.plot(
    g,
    target=ax,
    layout=layout,
    vertex_color=colors,
    vertex_label=g.vs['name'],
    vertex_size=30,
    edge_width=[1 + 3*w for w in g.es['weight']],
    bbox=(600, 600),
)
plt.show()




#<----------------------------------------------------------------------------------------------------
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

linked = sch.linkage(squareform(distance_matrix), method='ward')  # También puedes usar 'ward', 'complete', etc.
plt.figure(figsize=(8, 5))
dendro = sch.dendrogram(linked, labels=distance_matrix.index.tolist())
plt.title("Clustering jerárquico de jugadores por estilo")
plt.ylabel("Distancia")
plt.tight_layout()
plt.show()

#clusters = fcluster(linked, t = 7, criterion='distance')

hier_num_clusters = 4 # or set manually
hier_clusters = fcluster(linked, hier_num_clusters, criterion='maxclust')
hier_cluster_df = pd.DataFrame({'player': distance_matrix.index, 'hier_cluster': hier_clusters}).set_index('player').sort_values('hier_cluster')
hier_cluster_df['hier_cluster'] -= 1
hier_cluster_df
print(hier_cluster_df)

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
plt.show()

#<-----------------------------------------------------------------------------------------------------
games['favored_off_cluster'] = games['favored'].map(P_cluster_df.set_index('player')['attacker_cluster']).astype('Int64')
games['favored_def_cluster'] = games['favored'].map(Q_cluster_df.set_index('rival')['defender_cluster']).astype('Int64')
games['underdog_off_cluster'] = games['underdog'].map(P_cluster_df.set_index('player')['attacker_cluster']).astype('Int64')
games['underdog_def_cluster'] = games['underdog'].map(Q_cluster_df.set_index('rival')['defender_cluster']).astype('Int64')

full_games['player_off_cluster'] = full_games['player'].map(P_cluster_df.set_index('player')['attacker_cluster']).astype('Int64')
full_games['player_def_cluster'] = full_games['player'].map(Q_cluster_df.set_index('rival')['defender_cluster']).astype('Int64')
full_games['rival_off_cluster'] = full_games['rival'].map(P_cluster_df.set_index('player')['attacker_cluster']).astype('Int64')
full_games['rival_def_cluster'] = full_games['rival'].map(Q_cluster_df.set_index('rival')['defender_cluster']).astype('Int64')

test =full_games[((full_games['player_off_cluster'] == 4) & (full_games['rival_def_cluster'] == 6))][['player', 'rival', 'season', 'tourney_name', 'player_off_cluster', 'rival_def_cluster', 'win', 'game_score']]
full_games[full_games['season'] >= 2024]['game_score'].sum()


games_test = games[(~games['favored_off_cluster'].isna()) & (~games['underdog_def_cluster'].isna()) & (games['season'] >= 2024)].reset_index(drop=True)

games_test['favored_cluster_score'] = games_test.apply(lambda row: mean_cluster_effect[row['favored_off_cluster'], row['underdog_def_cluster']],axis=1)
games_test['underdog_cluster_score'] = games_test.apply(lambda row: mean_cluster_effect[row['underdog_off_cluster'], row['favored_def_cluster']],axis=1)
games_test['cluster_score_diff'] = games_test['favored_cluster_score']-games_test['underdog_cluster_score'] 

games_test['elo_diff_x_cluster_score_diff'] = games_test['elo_diff'] * games_test['cluster_score_diff']
games_test['elo_surface_diff_x_cluster_score_diff'] = games_test['elo_surface_diff'] * games_test['cluster_score_diff']
games_test['log_elo_diff_x_cluster_score_diff'] = games_test['log_elo_diff'] * games_test['cluster_score_diff']
games_test['log_elo_surface_diff_x_cluster_score_diff'] = games_test['log_elo_surface_diff'] * games_test['cluster_score_diff']

X = games_test[['elo_surface_diff', 'cluster_score_diff', 'favored_cluster_score', 'underdog_cluster_score',
                'elo_surface_diff_x_cluster_score_diff']].values
y = games_test['favored_win'].values

lr = LogisticRegression()
lr.fit(X, y)

y_pred_proba = lr.predict_proba(X)[:, 1]
print(f'Log Loss: {log_loss(y, y_pred_proba):.4f}')
print(f'Log Loss: {log_loss(y, elo_probability(games_test['favored_elo_surface'], games_test['underdog_elo_surface'])):.4f}')
