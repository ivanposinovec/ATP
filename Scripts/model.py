import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import statistics
import tensorflow.keras
from sklearn import metrics
import time
import tensorflow.keras.initializers
import tensorflow.keras
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from tensorflow.keras.layers import LeakyReLU,PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings("ignore",category =RuntimeWarning)

games = pd.read_csv('data.csv')
features_names = ['season', 'week', 'uncertainty'] + [feature for feature in games.columns if '_diff' in feature] 
dummies_features = [feature for feature in games.columns if feature.startswith(('tourney_series_', 'surface_', 'round_', 'best_of_', 'entry_match_', 'hand_match_', ))]


X = games[features_names+dummies_features]
y = games['favored_win']
odds = games[['favored', 'underdog', 'season', 'tourney_full_name', 'tourney_date', 'round', 'favored_win', 'favored_odds', 'underdog_odds', 'favored_max_odds', 'underdog_max_odds', 'favored_prob_implied', 'underdog_prob_implied']].copy()

train_index = games[(games['season'] >= 2004) & (games['season'] < 2016)].index
val_index = games[(games['season'] >= 2016) & (games['season'] < 2018)].index
train_val_index = games[(games['season'] >= 2004) & (games['season'] < 2018)].index
test_index = games[games['season'] >= 2018].index
X_train, X_val, X_train_val, X_test = X.loc[train_index], X.loc[val_index], X.loc[train_val_index], X.loc[test_index]
y_train, y_val, y_train_val, y_test = y.loc[train_index], y.loc[val_index], y.loc[train_val_index], y.loc[test_index]
odds_train, odds_val, odds_train_val, odds_test = odds.loc[train_index], odds.loc[val_index], odds.loc[train_val_index], odds.loc[test_index]


# High leverage values removal
features_to_remove_high_leverage = [feature for feature in X_train_val.columns if feature not in dummies_features+['uncertainty']]
indexes_to_drop = []
for col in features_to_remove_high_leverage:
    Q1 = X_train_val[col].quantile(0.2)
    Q3 = X_train_val[col].quantile(0.8)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    indexes_to_drop.append(list(X_train_val[(X_train_val[col] <= lower_bound) | (X_train_val[col] >= upper_bound)].index))
    print(f'{col} Number of indexes to drop: {len(list(X_train_val[(X_train_val[col] <= lower_bound) | (X_train_val[col] >= upper_bound)].index))}')

indexes_to_drop = list(set([index for sublist in indexes_to_drop for index in sublist]))
indexes = [index for index in X_train_val.index if index not in indexes_to_drop]
print(f'Number of indexes to drop: {len(indexes_to_drop)}')
print(f'Number of indexes kept: {len(indexes)}')

X_train = X_train.loc[[index for index in indexes if index in X_train.index]]
y_train = y.loc[[index for index in indexes if index in X_train.index]]
odds_train = odds.loc[[index for index in indexes if index in X_train.index]]

X_train_val = X_train_val.loc[indexes]
y_train_val = y_train_val.loc[indexes]
odds_train_val = odds_train_val.loc[indexes]


# Scaling
features_to_scale = [feature for feature in X_train_val.columns if X_train_val[feature].nunique() > 2]
features_not_to_scale = [feature for feature in X_train_val.columns if feature not in features_to_scale]

scaler = StandardScaler()
scaler.fit(X_train_val[features_to_scale])

X_train_scaled = pd.concat([X_train[features_not_to_scale], pd.DataFrame(scaler.transform(X_train[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_train.index)], axis = 1)
X_val_scaled = pd.concat([X_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_val[features_to_scale]), columns = X_val[features_to_scale].columns, index = X_val.index)], axis = 1)
X_train_val_scaled = pd.concat([X_train_val[features_not_to_scale], pd.DataFrame(scaler.transform(X_train_val[features_to_scale]), columns = X_train_val[features_to_scale].columns, index = X_train_val.index)], axis = 1)
X_test_scaled = pd.concat([X_test[features_not_to_scale], pd.DataFrame(scaler.transform(X_test[features_to_scale]), columns = X_train[features_to_scale].columns, index = X_test.index)], axis = 1)


def generate_model(dropout, neurons):
    model = Sequential([
        Input(shape=(X_train_val_scaled.shape[1],)),
        Dense(int(neurons), activation='relu'),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    return model

#model = generate_model(dropout=0.2, neurons = 500)
#model.summary()

SPLITS = 3
EPOCHS = 500
PATIENCE = 10
def evaluate_network(dropout,neurons,learning_rate):
    # Bootstrap for Classification
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.1) 
    
    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0
    
    # Loop through samples
    for train, test in boot.split(X_train_val_scaled, y_train_val):
        start_time = time.time()
        num+=1
        
        # Split train and test
        x_train = X_train_val_scaled.iloc[train]
        y_train = y_train_val.iloc[train]
        x_test = X_train_val_scaled.iloc[test]
        y_test = y_train_val.iloc[test]
        
        model = generate_model(dropout, neurons)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate))
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=PATIENCE, verbose=0, mode='auto', restore_best_weights=True)
        
        # Train on the bootstrap sample
        model.fit(x_train,y_train,validation_data=(x_test,y_test), callbacks=[monitor],verbose=0,epochs=EPOCHS)
        epochs = monitor.stopped_epoch
        epochs_needed.append(epochs)
        
        # Predict on the out of boot (validation)
        pred = model.predict(x_test)
        
        # Measure this bootstrap's log loss
        score = metrics.log_loss(y_test, pred)
        mean_benchmark.append(score)
        m1 = statistics.mean(mean_benchmark)
        m2 = statistics.mean(epochs_needed)
        mdev = statistics.pstdev(mean_benchmark)
        
        # Record this iteration
        time_took = time.time() - start_time
        
    tensorflow.keras.backend.clear_session()
    return (-m1)

#print(evaluate_network(dropout=0.2, neurons = 500, learning_rate = 0.01))

# Bounded region of parameter space
pbounds = {
    'neurons': (5, 250),
    'dropout': (0.1, 0.5),
    'learning_rate': (1e-4, 1e-2)
}

optimizer = BayesianOptimization(
    f=evaluate_network,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=42
)

start_time = time.time()
optimizer.maximize(init_points=10, n_iter=20,)
time_took = time.time() - start_time
print(optimizer.max)

# Best params
best_params = optimizer.max['params']

# Train the model with the best parameters
#evaluate_network(dropout=best_params['dropout'], neurons = best_params['learning_rate'], learning_rate = best_params['neurons'])
# Train the model with cross-validation using the best parameters

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = []

model = generate_model(dropout=best_params['dropout'], neurons=best_params['neurons'])
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=best_params['learning_rate']))
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=0, mode='auto', restore_best_weights=True)    
model.fit(X_train_val_scaled, y_train_val, validation_split=0.33, callbacks=[monitor], verbose=1, epochs=1000)


odds_test['favored_prob_nn'] = model.predict(X_test_scaled)
odds_test['underdog_prob_nn'] = 1-odds_test['favored_prob_nn']

print(f"Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_nn']):.5f}")
print(f"Odds Test Log Loss: {log_loss(odds_test['favored_win'], odds_test['favored_prob_implied']):.5f}")


