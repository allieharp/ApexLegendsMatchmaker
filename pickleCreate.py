import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from tqdm import tqdm
import pickle
import os

# Load player data from CSV file
csv_file = '/Matchmaker/Code/Players.csv'       # Update Pathname   
df = pd.read_csv(csv_file)

# Drop columns that are not needed for similarity calculations
df_nums = df.drop(columns=['EA ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL', 'Other Names'])
df_withoutmissingdata = df_nums.drop(columns=['grenades', 'grenades_avg', 'ults', 'ults_avg', 'tacts', 'tacts_avg', 'damage_taken', 'total_games'])
df_cleaned = df_withoutmissingdata.drop(columns=['damage taken*', 'damage taken*_avg', 'revives', 'revives_avg', 'respawns', 'respawns_avg', 'headshots_avg', 'hits_avg', 'shots_avg', 'downs_avg', 'totalgames_avg'])

# Calculate additional statistics
df_cleaned['headshot_accuracy'] = df_nums['headshots'] / df_nums['shots']
df_cleaned['shot_accuracy'] = df_nums['hits'] / df_nums['shots']
df_cleaned['finish_percentage'] = df_nums['kills'] / df_nums['downs']

# Drop rows with NaN values to clean the data
df_cleaned = df_cleaned.dropna()

# Sort by 'Overstat ID' and add a 'Second ID' column
df_sorted = df_cleaned.sort_values(by='Overstat ID').reset_index(drop=True)
df_sorted['Second ID'] = np.arange(1, len(df_sorted) + 1)
df_cleaned = pd.merge(df_cleaned, df_sorted[['Overstat ID', 'Second ID']], on='Overstat ID')

# Standardize the data to normalize the values
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cleaned)

# Create mappings and arrays for player statistics
player_ID = {row['Overstat ID']: idx for idx, row in df_cleaned.iterrows()}
players = df_cleaned['Overstat ID'].tolist()
stats = df_cleaned.drop(columns=['Overstat ID', 'Second ID']).values

# Function to get player stats vector
def getStats(overstat_id):
    idx = player_ID[overstat_id]
    return stats[idx, :]

# Function to compute cosine similarity between players
def similarity(player1, player2):
    return 1 - distance.cosine(getStats(player1), getStats(player2))

# Function to normalize an array to a scale of 0 to 100
def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return np.array([round((num - min_val) * 100 / (max_val - min_val), 2) for num in array])

# Create similarity engine
engine = {}
for query in tqdm(players):
    metric = []
    for player in players:
        value = similarity(query, player)
        metric.append(value)
    metric = normalize(metric)
    engine[query] = metric

# Save the similarity engine using pickle
with open('engine.pickle', 'wb') as file:
    pickle.dump(engine, file)

print("Similarity engine has been created and saved.")

# Load the similarity engine from pickle
with open('engine.pickle', 'rb') as file:
    loaded_engine = pickle.load(file)

# Create a dictionary to map Overstat IDs to player names
id_to_name = dict(zip(df['Overstat ID'], df['EA ID']))

# Function to find similar players
def find_similar_players(overstat_id, top_n=5):
    similarity_scores = loaded_engine[overstat_id]
    similar_players = np.argsort(similarity_scores)[-top_n:][::-1]
    return [(players[idx], similarity_scores[idx]) for idx in similar_players]

# Example usage to find similar players
hardcoded_overstat_id = 837458  # Replace with actual Overstat ID
top_n = 25

print(f"Calculated using cosine similarity")
similar_players = find_similar_players(hardcoded_overstat_id, top_n)
print(f"Top {top_n} similar players to {id_to_name[hardcoded_overstat_id]} ({hardcoded_overstat_id}):")
print(f"{'EA ID':<30} {'Overstat ID':<15} {'Similarity':<10}")
for player_id, score in similar_players:
    player_name = id_to_name.get(player_id, "Unknown")
    print(f"{player_name:<30} {player_id:<15} {score:<10}")

# Function to compute distance between two player vectors
def distance_metric(player1, player2, metric='mixed'):
    vec1 = getStats(player1)
    vec2 = getStats(player2)
    if metric == 'euclidean':
        return distance.euclidean(vec1, vec2)
    elif metric == 'manhattan':
        return distance.cityblock(vec1, vec2)
    elif metric == 'mixed':
        return (distance.euclidean(vec1, vec2) + distance.cityblock(vec1, vec2)) / 2
    else:
        raise ValueError("Unsupported metric. Use 'euclidean', 'manhattan', or 'mixed'.")

# Create distance-based similarity engine
metric_type = 'mixed'
engine = {}
for query in tqdm(players):
    metric = []
    for player in players:
        value = distance_metric(query, player, metric=metric_type)
        metric.append(value)
    metric = normalize(metric)
    engine[query] = metric

# Save the distance-based similarity engine
with open('engine_distance.pickle', 'wb') as file:
    pickle.dump(engine, file)

print("Distance-based similarity engine has been created and saved.")

# Load the distance-based similarity engine from pickle
with open('engine_distance.pickle', 'rb') as file:
    loaded_engine = pickle.load(file)

# Example usage with distance metric to find similar players
hardcoded_overstat_id = 1271
top_n = 25

print(f"Calculated using {metric_type} distance")
similar_players = find_similar_players(hardcoded_overstat_id, top_n)
print(f"Top {top_n} similar players to {id_to_name[hardcoded_overstat_id]} ({hardcoded_overstat_id}):")
print(f"{'Name':<30} {'Overstat ID':<15} {'Similarity':<10}")
for player_id, score in similar_players:
    player_name = id_to_name.get(player_id, "Unknown")
    print(f"{player_name:<30} {player_id:<15} {score:<10}")
