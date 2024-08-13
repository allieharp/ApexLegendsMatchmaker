import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from tqdm import tqdm
import pickle
import os

# Load player data from CSV file
csv_file = '/Matchmaker/Code/Players.csv'               # Update Pathname
df = pd.read_csv(csv_file)

# Drop columns that are not needed for similarity calculations
df_nums = df.drop(columns=['EA ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL', 'Other Names'])
df_cleaned = df_nums.drop(columns=['damage taken*', 'damage_taken', 'total_games', 'damage taken*_avg', 'revives', 'revives_avg', 'respawns', 'respawns_avg', 'headshots_avg', 'hits_avg', 'shots_avg', 'downs_avg', 'totalgames_avg'])

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
df_scaled = scaler.fit_transform(df_cleaned.drop(columns=['Overstat ID', 'Second ID']))

# Create mappings and arrays for player statistics
player_ID = {row['Overstat ID']: idx for idx, row in df_cleaned.iterrows()}
players = df_cleaned['Overstat ID'].tolist()
stats = df_scaled

# Function to get player stats vector
def getStats(overstat_id):
    idx = player_ID[overstat_id]
    return stats[idx, :]

# Function to compute cosine similarity between players
def similarity(player1, player2):
    return 1 - distance.cosine(getStats(player1), getStats(player2))

# Normalize cosine similarity values to a scale of 0 to 100
def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return np.array([round((num - min_val) * 100 / (max_val - min_val), 2) for num in array])

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

# Create similarity engine
def create_similarity_engine(metric='cosine'):
    engine = {}
    for query in tqdm(players):
        metric_values = []
        for player in players:
            if metric == 'cosine':
                value = similarity(query, player)
            else:
                value = distance_metric(query, player, metric=metric)
            metric_values.append(value)
        if metric == 'cosine':  # Normalize similarity values for cosine metric
            metric_values = normalize(metric_values)
        engine[query] = metric_values
    return engine

# Save the similarity engine using pickle
def save_engine(engine, filename):
    with open(filename, 'wb') as file:
        pickle.dump(engine, file)
    print(f"Similarity engine has been created and saved as {filename}.")

# Load the similarity engine from pickle
def load_engine(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Function to find similar players for cosine similarity
def find_similar_players_cosine(engine, overstat_id, top_n=5):
    similarity_scores = engine[overstat_id]
    similar_players = np.argsort(similarity_scores)[-top_n:][::-1]
    return [(players[idx], similarity_scores[idx]) for idx in similar_players]

# Function to find similar players for distance metrics
def find_similar_players_distance(engine, overstat_id, top_n=5):
    distance_scores = engine[overstat_id]
    similar_players = np.argsort(distance_scores)[:top_n]
    return [(players[idx], distance_scores[idx]) for idx in similar_players]

# Get user input for Overstat ID, number of similar players, and metric type
user_overstat_id = int(input("Enter Overstat ID: "))
num_similar_players = int(input("Enter number of similar players to find: "))
metric_type = input("Enter similarity metric (cosine, euclidean, manhattan, mixed): ").strip().lower()

# Always create a new similarity engine based on user input
engine = create_similarity_engine(metric=metric_type)
engine_filename = f'engine_{metric_type}.pickle'
save_engine(engine, engine_filename)

# Create a dictionary to map Overstat IDs to player names
id_to_name = dict(zip(df['Overstat ID'], df['EA ID']))

# Find and print similar players based on the metric type
if metric_type == 'cosine':
    similar_players = find_similar_players_cosine(engine, user_overstat_id, num_similar_players)
    print(f"Calculated using {metric_type} metric")
    print(f"Top {num_similar_players} similar players to {id_to_name[user_overstat_id]} ({user_overstat_id}):")
    print(f"{'EA ID':<30} {'Overstat ID':<15} {'Similarity':<10}")
    for player_id, score in similar_players:
        player_name = id_to_name.get(player_id, "Unknown")
        print(f"{player_name:<30} {player_id:<15} {score:<10}")
else:
    similar_players = find_similar_players_distance(engine, user_overstat_id, num_similar_players)
    print(f"Calculated using {metric_type} metric")
    print(f"Top {num_similar_players} similar players to {id_to_name[user_overstat_id]} ({user_overstat_id}):")
    print(f"{'EA ID':<30} {'Overstat ID':<15} {'Distance':<10}")
    for player_id, score in similar_players:
        player_name = id_to_name.get(player_id, "Unknown")
        print(f"{player_name:<30} {player_id:<15} {score:<10}")
