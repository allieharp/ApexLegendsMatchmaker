import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import plotly.express as px
import random
from sklearn.metrics import pairwise_distances_argmin_min

# Load player data
csv_file = '/Matchmaker/Code/RandomRoyalePlayers.csv'           # Update Pathname
df = pd.read_csv(csv_file)

# Data cleaning and feature engineering
df_cleaned = df.drop(columns=['EA ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL', 'Other Names'])
df_cleaned['headshot_accuracy'] = df['headshots'] / df['shots']
df_cleaned['shot_accuracy'] = df['hits'] / df['shots']
df_cleaned['finish_percentage'] = df['kills'] / df['downs']
df_cleaned = df_cleaned.dropna()

# Define weights for composite score calculation
weights = {
    'kills': 0.05,
    'assists': 0.05,
    'damage': 0.1,
    'kills_avg': 0.2,
    'assists_avg': 0.1,
    'damage_avg': 0.2,
    'shot_accuracy': 0.15,
    'headshot_accuracy': 0.1,
    'time_avg': 0.02,
    'total_games': 0.03
}


# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_cleaned)

# Perform T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=28)
tsne_results = tsne.fit_transform(df_scaled)

# Create a dataframe with the T-SNE results
df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
df_tsne['Overstat ID'] = df['Overstat ID'].astype(str)
df_tsne['EA ID'] = df['EA ID']

# Function to initialize cluster centers using kmeans++ method
def initialize_clusters(X, k):
    n_samples, n_features = X.shape
    centers = np.empty((k, n_features))
    centers[0] = X[np.random.randint(n_samples)]
    for i in range(1, k):
        distances = np.min(np.array([np.linalg.norm(X - center, axis=1)**2 for center in centers[:i]]), axis=0)
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_center = X[np.searchsorted(cumulative_probabilities, r)]
        centers[i] = next_center
    return centers

# Function to perform equal-size k-means clustering
def equal_size_kmeans(X, k, max_iter=100):
    n_samples = X.shape[0]
    cluster_size = n_samples // k
    centers = initialize_clusters(X, k)
    clusters = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        clusters, _ = pairwise_distances_argmin_min(X, centers)
        counts = np.bincount(clusters, minlength=k)
        excess = counts - cluster_size
        excess_clusters = np.where(excess > 0)[0]
        deficit_clusters = np.where(excess < 0)[0]
        for ex in excess_clusters:
            while excess[ex] > 0:
                sample_indices = np.where(clusters == ex)[0]
                distances = np.linalg.norm(X[sample_indices] - centers[ex], axis=1)
                sorted_indices = sample_indices[np.argsort(distances)]
                to_move = sorted_indices[0]
                best_deficit = deficit_clusters[np.argmin([np.linalg.norm(X[to_move] - centers[df]) for df in deficit_clusters])]
                clusters[to_move] = best_deficit
                excess[ex] -= 1
                excess[best_deficit] += 1
        centers = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
    return clusters

# Perform equal-size k-means clustering
n_clusters = 3
clusters = equal_size_kmeans(tsne_results, n_clusters)

# Create a dataframe with the T-SNE results and clusters
df_tsne['Cluster'] = clusters.astype(str)

# Function to print the number of players in each cluster
def print_cluster_sizes():
    cluster_sizes = df_tsne['Cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} players")

# Function to switch the clusters of two players by their Overstat IDs
def switch_clusters(id1, id2):
    cluster1 = df_tsne.loc[df_tsne['Overstat ID'] == id1, 'Cluster'].values[0]
    cluster2 = df_tsne.loc[df_tsne['Overstat ID'] == id2, 'Cluster'].values[0]
    df_tsne.loc[df_tsne['Overstat ID'] == id1, 'Cluster'] = cluster2
    df_tsne.loc[df_tsne['Overstat ID'] == id2, 'Cluster'] = cluster1
    print(f"Switched clusters of {id1} and {id2}")

# Display the clusters visually using Plotly
def plot_clusters():
    fig = px.scatter(
        df_tsne, x='TSNE1', y='TSNE2',
        color='Cluster',
        title='T-SNE with Equal-Size K-means Clustering of Player Stats',
        labels={'Cluster': 'Cluster'},
        hover_data=['Overstat ID', 'EA ID'],
        color_discrete_sequence=px.colors.qualitative.G10
    )
    fig.show()

# Initial plot
print_cluster_sizes()
plot_clusters()

# Function to adjust clusters manually
def adjust_clusters():
    print("Adjust clusters manually by switching players. Enter the Overstat IDs of the players to switch.")
    while True:
        id1 = input("Enter first Overstat ID to switch (or 'done' to finish): ")
        if id1.lower() == 'done':
            break
        id2 = input(f"Enter second Overstat ID to switch with {id1}: ")
        switch_clusters(str(id1), str(id2))
        print_cluster_sizes()
        plot_clusters()

# Allow manual adjustment of clusters
adjust_clusters()

# Calculate composite score for each player
composite_scores = pd.DataFrame(df_scaled, columns=df_cleaned.columns)[list(weights.keys())].mul(pd.Series(weights)).sum(axis=1)
df['Composite Score'] = composite_scores
df_tsne['Composite Score'] = composite_scores

# Rank players within each cluster based on their composite scores
df_tsne['Rank'] = df_tsne.groupby('Cluster')['Composite Score'].rank(method='first', ascending=False).astype(int)

# Print the ranking of players in each cluster
print("Player ranking within each cluster:")
for cluster in range(n_clusters):
    cluster_players = df_tsne[df_tsne['Cluster'] == str(cluster)].sort_values('Rank')
    print(f"\nCluster {cluster}:")
    print(cluster_players[['Overstat ID', 'EA ID', 'Rank', 'Composite Score']])

# Function to create teams
def create_teams(n_games, n_teams, df_tsne):
    teams = {f'Game {i+1}': [] for i in range(n_games)}
    all_players = set(df_tsne['Overstat ID'].astype(str))
    used_pairs = set()
    
    for game in range(n_games):
        remaining_players = all_players.copy()
        game_teams = []
        for _ in range(n_teams):
            team = []
            team_rank_sum = 0
            for cluster in range(n_clusters):
                cluster_players = df_tsne[df_tsne['Cluster'] == str(cluster)]
                available_players = cluster_players[cluster_players['Overstat ID'].isin(remaining_players)]
                if available_players.empty:
                    continue
                selected_player = available_players.sample().iloc[0]
                team.append(selected_player['Overstat ID'])
                team_rank_sum += selected_player['Rank']
                remaining_players.remove(selected_player['Overstat ID'])
            game_teams.append((team, team_rank_sum))
        # Sort teams by rank sum to balance skill level
        game_teams.sort(key=lambda x: x[1])
        teams[f'Game {game+1}'] = [team for team, _ in game_teams]
    return teams

# Assign players to teams for each game
n_games = 6
n_teams = len(df) // n_clusters

teams = create_teams(n_games, n_teams, df_tsne)

# Ensure no player has the same teammate more than once across all games
def validate_teams(teams):
    all_pairs = set()
    for game, game_teams in teams.items():
        for team in game_teams:
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    pair = tuple(sorted([team[i], team[j]]))
                    if pair in all_pairs:
                        return False
                    all_pairs.add(pair)
    return True

# Validate no player is on more than one team per game
def validate_unique_players(teams):
    validation_messages = []
    for game, game_teams in teams.items():
        player_to_teams = set()
        for team in game_teams:
            for player in team:
                if player in player_to_teams:
                    validation_messages.append(f"Player {player} is on more than one team in {game}")
                player_to_teams.add(player)
    return validation_messages

# Output the teams with average composite scores and validation
for game, game_teams in teams.items():
    print(f"\n{game}")
    for i, team in enumerate(game_teams, 1):
        player_names = [df.loc[df['Overstat ID'].astype(str) == player, 'EA ID'].values[0] for player in team]
        team_acs = df_tsne[df_tsne['Overstat ID'].isin(team)]['Composite Score'].mean()
        formatted_player_names = ', '.join([f"{name:<20}" for name in player_names])
        print(f"Team {i}: {formatted_player_names:<50} ACS: {team_acs:>2.2f}")

# Validate teams after printing
if not validate_teams(teams):
    print("All teams are valid: No player has the same teammate twice.")
else:
    print("Validation failed: Some players have the same teammates more than once.")

# Print validation for players on multiple teams
validation_messages = validate_unique_players(teams)
if validation_messages:
    for message in validation_messages:
        print(message)
else:
    print("Validation successful: No player is on more than one team per game.")
