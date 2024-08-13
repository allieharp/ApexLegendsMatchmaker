from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import plotly.express as px
import os

# Input parameters
csv_file = '/Matchmaker/Code/Players1.csv'                              # Update Pathname
n_clusters = 9          # Number of clusters for K-means
random_state = 42       # Random state for reproducibility

# Function to load and clean data from the CSV file
def load_and_clean_data(csv_file):
    df = pd.read_csv(csv_file)
    
    # Drop non-numeric columns for initial processing
    df_nums = df.drop(columns=['EA ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL', 'Other Names'])
    
    # Drop unnecessary columns
    df_cleaned = df_nums.drop(columns=['damage_taken', 'damage taken*', 'totalgames', 'damage taken*_avg', 'revives', 'revives_avg', 'respawns', 'respawns_avg', 'grenades', 'grenades_avg', 'headshots_avg', 'hits_avg', 'shots_avg', 'downs_avg', 'totalgames_avg'])

    # Calculate additional statistics
    df_cleaned['headshot_accuracy'] = df_nums['headshots'] / df_nums['shots'].replace(0, np.nan)
    df_cleaned['shot_accuracy'] = df_nums['hits'] / df_nums['shots'].replace(0, np.nan)
    df_cleaned['finish_percentage'] = df_nums['kills'] / df_nums['downs'].replace(0, np.nan)

    # Drop rows with NaN values to ensure clean data
    df_cleaned = df_cleaned.dropna()

    # Sort and create a sequential ID for players
    df_sorted = df_cleaned.sort_values(by='Overstat ID').reset_index(drop=True)
    df_sorted['Second ID'] = np.arange(1, len(df_sorted) + 1)
    df_cleaned = pd.merge(df_cleaned, df_sorted[['Overstat ID', 'Second ID']], on='Overstat ID')
    
    # Merge EA ID back into the cleaned dataframe
    df_cleaned = pd.merge(df_cleaned, df[['Overstat ID', 'EA ID']], on='Overstat ID')
    return df_cleaned

# Load and clean the data
df_cleaned = load_and_clean_data(csv_file)

# Select only numeric columns for scaling
numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
df_scaled = StandardScaler().fit_transform(df_cleaned[numeric_columns])

# Define options for colormap columns
colormap_options = [
    'Second ID', 'kills', 'headshots', 'assists', 'time', 'damage', 'hits', 'downs',
    'shots', 'ults', 'tacts', 'total_games', 'kills_avg', 'assists_avg',
    'time_avg', 'damage_avg', 'ults_avg', 'tacts_avg', 'headshot_accuracy', 'shot_accuracy'
]

# Function for T-SNE visualization
def tsne_visualization(colormap_column, kmeans=False):
    # Perform T-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=28)
    tsne_results = tsne.fit_transform(df_scaled)

    # Create a dataframe with the T-SNE results
    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['Overstat ID'] = df_cleaned['Overstat ID']
    df_tsne['Second ID'] = df_cleaned['Second ID']
    df_tsne[colormap_column] = df_cleaned[colormap_column]
    df_tsne['EA ID'] = df_cleaned['EA ID']

    if kmeans:
        # Perform K-means clustering on T-SNE results
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_model.fit(tsne_results)
        clusters = kmeans_model.labels_
        df_tsne['Cluster'] = clusters.astype(str)

        # T-SNE with K-means Clustering Visualization
        fig = px.scatter(
            df_tsne, x='TSNE1', y='TSNE2',
            color='Cluster',
            title='T-SNE with K-means Clustering of Player Stats',
            labels={'Cluster': 'Cluster'},
            hover_data=['Overstat ID', 'EA ID', colormap_column],
            color_discrete_sequence=px.colors.qualitative.G10
        )
    else:
        # T-SNE Visualization
        fig = px.scatter(
            df_tsne, x='TSNE1', y='TSNE2',
            color=colormap_column,
            title='T-SNE Visualization of Player Stats',
            labels={colormap_column: colormap_column.replace('_', ' ').title()},
            hover_data=['Overstat ID', 'EA ID'],
            color_continuous_scale=px.colors.sequential.Viridis
        )
    fig.show()

# Function for PCA visualization
def pca_visualization(colormap_column, kmeans=False):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(df_scaled)
    
    # Create a dataframe with the PCA results
    df_pca = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    df_pca['Overstat ID'] = df_cleaned['Overstat ID']
    df_pca['Second ID'] = df_cleaned['Second ID']
    df_pca[colormap_column] = df_cleaned[colormap_column]
    df_pca['EA ID'] = df_cleaned['EA ID']

    if kmeans:
        # Perform K-means clustering on PCA results
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_model.fit(pca_results)
        clusters = kmeans_model.labels_
        df_pca['Cluster'] = clusters.astype(str)

        # PCA with K-means Clustering Visualization
        fig = px.scatter(
            df_pca, x='PCA1', y='PCA2',
            color='Cluster',
            title='PCA with K-means Clustering of Player Stats',
            labels={'Cluster': 'Cluster'},
            hover_data=['Overstat ID', 'EA ID', colormap_column],
            color_discrete_sequence=px.colors.qualitative.G10
        )
    else:
        # PCA Visualization
        fig = px.scatter(
            df_pca, x='PCA1', y='PCA2',
            color=colormap_column,
            title='PCA Visualization of Player Stats',
            labels={colormap_column: colormap_column.replace('_', ' ').title()},
            hover_data=['Overstat ID', 'EA ID'],
            color_continuous_scale=px.colors.sequential.Viridis
        )
    fig.show()

# User prompts for visualization choices
print("Choose the type of graph:")
print("1. T-SNE")
print("2. PCA")
graph_type = input("Enter the number for the type of graph: ").strip()

use_kmeans = input("Do you want to use K-means clustering? (yes or no): ").strip().lower() == 'yes'

print("\nChoose the column for the colormap:")
for i, option in enumerate(colormap_options, 1):
    print(f"{i}. {option}")
colormap_choice = int(input("Enter the number for the colormap: ").strip()) - 1
colormap_column = colormap_options[colormap_choice]

# Generate the selected visualization
if graph_type == '1':
    tsne_visualization(colormap_column, kmeans=use_kmeans)
elif graph_type == '2':
    pca_visualization(colormap_column, kmeans=use_kmeans)
else:
    print("Invalid choice. Please choose either '1' for T-SNE or '2' for PCA.")
