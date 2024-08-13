import pandas as pd
import numpy as np
import pickle

# Load the similarity engine from the pickle file
with open('engine.pickle', 'rb') as file:
    loaded_engine = pickle.load(file)

# Reload the cleaned CSV file to get the full DataFrame with string columns
csv_file = '/Matchmaker/Code/Players.csv'                                                   # Update Pathname 
df = pd.read_csv(csv_file)

# Create a dictionary to map Overstat IDs to player names
id_to_name = dict(zip(df['Overstat ID'], df['EA ID']))

# Define the necessary columns and structures
players = df['Overstat ID'].tolist()

# Columns to focus on for imputation
impute_columns = ['grenades', 'ults', 'tacts', 'damage_taken', 'grenades_avg', 'ults_avg', 'tacts_avg']

# Function to find similar players based on Overstat ID
def find_similar_players(overstat_id, top_n=5):
    if overstat_id not in loaded_engine:
        return []
    similarity_scores = loaded_engine[overstat_id]
    similar_players = np.argsort(similarity_scores)[-top_n:][::-1]
    return [(players[idx], similarity_scores[idx]) for idx in similar_players]

# Iterate through the DataFrame to find and replace values
for idx, row in df.iterrows():
    overstat_id = row['Overstat ID']
    if all(row[column] == 0 or pd.isna(row[column]) for column in impute_columns):
        similar_players = find_similar_players(overstat_id, top_n=5)
        
        for column in impute_columns:
            # Calculate the average for each column to be imputed from the similar players
            similar_values = []
            for sim_player_id, _ in similar_players:
                sim_player_row = df[df['Overstat ID'] == sim_player_id]
                if not sim_player_row.empty and sim_player_row[column].iloc[0] != 0 and not pd.isna(sim_player_row[column].iloc[0]):  # Only use non-zero and non-NaN values
                    similar_values.append(sim_player_row[column].iloc[0])
            if similar_values:
                df.at[idx, column] = round(np.mean(similar_values), 2)

# Debugging: Print the last row before and after to verify changes
print("Before imputation:")
print(df.iloc[-1])

# Export the completed DataFrame to a new CSV file
output_csv_file = '/Matchmaker/Code/nomissing.csv'                                      # Update Pathname
df.to_csv(output_csv_file, index=False)

print("After imputation:")
print(df.iloc[-1])

print(f"Completed DataFrame has been saved to {output_csv_file}")
