import pandas as pd
import numpy as np
from collections import Counter
import re

# Load the original CSV file
csv_file = '/Matchmaker/Code/nomissing.csv'                         # Update Pathname
df = pd.read_csv(csv_file)

# Remove rows where 'damage_taken' is 0
df = df[df['damage_taken'] != 0]

# Remove columns that are not needed
columns_to_drop = ['damage taken*', 'totalgames', 'damage taken*_avg', 'revives', 'revives_avg', 'respawns', 'respawns_avg', 'headshots_avg', 'hits_avg', 'shots_avg', 'downs_avg', 'totalgames_avg']
df = df.drop(columns=columns_to_drop)

# List of words that should never be chosen
excluded_words = set([
    "ttv", "twitch", "tiktok", "tik", "tok", "youtube", "c9", "tsm", "nrg", "lg", "that", "ae", "crg", "lic", "love", "hate", "aim", "assist", "clg", "esa", "inf", "klg", "falcons", "col", "src", "ssg", "etr", "hke", "sir", "boy", "girl", "pfg",
    "faze", "nip", "navi", "gg", "flcn", "mst", "kick", "yt", "me", "i", ".", "dz", "9l", "the", "it", "iitz", "you", "she", "her", "he", "him", "jrx", "evo", "utft", "furia", "cce", "eec", "g2", "ftx", "123", "iam", "miss", "mr", "mrs"
])

# Function to determine the most common word (case-insensitive) excluding certain words
def most_common_word(other_names):
    if pd.isna(other_names) or len(other_names.split(',')) == 1:
        return other_names.strip()
    words = [word.strip().lower() for name in other_names.split(',') for word in re.split(r'\W+', name)]
    filtered_words = [word for word in words if word not in excluded_words and word]
    word_counts = Counter(filtered_words)
    
    # Filter words to those with length >= 3 if any exist
    words_3_or_more = [word for word in filtered_words if len(word) >= 3]
    if words_3_or_more:
        word_counts = Counter(words_3_or_more)
    
    most_common = word_counts.most_common(1)
    return most_common[0][0] if most_common else ''

# Track the number of names changed and store the changes
names_changed = 0
changes = []

# Update Name to the most common word from 'Other Names'
for idx, row in df.iterrows():
    original_name = row['EA ID']
    other_names = row.get('Other Names', '')
    if pd.isna(original_name) or original_name.strip() == '':
        most_common = most_common_word(other_names)
        if most_common:
            df.at[idx, 'EA ID'] = most_common
            names_changed += 1
            changes.append((original_name, most_common))

# Remove duplicate users based on 'Overstat ID' to ensure unique players
df = df.drop_duplicates(subset='Overstat ID')

# Define the numeric columns to retain and calculate additional statistics
numeric_columns = ['kills', 'headshots', 'assists', 'time', 'damage', 'hits', 'downs', 'shots', 'grenades', 'ults', 'tacts', 'damage_taken', 'total_games', 'kills_avg', 'assists_avg', 'time_avg', 'damage_avg', 'grenades_avg', 'ults_avg', 'tacts_avg']
df_nums = df[numeric_columns]

# Create new columns for headshot accuracy, shot accuracy, and finish percentage
df_nums = df_nums.copy()  # To avoid SettingWithCopyWarning
df_nums['headshot_accuracy'] = df['headshots'] / df['shots'].replace(0, np.nan)     # Calculate headshot accuracy
df_nums['shot_accuracy'] = df['hits'] / df['shots'].replace(0, np.nan)              # Calculate shot accuracy
df_nums['finish_percentage'] = df['kills'] / df['downs'].replace(0, np.nan)         # Calculate finish percentage

# Drop rows with NaN values to ensure clean data
#df_nums = df_nums.dropna()

# Add 'Overstat ID' and other non-numeric columns back to the DataFrame
string_columns = ['EA ID', 'Overstat ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL', 'Other Names']
df_cleaned = pd.concat([df_nums, df[string_columns]], axis=1)

# Ensure the order of columns
column_order = ['EA ID', 'Overstat ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL', 'kills', 'headshots', 'assists', 'time', 'damage', 'hits', 'downs', 'shots', 'grenades', 'ults', 'tacts', 'damage_taken', 'total_games', 'kills_avg', 'assists_avg', 'time_avg', 'damage_avg', 'grenades_avg', 'ults_avg', 'tacts_avg', 'headshot_accuracy', 'shot_accuracy', 'finish_percentage', 'Other Names']
df_cleaned = df_cleaned[column_order]

# Save the updated DataFrame to a CSV file
output_csv_file = '/Matchmaker/Code/PlayersFinal.csv'                       # Update Pathname

df_cleaned.to_csv(output_csv_file, index=False)

print("Data cleaned and saved successfully.")

# Print the number of names changed and the before and after
print(f"Number of names changed: {names_changed}")
for original, new in changes:
    print(f"Before: {original} -> After: {new}")

print(f"Updated DataFrame has been saved to {output_csv_file}")
