import os
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from datetime import datetime
from webdriver_manager.chrome import ChromeDriverManager

# Start the timer
start_time = datetime.now()

# Set the correct file paths for the CSV files
csv_files = [
    '/Matchmaker/Code/Players1.csv',                            # Update Pathnames
    '/Matchmaker/Code/newPlayers.csv',
    '/Matchmaker/Code/newRandomPlayers.csv'
]
output_csv_file = '/Matchmaker/Code/Players.csv'                # Update Pathname

# Path to your chromedriver
chromedriver_path = '/Code/chromedriver-mac-x64/chromedriver'   # Update Pathname and Chromedriver Version

# Read and merge the CSV files
dfs = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file)
        print(f"Read {len(df_temp)} rows from {file}")
        dfs.append(df_temp)
    except Exception as e:
        print(f"Failed to read {file}: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"Merged dataframe has {len(df)} rows")

# Ensure 'Overstat ID' column is of type string and clean any floating-point values
df['Overstat ID'] = df['Overstat ID'].astype(str).str.replace('.0', '', regex=False)

# Remove rows with 'nan' values in 'Overstat ID'
df = df[df['Overstat ID'] != 'nan']
print(f"Dataframe after removing NaN 'Overstat ID' rows has {len(df)} rows")

# Create URL column for each player
df['osURL'] = 'https://overstat.gg/player/' + df['Overstat ID'] + '.playername/overview'

# Add "Other Names" column to the DataFrame
df['Other Names'] = ''

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Verify that chromedriver exists at the specified path
if not os.path.exists(chromedriver_path):
    raise FileNotFoundError(f"chromedriver not found at {chromedriver_path}")

# Function to scrape statistics and other names from the HTML table using Selenium
def scrape_statistics_and_names(url, retries=5):
    print(f"Scraping URL: {url}")
    for attempt in range(retries):
        try:
            service = ChromeService(executable_path=chromedriver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)

            # Wait for the page to load
            driver.implicitly_wait(10)

            # Get page content and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()

            # Extract statistics
            statistics = {}
            averages = {}
            other_names = []

            # Extract the "In-game Names" section
            in_game_names_section = soup.find('h2', string='In-game Names')
            if in_game_names_section:
                name_divs = in_game_names_section.find_next_siblings('div', class_='text-center')
                other_names = [name_div.get_text().strip() for name_div in name_divs]

            # Extract the "Totals" table
            totals_table = soup.find('h2', string='Totals')
            if totals_table:
                totals_table = totals_table.find_next('table')
                if totals_table:
                    table_rows = totals_table.find_all('tr')
                    for row in table_rows:
                        key = row.find(class_='key').get_text().strip().lower()
                        value = row.find(class_='value').get_text().strip().replace(',', '')  # Remove commas from numbers
                        statistics[key] = value

            # Extract the "Average" table
            averages_table = soup.find('h2', string='Average')
            if averages_table:
                averages_table = averages_table.find_next('table')
                if averages_table:
                    table_rows = averages_table.find_all('tr')
                    for row in table_rows:
                        key = row.find(class_='key').get_text().strip().lower()
                        value = row.find(class_='value').get_text().strip().replace(',', '')  # Remove commas from numbers
                        averages[key + '_avg'] = value

            return statistics, averages, other_names
        except Exception as e:
            print(f"Attempt {attempt + 1} failed scraping URL {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait a bit before retrying
            else:
                return None, None, None

# Function to print statistics
def print_statistics(statistics, averages):
    for key, value in {**statistics, **averages}.items():
        print(f"{key.capitalize()}: {value}")

# Scrape statistics and other names for existing players
for index, row in df.iterrows():
    # Skip players who already have statistics
    if not pd.isna(row['kills']):
        print(f"Skipping player at index {index} with Overstat ID {row['Overstat ID']} as they already have statistics.")
        continue

    url = row['osURL']
    print(f"Scraping statistics for {row['EA ID']} from URL: {url}")
    statistics, averages, other_names = scrape_statistics_and_names(url)
    if statistics and averages:
        print(f"Statistics for {row['EA ID']}:")
        print_statistics(statistics, averages)

        # Update DataFrame with scraped statistics
        new_player_data = {
            'kills': int(statistics.get('kills', 0)),
            'revives': int(statistics.get('revives', 0)),
            'headshots': int(statistics.get('headshots', 0)),
            'assists': int(statistics.get('assists', 0)),
            'time': int(statistics.get('time', 0)),
            'respawns': int(statistics.get('respawns', 0)),
            'damage': int(statistics.get('damage', 0)),
            'hits': int(statistics.get('hits', 0)),
            'downs': int(statistics.get('downs', 0)),
            'shots': int(statistics.get('shots', 0)),
            'grenades': int(statistics.get('grenades', 0)),
            'ults': int(statistics.get('ults', 0)),
            'tacts': int(statistics.get('tacts', 0)),
            'damage taken*': int(statistics.get('damage taken*', 0)),
            'totalgames': int(statistics.get('totalgames', 0)),
            'kills_avg': float(averages.get('kills_avg', 0)),
            'revives_avg': float(averages.get('revives_avg', 0)),
            'headshots_avg': float(averages.get('headshots_avg', 0)),
            'assists_avg': float(averages.get('assists_avg', 0)),
            'time_avg': float(averages.get('time_avg', 0)),
            'respawns_avg': float(averages.get('respawns_avg', 0)),
            'damage_avg': float(averages.get('damage_avg', 0)),
            'hits_avg': float(averages.get('hits_avg', 0)),
            'downs_avg': float(averages.get('downs_avg', 0)),
            'shots_avg': float(averages.get('shots_avg', 0)),
            'grenades_avg': float(averages.get('grenades_avg', 0)),
            'ults_avg': float(averages.get('ults_avg', 0)),
            'tacts_avg': float(averages.get('tacts_avg', 0)),
            'damage taken*_avg': float(averages.get('damage taken*_avg', 0)),
            'totalgames_avg': float(averages.get('totalgames_avg', 0))
        }

        for key, value in new_player_data.items():
            df.at[index, key] = value
        
        df.at[index, 'Other Names'] = ', '.join(other_names)
    else:
        print(f"Failed to scrape statistics for {row['EA ID']} from URL: {url}")

    time.sleep(2)  # Add a 2-second delay after scraping each player's data

# Save the DataFrame after all scraping is done
df.to_csv(output_csv_file, index=False)

# End the timer
end_time = datetime.now()
execution_time = datetime.now() - start_time

print(f"Data for {len(df)} players has been saved to {output_csv_file}")
print(f"Total execution time: {execution_time}")
