import os
import pandas as pd
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Input parameters
csv_file = '/Matchmaker/Code/players1.csv'                                      # Update Pathnames
new_players_file = '/Matchmaker/Code/newRandomPlayers.csv'
chromedriver_path = '/code/chromedriver-mac-x64'                                # Update Pathname and ChromeDriver version

# Set up Selenium WebDriver options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Function to scrape statistics from the HTML table using Selenium
def scrape_statistics(url, retries=3):
    print(f"Scraping URL: {url}")
    for attempt in range(retries):
        try:
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
            driver.get(url)

            # Wait for the page to load
            driver.implicitly_wait(10)

            # Get page content and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()

            # Extract statistics
            statistics = {}
            table = soup.find('table')
            
            if table is not None:
                table_rows = table.find_all('tr')
                for row in table_rows:
                    key = row.find(class_='key').get_text().strip().lower()
                    value = row.find(class_='value').get_text().strip().replace(',', '')  # Remove commas from numbers
                    statistics[key] = value
                return statistics
            else:
                print("Error: Table not found in the HTML content")
                return None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed scraping URL {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait a bit before retrying
            else:
                return None

# Check if the input CSV file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found at {csv_file}")

# Import the data from the CSV file
df = pd.read_csv(csv_file, encoding='latin1')

# Ensure 'Overstat ID' column is of type string and clean any floating-point values
df['Overstat ID'] = df['Overstat ID'].astype(str).str.replace('.0', '', regex=False)

# Remove rows with 'nan' values in 'Overstat ID'
df = df[df['Overstat ID'] != 'nan']

# Function to randomize roles, input, and legends
def randomize_player_attributes():
    roles = ['IGL', 'Co-IGL', 'Frag', 'Re-frag', 'Anchor']
    inputs = ['Controller', 'Mouse/Keyboard']
    mainlegends = ['Bloodhound', 'Bangalore', 'Caustic', 'Wattson', 'Crypto', 'Rampart', 'Horizon', 'Fuse', 'Valkyrie', 'Seer', 'Mad Maggie', 'Newcastle', 'Catalyst']
    flexlegends = ['Wraith', 'Gibraltar', 'Bloodhound', 'Lifeline', 'Pathfinder', 'Bangalore', 'Caustic', 'Mirage', 'Octane', 'Wattson', 'Crypto', 'Revenant', 'Loba', 'Rampart', 'Horizon', 'Fuse', 'Valkyrie', 'Seer', 'Ash', 'Mad Maggie', 'Newcastle', 'Vantage', 'Catalyst']
    
    input_type = random.choice(inputs)
    
    if input_type == 'Controller':
        role1 = random.choices(roles, weights=[18, 18, 23, 23, 18], k=1)[0]
        remaining_roles = [role for role in roles if role != role1]
        role2 = random.choices(remaining_roles, weights=[20] * len(remaining_roles), k=1)[0]
    else:
        role1 = random.choices(roles, weights=[20, 20, 20, 20, 20], k=1)[0]
        remaining_roles = [role for role in roles if role != role1]
        role2 = random.choices(remaining_roles, weights=[20] * len(remaining_roles), k=1)[0]

    legend1 = random.choice(mainlegends)
    remaining_mainlegends = [legend for legend in mainlegends if legend != legend1]
    legend2 = random.choice(remaining_mainlegends)
    remaining_flexlegends = [legend for legend in flexlegends if legend != legend1 and legend != legend2]
    flex_legend = random.choice(remaining_flexlegends)
    
    return role1, role2, input_type, legend1, legend2, flex_legend

# Function to pick a unique random number (or player)
def pick_random_number_from_ranges():
    ranges = [
        (1, 1000),
        (100000, 110000),
        (200000, 210000),
        (300000, 310000),
        (400000, 410000),
        (300000, 510000),
        (600000, 610000),
        (700000, 710000),
        (800000, 810000),
        (900000, 910000)
    ]
    # Pick a random range
    chosen_range = random.choice(ranges)
    # Pick a random number within the chosen range
    random_number = random.randint(chosen_range[0], chosen_range[1])
    return random_number

# Function to pick a unique random number that hasn't been used
def pick_unique_random_number(picked_numbers):
    while True:
        number = pick_random_number_from_ranges()
        if number not in picked_numbers:
            picked_numbers.add(number)
            return number

# Function to gather info for new players
def gather_info_for_new_players(existing_df, total_players_needed):
    picked_numbers = set(existing_df['Overstat ID'].astype(int).tolist())
    new_players = []

    while len(new_players) < total_players_needed:
        possible_id = pick_unique_random_number(picked_numbers)
        os_url = f'https://overstat.gg/player/{possible_id}.playername/overview'
        
        # Validate Overstat ID by scraping the web page
        valid_id = scrape_statistics(os_url)
        
        if valid_id is not None:
            role1, role2, input_type, legend1, legend2, flex_legend = randomize_player_attributes()
            
            new_player_data = {
                'EA ID': '',
                'Overstat ID': str(possible_id),
                'Region': 'NA',  # All Regions are NA for now
                'Input': input_type,
                'Role 1': role1,
                'Role 2': role2,
                'Legend 1': legend1,
                'Legend 2': legend2,
                'Flex Legend': flex_legend,
                'osURL': os_url
            }
            
            new_players.append(new_player_data)
            print(f"Added new player with ID {possible_id}. Total new players: {len(new_players)}")
        else:
            print("Invalid Overstat ID. Please try again.")

    return new_players

# Number of new players needed
new_players_needed = int(input("Enter the number of new players needed: "))

# Gather info for new players
new_players_list = gather_info_for_new_players(df, new_players_needed)

# Convert new players list to DataFrame
new_players_df = pd.DataFrame(new_players_list)

# Check if new players CSV already exists
if os.path.exists(new_players_file):
    # Read existing new players data
    existing_new_players_df = pd.read_csv(new_players_file)
    # Concatenate with new data
    new_players_df = pd.concat([existing_new_players_df, new_players_df], ignore_index=True)

# Remove the 'Random' column if it exists
if 'Random' in new_players_df.columns:
    new_players_df = new_players_df.drop(columns=['Random'])

# Save new players DataFrame to CSV file with the correct column order
column_order = ['EA ID', 'Overstat ID', 'Region', 'Input', 'Role 1', 'Role 2', 'Legend 1', 'Legend 2', 'Flex Legend', 'osURL']
new_players_df = new_players_df[column_order]
new_players_df.to_csv(new_players_file, index=False)

print(f"New players have been saved to {new_players_file}")
