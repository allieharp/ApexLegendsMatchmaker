import pandas as pd
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Input parameters  
new_players_file = '/Matchmaker/Code/newPlayers.csv'            # Update Pathname
chromedriver_path = '/code/chromedriver-mac-x64'                # Update Pathname and ChromeDriver version

# Verify that chromedriver exists at the specified path
if not os.path.exists(chromedriver_path):
    raise FileNotFoundError(f"chromedriver not found at {chromedriver_path}")

# Set up Selenium WebDriver options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Function to gather player information from the user
def get_player_info():
    player_info = {}
    player_info['EA ID'] = input("Enter EA ID: ")
    player_info['Overstat ID'] = input("Enter Overstat ID: ")
    player_info['Region'] = input("Enter Region: ")
    player_info['Input'] = input("Enter Input (Controller/Mouse/Keyboard): ")
    player_info['Role 1'] = input("Enter Role 1: ")
    player_info['Role 2'] = input("Enter Role 2: ")
    player_info['Legend 1'] = input("Enter Legend 1: ")
    player_info['Legend 2'] = input("Enter Legend 2: ")
    player_info['Flex Legend'] = input("Enter Flex Legend: ")
    player_info['osURL'] = f'https://overstat.gg/player/{player_info["Overstat ID"]}.playername/overview'

    return player_info

# Function to scrape statistics and validate Overstat ID using Selenium
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

            # Check if the table exists
            table = soup.find('table')
            if table is not None:
                print("Table found.")
                return True
            else:
                print("Error: Table not found in the HTML content. Invalid Overstat ID.")
                return False
        except Exception as e:
            print(f"Attempt {attempt + 1} failed scraping URL {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # Wait a bit before retrying
            else:
                return False

# Number of new players to add
num_new_players = int(input("Enter the number of new players to add: "))

# List to store new player data
new_players_list = []

# Gather information for each new player
for _ in range(num_new_players):
    while True:
        new_player_info = get_player_info()
        
        # Validate Overstat ID by scraping the web page
        valid_id = scrape_statistics(new_player_info['osURL'])
        
        if valid_id:
            if os.path.exists(new_players_file):
                # Read existing new players data
                existing_new_players_df = pd.read_csv(new_players_file)
                if new_player_info['Overstat ID'] in existing_new_players_df['Overstat ID'].values:
                    print(f"Player with Overstat ID {new_player_info['Overstat ID']} already exists. Please claim the profile instead.")
                    continue
            new_players_list.append(new_player_info)
            break
        else:
            print("Invalid Overstat ID. Please enter a valid ID.")

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
