# Enhancing competitive play in Apex Legends through data-driven techniques and machine learning


# Abstract 
Our research aims to enhance competitive play in Apex Legends by developing a data-driven approach to player matching and lobby creation. We collected and cleaned player statistics from Overstat, a competitive lobby organizer and player database, addressing missing data and name inconsistencies. Dimensionality reduction techniques such as T-SNE and PCA were used for visualization, while K-means clustering facilitated the creation of skill tiers. A similarity engine was built to assess player similarity and facilitate the formation of balanced teams. This system considers player roles, input methods, and statistical performance, providing an effective solution for creating evenly matched teams and balanced lobbies in competitive scenarios. 

 
# Apex Team Maker Application 

Introduction 
Apex Legends, released in February 2019, is a battle royale game where 20 squads of 3 players compete until one team remains. Players choose from various legends, each with unique abilities. No team can have more than 1 of each legend. As the game progresses, the ‘ring’ or playable space shrinks forcing players to fight.


# Application Features 

Questionnaire: 
Users fill out a form with their details

# Information collected includes: 
EA ID: User's Electronic Arts ID for finding statistics. 
Overstat ID: User’s Overstat ID for finding competitive statistics.
Region: Options include North America, EMEA, APAC, etc. 
Input: Mouse and Keyboard, Controller. 
Role: Preferred position (IGL, Fragger, Anchor, etc.). 
Role 2: Secondary role choice
Main Legend: The character the user is most comfortable playing. 
Secondary Legend. 
Flex Legend. 

# Matchmaker: 
The core feature of the app, using user-submitted information to create balanced teams. 
Utilizes Overstat ID to look up player statistics on OverStat. 
Matches users regardless of input but avoids teams with all Controller or all MnK players. 
Prioritizes diversity in roles, ensuring no team has 3 IGLs, for example. 
Ensures variety in legend playability, avoiding teams with 3 of the same legend. 
Incorporates AI to keep team compositions up-to-date with current strategies and effective tactics. 
Users should be able to find similar players to them and then be able to filter out players based on role, region, input, legend selection, etc. 

# Random Royale Team Maker: 
For charity tournaments and similar events, where teams are randomized for each game. 
Players' statistics are considered solo in these tournaments. 
Teams are randomized every game with one player from each tier. Usually tiered players are pros, seasoned players and amateurs.  
Teams should be as equally balanced as possible 

 
# Code 
 
Short descriptions of all of the files are below.  

NewPlayer.py: Collects and validates new player information, ensuring no duplicates exist before saving. 

NewRandomPlayer.py: Adds new random players with generated attributes and validates their Overstat IDs. 

GetStatistics.py: Scrapes player statistics and names from specified URLs, combining new data with existing player data. 

PickleCreate.py: creates a pickle similarity engine for all of the players (removing columns that have missing data) 

MissingData.py: Imputes missing or zero values in player statistics using a similarity engine. 

CleanData.py: Processes and cleans the player data, calculating additional statistics and ensuring data validity. 

Graphs.py: Visualizes player statistics using T-SNE and PCA, optionally with K-means clustering. 

Graphs.ipynb: A Jupyter Notebook version of Graphs.py that displays all possible visualizations. 

mainPickleCreate.py: Processes player data, calculates similarity scores, and ranks players based on composite scores. 

RandomRoyaleLobbies.py: Clusters players into equally sized groups and assigns them to teams for games, allowing manual adjustments. 


# Future Adaptations  

Transform this project into a full stack.
Add a database, probably relational.
Allow users to create accounts.
Add Leagues to separate skill levels.
    For example, when filling out a questionnaire, players could select communities they are active in, this would help the machine get an idea of who belongs in what skill       tier and could be used in filtering matches. This could also be utilized in creating Random Royale lobbies.
    
I recently rebuilt my old PC into a server PC. I would like to further my skills and this project by using my server PC to make this project into a website where I can get practice on all aspects of the stack.
