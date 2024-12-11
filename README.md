# Deadlock Match Tracker Bot

### Overview
This bot provides real-time data about active matches in the game Deadlock for players in the top 250 leaderboard. Users can check their Steam profile to view match information and refresh data dynamically.

### Features:
- Check Steam profiles for active matches.
- Display detailed match data including team net worth, hero information, and winning probability.
- Refresh match data with a button click.

## Installation

**Requirements**:
- Python 3.10 or higher
- Required Python packages listed in `requirements.txt`

### Setup Steps:
1. Clone the repository:
   ```
   git clone https://github.com/1adore1/deadlock-matches
   cd deadlock-matches
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure the bot token:
     Add your bot's API token in `config.py`:
     ```
     API_TOKEN = "your_bot_token_here"
     ```
4. Run the bot:
   ```
   python tg_bot.py
   ```

## Bot Commands

### `/start`
- Displays the bot's welcome message and shows a button to link a Steam profile.

### `/check`
- Prompts the user to enter their Steam profile URL to check active matches.

## How to Use

1. Start the bot with `/start`.
2. Use the `Check account (Top 250)` button or `/check` command to check for active games by entering a Steam account.
3. If a match is found, the bot will display detailed match data.
4. Use the `Refresh Match Info` button to update match details dynamically.

## Code Structure

- **`tg_bot.py`**: Main script for running the bot.
- **`tools.py`**: Contains utility functions for fetching and processing Steam and match data.
- **`config.py`**: Holds bot configuration, such as the API token.
- **`models/model.joblib`**: Pretrained model for match predictions.
