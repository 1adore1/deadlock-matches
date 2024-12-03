import requests
import re
import pandas as pd
import json
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def transform_to_predict(match_info, train=True):
    match_info['net_worth_diff'] = match_info['net_worth_team_1'] - match_info['net_worth_team_0']
    features = ['net_worth_diff', 'hero_id_1', 'hero_id_2', 'hero_id_3',
       'hero_id_4', 'hero_id_5', 'hero_id_6', 'hero_id_7', 'hero_id_8',
       'hero_id_9', 'hero_id_10', 'hero_id_11', 'hero_id_12', 'tier1_lane1_0',
       'tier1_lane2_0', 'tier1_lane3_0', 'tier1_lane4_0', 'tier2_lane1_0',
       'tier2_lane2_0', 'tier2_lane3_0', 'tier2_lane4_0', 'titan_0', 'titan_shield_generator_1_0',
       'titan_shield_generator_2_0', 'barrack_boss_lane1_0',
       'barrack_boss_lane2_0', 'barrack_boss_lane3_0', 'barrack_boss_lane4_0',
       'tier1_lane1_1', 'tier1_lane2_1', 'tier1_lane3_1',
       'tier1_lane4_1', 'tier2_lane1_1', 'tier2_lane2_1', 'tier2_lane3_1',
       'tier2_lane4_1', 'titan_1', 'titan_shield_generator_1_1',
       'titan_shield_generator_2_1', 'barrack_boss_lane1_1',
       'barrack_boss_lane2_1', 'barrack_boss_lane3_1', 'barrack_boss_lane4_1']
    if train:
        features += ['winning_team']
    match_info = match_info[features]
    return match_info

def get_players(players):
    result = {}
    for i, player in enumerate(players, 1):
        for k, v in player.items():
            result[f'{k}_{i}'] = v
    return result

def extract_objectives(mask):
    return {
        'core': bool(mask & (1 << 0)),
        'tier1_lane1': bool(mask & (1 << 1)),
        'tier1_lane2': bool(mask & (1 << 2)),
        'tier1_lane3': bool(mask & (1 << 3)),
        'tier1_lane4': bool(mask & (1 << 4)),
        'tier2_lane1': bool(mask & (1 << 5)),
        'tier2_lane2': bool(mask & (1 << 6)),
        'tier2_lane3': bool(mask & (1 << 7)),
        'tier2_lane4': bool(mask & (1 << 8)),
        'titan': bool(mask & (1 << 9)),
        'titan_shield_generator_1': bool(mask & (1 << 10)),
        'titan_shield_generator_2': bool(mask & (1 << 11)),
        'barrack_boss_lane1': bool(mask & (1 << 12)),
        'barrack_boss_lane2': bool(mask & (1 << 13)),
        'barrack_boss_lane3': bool(mask & (1 << 14)),
        'barrack_boss_lane4': bool(mask & (1 << 15)),
    }

def preprocess_active(data):
    data = pd.DataFrame([data[0]])

    data.dropna(inplace=True)

    players_df = data['players'].apply(get_players).apply(pd.Series)
    data = pd.concat([data, players_df], axis=1)

    mask_columns0 = data['objectives_mask_team0'].apply(lambda mask: pd.Series(extract_objectives(mask))).add_suffix('_0')
    mask_columns1 = data['objectives_mask_team1'].apply(lambda mask: pd.Series(extract_objectives(mask))).add_suffix('_1')
    data = pd.concat([data, mask_columns0, mask_columns1], axis=1)

    data['net_worth_diff'] = data['net_worth_team_1'] - data['net_worth_team_0']

    features = ['match_id', 'net_worth_team_0', 'net_worth_team_1',
       'match_score', 'account_id_1', 'hero_id_1',
       'account_id_2', 'hero_id_2', 'account_id_3', 'hero_id_3',
       'account_id_4', 'hero_id_4', 'account_id_5', 'hero_id_5',
       'account_id_6', 'hero_id_6', 'account_id_7', 'hero_id_7',
       'account_id_8', 'hero_id_8', 'account_id_9', 'hero_id_9',
       'account_id_10', 'hero_id_10', 'account_id_11', 'hero_id_11',
       'account_id_12', 'hero_id_12', 'core_0', 'tier1_lane1_0',
       'tier1_lane2_0', 'tier1_lane3_0', 'tier1_lane4_0', 'tier2_lane1_0',
       'tier2_lane2_0', 'tier2_lane3_0', 'tier2_lane4_0', 'titan_0',
       'titan_shield_generator_1_0', 'titan_shield_generator_2_0',
       'barrack_boss_lane1_0', 'barrack_boss_lane2_0', 'barrack_boss_lane3_0',
       'barrack_boss_lane4_0', 'core_1', 'tier1_lane1_1', 'tier1_lane2_1',
       'tier1_lane3_1', 'tier1_lane4_1', 'tier2_lane1_1', 'tier2_lane2_1',
       'tier2_lane3_1', 'tier2_lane4_1', 'titan_1',
       'titan_shield_generator_1_1', 'titan_shield_generator_2_1',
       'barrack_boss_lane1_1', 'barrack_boss_lane2_1', 'barrack_boss_lane3_1',
       'barrack_boss_lane4_1']
    data = data[features]

    return data

def get_match_account_id(account_id):
    url = 'https://data.deadlock-api.com/v1/active-matches'
    response = requests.get(url, params={'account_id': account_id})
    if response.status_code != 200:
        return response.status_code

    match_info = json.loads(response.text)
    if match_info:
        return preprocess_active(match_info)
    else:
        return None

def get_match_info(match_info):
    if match_info is not None:
        match_id = match_info.at[0, 'match_id']
        net_worth_team_0 = match_info.at[0, 'net_worth_team_0']
        net_worth_team_1 = match_info.at[0, 'net_worth_team_1']
        match_score = match_info.at[0, 'match_score']
        heroes_0 = []
        heroes_1 = []
        for i in range(1, 7):
            heroes_0.append(match_info.at[0, f'hero_id_{i}'])
        for i in range(6, 13):
            heroes_1.append(match_info.at[0, f'hero_id_{i}'])
        return match_id, net_worth_team_0, net_worth_team_1, match_score, heroes_0, heroes_1
    else:
        return [None] * 6

def get_match_predict(match_info, model):
    match_info = transform_to_predict(match_info, False)
    softmax_arr = model.predict_proba(match_info)
    return softmax_arr

def get_steamid3(profile_url):
    # Если URL содержит SteamID64
    match = re.search(r'/profiles/(\d+)', profile_url)
    if match:
        steamid64 = int(match.group(1))
        steamid3 = steamid64 - 76561197960265728
        return steamid3
    
    # Если URL содержит кастомное имя
    match = re.search(r'/id/([\w\d_]+)', profile_url)
    if match:
        custom_name = match.group(1)
        api_url = f"https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"
        params = {'key': '7C54CEDE1547C7CF2C78B7FF287897F6', 'vanityurl': custom_name}
        response = requests.get(api_url, params=params)
        data = response.json()
        
        if data['response']['success'] == 1:
            steamid64 = int(data['response']['steamid'])
            steamid3 = steamid64 - 76561197960265728
            return steamid3
        else:
            raise ValueError("Custom name not found")
    
    raise ValueError("Invalid Steam profile URL")

def get_model(df):
    df = transform_to_predict(df)
    X, y = df.drop(['winning_team'], axis=1), df['winning_team']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    dump(gbc, 'models/model.joblib', compress=9)
    return gbc, acc, 'models/model.joblib'

def get_heroes(hero_ids):
    res = []
    for id in hero_ids:
        url = f'https://assets.deadlock-api.com/v2/heroes/{id}'
        response = requests.get(url)
        data = response.json()
        hero_name = data['name']
        res.append(hero_name)
    return res
