import numpy as np
import pandas as pd
import json
import typing as tp
from catboost import CatBoostClassifier

columns = ['home_full_time_goals', 'away_full_time_goals', 'home_half_time_goals', 'away_half_time_goals', 'home_shots',
           'away_shots', 'home_shots_on_target', 'away_shots_on_target', 'home_fouls', 'away_fouls', 'home_corners', 'away_corners',
           'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards']
columns_known_before_match = ['Division', 'Time', 'home_team', 'away_team', 'Referee', 'home_coef', 'draw_coef', 'away_coef']
stat_cols = ['full_time_goals', 'half_time_goals', 'shots', 'shots_on_target',
             'fouls', 'corners', 'yellow_cards', 'red_cards']

def get_team_stat(teams_stat: tp.Dict[int, tp.List], team_id, n_matches):
    if team_id in teams_stat:
        return np.nanmean(teams_stat[team_id][-n_matches:])
    else:
        return 0


def get_stat(teams_stat, row, n_matches=5):
    home_team_stat = get_team_stat(teams_stat, row['home_team'], n_matches)
    away_team_stat = get_team_stat(teams_stat, row['away_team'], n_matches)
    return home_team_stat, away_team_stat


def get_stat_multiple_n_matches(teams_stat, row, n_matches_range: tp.List):
    answer = []
    for n_matches in n_matches_range:
        home_team_stat, away_team_stat = get_stat(teams_stat, row, n_matches)
        answer.append(home_team_stat)
        answer.append(away_team_stat)
    return answer


def update_teams_stat(home_team, away_team, home_stat, away_stat, stat, stat_dict):
    if home_team in stat_dict[stat]:
        stat_dict[stat][home_team].append(home_stat)
    else:
        stat_dict[stat][home_team] = [home_stat]

    if away_team in stat_dict[stat]:
        stat_dict[stat][away_team].append(away_stat)
    else:
        stat_dict[stat][away_team] = [away_stat]


def float_for_time(a):
    try:
        return float(a)
    except:
        return -1


def make_bet(model, features: list, home_coef, draw_coef, away_coef):
    prob = model.predict_proba(features, verbose=False)
    EH = prob[2] * (home_coef - 1) - (1 - prob[2])
    ED = prob[1] * (draw_coef - 1) - (1 - prob[1])
    EA = prob[0] * (away_coef - 1) - (1 - prob[0])
    if EA > EH and EA > ED:
        return 'AWAY'
    elif ED > EH and ED > EA:
        return 'DRAW'
    else:
        return 'HOME'

def make_bet_prob(model, features: list):
    prob = model.predict_proba(features, verbose=False)
    if prob[0] > prob[1] and prob[0] > prob[2]:
        return 'AWAY'
    elif prob[1] > prob[0] and prob[1] > prob[2]:
        return 'DRAW'
    else:
        return 'HOME'


model = CatBoostClassifier(loss_function='MultiClass')
model.load_model('model.cbm')
#doesnt work otherwise
model._init_params['verbose'] = None

with open('stat_dict.json', 'r') as fp:
    stat_dict = json.load(fp)

n = int(input())
for _ in range(n):
    row = list(map(float_for_time, input().split()))
    row = pd.Series(row, index=columns_known_before_match)
    features = []
    for stat in stat_cols:
        all_stat = get_stat_multiple_n_matches(stat_dict[stat], row, list(range(1, 10, 3)))
        features += all_stat
    features.append(row['Division'])
    #print(make_bet(model, features, row['home_coef'], row['draw_coef'], row['away_coef']), flush=True)
    print(make_bet_prob(model, features), flush=True)

    home_team = row['home_team']
    away_team = row['away_team']
    row = list(map(float, input().split()))
    row = pd.Series(row, index=columns)
    for stat in stat_cols:
        update_teams_stat(home_team, away_team, row['home_' + stat], row['away_' + stat],
                          stat, stat_dict)
