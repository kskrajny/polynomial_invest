import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.express as px


def save_results(df, file_name, roll=50):
    df['USD_per_lot'] = df['gain'] / df['future_length']
    rolled = df.groupby(['model', 'key']).rolling(roll).mean().reset_index(level=[0, 1]).dropna()
    df['Cumulative_USD_per_lot'] = df.groupby(['model', 'key'])['USD_per_lot'].cumsum()
    df['Cumulative_max_USD_per_lot'] = df.groupby(['model', 'key'])['Cumulative_USD_per_lot'].cummax()
    df['Draw_down'] = df['Cumulative_max_USD_per_lot'] - df['Cumulative_USD_per_lot']
    with open(file_name + '.html', 'a') as f:
        for fig in [
            px.line(df, color='model', y='Cumulative_USD_per_lot', line_group='key'),
            px.line(rolled, color='model', y='USD_per_lot', line_group='key'),
            px.box(df, x='model', y="USD_per_lot", points="all", color='key'),
        ]:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def map_(x: bool):
    return 1 if x else -1


def predict_with_treshold(p, t):
    if p <= 1 - t:
        return 0
    elif p >= t:
        return 1
    else:
        return -2


path_to_dfs = 'results/20230109-113438/'
os.mkdir(path_to_dfs + 'stats')
tresholds = [0.5, 0.52]
spread_qs = [1, 0.7]

dirs = os.listdir(path_to_dfs)
final_pd = pd.DataFrame()

df_source = pd.read_csv(path_to_dfs + 'data.csv')
df_source.set_index(['decision_date', 'model', 'instrument'], inplace=True)
df_source = df_source[~df_source.index.duplicated()]

df_source.reset_index(inplace=True)
all_df = pd.DataFrame()

for treshold in tresholds:
    for spread_q in spread_qs:
        df = df_source.copy()
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        max_spread = df['spread'].quantile(spread_q)
        df['key'] = 't:' + str(treshold) + '_s:' + str(spread_q)
        df['treshold'] = treshold
        df['spread_q'] = spread_q
        df.set_index(['decision_date', 'model', 'instrument'], inplace=True)
        df['pred'] = [predict_with_treshold(p, treshold) for p in df['proba_1']]
        df['measurable_y'] = df['target'][(df['target'] != -1) & (df['pred'] != -2) & (df['spread'] < max_spread)]
        df['measurable_pred'] = df['pred'][(df['target'] != -1) & (df['pred'] != -2) & (df['spread'] < max_spread)]
        df['gain'] = df.apply(
            lambda x: x['usd_per_pips'] * map_(x['target'] == x['pred']) * x['potential_gain'] * 10000 - x['commission']
            if x['pred'] != -2 and x['spread'] < max_spread else 0, axis=1
        )
        all_df = pd.concat([df, all_df])

all_df.reset_index(inplace=True)
grouped = all_df.groupby(['decision_date', 'model', 'key']).mean().reset_index(level=[1, 2])[['model', 'key', 'gain',
                                                                                              'future_length']]
all_df.set_index('decision_date', inplace=True)

save_results(grouped, path_to_dfs + 'stats/main')
grouped = all_df.groupby(['instrument'])
for name, group in grouped:
    save_results(all_df.copy(), path_to_dfs + 'stats/' + str(name))

grouped = all_df.groupby(['model', 'treshold', 'spread_q', 'instrument'])

for name, group in grouped:
    group.dropna(inplace=True)
    matrix = confusion_matrix(group['measurable_y'].values, group['measurable_pred'].values)
    acc = accuracy_score(group['measurable_y'].values, group['measurable_pred'].values)
    usd_gain_per_lot = (group['gain'] / group['future_length']).sum()
    acc_up, acc_down, pred_up, pred_down, usd_gain_per_position = 0, 0, 0, 0, 0
    lost_test = len(group) - group['measurable_y'].count()

    try:
        acc_down = matrix[0][0] / (matrix[0][0] + matrix[1][0]) if matrix[0][0] + matrix[1][0] > 0 else 0.5
        acc_up = matrix[1][1] / (matrix[1][1] + matrix[0][1]) if matrix[1][1] + matrix[0][1] > 0 else 0.5
        pred_up = matrix[0][0] + matrix[1][0]
        pred_down = matrix[0][1] + matrix[1][1]
        usd_gain_per_position = group['gain'].sum() / (pred_up + pred_down)
    except IndexError:
        pass
    final_pd = final_pd.append({
        "Model": name[0], 'Treshold': name[1], 'Spread_q': name[2],
        'Instrument': name[3],
        "Acc": acc, "Acc_down": acc_down, "Acc_up": acc_up,
        "USD_per_lot": usd_gain_per_lot, "USD_per_position": usd_gain_per_position,
        "Pred_up": pred_up, "Pred_down": pred_down,
        "Lost_test_%": lost_test / all_df.shape[0] * 100
    }, ignore_index=True)

final_pd.set_index(['Instrument', 'Model', 'Treshold', 'Spread_q'])
final_pd.to_excel(path_to_dfs + 'stats/' + 'final.xlsx')
