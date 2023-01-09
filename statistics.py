import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

import plotly.express as px


def save_results(df, file_name, roll=500):
    df['Gain_in_pips_per_lot'] = df['gain'] / df['future_length']
    rolled = df.groupby(['model', 'key']).rolling(roll).mean().reset_index(level=[0, 1]).dropna()
    df['Cumulative_gain_in_pips_per_lot'] = df.groupby(['model', 'key'])['Gain_in_pips_per_lot'].cumsum()
    df['Cumulative_max_gain_in_pips_per_lot'] = df.groupby(['model', 'key'])['Cumulative_gain_in_pips_per_lot'].cummax()
    df['Draw_down'] = df['Cumulative_max_gain_in_pips_per_lot'] - df['Cumulative_gain_in_pips_per_lot']
    with open(file_name + '.html', 'a') as f:
        for fig in [
            px.line(df, color='model', y='Cumulative_gain_in_pips_per_lot', line_group='key'),
            px.line(rolled, color='model', y='Gain_in_pips_per_lot', line_group='key'),
            px.box(df, x='model', y="Gain_in_pips_per_lot", points="all", color='key'),
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


path_to_dfs = 'results/20230103-174702/'
os.mkdir(path_to_dfs + 'stats')
tresholds = [0.5, 0.52]
spread_qs = [1, 0.7]

dirs = os.listdir(path_to_dfs)
final_pd = pd.DataFrame()

for file in dirs:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    if ".csv" not in file:
        continue
    path_to_df = path_to_dfs + file
    instrument = file.split('.')[0]
    all_dfs = pd.DataFrame()
    for treshold in tresholds:
        for spread_q in spread_qs:
            df = pd.read_csv(path_to_df)
            max_spread = df['spread'].quantile(spread_q)
            df['key'] = 't:' + str(treshold) + '_s:' + str(spread_q)
            df.set_index(['decision_date', 'model'], inplace=True)
            df = df[~df.index.duplicated()]
            df['pred'] = [predict_with_treshold(p, treshold) for p in df['proba_1']]
            df['measurable_y'] = df['target'][(df['target'] != -1) & (df['pred'] != -2) & (df['spread'] < max_spread)]
            df['measurable_pred'] = df['pred'][(df['target'] != -1) & (df['pred'] != -2) & (df['spread'] < max_spread)]
            df['gain'] = df.apply(lambda x: map_(x['target'] == x['pred']) * x['potential_gain'] * 10000
                                            - x['commission']
            if x['pred'] != -2 and x['spread'] < max_spread else 0, axis=1)
            gain_per_lot = (df['gain'] / df['future_length']).sum()
            acc_up, acc_down, pred_up, pred_down, gain_per_position = 0, 0, 0, 0, 0
            lost_test = len(df) - df['measurable_y'].count()
            all_dfs = pd.concat([all_dfs, df.copy()])

            grouped = df.groupby(level=1)

            for name, group in grouped:
                group.dropna(inplace=True)
                matrix = confusion_matrix(group['measurable_y'].values, group['measurable_pred'].values)
                acc = accuracy_score(group['measurable_y'].values, group['measurable_pred'].values)
                try:
                    acc_down = matrix[0][0] / (matrix[0][0] + matrix[1][0]) if matrix[0][0] + matrix[1][0] > 0 else 0.5
                    acc_up = matrix[1][1] / (matrix[1][1] + matrix[0][1]) if matrix[1][1] + matrix[0][1] > 0 else 0.5
                    pred_up = matrix[0][0] + matrix[1][0]
                    pred_down = matrix[0][1] + matrix[1][1]
                    gain_per_position = df['gain'].sum() / (pred_up + pred_down)
                except IndexError:
                    pass
                final_pd = final_pd.append({
                    "Model": name, 'Instrument': instrument,
                    'Treshold': treshold, 'Spread_q': spread_q,
                    "Acc": acc, "Acc_down": acc_down, "Acc_up": acc_up,
                    "Gain_per_lot": gain_per_lot, "Gain_per_position": gain_per_position,
                    "Pred_up": pred_up, "Pred_down": pred_down,
                    "Lost_test_%": lost_test / df.shape[0] * 100, "Max_spread": max_spread
                }, ignore_index=True)
    all_dfs.reset_index(inplace=True)
    all_dfs.set_index('decision_date', inplace=True)
    save_results(all_dfs, path_to_dfs + 'stats/' + instrument)
final_pd.set_index(['Instrument', 'Model', 'Treshold', 'Spread_q'])
final_pd.to_excel(path_to_dfs + 'stats/' + 'final.xlsx')
