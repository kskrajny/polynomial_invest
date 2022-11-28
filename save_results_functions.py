import os
import pandas as pd
import plotly.express as px
import dalex as dx
from contans_main import results_dir, clf_names, classifiers


def save_text(writer, text):
    worksheet = writer.sheets['Sheet1']

    options = {
        'height': 400,
        'width': 150
    }

    worksheet.insert_textbox('O2', text, options)


def create_meta_text(instrument_list, delta_list, data_delta, tau, future_length, polynomial_degree, date_from=None,
                     skip_time_days=None):
    text = "Data delta:\n" + str(data_delta).split('.')[0]
    text += "\n\nTau:  " + str(tau)
    text += "\n\nFuture length:  " + str(future_length)
    text += "\n\nPolynomial degree:  " + str(polynomial_degree)
    text += "\n\nInstruments:"
    for name in instrument_list:
        name = name.split('_')[0]
        text += "\n" + name
    text += "\n\nDeltas:"
    for delta in delta_list:
        delta = str(delta).split('.')[0]
        text += "\n" + delta
    if date_from is not None:
        text += "\n\nStart date: " + date_from.strftime("%Y.%m.%d")
    if skip_time_days is not None:
        text += "\n\nTest days: " + str(skip_time_days)
    return text


def save_rolling_stats(sheet_name, df, writer):
    df = df.dropna()
    df.to_excel(writer, sheet_name=sheet_name)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    chart = workbook.add_chart({'type': 'line'})
    rolling_len = int(len(df) / df.index.get_level_values(0).nunique())
    for i in range(df.index.get_level_values(0).nunique()):
        chart.add_series({
            'values': [sheet_name, 1 + i * rolling_len, 2, (1 + i) * rolling_len, 2],
            'name': df.index.get_level_values(0).unique()[i]
        })
    chart.set_x_axis({'name': 'Index', 'position_axis': 'on_tick'})
    chart.set_y_axis({'name': 'Value', 'major_gridlines': {'visible': False}})
    worksheet.insert_chart('G2', chart)


def save_results(dfs, exp_df, file_names, meta_text, future_length, roll=2):
    os.mkdir(results_dir)
    for df, file_name, clf_name in zip(dfs, file_names, clf_names):
        df = df.set_index('Time_start')
        df['Gain_in_pips_per_lot'] = df['Gain'] / future_length
        rolled = df.groupby('Model').rolling(roll).mean().reset_index(level=0).dropna()
        df_ = df.groupby(['Model', 'Treshold'])[["Acc_down", "Acc_up", "Acc", "Gain_per_position", "Pred_up",
                                                 "Pred_down", "Lost_test_%"]].mean()
        df['Cumulative_gain_in_pips_per_lot'] = df.groupby('Model')['Gain_in_pips_per_lot'].cumsum()
        df['Cumulative_max_gain_in_pips_per_lot'] = df.groupby('Model')['Cumulative_gain_in_pips_per_lot'].cummax()
        df['Draw_down'] = df['Cumulative_max_gain_in_pips_per_lot'] - df['Cumulative_gain_in_pips_per_lot']
        df_['Draw_down'] = df.groupby(['Model', 'Treshold'])['Draw_down'].max()
        exp_df.groupby(['label', 'variable']).mean().reset_index()
        with open(file_name + '.html', 'a') as f:
            for fig in [
                px.line(df, color='Model', y='Cumulative_gain_in_pips_per_lot', line_group='Treshold'),
                px.line(rolled, color='Model', y='Acc', line_group='Treshold'),
                px.line(rolled, color='Model', y='Acc_up', line_group='Treshold'),
                px.line(rolled, color='Model', y='Acc_down', line_group='Treshold'),
                px.line(rolled, color='Model', y='Gain_in_pips_per_lot', line_group='Treshold'),
                px.box(df, x='Model', y="Acc", points="all", color='Treshold'),
                px.box(df, x='Model', y="Acc_up", points="all", color='Treshold'),
                px.box(df, x='Model', y="Acc_down", points="all", color='Treshold'),
                px.box(df, x='Model', y="Gain_in_pips_per_lot", points="all", color='Treshold'),
                px.box(exp_df, x='label', color='variable', y='dropout_loss', points="all")
            ]:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

        df.drop(columns=['Cumulative_max_gain_in_pips_per_lot', 'Gain', 'Draw_down', 'Cumulative_gain_in_pips_per_lot'],
                inplace=True)
        with pd.ExcelWriter(file_name + ".xlsx", engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="Sheet1")
            save_text(writer, meta_text)
            df_.to_excel(writer, sheet_name="Sheet2")
