import matplotlib.pyplot as plt
import pandas as pd
import os


def save_text(writer, text):
    worksheet = writer.sheets['Sheet1']

    options = {
        'height': 400,
        'width': 150
    }

    worksheet.insert_textbox('O2', text, options)


def create_meta_text(instrument_list, delta_list, data_delta, tau, future_length, polynomial_degree, date_from=None):
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


def save_to_excel(df, excel_name, meta_text):
    df_ = df.groupby('Model').mean()
    with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Sheet1")
        save_text(writer, meta_text)
        df_.to_excel(writer, sheet_name="Sheet2")
        save_rolling_stats("Sheet3", df.groupby("Model")["Acc"].rolling(5).mean(), writer)
        save_rolling_stats("Sheet4", df.groupby("Model")["Acc"].rolling(15).mean(), writer)
        save_rolling_stats("Sheet5", df.groupby("Model")["Acc_up"].rolling(5).mean(), writer)
        save_rolling_stats("Sheet6", df.groupby("Model")["Acc_up"].rolling(15).mean(), writer)
        save_rolling_stats("Sheet7", df.groupby("Model")["Acc_down"].rolling(5).mean(), writer)
        save_rolling_stats("Sheet8", df.groupby("Model")["Acc_down"].rolling(15).mean(), writer)
        save_rolling_stats("Sheet9", df.groupby("Model")["Gain_%"].rolling(5).mean(), writer)
        save_rolling_stats("Sheet10", df.groupby("Model")["Gain_%"].rolling(15).mean(), writer)
        save_rolling_stats("Sheet11", df.groupby("Model")["Gain_%_up"].rolling(5).mean(), writer)
        save_rolling_stats("Sheet12", df.groupby("Model")["Gain_%_up"].rolling(15).mean(), writer)
        save_rolling_stats("Sheet13", df.groupby("Model")["Gain_%_down"].rolling(5).mean(), writer)
        save_rolling_stats("Sheet14", df.groupby("Model")["Gain_%_down"].rolling(15).mean(), writer)
        pd.DataFrame().to_excel(writer, sheet_name='Sheet15')
        worksheet = writer.sheets['Sheet15']
        df.boxplot(column=['Acc', 'Acc_up', 'Acc_down'])
        plt.savefig('abc1.png')
        worksheet.insert_image('B2', 'abc1.png')
        df.boxplot(column=['Gain_%', 'Gain_%_up', 'Gain_%_down'])
        plt.savefig('abc2.png')
        worksheet.insert_image('O2', 'abc2.png')
    os.remove('abc1.png')
    os.remove('abc2.png')
