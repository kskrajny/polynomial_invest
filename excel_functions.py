import pandas as pd


def save_text(writer, text):
    worksheet = writer.sheets['Sheet1']

    options = {
        'height': 400,
        'width': 150
    }

    worksheet.insert_textbox('I2', text, options)


def create_meta_text(instrument_list, delta_list, data_delta, tau, future_length, polynomial_degree):
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
