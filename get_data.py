from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from data_downloader.data_downloader import DataDownloader
from dotenv import load_dotenv
import os

load_dotenv()
processed_data = None


def read_data(instrument_names, date_from=None, date_to=None, candles_num=5,
              tau=0.0003, future_length=20, normalize=True, delta_list=None, polynomial_degree=2,
              global_date_from=None, global_date_to=None, labels=None):
    global processed_data

    if processed_data is None:
        print("Get data")
        data_downloader = DataDownloader(username=os.environ.get("DB_AIT_USERNAME"),
                                         password=os.environ.get("DB_AIT_PASSWORD"))

        ohlc = [data_downloader.get_single_dataframe(instruments=[name],
                                                     date_from=global_date_from,
                                                     date_to=global_date_to) for name in instrument_names]
        print('Gather data')
        processed_data = gather(instrument_names, ohlc, candles_num, future_length, tau, normalize, delta_list,
                         polynomial_degree, labels)
        print('Data done')
    return processed_data[date_from: date_to]


def gather(instrument_names, ohlc_list, candles_num, future_length, tau, normalize, delta_list, polynomial_degree,
           labels):
    X = pd.DataFrame()
    for ohlc, name in zip(ohlc_list, instrument_names):
        i = 0
        while True:
            x, d, _ = get_ith_features(i, ohlc, candles_num, future_length, tau, normalize, delta_list,
                                        polynomial_degree)
            if x is None:
                break
            x = np.arctan([a[:-1] for a in x]).flatten()
            x_dict = {}
            for j in range(len(labels)):
                x_dict[labels[j]] = x[j]
            x_dict.update(d)
            x_dict['instrument'] = name
            X = X.append(x_dict, ignore_index=True)
            i += 1
    X.set_index('decision_date', inplace=True)
    X.sort_index(inplace=True)
    return X


def show_sample_data(instrument_name=None, date_from=datetime(2019, 1, 1), date_to=datetime(2019, 12, 28),
                     candles_num=5, tau=0.0005, future_length=20, normalize=False, delta_list=None,
                     polynomial_degree=2):
    if instrument_name is None:
        instrument_name = "EURUSD_FIVE_MINS"

    labels = []
    for delta in delta_list:
        for j in range(polynomial_degree):
            labels.append("{}_{}".format(str(delta).split('.')[0], polynomial_degree - j))

    data_downloader = DataDownloader(username=os.environ.get("DB_AIT_USERNAME"),
                                     password=os.environ.get("DB_AIT_PASSWORD"))

    ohlc = data_downloader.get_single_dataframe(instruments=[instrument_name],
                                                date_from=date_from,
                                                date_to=date_to)
    data = ohlc.close
    data = data[::-1]
    for i in np.arange(future_length, (data.size / 2), int(data.size / 10)):
        i = int(i)
        x, d, li = get_ith_features(i, ohlc, candles_num, future_length, tau, normalize, delta_list, polynomial_degree)
        barriers, data_i, Idx = li
        plot_single_example(x, d['target'], barriers, d['decision_date'], data_i, Idx, None, instrument_name,
                            future_length)
        plt.show()
        plt.close()
        plt.close()


def plot_single_example(x, target, barriers, decision_date, data_i, Idx, _, instrument_name, future_length, pred=None):
    plt.subplots_adjust(top=0.8)
    title = ""
    if instrument_name is not None:
        title += instrument_name + '\n'
    title += "target {}\n".format(target)
    title += "decision date {}\n".format(decision_date)
    if pred is not None:
        title += "pred {}".format(pred)
    plt.title(title)
    plt.plot(data_i.close.values[::-1])
    plt.plot(data_i.high.values[::-1])
    plt.plot(data_i.low.values[::-1])
    for idx, pol in zip(Idx, x):
        x_plot = np.linspace(idx[0], idx[-1])
        x = np.linspace(idx[0], idx[-1]) - idx[0]
        y = [np.polyval(pol, i) for i in x]
        plt.plot(data_i.shape[0] - x_plot, y, linewidth=3)
    for barrier in barriers:
        x = np.linspace(data_i.shape[0] - future_length, data_i.shape[0])
        y = [barrier for _ in x]
        plt.plot(x, y)


def get_ith_features(i, ohlc, candles_num, future_length, tau, normalize, delta_list, polynomial_degree, test=False):
    data = ohlc
    data = data[::-1]
    Idx = []
    for delta in delta_list:
        curr_date = data.index[i]
        idx = []
        for _ in range(candles_num):
            candles = data[data.index <= curr_date]
            if len(candles) == 0:
                return None, None, None
            curr_candle_date = candles.index[0]
            next_date = curr_candle_date - delta
            idx.append(data.index.get_loc(curr_candle_date) - i + future_length)
            curr_date = next_date
        Idx.append(idx)

    data_i = data.iloc[i + Idx[0][0] - future_length: i + Idx[0][-1]].copy()
    last_close = data_i.close.iloc[future_length]
    barriers = [last_close * (1 + sign * tau) for sign in [-1, 1]]
    if normalize:
        mean = data_i[future_length:].close.mean()
        std = data_i[future_length:].close.std()
        data_i = (data_i - mean) / std
        barriers = [(barrier - mean) / std for barrier in barriers]
    x = [np.polyfit(np.array(idx[:-1]) - idx[0], [data_i.close.iloc[a:b].mean()
                                                  for a, b in zip(idx[:-1], idx[1:])], polynomial_degree, full=True)[0]
         for idx in Idx]
    target, potential_gain = target_triple_barrier(ohlc.iloc[len(ohlc) - i - future_length - 1: len(ohlc) - i],
                                                   tau, test=test)
    if np.isnan(x).any():
        return None, None, None
    d_dict = {'target': int(target), 'decision_date': data.index[i], 'potential_gain': potential_gain}
    return x, d_dict, [barriers, data_i, Idx]


def target_triple_barrier(prices, tau, test=False):
    last_close = prices.iloc[0, 3]
    future_ohlc = prices.iloc[1:, :4].values
    changes = future_ohlc / last_close - 1
    up = (changes > tau).any(axis=1)
    down = (changes < -tau).any(axis=1)
    cha = np.stack([up, down]).any(axis=0)
    idx = np.where(cha)[0][0] if cha.any() else -1
    if idx == -1:
        class_id = int(future_ohlc[-1, 3] > last_close) if test else -1
        return class_id, future_ohlc[-1, 3] - last_close
    elif up[idx] and down[idx]:
        return -1, last_close * tau
    elif up[idx]:
        return 1, last_close * tau
    elif down[idx]:
        return 0, last_close * tau
