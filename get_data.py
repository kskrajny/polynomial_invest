from datetime import datetime, timedelta

import numpy as np
import torch
from matplotlib import pyplot as plt

from data_downloader.data_downloader import DataDownloader
from dotenv import load_dotenv
import os

load_dotenv()

data_downloader = DataDownloader(username=os.environ.get("DB_AIT_USERNAME"),
                                 password=os.environ.get("DB_AIT_PASSWORD"))

delta_list = [
    # timedelta(days=1),
    # timedelta(hours=4),
    timedelta(hours=1),
    timedelta(minutes=15),
    timedelta(minutes=5)
]


def read_data(instrument_names, date_from=datetime(2019, 1, 1), date_to=datetime(2019, 1, 28), candles_num=5,
              tau=0.0003, future_length=20, normalize=True):
    ohlc_list = [data_downloader.get_single_dataframe(instruments=[name],
                                                date_from=date_from,
                                                date_to=date_to) for name in instrument_names]
    X = []
    y = []
    for ohlc in ohlc_list:
        i = 0
        while True:
            features = get_ith_features(i, ohlc, candles_num, future_length, tau, normalize)
            if features is None:
                break
            X.append(np.array(features[0]).flatten())
            y.append(features[1])
            i += 1
    return X, y


def show_sample_data(instrument_names=None, date_from=datetime(2019, 1, 1), date_to=datetime(2019, 12, 28),
                     candles_num=5, tau=0.0003, future_length=20, normalize=False):
    if instrument_names is None:
        instrument_names = ["EURUSD_FIVE_MINS"]
    ohlc = data_downloader.get_single_dataframe(instruments=instrument_names,
                                                date_from=date_from,
                                                date_to=date_to)
    data = ohlc.close
    data = data[::-1]
    for i in np.arange(future_length, (data.size / 2), int(data.size / 4)):
        i = int(i)
        x, target, barriers, start_date, data_i, Idx = get_ith_features(i, ohlc, candles_num, future_length, tau,
                                                                        normalize)

        plt.subplots_adjust(top=0.8)
        plt.title(instrument_names[0] + "\ntarget {}".format(target) +
                  "\n{}  -  {}".format(start_date, data.index[i]))
        plt.plot(data_i.values[::-1])
        for idx, pol in zip(Idx, x):
            x = np.linspace(idx[0], idx[-1])
            y = [np.polyval(pol, i) for i in x]
            plt.plot(data_i.size - x, y)
        for barrier in barriers:
            x = np.linspace(data_i.size - future_length, data_i.size)
            y = [barrier for _ in x]
            plt.plot(x, y)
        plt.show()
        plt.close()


def get_ith_features(i, ohlc, candles_num, future_length, tau, normalize):
    data = ohlc.close
    data = data[::-1]
    Idx = []
    curr_date = data.index[i]
    for delta in delta_list:
        curr_date = data.index[i]
        idx = []
        for _ in range(candles_num):
            candles = data[data.index <= curr_date]
            if len(candles) == 0:
                return None
            curr_candle_date = candles.index[0]
            next_date = curr_candle_date - delta
            idx.append(data.index.get_loc(curr_candle_date) - i + future_length)
            curr_date = next_date
        Idx.append(idx)

    data_i = data.iloc[i + Idx[0][0] - future_length: i + Idx[0][-1]].copy()
    last_close = data_i.iloc[future_length]
    barriers = [last_close * (1 + sign * tau) for sign in [-1, 1]]
    if normalize:
        mean = data_i.mean()
        std = data_i.std()
        data_i = (data_i - mean) / std
        barriers = [(barrier - mean) / std for barrier in barriers]
    x = [np.polyfit(idx[1:], [data_i.iloc[a:b].mean()
                              for a, b in zip(idx[:-1], idx[1:])], 2, full=True)[0]
         for idx in Idx]
    target = target_triple_barrier(ohlc.iloc[len(ohlc) - i - future_length: len(ohlc) - i], tau)
    if np.isnan(x).any():
        return None
    return [x, target, barriers, curr_date, data_i, Idx]


def target_triple_barrier(prices, tau):
    last_close = prices.iloc[0, 3]
    future_ohlc = prices.iloc[1:, :3].values
    changes = future_ohlc / last_close - 1
    up = (changes > tau).any(axis=1)
    down = (changes < -tau).any(axis=1)
    cha = np.stack([up, down]).any(axis=0)
    idx = np.where(cha)[0][0] if cha.any() else -1
    if idx == -1:
        return 1
    elif up[idx] and down[idx]:
        if changes[idx][0] > tau:
            return 2
        elif changes[idx][0] < -tau:
            return 0
        else:
            return 1
    elif up[idx]:
        return 2
    elif down[idx]:
        return 0
    else:
        return 1
