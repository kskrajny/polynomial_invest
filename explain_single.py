from datetime import timedelta, datetime

from single_training import single_training

instrument_list = ["EURUSD_FIVE_MINS", "EURCHF_FIVE_MINS", "AUDCAD_FIVE_MINS", "CADCHF_FIVE_MINS"]
delta_list = [
    timedelta(hours=1),
    timedelta(minutes=20),
    timedelta(minutes=5)
]
clf_names = [
    "XGBoost",
    "MLP",
    "CatBoost"
]
tau = 0.0003
future_length = 20
polynomial_degree = 2
date_from = datetime(2013, 1, 1)
data_delta = timedelta(days=3)

single_training(instrument_list, date_from, date_from + data_delta, delta_list, clf_names[1], tau, future_length,
                polynomial_degree, explain=True)

