from datetime import timedelta, datetime
from classifier import ANN, CustomTabNet

time_stamp = "_ONE_HOUR"
instrument_list = [x + time_stamp for x in ["EURUSD", "EURCHF", "AUDCAD", "CADCHF"]]

now = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = "results/{}/".format(now)

file_names = [
    results_dir + instrument for instrument in instrument_list
]

delta_list = [
    timedelta(hours=16),
    timedelta(hours=8),
    timedelta(hours=4)
]

clf_names = [
    "TabNet",
    "MLP",
    "MLP1",
    "MLP2"
]

classifiers = {
    'MLP': lambda: ANN([len(delta_list) * polynomial_degree, 128, 2]),
    'MLP1': lambda: ANN([len(delta_list) * polynomial_degree, 64, 2]),
    'MLP2': lambda: ANN([len(delta_list) * polynomial_degree, 16, 2]),
    'TabNet': lambda: CustomTabNet()
}

data_delta_days = 180
tau = 0.0017
future_length = 15
polynomial_degree = 2

start_date = datetime(2013, 1, 1)
final_date = datetime(2021, 12, 31)

skip_time_days = 60
explain = 1

tresholds = [0.5, 0.504, 0.508, 0.512]
