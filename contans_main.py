from datetime import timedelta, datetime
from classifier import ANN, CustomTabNet
from get_data import feature_eng_partial_square

time_stamp = "_FIFTEEN_MINS"
instrument_list = [x + time_stamp for x in ["US500USD", "AUS200AUD"]]

now = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = "results/{}/".format(now)

file_names = [
    results_dir + instrument for instrument in instrument_list
]

delta_list = [
    timedelta(hours=12),
    timedelta(hours=6),
    timedelta(hours=1)
]

clf_names = [
    "TabNet",
    "MLP",
    "MLP2",
    "MLPD",
    "MLPD2",
    'MLP_EASY'
]

classifiers = {
    'MLP': lambda: ANN([len(delta_list) * (polynomial_degree + 1), 128, 2], .00005),
    'MLP1': lambda: ANN([len(delta_list) * (polynomial_degree + 1), 64, 2], .00005),
    'MLP2': lambda: ANN([len(delta_list) * (polynomial_degree + 1), 16, 2], .00005),
    'MLPD': lambda: ANN([len(delta_list) * (polynomial_degree + 1), 16, 64, 16, 2], .00001),
    'MLPD2': lambda: ANN([len(delta_list) * (polynomial_degree + 1), 64, 64, 2], .00005),
    'MLP_EASY': lambda: ANN([len(delta_list) * (polynomial_degree + 1), 2], .00005),
    'TabNet': lambda: CustomTabNet()
}

data_delta_days = 50
tau = 0.005
future_length = 40
polynomial_degree = 2
candle_num = 10

start_date = datetime(2017, 1, 1)
final_date = datetime(2021, 12, 31)

skip_time_days = 20
explain = 1

tresholds = [0.5]

class_balance = True

feature_eng_name = "feature_eng_partial_square"

feature_eng = {
    "None": lambda x: x,
    "feature_eng_partial_square": feature_eng_partial_square
}

labels = []
for delta in delta_list:
    for j in range(polynomial_degree):
        labels.append("{}_{}".format(str(delta).split('.')[0], polynomial_degree - j))
    labels.append("{}_mul".format(str(delta).split('.')[0]))

spread_q = 0.75