from datetime import timedelta, datetime

instrument_list = ["EURUSD_FIVE_MINS", "EURCHF_FIVE_MINS", "AUDCAD_FIVE_MINS", "CADCHF_FIVE_MINS"]
excel_name = "results/{}.xlsx".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
delta_list = [
    # timedelta(days=1),
    # timedelta(hours=4),
    timedelta(hours=1),
    timedelta(minutes=20),
    timedelta(minutes=5)
]
clf_names = [
    "XGBoost",
    "MLP",
    "CatBoost"
]
data_delta = timedelta(days=16)
tau = 0.0003
future_length = 20
polynomial_degree = 2

start_date = datetime(2013, 1, 1)
final_date = datetime(2021, 12, 31)

skip_time = timedelta(days=4)
