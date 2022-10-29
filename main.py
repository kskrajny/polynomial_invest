from datetime import datetime, timedelta
from excel_functions import save_text, create_meta_text, save_rolling_stats
from single_training import single_training
import pandas as pd
import pprint

pp = pprint.PrettyPrinter(indent=4)

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
data_delta = timedelta(weeks=1)
tau = 0.0003
future_length = 20
polynomial_degree = 2
meta_text = create_meta_text(instrument_list, delta_list, data_delta, tau, future_length, polynomial_degree)

df = pd.DataFrame()

date_from = datetime(2013, 1, 1)

while date_from < datetime(2021, 1, 1):
    print(date_from.strftime("%d.%m.%Y"))
    date_to = date_from + data_delta
    for clf_name in clf_names:
        stats = single_training(instrument_list, date_from, date_to, delta_list, clf_name, tau, future_length,
                                polynomial_degree)
        df = df.append(stats, ignore_index=True)
        pp.pprint(stats)
    date_from += timedelta(weeks=3)

df_ = df.groupby('Model').mean()
print(df_)

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
