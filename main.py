from datetime import datetime, timedelta
from excel_functions import save_text, create_meta_text
from single_training import single_training
import pandas as pd


instrument_list = ["EURUSD_FIVE_MINS", "EURCHF_FIVE_MINS", "AUDCAD_FIVE_MINS", "CADCHF_FIVE_MINS"]
lr = 0.001
excel_name = "results/{}.xlsx".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
delta_list = [
    #timedelta(days=1),
    #timedelta(hours=4),
    timedelta(hours=1),
    timedelta(minutes=20),
    timedelta(minutes=5)
]
data_delta = timedelta(weeks=30)
meta_text = create_meta_text(instrument_list, delta_list, data_delta, lr)

df = pd.DataFrame()

date_from = datetime(2017, 1, 1)

while date_from < datetime(2021, 1, 1):
    print(date_from.strftime("%d.%m.%Y"))
    date_to = date_from + data_delta
    stats = single_training(instrument_list, lr, date_from, date_to, delta_list)
    df = df.append(stats, ignore_index=True)
    date_from = date_to

with pd.ExcelWriter(excel_name, engine='xlsxwriter') as writer:
    df.to_excel(writer)
    save_text(writer, meta_text)
