import pandas as pd
from excel_functions import create_meta_text, save_to_excel
from single_training import single_training
from contans_main import data_delta, clf_names, instrument_list, delta_list, tau, future_length, polynomial_degree,\
    excel_name, start_date, final_date, skip_time


df = pd.DataFrame()
date_from = start_date
while date_from < final_date:
    print(date_from.strftime("%d.%m.%Y"))
    date_to = date_from + data_delta
    for clf_name in clf_names:
        stats = single_training(instrument_list, date_from, date_to, delta_list, clf_name, tau, future_length,
                                polynomial_degree)
        df = df.append(stats, ignore_index=True)
    date_from += skip_time

meta_text = create_meta_text(instrument_list, delta_list, data_delta, tau, future_length, polynomial_degree)
save_to_excel(df, excel_name, meta_text)
