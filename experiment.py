from datetime import timedelta
import pandas as pd
from save_results_functions import create_meta_text, save_results
from single_training import single_training
from contans_main import data_delta_days, clf_names, instrument_list, delta_list, tau, future_length, \
    polynomial_degree, file_names, start_date, final_date, skip_time_days, explain, results_dir
import warnings
warnings.filterwarnings("ignore")

dfs = [pd.DataFrame() for x in instrument_list]
exp_df = pd.DataFrame()
date_from_train = start_date + timedelta(days=(7 - start_date.weekday()) % 7)

labels = []
for delta in delta_list:
    for j in range(polynomial_degree):
        labels.append("{}_{}".format(str(delta).split('.')[0], polynomial_degree - j))

date_to_test = None
while date_to_test is None or date_to_test < final_date:
    print(date_from_train.strftime("%d.%m.%Y"))
    days = [date_from_train + timedelta(days=x) for x in range(5 * (data_delta_days + skip_time_days))]
    days = [x for x in days if x.weekday() < 5]
    date_from_test = days[data_delta_days]
    date_to_test = days[data_delta_days + skip_time_days]
    for clf_name in clf_names:
        stats, exp_res = single_training(instrument_list, date_from_train, date_from_test, date_to_test, delta_list,
                                         clf_name, tau, future_length, polynomial_degree, labels=labels,
                                         global_date_from=start_date, global_date_to=final_date, explain=explain)
        for i in range(len(instrument_list)):
            dfs[i] = dfs[i].append(stats[i], ignore_index=True, sort=False)
        exp_df = pd.concat([exp_df, exp_res], ignore_index=True)
    date_from_train = days[skip_time_days]
meta_text = create_meta_text(instrument_list, delta_list, data_delta_days, tau, future_length, polynomial_degree,
                             skip_time_days=skip_time_days)
save_results(dfs, exp_df, file_names, meta_text, future_length)
