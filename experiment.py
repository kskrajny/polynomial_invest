import os
from datetime import timedelta
import pandas as pd
from single_training import single_training
import warnings

warnings.filterwarnings("ignore")


def create_meta_text(instrument_list, delta_list, data_delta, tau, future_length, polynomial_degree, candle_num,
                     class_balance, feature_eng_name, date_from=None, skip_time_days=None):
    text = "Data delta:\n" + str(data_delta).split('.')[0]
    text += "\n\nTau:  " + str(tau)
    text += "\n\nFuture length:  " + str(future_length)
    text += "\n\nCandle num:  " + str(candle_num)
    text += "\n\nPolynomial degree:  " + str(polynomial_degree)
    text += "\n\nFeature engineering:  " + feature_eng_name
    text += "\n\nClass_balance:  " + str(class_balance)
    text += "\n\nInstruments:"
    for name in instrument_list:
        name = name.split('_')[0]
        text += "\n" + name
    text += "\n\nDeltas:"
    for delta in delta_list:
        delta = str(delta).split('.')[0]
        text += "\n" + delta
    if date_from is not None:
        text += "\n\nStart date: " + date_from.strftime("%Y.%m.%d")
    if skip_time_days is not None:
        text += "\n\nTest days: " + str(skip_time_days)
    return text


def experiment(data_delta_days, clf_names, instrument_list, delta_list, tau, future_length, polynomial_degree,
               start_date, final_date, skip_time_days, candle_num, class_balance, feature_eng, feature_eng_name, labels,
               results_dir, classifiers):

    df = pd.DataFrame()
    exp_df = pd.DataFrame()
    date_from_train = start_date + timedelta(days=(7 - start_date.weekday()) % 7)

    date_to_test = None
    while date_to_test is None or date_to_test < final_date:
        print(date_from_train.strftime("%d.%m.%Y"))
        days = [date_from_train + timedelta(days=x) for x in range(5 * (data_delta_days + skip_time_days))]
        days = [x for x in days if x.weekday() < 5]
        date_from_test = days[data_delta_days]
        date_to_test = days[data_delta_days + skip_time_days]
        for clf_name in clf_names:
            # noinspection PyBroadException
            try:
                stats, exp_res = single_training(instrument_list, date_from_train, date_from_test, date_to_test,
                                                 delta_list, clf_name, tau, future_length, polynomial_degree,
                                                 classifiers, labels=labels, global_date_from=start_date,
                                                 global_date_to=final_date, candle_num=candle_num,
                                                 feature_eng=feature_eng[feature_eng_name], class_balance=class_balance)
                for i in range(len(instrument_list)):
                    df = pd.concat([df, stats[i]], ignore_index=True)
                exp_df = pd.concat([exp_df, exp_res], ignore_index=True)
            except Exception as e:
                print(e)
        date_from_train = days[skip_time_days]
    os.mkdir(results_dir)
    df.set_index(['decision_date', 'instrument', 'model'], inplace=True)
    df.to_csv(results_dir + 'data.csv')
    meta_text = create_meta_text(instrument_list, delta_list, data_delta_days, tau, future_length, polynomial_degree,
                                 candle_num, class_balance, feature_eng_name, skip_time_days=skip_time_days)
    with open(results_dir + 'meta', 'w') as f:
        f.write(meta_text)
