from contans_test import data_delta_days, clf_names, instrument_list, delta_list, tau, future_length, \
    polynomial_degree, file_names, start_date, final_date, skip_time_days, explain, candle_num, class_balance, \
    feature_eng, feature_eng_name, labels, results_dir, classifiers, tresholds, spread_q
from experiment import experiment

experiment(data_delta_days, clf_names, instrument_list, delta_list, tau, future_length,
    polynomial_degree, file_names, start_date, final_date, skip_time_days, explain, candle_num, class_balance,
    feature_eng, feature_eng_name, labels, results_dir, classifiers, tresholds, spread_q
)