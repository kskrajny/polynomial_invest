import numpy as np
from get_data import read_data
import dalex as dx


def single_training(instrument_list, date_from_train, date_from_test, date_to_test, delta_list, clf_name, tau,
                    future_length, polynomial_degree, classifiers, tresholds, explain=0, labels=None,
                    global_date_from=None, global_date_to=None, candle_num=5, feature_eng=None, class_balance=True,
                    spread_q=0):

    data_train = read_data(instrument_list, date_from=date_from_train, date_to=date_from_test,
                                          delta_list=delta_list, tau=tau, future_length=future_length,
                                          polynomial_degree=polynomial_degree, labels=labels,
                           global_date_from=global_date_from, global_date_to=global_date_to,
                           candle_num=candle_num, feature_eng=feature_eng)

    data_test = read_data(instrument_list, date_from=date_from_test, date_to=date_to_test,
                                            delta_list=delta_list, tau=tau, future_length=future_length,
                                                   polynomial_degree=polynomial_degree, labels=labels,
                          global_date_from=global_date_from, global_date_to=global_date_to,
                          candle_num=candle_num, feature_eng=feature_eng)

    X_train = data_train[labels]
    y_train = data_train['target'].values
    X_train = X_train.iloc[np.where(y_train != -1)]
    y_train = np.array([y for y in y_train if y != -1])
    clf = classifiers[clf_name]().fit(X_train.values, y_train, class_balance)

    st = []
    exp_df = None
    if explain > 0:
        explainer_train = dx.Explainer(clf, X_train.reset_index().drop(columns=['decision_date']), y_train,
                                       verbose=False, label=clf_name)
        exp_df = explainer_train.model_parts().result
        exp_df.dropout_loss -= exp_df[exp_df['variable'] == '_full_model_']['dropout_loss'].values[0]
        exp_df.drop(exp_df[exp_df.variable == '_full_model_'].index, inplace=True)
        exp_df.drop(exp_df[exp_df.variable == '_baseline_'].index, inplace=True)

    for i in range(len(instrument_list)):
        max_spread = data_train[data_train['instrument'] == instrument_list[i]]['spread'].quantile(spread_q)\
            if spread_q else 1000
        instrument_data = data_test[data_test['instrument'] == instrument_list[i]]
        X_test = instrument_data[labels]
        y_test = instrument_data['target'].values
        test_len = len(y_test)
        st_t = []
        for t in tresholds:
            acc, matrix, lost_test, gains = clf.statistics(X_test.values, y_test, instrument_data['potential_gain'],
                                                           instrument_data['spread'], t, max_spread)
            gain = np.sum(gains)
            acc_up, acc_down, pred_up, pred_down, gain_per_position = 0, 0, 0, 0, 0
            try:
                acc_down = matrix[0][0] / (matrix[0][0] + matrix[1][0]) if matrix[0][0] + matrix[1][0] > 0 else 0.5
                acc_up = matrix[1][1] / (matrix[1][1] + matrix[0][1]) if matrix[1][1] + matrix[0][1] > 0 else 0.5
                pred_up = matrix[0][0] + matrix[1][0]
                pred_down = matrix[0][1] + matrix[1][1]
                gain_per_position = gain / (pred_up + pred_down)
            except IndexError:
                pass
            stats = {"Model": clf_name, "Time_start": date_from_test.strftime("%d.%m.%y"),
                     "Time_end": date_to_test.strftime("%d.%m.%y"),
                     "Acc_down": acc_down, "Acc_up": acc_up,
                     "Acc": acc, "Gain": gain, "Pred_up": pred_up,
                     "Pred_down": pred_down, "Gain_per_position": gain_per_position,
                     "Lost_test_%": lost_test / test_len * 100, "Treshold": t, "Max_spread": max_spread}
            st_t.append(stats)
        st.append(st_t)
    return st, exp_df
