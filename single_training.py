import numpy as np
from get_data import read_data


def single_training(instrument_list, date_from_train, date_from_test, date_to_test, delta_list, clf_name, tau,
                    future_length, polynomial_degree, classifiers, explain=0, labels=None, global_date_from=None,
                    global_date_to=None, candle_num=None, feature_eng=None, class_balance=True):

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

    results = []
    exp_df = None
    '''
    if explain > 0:
        explainer_train = dx.Explainer(clf, X_train.reset_index().drop(columns=['decision_date']), y_train,
                                       verbose=False, label=clf_name)
        exp_df = explainer_train.model_parts().result
        exp_df.dropout_loss -= exp_df[exp_df['variable'] == '_full_model_']['dropout_loss'].values[0]
        exp_df.drop(exp_df[exp_df.variable == '_full_model_'].index, inplace=True)
        exp_df.drop(exp_df[exp_df.variable == '_baseline_'].index, inplace=True)
    '''
    for i in range(len(instrument_list)):
        instrument_data = data_test[data_test['instrument'] == instrument_list[i]]
        X_test = instrument_data[labels]
        proba = clf.predict_proba(X_test.values)
        instrument_data['proba_1'] = proba[:, 1]
        instrument_data['model'] = clf_name
        instrument_data['future_length'] = future_length
        instrument_data.reset_index(inplace=True)
        results.append(
            instrument_data[['decision_date', 'target', 'potential_gain', 'commission', 'spread', 'proba_1', 'model',
                             'future_length', 'instrument', 'usd_per_pips']]
        )
    return results, exp_df
