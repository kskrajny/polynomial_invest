import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from classifier import CustomMLPClassifier, CustomGradientBoostingClassifier, CustomCatBoostClassifier
from excel_functions import create_meta_text
from get_data import read_data
from sklearn.model_selection import train_test_split
import dalex as dx

classifiers = {
    "XGBoost": lambda: CustomGradientBoostingClassifier(learning_rate=0.01, max_depth=10),
    "MLP": lambda: CustomMLPClassifier(random_state=1, max_iter=5000, learning_rate="invscaling",
                                       learning_rate_init=.001, early_stopping=True),
    "CatBoost": lambda: CustomCatBoostClassifier(iterations=500, early_stopping_rounds=20, learning_rate=0.1, depth=8,
                                                 verbose=False)
}


def single_training(instrument_list, date_from, date_to, delta_list, clf_name, tau, future_length, polynomial_degree,
                    explain=False):
    X, y = read_data(instrument_list, date_from=date_from, date_to=date_to, delta_list=delta_list, tau=tau,
                     future_length=future_length, polynomial_degree=polynomial_degree)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    clf = classifiers[clf_name]().fit(X_train, y_train)
    acc, matrix, gain, gain_up, gain_down = clf.statistics(X_test, y_test, tau)
    stats = {"Model": clf_name, "Start": date_from.strftime("%d.%m.%y"), "End": date_to.strftime("%d.%m.%y"),
             "Acc_down": matrix[0][0] / (matrix[0][0] + matrix[0][1]),
             "Acc_up": matrix[1][1] / (matrix[1][0] + matrix[1][1]),
             "Acc": acc, "Gain_%": gain * 100, "Gain_%_up": gain_up * 100, "Gain_%_down": gain_down * 100,
             "Down": matrix[0][0] + matrix[0][1], "Up": matrix[1][0] + matrix[1][1]}
    if explain:
        print("Explaining")
        print(type(X), np.array(X))
        explainer = dx.Explainer(clf, np.array(X), verbose=False)
        explanation_dir_name = "results/exp_{}/".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.mkdir(explanation_dir_name)
        txt = create_meta_text(instrument_list, delta_list, date_to - date_from, tau, future_length, polynomial_degree,
                         date_from)
        text_file = open(explanation_dir_name + "meta", "w")
        text_file.write(txt)
        text_file.close()
        print("Plotting")
        for i in range(len(X_test)):
            explainer.predict_parts(X_test[i], type="shap").plot(max_vars=5)
            plt.show()
            #plt.savefig(explanation_dir_name + 'shap_' + str(i))
    return stats
