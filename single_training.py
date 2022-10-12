from classifier import CustomMLPClassifier
from get_data import read_data
from sklearn.model_selection import train_test_split


def single_training(instrument_list, lr, date_from, date_to, delta_list):
    X, y = read_data(instrument_list, date_from=date_from, date_to=date_to, delta_list=delta_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    clf = CustomMLPClassifier(random_state=1, max_iter=200, learning_rate="invscaling", learning_rate_init=lr,
                              early_stopping=True).fit(X_train, y_train)

    acc, matrix = clf.statistics(X_test, y_test)
    stats = {"Start": date_from.strftime("%d.%m.%y"), "End": date_to.strftime("%d.%m.%y")}
    for i, row in enumerate(matrix):
        stats["Acc_%d" % i] = row[i] / sum(row)
    stats["Acc"] = acc

    return stats
