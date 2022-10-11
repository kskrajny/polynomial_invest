from classifier import CustomMLPClassifier
from get_data import read_data
from sklearn.model_selection import train_test_split

X, y = read_data(["EURUSD_FIVE_MINS"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = CustomMLPClassifier(random_state=1, max_iter=1000, learning_rate="invscaling", learning_rate_init=0.001,
                          early_stopping=True, verbose=True).fit(X_train, y_train)

acc, matrix = clf.statistics(X_test, y_test)

for i, row in enumerate(matrix):
    print("ACC_{}: {}".format(i, row[i] / sum(row)))

print("ACC: {}".format(acc))
