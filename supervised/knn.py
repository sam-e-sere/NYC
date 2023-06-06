from locale import normalize
import pandas as pd
from sklearn import clone, model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# for average -> "macro" or "micro"
def print_classifier_scores(true_y, pred_y, beta=1.0, average="macro"):
    acc, pr, rec, f_sc = calc_classifier_scores(true_y, pred_y, beta=beta, average=average)
    print("Accuracy:\t" + str(acc))
    print("Precision:\t" + str(pr))
    print("Recall:\t\t" + str(rec))
    print("F-measure" + ":\t" + str(f_sc))
    print("(beta " + str(beta) + ")")

def calc_classifier_scores(true_y, pred_y, beta=1.0, average="macro"):
    (pr, rec, f_sc, su) = precision_recall_fscore_support(y_true=true_y, y_pred=pred_y, beta=beta, average=average)
    acc = accuracy_score(y_true=true_y, y_pred=pred_y)
    return acc, pr, rec, f_sc


def k_fold(X: pd.DataFrame, y: pd.Series, n_folds: int, classifier, verbose=False):
    kf = model_selection.KFold(n_splits=n_folds, shuffle=True)
    i_train = 0  # accuracy accumulator for training
    i_test = 0  # accuracy accumulator for test
    j = 1  # counter
    max_score = 0
    curr_test_score = 0

    test_p_acc = 0  # precision accumulator for test
    test_r_acc = 0  # recall accumulator for test
    test_f_acc = 0  # f measure accumulator for test

    for train_indexes, test_indexes in kf.split(X, y):
        curr_classifier = clone(classifier)
        curr_classifier = curr_classifier.fit(X.iloc[train_indexes], y[train_indexes])

        # score == acc
        curr_train_score = curr_classifier.score(X.iloc[train_indexes], y[train_indexes])
        curr_test_score = curr_classifier.score(X.iloc[test_indexes], y[test_indexes])

        true_y = y[test_indexes]
        pred_y = curr_classifier.predict(X.iloc[test_indexes])

        acc, curr_test_p, curr_test_r, curr_test_f = calc_classifier_scores(true_y=true_y, pred_y=pred_y)

        if verbose:
            print("Fold " + str(j) + "/" + str(n_folds))
            print("--------MODEL " + str(j) + " QUALITY--------")

            print("-------| Training |-----------")

            print_classifier_scores(true_y=y[train_indexes],
                                    pred_y=curr_classifier.predict(X.iloc[train_indexes]))

            print("-------|   Test   |-----------")
            print_classifier_scores(true_y=true_y, pred_y=pred_y)

        if curr_test_score > max_score:
            best_classifier = curr_classifier
            max_score = curr_test_score

        j += 1
        i_train += curr_train_score
        i_test += curr_test_score

        test_p_acc += curr_test_p
        test_r_acc += curr_test_r
        test_f_acc += curr_test_f

    mean_train_score = i_train / n_folds
    mean_test_score = i_test / n_folds
    mean_p_score = test_p_acc / n_folds
    mean_r_score = test_r_acc / n_folds
    mean_f_score = test_f_acc / n_folds

    return best_classifier, mean_train_score, mean_test_score, mean_p_score, mean_r_score, mean_f_score


def my_knn():

    df1 = pd.read_csv("data/Selected Accidents.csv")
    df2 = pd.read_csv("kb/generated_dataset.csv")
    x = pd.merge(df1, df2, on="COLLISION_ID")

    #x = pd.DataFrame(normalize(df[["Latitude", "Longitude"]], axis=0))

    y = x["IS_NOT_DANGEROUS"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=33)
    knn = KNeighborsClassifier()
    k_scores = []

    for k in range(1, 31, 2):
        best_model, train_score, test_score, test_prec, test_rec, test_f = k_fold(x, y, 10, classifier=KNeighborsClassifier(k),verbose=False)
        print("For: ", k)
        print("Test: ", str(test_score))
        print("Train: ", str(train_score))

    pass

my_knn()