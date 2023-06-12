import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt

def gradientBoost(data, categorical_features, numeric_features, target):
    X = data[categorical_features + numeric_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

    # Divisione del dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_estimators_range = [10, 50, 100, 150]

    mean_train_score = []
    mean_test_score = []
    mean_test_p = []
    mean_test_r = []
    mean_test_f = []

    for i in n_estimators_range:
        clf = GradientBoostingClassifier(n_estimators=i, learning_rate=0.05, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
        cv_results = cross_validate(clf, X, y, cv=10, scoring=scoring, return_train_score=True)
        mean_train_score.append(np.mean(cv_results['train_accuracy']))
        mean_test_score.append(np.mean(cv_results['test_accuracy']))
        mean_test_p.append(np.mean(cv_results['test_precision']))
        mean_test_r.append(np.mean(cv_results['test_recall']))
        mean_test_f.append(np.mean(cv_results['test_f1']))

    plt.plot(n_estimators_range, mean_train_score, label="Training score")
    plt.plot(n_estimators_range, mean_test_score, label="Test score")
    plt.xlabel("Number of Trees")
    plt.ylabel("Score")
    plt.legend()
    plt.ylim([0.2, 1.0])
    #plt.show()

    print(f"Media test acc: {np.mean(mean_test_score)}")
    print(f"Media test prec: {np.mean(mean_test_p)}")
    print(f"Media test rec: {np.mean(mean_test_r)}")
    print(f"Media test f-measure: {np.mean(mean_test_f)}")

    return clf