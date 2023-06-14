import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt

def randomForest(data, categorical_features, numeric_features, target):
    X = data[categorical_features + numeric_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

    # Divisione del dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    max_depth_range = range(3, 26)
    n_estimators_range = [10, 50, 100, 150]

    mean_train_score_depth = []
    mean_test_score_depth = []
    mean_train_score_trees = []
    mean_test_score_trees = []
    mean_test_p = []
    mean_test_r = []
    mean_test_f = []
    
    for i in max_depth_range:
        clf = RandomForestClassifier(n_estimators=100, max_depth=i, criterion="entropy", random_state=0)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        mean_train_score_depth.append(accuracy_score(y_train, y_train_pred))
        mean_test_score_depth.append(accuracy_score(y_test, y_test_pred))
        mean_test_p.append(precision_score(y_test, y_test_pred, average='macro', zero_division = 0))
        mean_test_r.append(recall_score(y_test, y_test_pred, average='macro'))
        mean_test_f.append(f1_score(y_test, y_test_pred, average='macro'))
    
    for i in n_estimators_range:
        clf = RandomForestClassifier(n_estimators=i, max_depth=10, criterion="entropy", random_state=0)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        mean_train_score_trees.append(accuracy_score(y_train, y_train_pred))
        mean_test_score_trees.append(accuracy_score(y_test, y_test_pred))
        

    plt.figure(figsize=(8, 6))
    plt.title('Train and Test score')
    plt.plot(max_depth_range, mean_train_score_depth, label="Training score")
    plt.plot(max_depth_range, mean_test_score_depth, label="Test score")
    plt.xlabel("Tree Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.ylim([0.2, 1.0])
    # Specifica il percorso completo del file in cui salvare il grafico 
    path = "images/rf_score_depth.png" 
    plt.savefig(path)
    
    plt.figure(figsize=(8, 6))
    plt.title('Train and Test score')
    plt.plot(n_estimators_range, mean_train_score_trees, label="Training score")
    plt.plot(n_estimators_range, mean_test_score_trees, label="Test score")
    plt.xlabel("Number of Trees")
    plt.ylabel("Score")
    plt.legend()
    plt.ylim([0.2, 1.0])
    # Specifica il percorso completo del file in cui salvare il grafico 
    path = "images/rf_score_number_of_trees.png" 
    plt.savefig(path)

    print(f"Media test acc: {np.mean(mean_test_score_depth)}")
    print(f"Media test prec: {np.mean(mean_test_p)}")
    print(f"Media test rec: {np.mean(mean_test_r)}")
    print(f"Media test f-measure: {np.mean(mean_test_f)}")

    return clf