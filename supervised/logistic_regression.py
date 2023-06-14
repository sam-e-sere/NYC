import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def logisticRegression(data, categorical_features, numeric_features, target):
    # Definizione delle feature binarie
    binary_features = ["CLOUDCOVER"]
    # Rimozione della feature binaria dalle feature numeriche
    numeric_features = [f for f in numeric_features if f != "CLOUDCOVER"]

    X = data[categorical_features + numeric_features + binary_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

    # Normalizzazione delle feature numeriche
    scaler = StandardScaler()
    X.loc[:, numeric_features] = scaler.fit_transform(X[numeric_features])

    # Divisione del dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Prova diversi valori di C per la regressione logistica
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    
    clf = LogisticRegression(max_iter=1000, random_state=0)
    grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5, return_train_score=True)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    mean_train_score = accuracy_score(y_train, y_train_pred)
    mean_test_score = accuracy_score(y_test, y_test_pred)
    mean_test_p = precision_score(y_test, y_test_pred, average='macro', zero_division = 0)
    mean_test_r = recall_score(y_test, y_test_pred, average='macro')
    mean_test_f = f1_score(y_test, y_test_pred, average='macro')

    plt.plot(param_grid['C'], grid_search.cv_results_['mean_train_score'], label="Training score")
    plt.plot(param_grid['C'], grid_search.cv_results_['mean_test_score'], label="Test score")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.title("Accuracy Scores for Different C Values")
    plt.legend()
    plt.show()

    print(f"Media test acc: {mean_test_score}")
    print(f"Media test prec: {mean_test_p}")
    print(f"Media test rec: {mean_test_r}")
    print(f"Media test f-measure: {mean_test_f}")

    return best_model