import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt

def decisionTree(data, categorical_features, numeric_features, target):
    X = data[categorical_features + numeric_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

    # Divisione del dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    #for crit in {"gini", "entropy", "log_loss"}:
        #print(f"Criterion: {crit}")

    mean_train_score = []
    mean_test_score = []
    mean_test_p = []
    mean_test_r = []
    mean_test_f = []

    for i in range(3, 26):
        clf = DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=0)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        mean_train_score.append(accuracy_score(y_train, y_train_pred))
        mean_test_score.append(accuracy_score(y_test, y_test_pred))
        mean_test_p.append(precision_score(y_test, y_test_pred, average='macro'))
        mean_test_r.append(recall_score(y_test, y_test_pred, average='macro'))
        mean_test_f.append(f1_score(y_test, y_test_pred, average='macro'))

    plt.title('Training and Test score')
    plt.plot(range(3, 26), mean_train_score, label="Training score")
    plt.plot(range(3, 26), mean_test_score, label="Test score")
    plt.xlabel("Tree Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.ylim([0.2, 1.0])
    
    # Specifica il percorso completo del file in cui salvare il grafico 
    path = "images/dt_score.png" 
    # Salva il grafico nella directory specificata 
    plt.savefig(path)

    print(f"Media test acc: {np.mean(mean_test_score)}")
    print(f"Media test prec: {np.mean(mean_test_p)}")
    print(f"Media test rec: {np.mean(mean_test_r)}")
    print(f"Media test f-measure: {np.mean(mean_test_f)}")

    return clf
