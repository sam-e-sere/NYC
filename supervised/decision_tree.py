from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

    # dopo aver usato i tre criteri, è emerso che non vi è nessuna differenza significativa tra le prestazioni
    # è stato utilizzato il criterio entropy
    
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

    #creazione del grafico
    plt.clf()
    plt.title('Training and Test score')
    plt.plot(range(3, 26), mean_train_score, label="Training score")
    plt.plot(range(3, 26), mean_test_score, label="Test score")
    plt.xlabel("Tree Depth")
    plt.ylabel("Score")
    plt.legend()
    plt.ylim([0.2, 1.0])
    
    path = "images/dt_score.png" 
    plt.savefig(path)

    print(f"Average Test Accuracy: {np.mean(mean_test_score)}")
    print(f"Average Test Precision: {np.mean(mean_test_p)}")
    print(f"Average Test Recall: {np.mean(mean_test_r)}")
    print(f"Average Test F1-score: {np.mean(mean_test_f)}")

    return clf
