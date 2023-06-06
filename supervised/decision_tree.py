import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import matplotlib.pyplot as plt

# Caricamento del dataset
df1 = pd.read_csv("data/Selected Accidents.csv")
df2 = pd.read_csv("kb/generated_dataset.csv")
data = pd.merge(df1, df2, on="COLLISION_ID")

# Selezionare le feature e la variabile target
categorical_features = ["BOROUGH", "TRAFFIC STREET", "TIME_OF_DAY", "TEMPERATURE", "RAIN_INTENSITY", "WIND_INTENSITY", "TRAFFIC_VOLUME", "DAY OF WEEK"]
numeric_features = ["M", "CLOUDCOVER", "AVERAGE_VOLUME"]
target = "IS_NOT_DANGEROUS"

X = data[categorical_features + numeric_features]
y = data[target]

# Encoding delle variabili categoriche
encoder = OrdinalEncoder()
X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

# Divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for crit in {"gini", "entropy", "log_loss"}:
    print(f"Criterion: {crit}")

    mean_train_score = []
    mean_test_score = []
    mean_test_p = []
    mean_test_r = []
    mean_test_f = []

    for i in range(3, 26):
        clf = DecisionTreeClassifier(max_depth=i, criterion=crit, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
        cv_results = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
        mean_train_score.append(np.mean(cv_results['train_accuracy']))
        mean_test_score.append(np.mean(cv_results['test_accuracy']))
        mean_test_p.append(np.mean(cv_results['test_precision']))
        mean_test_r.append(np.mean(cv_results['test_recall']))
        mean_test_f.append(np.mean(cv_results['test_f1']))

    plt.plot(range(3, 26), mean_train_score, label="Training score")
    plt.plot(range(3, 26), mean_test_score, label="Test score")
    plt.xlabel("Tree Depth")
    plt.ylabel("Score")
    plt.legend()
    #plt.savefig(f"{directory}tree_{crit}.png")
    plt.show()

    print(f"Media test acc: {np.mean(mean_test_score)}")
    print(f"Media test prec: {np.mean(mean_test_p)}")
    print(f"Media test rec: {np.mean(mean_test_r)}")
    print(f"Media test f-measure: {np.mean(mean_test_f)}")

# Visualizzazione delle feature più importanti
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(0, 4):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plotting dell'importanza delle feature
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), [categorical_features[i] if i<len(categorical_features) else numeric_features[i-len(categorical_features)] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()