from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from decision_tree import decisionTree
from random_forest import randomForest
from logistic_regression import logisticRegression
from ada_boost import adaBoost
from gradient_boost import gradientBoost
from naiveBayesCategorico import naiveBayesCategorical

def printFeatureRanking(clf, X):
    # Visualizzazione delle feature più importanti
    if 'LogisticRegression' in clf.__class__.__name__:
        importances = np.abs(clf.coef_)[0]
    else:
        importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(0, 4):
        print("%d. %s (%f)" % (f+1, X.columns[indices[f]], importances[indices[f]]))

    # Plotting dell'importanza delle feature
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]), [categorical_features[i] if i<len(categorical_features) else numeric_features[i-len(categorical_features)] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

# Caricamento del dataset
df1 = pd.read_csv("data/Selected Accidents.csv")
df2 = pd.read_csv("kb/generated_dataset.csv")
data = pd.merge(df1, df2, on="COLLISION_ID")

# Selezionare le feature e la variabile target
categorical_features = ["BOROUGH", "TRAFFIC STREET", "TIME_OF_DAY", "TEMPERATURE", "RAIN_INTENSITY", "WIND_INTENSITY", "TRAFFIC_VOLUME", "DAY OF WEEK"]
numeric_features = ["M", "CLOUDCOVER", "AVERAGE_VOLUME", 'NUM_ACCIDENTS_BOROUGH', 'NUM_ACCIDENTS_ON_STREET']
target = "IS_NOT_DANGEROUS"
X = data[categorical_features + numeric_features]

"""
print("---DECISION TREE---")
clf = decisionTree(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X)

"""
print("---RANDOM FOREST---")
clf = randomForest(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X)
"""
print("---LOGISTIC REGRESSION---")
clf = logisticRegression(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X)


print("---ADA BOOST---")
clf = adaBoost(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X)

print("---GRADIENT BOOST---")
clf = gradientBoost(data, categorical_features, numeric_features, target)
printFeatureRanking(clf, X)

print("---NAIVE BAYES CATEGORICAL---")
clf = naiveBayesCategorical(data, categorical_features, target)
"""
