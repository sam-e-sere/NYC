import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Carica il dataset
dataset = pd.read_csv("data/New NYC Accidents.csv")

# Seleziona le variabili predittive codificate (strada, latitudine e longitudine) e la variabile target (borgo)
X_train = dataset[dataset["BOROUGH"].notnull()][["LATITUDE", "LONGITUDE"]]
y_train = dataset[dataset["BOROUGH"].notnull()]["BOROUGH"]
X_test = dataset[dataset["BOROUGH"].isnull()][["LATITUDE", "LONGITUDE"]]

# Addestra il modello con KNN
knn = KNeighborsClassifier(n_neighbors=5) # Numero di vicini da considerare
knn.fit(X_train, y_train)

# Utilizza il modello per fare previsioni sul set di test
y_pred = knn.predict(X_test)

# Aggiungi i valori predetti al dataset originale
dataset.loc[dataset["BOROUGH"].isnull(), "BOROUGH"] = y_pred

# visualizza il dataframe risultante
dataset.to_csv("data/Complete Accidents.csv", index=False, mode='w')

# Valuta la precisione del modello
accuracy = knn.score(X_test, y_pred)
print("Accuracy:", accuracy)