import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Carica il dataset
dataset = pd.read_csv("data/New NYC Accidents.csv")

# Seleziona le variabili predittive codificate (strada, latitudine e longitudine) e la variabile target (borgo)
X_train = dataset[dataset["BOROUGH"].notnull()][["LATITUDE", "LONGITUDE"]]
y_train = dataset[dataset["BOROUGH"].notnull()]["BOROUGH"]
X_test = dataset[dataset["BOROUGH"].isnull()][["LATITUDE", "LONGITUDE"]]

# Crea un modello di KNN
knn = KNeighborsClassifier()

# Definisci i valori di k da testare
param_grid = {'n_neighbors': range(1, 31, 2)}

# Crea un oggetto GridSearchCV con il modello di KNN e i valori di k da testare
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Addestra il modello utilizzando GridSearchCV
grid.fit(X_train, y_train)

best_k=grid.best_params_['n_neighbors']

# Stampa il miglior valore di k e lo score corrispondente
print("Miglior valore di k: ", best_k)

# Crea un'istanza del modello KNN con il miglior k
knn = KNeighborsClassifier(n_neighbors=best_k) # Numero di vicini da considerare

# Addestra il modello sull'intero dataset
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