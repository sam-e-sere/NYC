from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve

def borough_prediction():
    # Carica il dataset
    dataset = pd.read_csv("data/working_dataset/New NYC Accidents.csv")

    # Separa i dati con BOROUGH not null e BOROUGH null
    dataset_train = dataset.dropna(subset=["BOROUGH"])  # Dati con BOROUGH not null
    dataset_missing = dataset[dataset["BOROUGH"].isnull()]  # Dati con BOROUGH null


    # Dividi il dataset con BOROUGH not null in set di addestramento e set di test
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_train[["LATITUDE", "LONGITUDE"]],
        dataset_train["BOROUGH"],
        test_size=0.2,
        random_state=42
    )

    # Definisci i valori di k da testare
    k_values = range(1, 31)

    # Crea un modello di KNN
    knn = KNeighborsClassifier()

    # Definisci i valori di k da testare
    param_grid = {'n_neighbors': k_values}

    # Crea un oggetto GridSearchCV con il modello di KNN e i valori di k da testare
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

    # Addestra il modello utilizzando GridSearchCV
    grid.fit(X_train, y_train)

    # Estrai i risultati della ricerca dei parametri
    results = grid.cv_results_

    # Calcola l'andamento dell'accuratezza al variare di k
    train_scores, test_scores = validation_curve(knn, X_train, y_train, param_name="n_neighbors", param_range=k_values, cv=5, scoring='accuracy')

    # Calcola l'accuratezza media e la deviazione standard dell'accuratezza per ogni valore di k
    mean_train_scores = np.mean(train_scores, axis=1)
    std_train_scores = np.std(train_scores, axis=1)
    mean_test_scores = np.mean(test_scores, axis=1)
    std_test_scores = np.std(test_scores, axis=1)

    # Visualizza l'andamento dell'accuratezza al variare di k
    plt.plot(k_values, mean_train_scores, label='Training accuracy')
    plt.fill_between(k_values, mean_train_scores - std_train_scores, mean_train_scores + std_train_scores, alpha=0.1)
    plt.plot(k_values, mean_test_scores, label='Test accuracy')
    plt.fill_between(k_values, mean_test_scores - std_test_scores, mean_test_scores + std_test_scores, alpha=0.1)
    plt.title("Andamento dell'accuratezza al variare di k")
    plt.xlabel("k")
    plt.xticks(np.arange(min(k_values), max(k_values)+1, 2.0))  # Imposta gli intervalli dell'asse x
    plt.ylabel("Accuratezza")
    plt.legend()
    plt.show()

    best_k=grid.best_params_['n_neighbors']

    # Stampa il valore di k ottimale e l'accuratezza corrispondente
    print("Valore ottimale di k:", best_k)
    print("Accuratezza:", grid.best_score_)

    # Crea un'istanza del modello KNN
    knn = KNeighborsClassifier(n_neighbors=best_k)

    # Addestra il modello sul set di addestramento
    knn.fit(X_train, y_train)

    # Utilizza il modello per fare previsioni sul set di test
    y_pred = knn.predict(X_test)

    # Calcola l'accuratezza
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Utilizza il modello per fare previsioni sul dataset con BOROUGH null
    X_missing = dataset_missing[["LATITUDE", "LONGITUDE"]]
    y_pred_missing = knn.predict(X_missing)

    # Aggiungi i valori predetti al dataset originale
    dataset.loc[dataset["BOROUGH"].isnull(), "BOROUGH"] = y_pred_missing

    dataset.to_csv("data/working_dataset/Complete Accidents.csv", index=False, mode='w')