from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder

# Carica i dati sui due dataset e utilizza il dataset su cui è stato effettuato il merge
df1 = pd.read_csv("data/Selected Accidents.csv")
df2 = pd.read_csv("kb/generated_dataset.csv")
data = pd.merge(df1, df2, on="COLLISION_ID")

# Selezionare le feature e la variabile target
categorical_features = ["TEMPERATURE", "RAIN_INTENSITY", "WIND_INTENSITY"]
numeric_features = ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED", "CLOUDCOVER", "AVERAGE_VOLUME"]

X = data[categorical_features + numeric_features]

# Encoding delle variabili categoriche
encoder = OrdinalEncoder()
X.loc[:, categorical_features] = encoder.fit_transform(X[categorical_features])

# Calcola l'inerzia (somma dei quadrati delle distanze tra ogni osservazione e il centroide del suo cluster) per diversi valori di k
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0,  n_init=10).fit(X)
    inertias.append(kmeans.inertia_)
    """"
    print("con K = ", k)
    # Ottieni le etichette di cluster assegnate a ciascun esempio
    cluster_labels = kmeans.labels_

    # Calcola il numero di esempi in ogni cluster
    cluster_counts = np.bincount(cluster_labels)

    # Stampa il numero di esempi in ogni cluster
    for cluster, count in enumerate(cluster_counts):
        print(f"Cluster {cluster}: {count} esempi")
    """

# Traccia la curva di elbow
plt.plot(range(1, 11), inertias)
plt.title('Curva di elbow')
plt.xlabel('Numero di cluster')
plt.ylabel('Inerzia')
path = "images/elbow.png" 
plt.savefig(path)

# Esegui il clustering con l'algoritmo k-means
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)

kmeans.fit(X)

# Aggiungi i cluster al dataset
data["Cluster"] = kmeans.predict(X)

# Seleziona le feature per il grafico scatter
feature1 = "LONGITUDE"
feature2 = "LATITUDE"

#il grafico scatter mostra eventuali aree della città dove si verificano incidenti stradali con caratteristiche simili, 
# come ad esempio un alto numero di persone ferite o uccise.

# Crea il grafico scatter colorato in base all'etichetta di cluster
cmap = matplotlib.cm.get_cmap('viridis', len(data['Cluster'].unique()))
plt.title('Clustering')
plt.scatter(data[feature1], data[feature2], c=data['Cluster'], cmap=cmap)
plt.xlabel(feature1)
plt.ylabel(feature2)
# Imposta gli intervalli sull'asse x e sull'asse y
plt.xlim(-74.3, -73.6)
plt.ylim(40.4, 40.99)


# Aggiungi la legenda dei cluster
cluster_labels = sorted(data['Cluster'].unique())
for label in cluster_labels:
    plt.scatter([], [], color=cmap(label), alpha=0.5, label='Cluster {}'.format(label))
plt.legend(title="Cluster", loc="upper right", markerscale=1, fontsize=10)

path = "images/clustering.png" 
plt.savefig(path)

# Calcola l'indice di silhouette (valutazione) per il clustering
silhouette_avg = silhouette_score(X, kmeans.labels_)

# Stampa l'indice di silhouette medio
print("Indice di silhouette medio:", silhouette_avg)