from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OrdinalEncoder

def naiveBayesCategorical(data, categorical_features, target):
    X = data[categorical_features]
    y = data[target]

    # Encoding delle variabili categoriche
    encoder = OrdinalEncoder()
    X = encoder.fit_transform(X)

    # Divisione del dataset in training set e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Creazione e addestramento del modello Naive Bayes per dati categorici
    clf = CategoricalNB()
    clf.fit(X_train, y_train)

    # Valutazione del modello sui dati di test
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    
    print(f"Accuracy: {train_acc}")
    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1-score: {f1}")

    return clf