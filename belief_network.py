from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Carica i dati sui due dataset e utilizza il dataset su cui è stato effettuato il merge
incidenti = pd.read_csv("data/Selected Accidents.csv")
generated = pd.read_csv("kb/generated_dataset.csv")
merge = pd.merge(incidenti, generated, on="COLLISION_ID")

categorical_features = ["BOROUGH", "TRAFFIC STREET", "TEMPERATURE", "RAIN_INTENSITY", "WIND_INTENSITY"]
boolean_features = ["CLOUDCOVER", "IS_NOT_DANGEROUS"]

X = merge[categorical_features + boolean_features]

# Encoding delle variabili categoriche utilizzando la codifica ordinale
encoder = OrdinalEncoder()
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_features]), columns=categorical_features)

# Combinazione delle variabili codificate e delle variabili booleane
X_encoded = pd.concat([X_encoded, X[boolean_features]], axis=1)

# Crea la rete bayesiana
model = BayesianModel()

model.add_nodes_from(X_encoded.columns)

# Aggiungi gli archi per modellare le dipendenze tra le variabili
model.add_edges_from([('BOROUGH', 'TRAFFIC STREET')])

# Definisci il numero di possibili valori per ogni variabile
variable_card = {'BOROUGH': 5, 'TRAFFIC STREET': 264, 'TEMPERATURE': 3, 'RAIN_INTENSITY': 5, 'WIND_INTENSITY': 3, 'CLOUDCOVER': 2, 'IS_NOT_DANGEROUS': 2}

# Definisci le CPD per ogni variabile
cpds = []

for variable in X_encoded.columns:
    parents = model.get_parents(variable)

    # Conta il numero di volte in cui ogni combinazione di valori si verifica
    counts = X_encoded.groupby(parents + [variable]).size().reset_index(name='counts')

    # Normalizza le frequenze relative per ottenere le probabilità condizionali
    cpd_values = counts.pivot_table(values='counts', index=parents, columns=variable, fill_value=0)
    cpd_values = cpd_values.div(cpd_values.sum(axis=1), axis=0)

    # Crea la CPD solo se ci sono combinazioni di valori nel dataset
    if not cpd_values.empty:
        # Crea la CPD utilizzando la classe TabularCPD di pgmpy
        cpd = TabularCPD(variable=variable, variable_card=variable_card[variable],
                        values=cpd_values.values.T.tolist(), evidence=parents,
                        evidence_card=[variable_card[p] for p in parents])
    else:
        # Crea una CPD vuota con probabilità zero
        cpd = TabularCPD(variable=variable, variable_card=variable_card[variable],
                        values=np.zeros(variable_card[variable]), evidence=parents,
                        evidence_card=[variable_card[p] for p in parents])
    cpds.append(cpd)

# Aggiungi le CPD alla rete bayesiana
for cpd in cpds:
    model.add_cpds(cpd)

if model.check_model():
    print("rete valida")
else:
    print("rete non valida")

# Effettua l'inferenza per calcolare la probabilità che l'evento "IS_NOT_DANGEROUS" si verifichi dati alcuni valori
infer = VariableElimination(model)

# Definisci le evidenze
evidence = {'BOROUGH': "BROOKLYN", 'TRAFFIC STREET': "RALPH AVENUE", 'TEMPERATURE':"hot", 'RAIN_INTENSITY':"weak", 'WIND_INTENSITY':"weak", 'CLOUDCOVER':1}

# Crea un DataFrame per l'evidenza
evidence_df = pd.DataFrame([evidence])

# Effettua l'encoding per le variabili categoriche eccetto CLOUDCOVER
encoded_evidence = pd.DataFrame(encoder.transform(evidence_df[categorical_features]), columns=categorical_features)

# Combina l'evidenza codificata e la variabile CLOUDCOVER non codificata
final_evidence = pd.concat([encoded_evidence, evidence_df['CLOUDCOVER']], axis=1)

# Effettua l'inferenza
prob = infer.query(['IS_NOT_DANGEROUS'], evidence=final_evidence.iloc[0])

# Stampa i risultati
print(prob)   

