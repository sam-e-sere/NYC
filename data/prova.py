import pandas as pd
import opencage.geocoder

# carica il dataset con latitudine e longitudine
dataset = pd.read_csv('data/New NYC Accidents.csv')

# inizializza il geocoder con la tua chiave API
geocoder = opencage.geocoder.OpenCageGeocode("a034ec067938477594716cd36b1404b8")

# definisci una funzione per ottenere il nome del quartiere e il codice postale
def reverse_geocode(lat, lng):
    results = geocoder.reverse_geocode(lat, lng)
    components = results[0]['components']
    neighborhood = components.get('neighborhood', '')
    return neighborhood

dataset['borgo'] = ''

# applica la funzione al DataFrame per ottenere il nome del quartiere eil codice postale per ogni coppia di coordinate
dataset['borgo'] = dataset.apply(lambda row: pd.Series(reverse_geocode(row['LATITUDE'], row['LONGITUDE'])), axis=1)


# visualizza il dataframe risultante
dataset.to_csv("data/FINALE.csv", index=False, mode='w')