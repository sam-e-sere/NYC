import pandas as pd
from shapely.wkt import loads
import pyproj

#Estrazione delle informazioni necessarie del file 'NYC weather'
def extract_weather():
    # Caricamento del dataset
    weather = pd.read_csv("data/NYC Weather.csv")
    # crea le nuove colonne vuote
    weather['Y'] = ''
    weather['M'] = ''
    weather['D'] = ''
    weather['TIME'] = ''

    # itera su ogni riga del dataframe e separa la colonna "time"
    for index, row in weather.iterrows():
        time = row['time'].split('T')
        date = time[0]
        time = time[1]
        year, month, day = date.split('-')
        
        # aggiorna le nuove colonne con i valori corretti
        weather.at[index, 'Y'] = year
        weather.at[index, 'M'] = month
        weather.at[index, 'D'] = day
        weather.at[index, 'TIME'] = time

    # rimuovi la colonna "time" originale
    weather = weather.drop('time', axis=1)

    # crea le nuove colonne vuote
    weather['HH'] = ''

    # itera su ogni riga del dataframe e separa il tempo
    for index, row in weather.iterrows():
        hour, minutes = row['TIME'].split(':')
        
        # aggiorna le nuove colonne con i valori corretti
        weather.at[index, 'HH'] = hour

    # rimuovi la colonna "time" originale
    weather = weather.drop('TIME', axis=1)

    # riordina le colonne in base all'ordine desiderato
    weather = weather.reindex(columns=['Y', 'M', 'D', 'HH'] + list(weather.columns[:-4]))

    # seleziona solo le righe con l'anno 2020 
    weather = weather.loc[(weather['Y'] == '2019') | (weather['Y'] == '2020')]

    # rimuovi le tre colonne che riguardano il cloudcover -> è utile solo cloudcover_low
    weather = weather.drop(['cloudcover (%)','cloudcover_mid (%)','cloudcover_high (%)'], axis=1)



    # visualizza il dataframe risultante
    weather.to_csv("data/New NYC Weather.csv", index=False, mode='w')




#Estrazione delle informazioni necessarie del file 'NYC Accidents'
def extract_accidents():
    # Caricamento del dataset
    accidents = pd.read_csv("data/NYC Accidents.csv")
    # crea le nuove colonne vuote
    accidents['Y'] = ''
    accidents['M'] = ''
    accidents['D'] = ''

    # itera su ogni riga del dataframe e separa la data
    for index, row in accidents.iterrows():
        month, day, year = row['CRASH DATE'].split('/')
        
        # aggiorna le nuove colonne con i valori corretti
        accidents.at[index, 'Y'] = year
        accidents.at[index, 'M'] = month
        accidents.at[index, 'D'] = day


    # rimuovi la colonna "time" originale
    accidents = accidents.drop('CRASH DATE', axis=1)

    # crea le nuove colonne vuote
    accidents['HH'] = ''
    accidents['MM'] = ''

    # itera su ogni riga del dataframe e separa la data
    for index, row in accidents.iterrows():
        hour, minutes = row['CRASH TIME'].split(':')
        
        # aggiorna le nuove colonne con i valori corretti
        accidents.at[index, 'HH'] = hour
        accidents.at[index, 'MM'] = minutes

    # rimuovi la colonna "time" originale
    accidents = accidents.drop('CRASH TIME', axis=1)

    # riordina le colonne in base all'ordine desiderato
    accidents = accidents.reindex(columns=['Y', 'M', 'D','HH','MM'] + list(accidents.columns[:-5]))
    
    # rimuovi le colonne non necessarie
    accidents = accidents.drop(['ZIP CODE','LOCATION', 'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED','CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2','CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','COLLISION_ID','VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5'], axis=1)

    accidents = accidents.apply(lambda x: x.str.upper() if x.dtype == "object" else x)


    # visualizza il dataframe risultante
    accidents.to_csv("data/New NYC Accidents.csv", index=False, mode='w')


#Estrazione delle informazioni necessarie del file 'NYC Accidents'
def extract_traffic():
    # Caricamento del dataset
    traffic = pd.read_csv("data/NYC Traffic.csv")

    """
    # estrai le coordinate x e y dalla colonna WktGeom e crea una serie geografica
    geometry = traffic['WktGeom'].apply(lambda x: loads(x))
    geo_series = pd.Series(geometry)

    # converte le coordinate geografiche in latitudine e longitudine
    traffic['X'] = geo_series.apply(lambda x: x.x)
    traffic['Y'] = geo_series.apply(lambda x: x.y)

    
    # imposta la proiezione cartografica utlizzata nel dataset
    crs = 'EPSG:2263'  #proiezione cartografica UTM

    # crea un oggetto Transformer per la conversione delle coordinate
    transformer = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

    # converte le coordinate in latitudine e longitudine
    lon, lat = transformer.transform(traffic['X'], traffic['Y'])

    # aggiungi le colonne di latitudine e longitudine al dataset
    traffic['LATITUDE'] = lat
    traffic['LONGITUDE'] = lon

    # rimuovi le colonne non necessarie
    traffic = traffic.drop(['WktGeom','X', 'Y'], axis=1)
    """

    # crea un nuovo DataFrame con la media dei valori di "Vol" per ogni gruppo di righe duplicate
    traffic = traffic.groupby(['Boro', 'Yr', 'M', 'D', 'HH', 'MM', 'street'], as_index=False)['Vol'].mean()

    #rinomina delle colonne del dataset
    traffic = traffic.rename(columns={'Boro':'BOROUGH', 'Yr':'Y','Vol':'VOL','street':'STREET NAME'})
    
    # riordina le colonne in base all'ordine desiderato
    traffic = traffic.reindex(columns=['BOROUGH','Y','M','D', 'HH', 'MM', 'VOL','STREET NAME'])

    # Trasforma i valori  in maiuscolo
    traffic = traffic.apply(lambda x: x.str.upper() if x.dtype == "object" else x)


    # visualizza il dataframe risultante
    traffic.to_csv("data/New NYC Traffic.csv", index=False, mode='w')


def union_dataset():

    incidenti = pd.read_csv("data/New NYC Accidents.csv")
    meteo = pd.read_csv("data/New NYC Weather.csv")

    incidenti['MM'] = incidenti['MM'].astype(int)

    trasformazione = lambda x: '00' if x<8 else ('15' if x<24 else ('30' if x<38 else '45'))
    incidenti['MM'] = incidenti['MM'].apply(trasformazione)


    incidente_meteo = pd.merge(incidenti, meteo, on=['Y','M','D','HH'], how='inner')

    # visualizza il dataframe risultante
    incidente_meteo.to_csv("data/Accidents Weather.csv", index=False, mode='w')
    

def traffico_incidenti():

    incidenti_meteo = pd.read_csv("data/Accidents Weather.csv")
    traffico = pd.read_csv("data/New NYC Traffic.csv")

    #rinomina delle colonne del dataset
    incidenti_meteo = incidenti_meteo.rename(columns={'ON STREET NAME':'STREET NAME'})

    merged1 = pd.merge(incidenti_meteo, traffico, on=['Y','M','D', 'HH', 'MM', 'STREET NAME'], how='inner')

    # visualizza il dataframe risultante
    merged1.to_csv("data/working_dataset/merge_on_street.csv", index=False, mode='w')

    #rinomina delle colonne del dataset
    traffico = traffico.rename(columns={'STREET NAME':'CROSS STREET NAME'})

    merged2 = pd.merge(incidenti_meteo, traffico, on=['Y','M','D', 'HH', 'MM', 'CROSS STREET NAME'], how='inner')

    # visualizza il dataframe risultante
    merged2.to_csv("data/working_dataset/merge_cross_street.csv", index=False, mode='w')

    #rinomina delle colonne del dataset
    traffico = traffico.rename(columns={'CROSS STREET NAME':'OFF STREET NAME'})

    merged3 = pd.merge(incidenti_meteo, traffico, on=['Y','M','D', 'HH', 'MM', 'OFF STREET NAME'], how='inner')

    # visualizza il dataframe risultante
    merged3.to_csv("data/working_dataset/merge_off_street.csv", index=False, mode='w')

    # concatena i due DataFrame lungo l'asse delle righe
    concatenaz = pd.concat([merged1, merged2, merged3], axis=0)

    # identifica le righe duplicate escludendo la colonna TRAFFICO
    duplicated_rows = concatenaz.duplicated(subset=['Y','M','D','HH','MM','STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME'], keep='first')

    # crea un DataFrame con le righe duplicate e le rispettive occorrenze di TRAFFICO
    duplicated_df = concatenaz[duplicated_rows]

    # elimina le righe duplicate, tranne la prima occorrenza
    concatenaz.drop_duplicates(subset=['Y','M','D','HH','MM','STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME'], keep='first', inplace=True)

    # visualizza il dataframe risultante
    concatenaz.to_csv("data/Final Dataset.csv", index=False, mode='w')



def main():
    extract_weather()
    extract_accidents()
    extract_traffic()
    union_dataset()
    traffico_incidenti()

main()