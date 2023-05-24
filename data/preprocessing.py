import datetime
import pandas as pd
from shapely.wkt import loads
import pyproj

#Estrazione delle informazioni necessarie del file 'NYC weather'
def extract_weather():
    # Caricamento del dataset
    weather = pd.read_csv("data/NYC weather.csv")
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

    # riordina le colonne in base all'ordine desiderato
    weather = weather.reindex(columns=['Y', 'M', 'D', 'TIME'] + list(weather.columns[:-4]))

    # seleziona solo le righe con l'anno 2020 
    weather_2020 = weather.loc[weather['Y'] == '2020']

    # rimuovi le tre colonne che riguardano il cloudcover -> è utile solo cloudcover_low
    weather_2020 = weather_2020.drop(['cloudcover (%)','cloudcover_mid (%)','cloudcover_high (%)'], axis=1)

    # visualizza il dataframe risultante
    weather_2020.to_csv("data/New NYC weather.csv", index=False, mode='w')




#Estrazione delle informazioni necessarie del file 'NYC Accidents'
def extract_accidents():
    # Caricamento del dataset
    accidents = pd.read_csv("data/NYC Accidents 2020.csv")
    # crea le nuove colonne vuote
    accidents['Y'] = ''
    accidents['M'] = ''
    accidents['D'] = ''

    # itera su ogni riga del dataframe e separa la data
    for index, row in accidents.iterrows():
        year, month, day = row['CRASH DATE'].split('-')
        
        # aggiorna le nuove colonne con i valori corretti
        accidents.at[index, 'Y'] = year
        accidents.at[index, 'M'] = month
        accidents.at[index, 'D'] = day


    # rimuovi la colonna "time" originale
    accidents = accidents.drop('CRASH DATE', axis=1)

    # riordina le colonne in base all'ordine desiderato
    accidents = accidents.reindex(columns=['Y', 'M', 'D'] + list(accidents.columns[:-3]))

    # rimuovi gli ultimi tre caratteri (i secondi) dalla colonna 'time'
    accidents['CRASH TIME'] = accidents['CRASH TIME'].str.slice(stop=-3)

    
    # rimuovi le colonne non necessarie
    accidents = accidents.drop(['ZIP CODE','LOCATION', 'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED','CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2','CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','COLLISION_ID','VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5'], axis=1)

    # visualizza il dataframe risultante
    accidents.to_csv("data/New NYC Accidents 2020.csv", index=False, mode='w')


#Estrazione delle informazioni necessarie del file 'NYC Accidents'
def extract_traffic():
    # Caricamento del dataset
    traffic = pd.read_csv("data/NYC Traffic Volume Counts.csv")

    
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
    traffic['LONGITUDE'] = lon
    traffic['LATITUDE'] = lat

    # rimuovi le colonne non necessarie
    # accidents = accidents.drop(['ZIP CODE','LOCATION', 'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED','CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2','CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','COLLISION_ID','VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5'], axis=1)

    # visualizza il dataframe risultante
    traffic.to_csv("data/New NYC Traffic Volume.csv", index=False, mode='w')





def main():
    extract_traffic()

main()