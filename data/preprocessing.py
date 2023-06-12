from datetime import datetime
import pandas as pd
from shapely.wkt import loads
from knn import borough_prediction

#Estrazione delle informazioni necessarie del file 'NYC weather'
def extract_weather():
    # Caricamento del dataset
    weather = pd.read_csv("data/old dataset/NYC Weather.csv")
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

    weather['Y'] = weather['Y'].astype(int)
    weather['M'] = weather['M'].astype(int)
    weather['D'] = weather['D'].astype(int)
    weather['HH'] = weather['HH'].astype(int)

    # riordina le colonne in base all'ordine desiderato
    weather = weather.reindex(columns=['Y', 'M', 'D', 'HH'] + list(weather.columns[:-4]))

    # seleziona solo le righe con l'anno 2020 
    weather = weather.loc[(weather['Y'] == 2019) | (weather['Y'] == 2020)]

    # rimuovi le tre colonne che riguardano il cloudcover -> è utile solo cloudcover_low
    weather = weather.drop(['cloudcover (%)','cloudcover_mid (%)','cloudcover_high (%)'], axis=1)



    # visualizza il dataframe risultante
    weather.to_csv("data/working_dataset/New NYC Weather.csv", index=False, mode='w')




#Estrazione delle informazioni necessarie del file 'NYC Accidents'
def extract_accidents():
    # Caricamento del dataset
    accidents = pd.read_csv("data/old dataset/NYC Accidents.csv")
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
    
    # rimuovi le colonne non necessarie
    accidents = accidents.drop(['ZIP CODE','LOCATION', 'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED','CONTRIBUTING FACTOR VEHICLE 1','CONTRIBUTING FACTOR VEHICLE 2','CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','VEHICLE TYPE CODE 1','VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5'], axis=1)

    column_order = ['COLLISION_ID', 'Y', 'M', 'D', 'HH', 'MM', 'BOROUGH', 'LATITUDE', 'LONGITUDE', 'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME', 'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED']
    accidents = accidents.reindex(columns=column_order)

    # rimuove le righe con valori mancanti nelle colonne "LATITUDE" e "LONGITUDE"
    accidents = accidents.dropna(subset=['LATITUDE', 'LONGITUDE'])

    # visualizza il dataframe risultante
    accidents.to_csv("data/working_dataset/New NYC Accidents.csv", index=False, mode='w')


#Estrazione delle informazioni necessarie del file 'NYC Accidents'
def extract_traffic():
    # Caricamento del dataset
    traffic = pd.read_csv("data/old dataset/NYC Traffic.csv")

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

    # Aggiungi una colonna ID con valori sequenziali
    traffic.insert(0, 'TRAFFIC ID', range(1, len(traffic)+1))
    
    # riordina le colonne in base all'ordine desiderato
    traffic = traffic.reindex(columns=['TRAFFIC ID','BOROUGH','Y','M','D', 'HH', 'MM', 'VOL','STREET NAME'])

    # Trasforma i valori  in maiuscolo
    traffic = traffic.apply(lambda x: x.str.upper() if x.dtype == "object" else x)


    # visualizza il dataframe risultante
    traffic.to_csv("data/working_dataset/New NYC Traffic.csv", index=False, mode='w')


def union_dataset():

    incidenti = pd.read_csv("data/working_dataset/Complete Accidents.csv")
    meteo = pd.read_csv("data/working_dataset/New NYC Weather.csv")
    traffico = pd.read_csv("data/working_dataset/New NYC Traffic.csv")

    incidenti['MM'] = incidenti['MM'].astype(int)

    trasformazione = lambda x: '00' if x<8 else ('15' if x<24 else ('30' if x<38 else '45'))
    incidenti['MM'] = incidenti['MM'].apply(trasformazione)

    incidenti['MM'] = incidenti['MM'].astype(int)


    incidenti_meteo = pd.merge(incidenti, meteo, on=['Y','M','D','HH'], how='inner')

    #rinomina delle colonne del dataset
    incidenti_meteo = incidenti_meteo.rename(columns={'ON STREET NAME':'STREET NAME'})

    merged1 = pd.merge(incidenti_meteo, traffico, on=['Y','M','D', 'HH', 'MM', 'BOROUGH','STREET NAME'], how='inner')

    #rinomina delle colonne del dataset
    traffico = traffico.rename(columns={'STREET NAME':'CROSS STREET NAME'})

    merged2 = pd.merge(incidenti_meteo, traffico, on=['Y','M','D', 'HH', 'MM', 'BOROUGH', 'CROSS STREET NAME'], how='inner')

    #rinomina delle colonne del dataset
    traffico = traffico.rename(columns={'CROSS STREET NAME':'OFF STREET NAME'})

    merged3 = pd.merge(incidenti_meteo, traffico, on=['Y','M','D', 'HH', 'MM', 'BOROUGH', 'OFF STREET NAME'], how='inner')

    # concatena i due DataFrame lungo l'asse delle righe
    concatenaz = pd.concat([merged1, merged2, merged3], axis=0)

    # identifica le righe duplicate escludendo la colonna TRAFFICO
    duplicated_rows = concatenaz.duplicated(subset=['Y','M','D','HH','MM', 'BOROUGH','STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME'], keep='first')

    # crea un DataFrame con le righe duplicate e le rispettive occorrenze di TRAFFICO
    duplicated_df = concatenaz[duplicated_rows]

    # elimina le righe duplicate, tranne la prima occorrenza
    concatenaz.drop_duplicates(subset=['Y','M','D','HH','MM', 'BOROUGH','STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME'], keep='first', inplace=True)

    # aggiorno dataset di traffico solo con TRAFFIC ID e STREET NAME
    traffico = traffico.drop(['BOROUGH','Y','M','D','HH','MM','VOL'],axis=1)
    
    traffico = traffico.rename(columns={'OFF STREET NAME':'TRAFFIC STREET'})

    final = pd.merge(concatenaz, traffico, on='TRAFFIC ID', how='inner')

    selected_weather = final.loc[:, ['Y', 'M', 'D', 'HH', 'temperature_2m (°C)','precipitation (mm)','rain (mm)','cloudcover_low (%)','windspeed_10m (km/h)','winddirection_10m (°)']]
    
    selected_accidents = final.loc[:,['COLLISION_ID','Y', 'M', 'D', 'HH','MM','BOROUGH','LATITUDE','LONGITUDE','TRAFFIC STREET','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']]

    selected_traffic = final.loc[:,['TRAFFIC ID','Y', 'M', 'D', 'HH','MM','BOROUGH','VOL','TRAFFIC STREET']]

    selected_accidents.fillna('unknown', inplace=True)

    selected_weather = selected_weather.drop_duplicates()
    selected_traffic = selected_traffic.drop_duplicates()

    # visualizza i dataframe risultanti
    selected_weather.to_csv("data/Selected Weather.csv", index=False, mode='w')
    selected_accidents.to_csv("data/Selected Accidents.csv", index=False, mode='w')
    selected_traffic.to_csv("data/Selected Traffic.csv", index=False, mode='w')



def day_of_week():

    # Carica il tuo dataset CSV in un dataframe Pandas
    df = pd.read_csv("data/Selected Accidents.csv")  # Aggiorna il percorso al tuo dataset CSV

    # Per ogni riga nel dataframe
    for index, row in df.iterrows():
        # Estrai le informazioni sulla data dalle colonne Y, M e D
        year, month, day = row["Y"], row["M"], row["D"]
        
        # Costruisci la stringa data nel formato "YYYY-MM-DD"
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        
        # Converti la stringa data in un oggetto datetime
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Ottieni il giorno della settimana dalla data
        day_of_week = date_obj.strftime("%A")
        
        # Aggiungi l'informazione del giorno della settimana al dataframe
        df.loc[index, "DAY OF WEEK"] = day_of_week

    # Salva il dataframe con l'informazione aggiuntiva sul giorno della settimana in un nuovo file CSV
    df.to_csv("data/Selected Accidents.csv", index=False, mode = 'w')  # Aggiorna il percorso al tuo nuovo file CSV

def main():
    extract_weather()
    extract_accidents()
    extract_traffic()
    borough_prediction()
    union_dataset()
    day_of_week()


main()