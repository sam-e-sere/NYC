import datetime
import pandas as pd

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


    # rimuovi le tre colonne che riguardano il cloudcover -> è utile solo cloudcover_low
    #weather_2020 = weather_2020.drop(['cloudcover (%)','cloudcover_mid (%)','cloudcover_high (%)'], axis=1)

    # converte la colonna del tempo in formato datetime
    accidents['CRASH TIME'] = pd.to_datetime(accidents['CRASH TIME'], format='%H:%M:%S')

    # formatta la colonna del tempo nel nuovo formato
    accidents['CRASH TIME'] = accidents['CRASH TIME'].apply(lambda x: datetime.strftime(x, '%H:%M'))

    # visualizza il dataframe risultante
    accidents.to_csv("data/New NYC Accidents 2020.csv", index=False, mode='w')





def main():
   extract_accidents()

main()