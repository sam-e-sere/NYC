import pandas as pd

#
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
    weather.drop('time', axis=1, inplace=True)

    # riordina le colonne in base all'ordine desiderato
    weather = weather.reindex(columns=['Y', 'M', 'D', 'TIME'] + list(weather.columns[:-4]))

    # seleziona solo le righe con l'anno 2020 
    weather_2020 = weather.loc[weather['Y'] == '2020']

    # visualizza il dataframe risultante
    weather_2020.to_csv("data/New NYC weather.csv", index=False, mode='w')



def main():
   extract_weather()

main()