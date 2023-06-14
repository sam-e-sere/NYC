import datetime
import pandas as pd

def load_data_in_kb(accidents: pd.DataFrame, traffic: pd.DataFrame, weather: pd.DataFrame, kb=None):

    prolog_file = None

    if kb is None:
        prolog_file = open("kb/facts.pl", "w")
        action = lambda fact_list: assert_all_in_file(fact_list, prolog_file)
    else:
        action = lambda fact_list: assert_all(fact_list, kb)

    action([":-style_check(-discontiguous)"])

    #Inserimento dati per gli Incidenti
    for index, row in accidents.iterrows():
        collision_id = f"accident({row['COLLISION_ID']})"
        data = f"{row['Y']},{row['M']},{row['D']},{row['HH']},{row['MM']},0"
        info = [f"year({collision_id}, {row['Y']})",
                 f"month({collision_id},{row['M']})",
                 f"day({collision_id},{row['D']})",
                 f"hour({collision_id},{row['HH']})",
                 f"minutes({collision_id},{row['MM']})",
                 f"accident_date({collision_id}, {datetime_to_prolog_fact(data)})",
                 f"borough({collision_id},'{row['BOROUGH']}')",
                 f"location({collision_id},{row['LATITUDE']}, {row['LONGITUDE']})",
                 f"street_name({collision_id},'{row['TRAFFIC STREET']}')",
                 f"num_injured({collision_id},{row['NUMBER OF PERSONS INJURED']})",
                 f"num_killed({collision_id},{row['NUMBER OF PERSONS KILLED']})",
                 f"day_of_week({collision_id},'{row['DAY OF WEEK']}')"] 
                
        # individua il traffico corrispondente all'incidente
        traffic_row= traffic.loc[(traffic['Y'] == row['Y']) &
                            (traffic['M'] == row['M']) &
                            (traffic['D'] == row['D']) &
                            (traffic['HH'] == row['HH']) &
                            (traffic['MM'] == row['MM']) &
                            (traffic['BOROUGH'] == row['BOROUGH']) &
                            ((traffic['TRAFFIC STREET'] == row['TRAFFIC STREET']))]
    
        if not traffic_row.empty:
            traffic_id = f"traffic({traffic_row['TRAFFIC ID'].values[0]})"
            info.append(f"has_Traffic({collision_id}, {traffic_id})")
        
        # individua il meteo corrispondente all'incidente
        weather_row= weather.loc[(weather['Y'] == row['Y']) &
                            (weather['M'] == row['M']) &
                            (weather['D'] == row['D']) &
                            (weather['HH'] == row['HH'])]
    
        if not weather_row.empty:
            weather_date = f"date({row['Y']},{row['M']},{row['D']},{row['HH']},0,0)"
            info.append(f"has_Weather({collision_id}, {weather_date})")

        action(info)
    
    #Inserimento dati per il Traffico
    for index, row in traffic.iterrows():
        traffic_id = f"traffic({row['TRAFFIC ID']})"
        data = f"{row['Y']},{row['M']},{row['D']},{row['HH']},{row['MM']},0"
        info = [f"year({traffic_id}, {row['Y']})",
                f"month({traffic_id},{row['M']})",
                f"day({traffic_id},{row['D']})",
                f"hour({traffic_id},{row['HH']})",
                f"minutes({traffic_id},{row['MM']})",
                f"traffic_date({traffic_id}, {datetime_to_prolog_fact(data)})",
                f"borough({traffic_id},'{row['BOROUGH']}')",
                f"volume({traffic_id},{row['VOL']})",
                f"traffic_street({traffic_id},'{row['TRAFFIC STREET']}')"]

        action(info)

    #Inserimento dati per il Meteo
    for index, row in weather.iterrows():
        data = f"{int(row['Y'])},{int(row['M'])},{int(row['D'])},{int(row['HH'])},0,0"
        info = [f"temperature({datetime_to_prolog_fact(data)}, {row['temperature_2m (°C)']})",
                f"precipitation({datetime_to_prolog_fact(data)},{row['precipitation (mm)']})",
                f"rain({datetime_to_prolog_fact(data)},{row['rain (mm)']})",
                f"cloudcover({datetime_to_prolog_fact(data)},{row['cloudcover_low (%)']})",
                f"windspeed({datetime_to_prolog_fact(data)},{row['windspeed_10m (km/h)']})",
                f"winddirection({datetime_to_prolog_fact(data)},{row['winddirection_10m (°)']})"]

        action(info)

    if kb is not None:
        prolog_file.close()


def assert_all(info, kb):
    for fact in info:
        kb.asserta(fact)


def assert_all_in_file(info, kb_file):
    kb_file.writelines(".\n".join(info) + ".\n")


def create_prolog_kb():
    accidents = pd.read_csv("data/Selected Accidents.csv")
    traffic = pd.read_csv("data/Selected Traffic.csv")
    weather = pd.read_csv("data/Selected Weather.csv")

    load_data_in_kb(accidents, traffic, weather)


def datetime_to_prolog_fact(datetime_str: str) -> str:
    dt = date_time_from_dataset(datetime_str)
    datetime_str = "date({},{},{},{},{},{})".format(dt.year, dt.month, dt.day,
                                                         dt.hour, dt.minute, dt.second)
    return f"{datetime_str}"


def date_time_from_dataset(datetime_str: str) -> datetime:
    return datetime.datetime.strptime(datetime_str, '%Y,%m,%d,%H,%M,%S')

