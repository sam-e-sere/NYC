import datetime
import pandas as pd

def load_data_in_kb(accidents: pd.DataFrame, traffic: pd.DataFrame, weather: pd.DataFrame, kb=None):

    prolog_file = None

    if kb is None:
        prolog_file = open("data/informazioni.pl", "w")
        action = lambda fact_list: assert_all_in_file(fact_list, prolog_file)
    else:
        action = lambda fact_list: assert_all(fact_list, kb)

    action([":-style_check(-discontiguous)"])

    #Inserimento dati per gli Incidenti
    for index, row in accidents.iterrows():
        collision_id = f"accident({row['COLLISION_ID']})"
        data = f"{row['Y']}-{row['M']}-{row['D']}T{row['HH']}:{row['MM']}:00"
        info = [f"year({collision_id}, {row['Y']})",
                 f"month({collision_id},{row['M']})",
                 f"day({collision_id},{row['D']})",
                 f"hour({collision_id},{row['HH']})",
                 f"minutes({collision_id},{row['MM']})",
                 f"accident-date({collision_id}, {datetime_to_prolog_fact(data)})",
                 f"borough({collision_id},{row['BOROUGH']})",
                 f"location({collision_id},{row['LATITUDE']}, {row['LONGITUDE']})",
                 f"street name({collision_id},{row['STREET NAME']})",
                 f"cross street name({collision_id},{row['CROSS STREET NAME']})",
                 f"off street name({collision_id},{row['OFF STREET NAME']})",
                 f"num_injured({collision_id},{row['NUMBER OF PERSON INJURED']})",
                 f"num_killed({collision_id},{row['NUMBER OF PERSON KILLED']})",]  # due to initial number

        action(info)
    
    #Inserimento dati per il Traffico
    for index, row in traffic.iterrows():
        traffic_id = f"traffic({row['TRAFFIC ID']})"
        data = f"{row['Y']}-{row['M']}-{row['D']}T{row['HH']}:{row['MM']}:00"
        info = [f"year({traffic_id}, {row['Y']})",
                f"month({traffic_id},{row['M']})",
                f"day({traffic_id},{row['D']})",
                f"hour({traffic_id},{row['HH']})",
                f"minutes({traffic_id},{row['MM']})",
                f"traffic-date({traffic_id}, {datetime_to_prolog_fact(data)})",
                f"borough({traffic_id},{row['BOROUGH']})",
                f"volume({traffic_id},{row['VOL']})",
                f"traffic street({traffic_id},{row['TRAFFIC STREET']})"]

        action(info)

    #Inserimento dati per il Meteo
    for index, row in weather.iterrows():
        data = f"{row['Y']}-{row['M']}-{row['D']}T{row['HH']}:00:00"
        info = [f"temperature({data}, {row['temperature_2m (°C)']})",
                f"precipitation({data},{row['precipitation (mm)']})",
                f"rain({data},{row['rain (mm)']})",
                f"cloudcover({data},{row['cloudcover_low (%)']})",
                f"windspeed({data},{row['windspeed_10m (km/h)']})",
                f"winddirection({data},{row['winddirection_10m (°)']})"]

        action(info)

    if kb is not None:
        prolog_file.close()


def assert_all(facts, kb):
    for fact in facts:
        kb.asserta(fact)


def assert_all_in_file(facts, kb_file):
    kb_file.writelines(".\n".join(facts) + ".\n")


def create_prolog_kb():
    accidents = pd.read_csv("data/Selected Accidents.csv")
    traffic = pd.read_csv("data/Selected Traffic.csv")
    weather = pd.read_csv("data/Selected Weather.csv")

    load_data_in_kb(accidents, traffic, weather)


def datetime_to_prolog_fact(datetime_str: str) -> str:
    dt = date_time_from_dataset(datetime_str)
    datetime_str = "date({}, {}, {}, {}, {}, {})".format(dt.year, dt.month, dt.day,
                                                         dt.hour, dt.minute, dt.second)
    return f"datime({datetime_str})"


def date_time_from_dataset(datetime_str: str) -> datetime:
    return datetime.datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S')


def main():
    create_prolog_kb()


main()

