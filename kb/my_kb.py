import time
from load_kb import create_prolog_kb 
import pandas as pd
from pyswip import Prolog

#creazione delle clausole
def create_kb() -> Prolog:
    prolog = Prolog()

    prolog.consult("kb/facts.pl")
    kb=prolog

    # conteggi geografici
    prolog.assertz("accidents_same_borough(accident(ID1), accident(ID2)) :- borough(accident(ID1), Borough), borough(accident(ID2), Borough)")
    prolog.assertz("accidents_same_street(accident(ID1), accident(ID2)) :- street_name(accident(ID1), Street), street_name(accident(ID2), Street)")

    prolog.assertz("num_of_accidents_in_borough(accident(ID), Count) :- findall(ID1, accidents_same_borough(accident(ID), accident(ID1)), L), length(L, Count)")
    prolog.assertz("num_of_accidents_on_street(accident(ID), Count) :- street_name(accident(ID), OnStreet), borough(accident(ID), Borough), (OnStreet = 'unknown' -> Count = 'unknown' ; OnStreet \\= 'unknown', findall(StreetID, (street_name(accident(StreetID), OnStreet), borough(accident(StreetID), Borough)), StreetIDs), sort(StreetIDs, UniqueStreetIDs), length(UniqueStreetIDs, Count))")
   
    # conteggi sulla data/tempo
    prolog.assertz("time_of_day(accident(ID), TimeOfDay) :- hour(accident(ID), Hour), (Hour >= 4, Hour < 7, TimeOfDay = 'dawn'; Hour >= 7, Hour < 10, TimeOfDay = 'early morning'; Hour >= 10, Hour < 12, TimeOfDay = 'late morning'; Hour >= 12, Hour < 15, TimeOfDay = 'early afternoon'; Hour >= 15, Hour < 18, TimeOfDay = 'late afternoon'; Hour >= 18, Hour < 20, TimeOfDay = 'early evening'; Hour >= 20, Hour < 22, TimeOfDay = 'late evening'; Hour >= 22, Hour < 24, TimeOfDay = 'early night'; Hour >= 0, Hour < 4, TimeOfDay = 'late night')")
    
    #gravità (0 feriti e 0 morti = lieve, 1/+ feriti e 0 morti = moderato, 0/+ feriti e 1/+ morti = grave)
    prolog.assertz("severity(accident(ID), Severity) :- num_injured(accident(ID), NumInjured), num_killed(accident(ID), NumKilled), (NumInjured = 0, NumKilled = 0, Severity = 'minor'; NumInjured >= 1, NumKilled = 0, Severity = 'moderate'; NumInjured >= 0, NumKilled > 0, Severity = 'major')")
    
    
    #meteo
    prolog.assertz("temperature_classification(accident(ID), Temperature) :- has_Weather(accident(ID), Data), temperature(Data,Temp), (Temp < 10.0, Temperature = 'cold'; Temp > 26.0, Temperature = 'hot'; Temp >= 10.0, Temp =< 26.0, Temperature = 'mild')")
    prolog.assertz("rain_intensity(accident(ID), RainIntensity) :- has_Weather(accident(ID), Data), rain(Data,Rain), (Rain >= 0.1, Rain =< 4, RainIntensity = 'weak'; Rain > 4, Rain =< 6, RainIntensity = 'moderate'; Rain > 6, Rain =< 10, RainIntensity = 'heavy'; Rain > 10, RainIntensity = 'shower activity'; Rain = 0.0, RainIntensity = 'unknown')")
    prolog.assertz("cloudcover(accident(ID)) :- has_Weather(accident(ID), Data), cloudcover(Data,Cloud), Cloud > 70")
    prolog.assertz("wind_intensity(accident(ID), WindIntensity) :- has_Weather(accident(ID), Data), windspeed(Data,Wind), (Wind > 0, Wind =< 19, WindIntensity = 'weak'; Wind > 19, Wind =< 39, WindIntensity = 'moderate'; Wind > 39, Wind =< 59, WindIntensity = 'strong'; Wind > 59, Wind =< 74, WindIntensity = 'gale'; Wind > 74, Wind =< 89, WindIntensity = 'strong gale'; Wind >= 90, WindIntensity = 'storm'; Wind = 0.0, WindIntensity = 'unknown')")
    
    #traffico
    prolog.assertz("traffic_volume(accident(ID), Volume) :- has_Traffic(accident(ID), traffic(TrafficID)), volume(traffic(TrafficID), Vol), (Vol < 100, Volume = 'light'; Vol > 500, Volume = 'heavy'; Vol >= 100, Vol =< 500, Volume = 'medium')")
    prolog.assertz("volume_sum_same_location(accident(ID), TotalVolume) :- findall(Vol, (accidents_same_borough(accident(ID), accident(ID1)), accidents_same_street(accident(ID), accident(ID1)), borough(accident(ID1), Borough), street_name(accident(ID1), Street), has_Traffic(accident(ID1), traffic(TrafficID)), volume(traffic(TrafficID), Vol)), Volumes), sum_list(Volumes, TotalVolume)")
    prolog.assertz("average_volume_same_location(accident(ID), AvgVolume) :- num_of_accidents_on_street(accident(ID), NumAccidents), (NumAccidents = 'unknown' -> AvgVolume = 'unknown'; volume_sum_same_location(accident(ID), TotalVolume), AvgVolume is TotalVolume / NumAccidents)")

    #incidente senza feriti e morti
    prolog.assertz("is_not_dangerous(accident(ID)) :- severity(accident(ID), 'minor')")

    return prolog


#query sulla KB
def calculate_features(kb, accident_id, final=False) -> dict:
    features_dict = {}

    features_dict["COLLISION_ID"] = accident_id

    accident_id = f"accident({accident_id})"

    features_dict["NUM_ACCIDENTS_BOROUGH"] = list(kb.query(f"num_of_accidents_in_borough({accident_id}, Count)"))[0]["Count"]
    features_dict["NUM_ACCIDENTS_ON_STREET"] = list(kb.query(f"num_of_accidents_on_street({accident_id}, Count)"))[0]["Count"]
    features_dict["TIME_OF_DAY"] = list(kb.query(f"time_of_day({accident_id}, TimeOfDay)"))[0]["TimeOfDay"]
    features_dict["SEVERITY"] = list(kb.query(f"severity({accident_id}, Severity)"))[0]["Severity"]

    features_dict["TEMPERATURE"] = list(kb.query(f"temperature_classification({accident_id}, Temperature)"))[0]["Temperature"]
    features_dict["RAIN_INTENSITY"] = list(kb.query(f"rain_intensity({accident_id}, RainIntensity)"))[0]["RainIntensity"]
    features_dict["CLOUDCOVER"] = query_boolean_result(kb, f"cloudcover({accident_id})")
    features_dict["WIND_INTENSITY"] = list(kb.query(f"wind_intensity({accident_id}, WindIntensity)"))[0]["WindIntensity"]

    features_dict["TRAFFIC_VOLUME"] = list(kb.query(f"traffic_volume({accident_id}, Volume)"))[0]["Volume"]
    features_dict["AVERAGE_VOLUME"] = round(list(kb.query(f"average_volume_same_location({accident_id}, AvgVolume)"))[0]["AvgVolume"],2)

    features_dict["IS_NOT_DANGEROUS"] = query_boolean_result(kb, f"is_not_dangerous({accident_id})")

    return features_dict


def query_boolean_result(kb, query_str: str):
    return min(len(list(kb.query(query_str))), 1)

#creazione del dataset con la nuove feature
def produce_working_dataset(kb: Prolog, path: str, final=False):
    accidents_complete: pd.DataFrame = pd.read_csv("data/Selected Accidents.csv")

    extracted_values_df = None

    first = True
    for accident_id in accidents_complete["COLLISION_ID"]:

        features_dict = calculate_features(kb, accident_id, final)
        if first:
            extracted_values_df = pd.DataFrame([features_dict])
            first = False
        else:
            extracted_values_df = pd.concat([extracted_values_df, pd.DataFrame([features_dict])], ignore_index=True)

    extracted_values_df.to_csv(path, index=False, mode ="w")


def main():
    create_prolog_kb()
    knowledge_base = create_kb()
    produce_working_dataset(knowledge_base, "kb/generated_dataset.csv")
    print("Created generated_dataset")

main()