import time

import pandas as pd
from pyswip import Prolog


"""
def assert_clauses_facts(kb):
    with open("clauses.pl", "r") as loc_file:
        lines = loc_file.readlines()
        for line in lines:
            kb.assertz(line)
"""

def create_kb() -> Prolog:
    prolog = Prolog()

    prolog.consult("data/facts.pl")
    #assert_clauses_facts(kb=prolog)
    kb=prolog

    # conteggi geografici
    prolog.assertz("accidents_same_borough(accident(ID1), accident(ID2)) :- borough(accident(ID1), Borough), borough(accident(ID2), Borough), ID1 \= ID2")
    prolog.assertz("accidents_same_street(accident(ID1), accident(ID2)) :- street_name(accident(ID1), Street), street_name(accident(ID2), Street), ID1 \= ID2")

    prolog.assertz("num_of_accidents_in_borough(accident(ID), Count) :- findall(ID1, accidents_same_borough(accident(ID), accident(ID1)), L), length(L, Count)")
    prolog.assertz("num_of_accidents_on_street(accident(ID), Count) :- street_name(accident(ID), OnStreet), (OnStreet = 'unknown' -> Count = 'null' ; OnStreet \\= 'unknown', findall(ID1, accidents_same_street(accident(ID), accident(ID1)), L), length(L, Count))")

    # conteggi sulla data/tempo
    prolog.assertz("time_of_day(accident(ID), TimeOfDay) :- hour(accident(ID), Hour), (Hour >= 6, Hour < 12, TimeOfDay = 'mattina'; Hour >= 12, Hour < 18, TimeOfDay = 'pomeriggio'; Hour >= 18, Hour < 24, TimeOfDay = 'sera'; Hour >= 0, Hour < 6, TimeOfDay = 'notte')")
    
    #gravità (0 feriti e 0 morti = lieve, 1/+ feriti e 0 morti = moderato, 0/+ feriti e 1/+ morti = grave)
    #prolog.assertz("severity(accident(ID), Severity) :- num_injured(accident(ID), NumInjured), num_killed(accident(ID), NumKilled), (NumInjured = 0, NumKilled = 0, Severity = 'lieve'); (NumInjured > 1, NumKilled = 0, Severity = 'moderato'); (NumInjured >= 0, NumKilled > 0, Severity = 'grave')")
    #prolog.assertz("is_fatal(accident(ID)) :- severity(accident(ID), 'grave')")

    return prolog


# suppongo che ci sia già
def calculate_features(kb, accident_id, final=False) -> dict:
    features_dict = {}

    features_dict["COLLISION_ID"] = accident_id

    accident_id = f"accident({accident_id})"
    
    features_dict["NUM_ACCIDENTS_BOROUGH"] = list(kb.query(f"num_of_accidents_in_borough({accident_id}, Count)"))[0]["Count"]
    features_dict["NUM_ACCIDENTS_ON_STREET"] = list(kb.query(f"num_of_accidents_on_street({accident_id}, Count)"))[0]["Count"]
    features_dict["TIME_OF_DAY"] = list(kb.query(f"time_of_day({accident_id}, TimeOfDay)"))[0]["TimeOfDay"]
    #features_dict["SEVERITY"] = list(kb.query(f"severity({accident_id}, Severity)"))[0]["Severity"]

    #features_dict["IS_FATAL"] = query_boolean_result(kb, f"is_fatal({accident_id})")

    """
    if final:
        # added after Naive Bayes Categorical results
        for value in ["vehicle", "private_vehicle", "public_vehicle", "public_place", "parking", "store_pub", "gas_station",
                      "park", "outside", "street", "sidewalk", "alley", "residential", "apartment", "house", "residence",
                      "residential_outside"]:
            features_dict[f"LOCATION_{value}"] = query_boolean_result(kb, f"location_{value}({crime_id})")

    """
    return features_dict


def query_boolean_result(kb, query_str: str):
    return min(len(list(kb.query(query_str))), 1)


def produce_working_dataset(kb: Prolog, path: str, final=False):
    print(f"Producing dataset at {path}")
    start = time.time()
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
    end = time.time()
    print("Total time: ", end-start)


knowledge_base = create_kb()
produce_working_dataset(knowledge_base, "working_dataset.csv")
#produce_working_dataset(knowledge_base, "working_dataset_final.csv", final=True)
