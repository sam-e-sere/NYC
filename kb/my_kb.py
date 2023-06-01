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
    prolog.assertz("num_of_accidents_in_borough(accident(ID), Count) :- borough(accident(ID), Borough), findall(ID, borough(accident(ID), Borough), IDs), length(IDs, Count)")
    prolog.assertz("num_of_accidents_on_street(accident(ID), Count) :- street_name(accident(ID), Street), findall(ID, street_name(accident(ID), Street), IDs), length(IDs, Count)")
    prolog.assertz("num_of_accidents_on_cross_street(accident(ID), Count) :- cross_street_name(accident(ID), CrossStreet), findall(ID, cross_street_name(accident(ID), CrossStreet), IDs), length(IDs, Count)")
    prolog.assertz("num_of_accidents_on_off_street(accident(ID), Count) :- off_street_name(accident(ID), OffStreet), findall(ID, off_street_name(accident(ID), OffStreet), IDs), length(IDs, Count)")
    prolog.assertz("accidents_same_borough(accident(ID1), accident(ID2)) :- borough(accident(ID1), Borough), borough(accident(ID2), Borough), ID1 \= ID2")
    prolog.assertz("accidents_same_street(accident(ID1), accident(ID2)) :- street_name(accident(ID1), Street), street_name(accident(ID2), Street), ID1 \= ID2")
    prolog.assertz("accidents_same_cross_street(accident(ID1), accident(ID2)) :- cross_street_name(accident(ID1), CrossStreet), cross_street_name(accident(ID2), CrossStreet), ID1 \= ID2")
    prolog.assertz("accidents_same_off_street(accident(ID1), accident(ID2)) :- off_street_name(accident(ID1), OffStreet), off_street_name(accident(ID2), OffStreet), ID1 \= ID2")

    # conteggi sulla data/tempo
    prolog.assertz("time_of_day(accident(ID), TimeOfDay) :- hour(accident(ID), Hour), (Hour >= 6, Hour < 12, TimeOfDay = 'mattina'; Hour >= 12, Hour < 18, TimeOfDay = 'pomeriggio'; Hour >= 18, Hour < 24, TimeOfDay = 'sera'; Hour >= 0, Hour < 6, TimeOfDay = 'notte')")
    
    #gravità (0 feriti e 0 morti = lieve, 1/+ feriti e 0 morti = moderato, 0/+ feriti e 1/+ morti = grave)
    prolog.assertz("severity(accident(ID), Severity) :- num_injured(accident(ID), NumInjured), num_killed(accident(ID), NumKilled), (NumInjured = 0, NumKilled = 0, Severity = 'lieve'); (NumInjured > 1, NumKilled = 0, Severity = 'moderato'); (NumInjured >= 0, NumKilled > 0, Severity = 'grave')")
    #prolog.assertz("is_fatal(accident(ID)) :- severity(accident(ID), 'grave')")

    """
    
    prolog.assertz("num_of_crimes_in_zip_code(crime(C), N) :- "
                   "findall(C1, same_zip_code(crime(C), crime(C1)), L), length(L, N)")

    # CLAUSES ABOUT STREET ORGANIZATIONS from VICTIMIZATION
    prolog.assertz("street_organization(crime(C), O) :- "
                   "victimization(crime(C), victim(V), T), street_org(victim(V), O)")
    prolog.assertz("has_street_organization(crime(C)) :- street_organization(crime(C), O)")
    prolog.assertz("same_street_organization(crime(C1), crime(C2)) :- "
                   "street_organization(crime(C1), O), street_organization(crime(C2), O)")
    prolog.assertz("num_of_crimes_street_organization(crime(C), N) :- "
                   "findall(C1, same_street_organization(crime(C), crime(C1)), L), length(L, N)")

    # PROPERTIES OF COMMUNITY AREA
    prolog.assertz("crime_area_income(crime(C), I) :- comm_area(crime(C), COM), comm_income(COM, I)")
    prolog.assertz("crime_area_assault_homicide(crime(C), I) :- comm_area(crime(C), COM), comm_assault_homicide(COM, I)")
    prolog.assertz("crime_area_firearm(crime(C), I) :- comm_area(crime(C), COM), comm_firearm(COM, I)")
    prolog.assertz("crime_area_poverty_level(crime(C), I) :- comm_area(crime(C), COM), comm_poverty_level(COM, I)")
    prolog.assertz("crime_area_hs_diploma(crime(C), I) :- comm_area(crime(C), COM), comm_hs_diploma(COM, I)")
    prolog.assertz("crime_area_unemployment(crime(C), I) :- comm_area(crime(C), COM), comm_unemployment(COM, I)")
    prolog.assertz("crime_area_birth_rate(crime(C), I) :- comm_area(crime(C), COM), comm_birth_rate(COM, I)")

    # RACE AND SEX DATA
    prolog.assertz("crime_victim_sex(crime(C), S) :- "
                   "victimization(crime(C), victim(V), T), victim_sex(victim(V), S)")

    prolog.assertz("crime_victim_race(crime(C), VR) :- "
                   "victimization(crime(C), victim(V), T), victim_race(victim(V), VR)")
    prolog.assertz("crime_arrested_race(crime(C), PR) :- "
                   "has_arrest(crime(C), arrest(P)), criminal_race(arrest(P), PR) ")

    prolog.assertz("is_ratial(crime(C)) :- "
                   "crime_arrested_race(crime(C), PR), crime_victim_race(crime(C), VR), dif(VR, PR)")

    # Victim aver_age
    prolog.assertz("aver_age(crime(C), Avg) :- findall(A, (victimization(crime(C), victim(V), T), "
                   "victim_age(victim(V), A)), L), "
                   "sumlist(L, Sum), length(L, Length), Length > 0, Avg is Sum / Length")

    # features number of X
    prolog.assertz("num_of_victims(crime(C), N) :- findall(V, victimization(crime(C), victim(V), T), L), length(L, N)")
    prolog.assertz("num_of_dead(crime(C), N) :- "
                   "findall(V, victimization(crime(C), victim(V), homicide), L), length(L, N)")
    prolog.assertz("num_of_arrest(crime(C), N) :- findall(P, has_arrest(crime(C), arrest(P)), L), length(L, N)")
    prolog.assertz("is_homicide(crime(C)) :- victimization(crime(C), victim(V), homicide)")

    prolog.assertz("is_killed(victim(V)) :- victimization(crime(C), victim(V), homicide)")
    prolog.assertz("is_domestic(crime(C)) :- "
                   "location_description(crime(C), apartment); location_description(crime(C), house); "
                   "location_description(crime(C), residence); location_description(crime(C), driveway)")
    prolog.assertz("night_crime(crime(C)) :- crime_date(crime(C), datime(date(Y, M, D, H, M, S))), ((H >= 20; H =< 6))")
    prolog.assertz("crime_date_arrest(crime(C), D) :- has_arrest(crime(C), arrest(A)), arrest_date(arrest(A), D)")
    prolog.assertz("immediate_arrest(crime(C)) :- crime_date(crime(C), D), crime_date_arrest(crime(C), D)")
    prolog.assertz("same_month_arrest(crime(C)) :- crime_date(crime(C), datime(date(Y, M, D, H, M, S))), "
                   "crime_date_arrest(crime(C), datime(date(Y, M, D1, H1, M1, S1)))")

    prolog.assertz("is_there_a_child(crime(C), T) :- "
                   "victimization(crime(C), victim(V), T), victim_age(victim(V), A), A =< 15")
    prolog.assertz("is_killed_a_child(crime(C)) :- is_there_a_child(crime(C), homicide)")

    prolog.assertz("crime_by_group(crime(C)) :- num_of_arrest(crime(C), N), N >= 2")


    prolog.assertz("avg_num_charge(crime(C), Avg) :- "
                   "findall(NC, (has_arrest(crime(C), arrest(A)), num_of_charges(arrest(A), NC)), L), "
                   "sumlist(L, Sum), length(L, Length), Length > 0, Avg is Sum / Length")

    # added after Naive Bayes Categorical results

    # high abstracted categories
    prolog.assertz("is_vehicle(location(L)) :- is_private_vehicle(location(L)); is_public_vehicle(location(L))")
    prolog.assertz("is_public_place(location(L)) :- is_parking(location(L)); is_store_pub(location(L)); "
                   "is_gas_station(location(L)); is_park(location(L))")
    prolog.assertz("is_outside(location(L)) :- is_street(location(L)); is_sidewalk(location(L)); "
                   "is_alley(location(L))")
    prolog.assertz("is_residential(location(L)) :- is_apartment(location(L)); is_house(location(L)); "
                   "is_residence(location(L)); is_residential_outside(location(L))")

    for value in ["vehicle", "private_vehicle", "public_vehicle", "public_place", "parking", "store_pub", "gas_station",
                  "park", "outside", "street", "sidewalk", "alley", "residential", "apartment", "house", "residence",
                  "residential_outside"]:
        prolog.assertz(f"location_{value}(crime(C)) :- location_description(crime(C), location(L)), "
                       f"is_{value}(location(L))")

    """
    return prolog


# suppongo che ci sia già
def calculate_features(kb, accident_id, final=False) -> dict:
    features_dict = {}

    features_dict["COLLISION_ID"] = accident_id

    accident_id = f"accident({accident_id})"
    
    features_dict["NUM_ACCIDENTS_BOROUGH"] = list(kb.query(f"num_of_accidents_in_borough({accident_id}, Count)"))[0]["Count"]
    features_dict["NUM_ACCIDENTS_ON_STREET"] = list(kb.query(f"num_of_accidents_on_street({accident_id}, Count)"))[0]["Count"]
    features_dict["NUM_ACCIDENTS_CROSS_STREET"] = list(kb.query(f"num_of_accidents_on_cross_street({accident_id}, Count)"))[0]["Count"]
    features_dict["NUM_ACCIDENTS_ON_OFF_STREET"] = list(kb.query(f"num_of_accidents_on_off_street({accident_id}, Count)"))[0]["Count"]
    features_dict["TIME_OF_DAY"] = list(kb.query(f"time_of_day({accident_id}, TimeOfDay)"))[0]["TimeOfDay"]
    features_dict["SEVERITY"] = list(kb.query(f"severity({accident_id}, Severity)"))[0]["Severity"]

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

    extracted_values_df.to_csv(path, index=False)
    end = time.time()
    print("Total time: ", end-start)


knowledge_base = create_kb()
produce_working_dataset(knowledge_base, "working_dataset.csv")
#produce_working_dataset(knowledge_base, "working_dataset_final.csv", final=True)
