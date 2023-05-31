import pandas as pd

def load_data_in_kb(dataset: pd.DataFrame, kb=None):

    prolog_file = None

    if kb is None:
        prolog_file = open("informazioni.pl", "w")
        action = lambda fact_list: assert_all_in_file(fact_list, prolog_file)
    else:
        action = lambda fact_list: assert_all(fact_list, kb)

    action([":-style_check(-discontiguous)"])

    #Inserimento dati per gli Incidenti
    for index, row in dataset.iterrows():
        case_num = f"crime({row['CASE_NUMBER']})"
        facts = [f"location_description({case_num}, location({row['Location Description']}))",
                 f"beat({case_num},{row['Beat']})",
                 f"district({case_num},{row['District']})",
                 f"comm_area({case_num},{row['Community Area']})",
                 f"ward({case_num},{row['Ward']})",
                 f"crime_date({case_num}, {datetime_to_prolog_fact(row['Date'])})",
                 f"block({case_num}, {'block_' + row['Block']})"]  # due to initial number

        action(facts)

    # insert data for arrests
    for index, row in arrest_df.iterrows():
        arrest_num = f"arrest({row['ARREST_NUMBER']})"
        facts = [f"has_arrest(crime({row['CASE_NUMBER']}), {arrest_num})",
                 f"arrest_date({arrest_num}, {datetime_to_prolog_fact(row['ARREST DATE'])})",
                 f"criminal_race({arrest_num},{row['criminal_race']})"]

        num_charges = 0
        for i in range(1, 5):
            if not pd.isnull(row[f"CHARGE {i} STATUTE"]):
                num_charges += 1
            else:
                break
        # note: num charges is always >= 1
        facts.append(f"num_of_charges({arrest_num}, {num_charges})")

        action(facts)

    # insert data for shoot
    # Add info about gunshot injury

    for index, row in shoot_df.iterrows():
        victim_code = f"victim({row['VICTIM_CODE']})"
        facts = [f"victimization(crime({row['CASE_NUMBER']}), {victim_code}, {row['VICTIMIZATION']})",
                 f"date_shoot({victim_code}, {datetime_to_prolog_fact(row['DATE_SHOOT'])})",
                 f"victim_race({victim_code},{row['victim_race']})",
                 f"victim_sex({victim_code}, {row['SEX']})",
                 f"incident({victim_code}, {row['INCIDENT']})",
                 f"zip_code({victim_code}, {row['ZIP_CODE']})",
                 f"victim_area({victim_code}, {row['AREA']})",
                 f"victim_day_of_week({victim_code}, {row['DAY_OF_WEEK']})",
                 f"state_house_district({victim_code}, {row['STATE_HOUSE_DISTRICT']})",
                 f"state_senate_district({victim_code}, {row['STATE_SENATE_DISTRICT']})"]

        # street outreach
        if row['STREET_OUTREACH_ORGANIZATION'] != 'none':
            facts.append(f"street_org({victim_code}, {row['STREET_OUTREACH_ORGANIZATION']})")

        if not pd.isnull(row['AGE']):
            facts.append(f"victim_age({victim_code}, {row['AGE']})")

        action(facts)

    # insert data for health

    for index, row in health_df.iterrows():
        comm_area_code = float(row["Community Area"])
        facts = [f"comm_birth_rate({comm_area_code}, {row['Birth Rate']})",
                 f"comm_assault_homicide({comm_area_code}, {row['Assault (Homicide)']})",
                 f"comm_firearm({comm_area_code}, {row['Firearm-related']})",
                 f"comm_poverty_level({comm_area_code}, {row['Below Poverty Level']})",
                 f"comm_hs_diploma({comm_area_code}, {row['No High School Diploma']})",
                 f"comm_income({comm_area_code}, {row['Per Capita Income']})",
                 f"comm_unemployment({comm_area_code}, {row['Unemployment']})"]

        action(facts)

    if kb is not None:
        prolog_file.close()


def assert_all(facts, kb):
    for fact in facts:
        kb.asserta(fact)


def assert_all_in_file(facts, kb_file):
    kb_file.writelines(".\n".join(facts) + ".\n")


def create_prolog_kb():
    crimes_df = pd.read_csv(CLEAN_CRIME_PATH)
    arrest_df = pd.read_csv(CLEAN_ARREST_PATH)
    shoot_df = pd.read_csv(CLEAN_SHOOT_PATH)
    health_df = pd.read_csv(HEALTH_DATASET_PATH)

    load_data_in_kb(crimes_df=crimes_df, arrest_df=arrest_df, shoot_df=shoot_df, health_df=health_df)


def datetime_to_prolog_fact(datetime_str: str) -> str:
    dt = date_time_from_dataset(datetime_str)
    datetime_str = "date({}, {}, {}, {}, {}, {})".format(dt.year, dt.month, dt.day,
                                                         dt.hour, dt.minute, dt.second)
    return f"datime({datetime_str})"


def date_time_from_dataset(datetime_str: str) -> datetime:
    return datetime.strptime(datetime_str, '%m/%d/%Y %I:%M:%S %p')


def main():
    crime_codes = extract_crime_codes()
    clean_crime_dataset: pd.DataFrame = preprocess_crimes_dataset(extract_crime_dataset(crime_codes))
    clean_crime_dataset.to_csv(CLEAN_CRIME_PATH, index=False)

    clean_arrest_dataset: pd.DataFrame = preprocess_arrest_dataset(extract_arrest_dataset(crime_codes))
    clean_arrest_dataset.to_csv(CLEAN_ARREST_PATH, index=False)

    clean_shoot_dataset: pd.DataFrame = preprocess_shoot_dataset(extract_shoot_dataset(crime_codes))
    clean_shoot_dataset.to_csv(CLEAN_SHOOT_PATH, index=False)


main()
create_prolog_kb()
