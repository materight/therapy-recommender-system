import numpy as np
import pandas as pd
import requests, json
import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from faker import Faker
from faker.providers import DynamicProvider

def fetch_conditions():
    conditions = None
    URL = 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z'
    r = requests.get(URL)
    if r.ok:
        html = BeautifulSoup(r.text, 'html.parser')
        raw_ls = html.select('main div.row a h2')
        conditions = [c.text.strip().lower() for c in raw_ls]
    else:
        raise RuntimeError(f'Error while fetching list of conditions: {r.status_code}')
    return conditions


def fetch_therapies():
    therapies = None
    URL = 'https://en.wikipedia.org/wiki/List_of_therapies'
    r = requests.get(URL)
    if r.ok:
        html = BeautifulSoup(r.text, 'html.parser')
        raw_ls = html.select_one('#bodyContent ul').select('li')
        therapies = [t.text.split('(')[0].strip().lower() for t in raw_ls]
    else:
        raise RuntimeError(f'Error while fetching list of therapies: {r.status_code}')
    return therapies


def gen_conditions(conditions):
    ls = []
    for i, name in enumerate(tqdm(conditions, desc='Generating conditions')):
        ls.append({'id': f'Cond{i}', 'name': name, 'type': name})
    return ls


def gen_therapies(therapies):
    ls = []
    for i, name in enumerate(tqdm(therapies, desc='Generating therapies')):
        ls.append({'id': f'Th{i}', 'name': name, 'type': name})
    return ls


def gen_patients(faker, conditions, therapies, how_many_patients, conditions_per_patient_range, trials_per_condition_range):
    ls = []
    patient_condition_id, trials_id = 0, 0
    for patient_id in tqdm(range(how_many_patients), desc='Generating patients'):
        name = faker.name()
        patient_conditions, patient_trials = [], []
        # Generate conditions of patient
        for _ in range(np.random.randint(*conditions_per_patient_range)): # How many conditions per patient
            diagnosed = faker.date_between(start_date=datetime.date(2019,1,1), end_date=datetime.date(2021,12,1))
            cured = faker.date_between(start_date=diagnosed, end_date=datetime.date(2021,12,1))
            condition = np.random.choice(conditions)
            patient_conditions.append({'id': f'pc{patient_condition_id}', 'diagnosed': diagnosed.strftime('%Y%m%d'), 'cured': cured.strftime('%Y%m%d'), 'kind': condition['id']})
            patient_condition_id += 1
            # Generate trials of condition
            for _ in range(np.random.randint(*trials_per_condition_range)): # How many trials per condition 
                start = faker.date_between(start_date=diagnosed, end_date=cured)
                end = faker.date_between(start_date=start, end_date=cured)
                therapy = np.random.choice(therapies)
                successfull = np.random.randint(0, 100)
                patient_trials.append({'id': f'tr{trials_id}', 'start': start.strftime('%Y%m%d'), 'end': end.strftime('%Y%m%d'), 'condition': condition['id'], 'therapy': therapy['id'], 'successfull': f'{successfull}'})
                trials_id += 1
        ls.append({'id': patient_id, 'name': name, 'conditions': patient_conditions, 'trials': patient_trials})
    return ls


if __name__ == '__main__':
    # Initialize fake data generation
    faker = Faker()
    faker.seed_instance(0)
    np.random.seed(0)
    conditions_names = fetch_conditions()
    therapies_names = fetch_therapies()

    # Generate main tables
    conditions = gen_conditions(conditions_names)
    therapies = gen_conditions(therapies_names)
    patients = gen_patients(faker, conditions, therapies, how_many_patients=100, conditions_per_patient_range=(3,10), trials_per_condition_range=(0,5))
    
    # Save to json file
    data = {'Conditions': conditions, 'Therapies': therapies, 'Patients': patients}
    with open('data/generated.json', 'w') as f:
        json.dump(data, f)