import os, argparse
import numpy as np
import requests, json
import datetime
from tqdm import tqdm
from bs4 import BeautifulSoup
from faker import Faker


def fetch_conditions():
    conditions = None
    URL = 'https://www.nhsinform.scot/illnesses-and-conditions/a-to-z'
    r = requests.get(URL)
    if r.ok:
        html = BeautifulSoup(r.text, 'html.parser')
        raw_ls = html.select('main div.row a.nhs-uk__az-link')
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


def fetch_drugs():
    drugs = None
    URL = 'https://en.wikipedia.org/wiki/WHO_Model_List_of_Essential_Medicines'
    r = requests.get(URL)
    if r.ok:
        html = BeautifulSoup(r.text, 'html.parser')
        raw_ls = html.select('div.mw-parser-output > ul li > a')
        drugs = [d.text.strip().lower() for d in raw_ls]
    else:
        raise RuntimeError(f'Error while fetching list of drugs: {r.status_code}')
    return drugs


def fetch_data(filename):
    os.makedirs(f'data/scraped', exist_ok=True)
    if not os.path.exists(f'data/scraped/{filename}.csv'):
        print(f'Fetching {filename}...')
        if filename == 'conditions':
            scraped = fetch_conditions()
        elif filename == 'therapies':
            scraped = fetch_therapies()
        elif filename == 'drugs':
            scraped = fetch_drugs()
        with open(f'data/scraped/{filename}.csv', 'wb') as f:
            for item in scraped: f.write(f'{item}\n'.encode('utf8'))
    else:
        scraped = []
        with open(f'data/scraped/{filename}.csv', 'r') as f:
            for item in f: scraped.append(item.rstrip())
    return scraped


def gen_conditions(conditions_names, n_conditions, no_additional_info):
    ls = []
    if n_conditions == np.inf: n_conditions = None 
    for i, name in enumerate(tqdm(conditions_names[:n_conditions], desc='Generating conditions')):
        c = {'id': f'Cond{i}', 'name': name, 'type': name}
        if not no_additional_info:
            c['symptom'] = np.random.choice(['constipation', 'dermatitis', 'diziness', 'drowsiness', 'fatigue', 'headache', 'insomnia', 'nausea', 'weight loss', 'dry mouth'])
            c['incidence'] = np.random.randint(0, 100)
            c['prognosis_prob_cured'] = np.random.randint(0, 100)
        ls.append(c)
    return ls


def gen_therapies(therapies_names, drugs_names, n_therapies, no_additional_info):
    ls = []
    if n_therapies == np.inf: n_therapies = None 
    for i, name in enumerate(tqdm(therapies_names[:n_therapies], desc='Generating therapies')):
        t = {'id': f'Th{i}', 'name': name, 'type': name}
        if not no_additional_info:
            has_medicine = np.random.choice([True, False], p=[0.8, 0.2])
            t['medicine'] = np.random.choice(drugs_names) if has_medicine else None
            t['daily_dosage'] = str(np.random.randint(1, 100) * 10) + 'mg' if has_medicine else None
            t['duration_days'] = np.random.randint(1, 500)
            t['efficacy'] = np.random.randint(0, 100)
        ls.append(t)
    return ls


def gen_patients(faker, conditions, therapies, n_patients, n_test_patients, conditions_per_patient_range, trials_per_condition_range, prob_cured_condition, no_additional_info, min_date=datetime.date(2019,1,1), max_date=datetime.date(2021,12,1)):
    patients, test_cases = [], []
    patient_condition_id, trials_id = 0, 0
    for patient_id in tqdm(range(n_patients + n_test_patients), desc='Generating patients'):
        name = faker.name()
        patient_conditions, patient_trials = [], []
        is_test_patient = patient_id >= n_patients
        # Generate conditions of patient
        n_conditions = np.random.randint(*conditions_per_patient_range) + (1 if is_test_patient else 0)
        for i in range(n_conditions): # How many conditions per patient
            is_test_condition = is_test_patient and i == n_conditions-1
            is_cured = faker.boolean(chance_of_getting_true=prob_cured_condition) if not is_test_condition else False
            diagnosed = faker.date_between(start_date=min_date, end_date=max_date)
            cured = faker.date_between(start_date=diagnosed, end_date=max_date) if is_cured else None
            condition = np.random.choice(conditions)
            patient_conditions.append({'id': f'pc{patient_condition_id}', 'diagnosed': diagnosed.strftime('%Y%m%d'), 'cured': cured.strftime('%Y%m%d') if is_cured else None, 'kind': condition['id']})
            if is_test_condition:
                test_cases.append((patient_id, f'pc{patient_condition_id}'))
            # Generate trials for the current condition
            for _ in range(np.random.randint(*trials_per_condition_range)): # How many trials per condition 
                start = faker.date_between(start_date=diagnosed, end_date=cured if is_cured else max_date)
                end = faker.date_between(start_date=start, end_date=cured if is_cured else max_date)
                therapy = np.random.choice(therapies)
                successfull = np.random.randint(0, 100)
                trial = {'id': f'tr{trials_id}', 'start': start.strftime('%Y%m%d'), 'end': end.strftime('%Y%m%d'), 'condition': f'pc{patient_condition_id}', 'therapy': therapy['id'], 'successfull': f'{successfull}'}
                if not no_additional_info:
                    trial['side_effect'] = np.random.choice(['constipation', 'dermatitis', 'diziness', 'drowsiness', 'fatigue', 'headache', 'insomnia', 'nausea', 'weight loss', 'dry mouth'])
                patient_trials.append(trial)
                trials_id += 1
            patient_condition_id += 1
        # Add generated patients to the list with additional information if needed
        p = {'id': str(patient_id), 'name': name}
        if not no_additional_info:
            p['gender'] = np.random.choice(['M', 'F'])
            p['age'] = np.random.randint(1, 100)
            p['nationality'] = faker.country()
            p['job'] = faker.job()
            p['blood_group'] = np.random.choice(['A', 'B', 'AB', 'O']) + np.random.choice(['+', '-'])
        p['conditions'] = patient_conditions
        p['trials'] = patient_trials
        patients.append(p)
    return patients, test_cases



if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset of patients with conditions.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    parser.add_argument('-o', '--out', dest='out_dir', type=str, default='./data/generated', help='path of the output file (default: %(default)s).', metavar='PATH')
    parser.add_argument('--n-patients', type=int, default=50000, help='number of patients to be generated (default: %(default)s).', metavar='N')
    parser.add_argument('--n-test', type=int, default=3, help='number of test patients with an uncured condition to be generated (default: %(default)s).', metavar='N')
    parser.add_argument('--n-conditions', type=int, default=np.inf, help='maximum number of conditions to be generated (default: %(default)s).', metavar='N')
    parser.add_argument('--n-therapies', type=int, default=np.inf, help='maximum number of therapies to be generated (default: %(default)s).', metavar='N')
    parser.add_argument('--conditions-per-patient', type=tuple, nargs=2, default=(1, 15), help='min and max number of conditions to be generated for each patient (default: %(default)s).', metavar=('MIN', 'MAX'))
    parser.add_argument('--trials-per-conditions', type=tuple, nargs=2, default=(0, 10), help='min and max number of trials to be generated for each condition of a patient (default: %(default)s).', metavar=('MIN', 'MAX'))
    parser.add_argument('--prob-cured', type=int, default=90, help='probability for a condition to be cured (default: %(default)s). Must be a value between 0 and 100.', metavar='P')
    parser.add_argument('--no-additional-info', action='store_true', help='if specified no additional information about patients, conditions and therapies will be added to the dataset.')
    parser.add_argument('--seed', type=int, default=0, help='random seed used in the generation (default: %(default)s).', metavar='S')
    args = parser.parse_args()
   
    # Initialize fake data generation
    faker = Faker()
    faker.seed_instance(args.seed)
    np.random.seed(args.seed)

    # Scrape data from the web
    conditions_names = fetch_data('conditions')
    therapies_names = fetch_data('therapies')
    drugs_names = fetch_data('drugs')

    # Generate main tables
    conditions = gen_conditions(conditions_names, args.n_conditions, args.no_additional_info)
    therapies = gen_therapies(therapies_names, drugs_names, args.n_therapies, args.no_additional_info)
    patients, test_cases = gen_patients(faker, conditions, therapies, n_patients=args.n_patients, n_test_patients=args.n_test, conditions_per_patient_range=args.conditions_per_patient, trials_per_condition_range=args.trials_per_conditions, prob_cured_condition=args.prob_cured, no_additional_info=args.no_additional_info)
    
    # Save to json file
    data = {'Conditions': conditions, 'Therapies': therapies, 'Patients': patients}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f'{args.out_dir}/dataset.json', 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    with open(f'{args.out_dir}/test.csv', 'w') as f:
        f.write('patient_id,condition_id\n')
        for patient_id, condition_id in test_cases: 
            f.write(f'{patient_id},{condition_id}\n')
    print(f'Generated dataset saved into {args.out_dir}.')