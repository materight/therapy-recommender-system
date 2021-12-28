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


def gen_patients(faker, conditions, therapies, n_patients, n_test_patients, conditions_per_patient_range, trials_per_condition_range, prob_cured_condition, min_date=datetime.date(2019,1,1), max_date=datetime.date(2021,12,1)):
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
            patient_condition_id += 1
            # Generate trials of condition
            for _ in range(np.random.randint(*trials_per_condition_range)): # How many trials per condition 
                start = faker.date_between(start_date=diagnosed, end_date=cured if is_cured else max_date)
                end = faker.date_between(start_date=start, end_date=cured if is_cured else max_date)
                therapy = np.random.choice(therapies)
                successfull = np.random.randint(0, 100)
                patient_trials.append({'id': f'tr{trials_id}', 'start': start.strftime('%Y%m%d'), 'end': end.strftime('%Y%m%d'), 'condition': condition['id'], 'therapy': therapy['id'], 'successfull': f'{successfull}'})
                trials_id += 1
        patients.append({'id': patient_id, 'name': name, 'conditions': patient_conditions, 'trials': patient_trials})
    return patients, test_cases


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset of patients with conditions.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    parser.add_argument('-o', '--out', dest='out_dir', type=str, default='./data/generated', help='path of the output file (default: %(default)s).', metavar='PATH')
    parser.add_argument('--n_patients', type=int, default=1000, help='number of patients to be generated (default: %(default)s).', metavar='N')
    parser.add_argument('--n_test', type=int, default=3, help='number of test patients to be generated, with an uncured condition (default: %(default)s).', metavar='N')
    parser.add_argument('--conditions_per_patient', type=tuple, nargs=2, default=(3, 10), help='min and max number of conditions to be generated for each patient (default: %(default)s).', metavar=('MIN', 'MAX'))
    parser.add_argument('--trials_per_conditions', type=tuple, nargs=2, default=(0, 5), help='min and max number of trials to be generated for each condition of a patient (default: %(default)s).', metavar=('MIN', 'MAX'))
    parser.add_argument('--prob_cured', type=int, default=90, help='probability for a condition to be cured (default: %(default)s). Must be a value between 0 and 100.', metavar='P')
    parser.add_argument('--seed', type=int, default=0, help='random seed used in the generation (default: %(default)s).', metavar='S')
    args = parser.parse_args()

    # Initialize fake data generation
    faker = Faker()
    faker.seed_instance(args.seed)
    np.random.seed(args.seed)
    conditions_names = fetch_conditions()
    therapies_names = fetch_therapies()

    # Generate main tables
    conditions = gen_conditions(conditions_names)
    therapies = gen_therapies(therapies_names)
    patients, test_cases = gen_patients(faker, conditions, therapies, n_patients=args.n_patients, n_test_patients=args.n_test, conditions_per_patient_range=args.conditions_per_patient, trials_per_condition_range=args.trials_per_conditions, prob_cured_condition=args.prob_cured)
    
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