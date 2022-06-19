import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager

from recommender.dataset import Dataset

class Plotter():
    """Helper class to generate data plots."""
    def __init__(self, dataset: Dataset, out_dir: str):
        self.dataset = dataset
        self.out_dir = out_dir

    @contextmanager
    def plot(self, filepath: str, **kwargs):
        fig, ax = plt.subplots(**kwargs)
        yield fig, ax
        filepath = os.path.join(self.out_dir, filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)

    def plt_distributions(self):
        """Plot various data distributions."""
        # Condition status of patients
        with self.plot('distributions/conditions_per_patient') as (_, ax):
            p_cond = self.dataset.p_conditions['cured'].isna()
            conds_per_patient = p_cond.groupby(['patient', p_cond]).count().rename('count')
            conds_per_patient = conds_per_patient.reset_index().pivot(index='patient', columns='cured', values='count')
            for is_cured in [True, False]:
                conds_per_patient[is_cured].plot.hist(alpha=0.5, bins=range(conds_per_patient[is_cured].max()+1), ax=ax)
            ax.legend(['Cured', 'Not cured'])
            ax.set_xlabel('Number of conditions per patient')
            ax.set_ylabel('Frequency')
        # Number of trials per patient
        with self.plot('distributions/trials_per_patient') as (_, ax):
            trials_per_patient = self.dataset.p_trials.groupby('patient', observed=True)['therapy'].count().rename('count')
            trials_per_patient.plot.hist(alpha=0.5, bins=range(trials_per_patient.max()+1), ax=ax)
            ax.set_xlabel('Number of trials per patient')
            ax.set_ylabel('Frequency')
        # Number of trials per condition
        with self.plot('distributions/trials_per_condition') as (_, ax):
            trials_per_condition = self.dataset.p_trials.groupby('condition', observed=True)['therapy'].count().rename('count')
            trials_per_condition.plot.hist(alpha=0.5, bins=range(trials_per_condition.max()+1), ax=ax)
            ax.set_xlabel('Number of trials per condition')
            ax.set_ylabel('Frequency')
        # Distribution of success score
        with self.plot('distributions/success_scores') as (_, ax):
            trials_success = self.dataset.p_trials['successful']
            trials_success.plot.hist(alpha=0.5, bins=100, ax=ax)
            ax.set_xlabel('Success score of trials')
            ax.set_ylabel('Frequency')

if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser(description='Generate plots and stats about the given dataset. The plots are saved in the ./plots subdirectory.', formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42))
    # parser.add_argument('dataset_path', type=str, help='path to the dataset file.', default='./data/generated/dataset.json')

    #TODO: remove
    dataset_path = './data/final'
    parser.set_defaults(dataset_path=f'{dataset_path}/dataset.json')
    #TODO: remove

    # Compute plot folder path
    args = parser.parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dirname = os.path.basename(os.path.dirname(os.path.abspath(args.dataset_path)))
    out_dir = os.path.join(project_root, 'plots', dataset_dirname)
    os.makedirs(out_dir, exist_ok=True)

    print('Loading dataset...')
    dataset = Dataset(args.dataset_path, val_ratio=0)

    print('Generating plots...')
    plotter = Plotter(dataset, out_dir)
    plotter.plt_distributions()
