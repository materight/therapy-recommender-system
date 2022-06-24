# Therapy Recommender System

## Get started
Clone the repository and install the required dependencies:
```
git clone https://github.com/materight/therapy-recommender-system.git
cd therapy-recommender-system
pip install -r requirements.txt
```

## Run recommender
To produce recommendations for a patient and a condition, run:
```
python main.py -d [dataset_path] -p [patient_id] -c [condition_id]
```

Alternatively, it is possible to specify a test `csv` file containing the patient and condition ids to predict: 
```
python main.py -d [dataset_path] -t [test_path]
```

Additional options can be specified (run with `--help` to see the available ones).


## Hyperparameter search
To evaluate multiple configurations for the recommender and perform an ablation study,run: 
```
python benchmark.py -d [dataset_path]
```
The results will be saved in a `csv` file under the `./results` folder.

The `--val_split [fraction]` option can be specified to customize the size of the validation split to use for evaluation.


## Data generation
A synthetic dataset can be generated by simply running: 
```
python data_generator.py --n-patients [num_generated_patients] -o [output_path]
```

The generation can be customized with additional options (use `--help` to see the available ones).