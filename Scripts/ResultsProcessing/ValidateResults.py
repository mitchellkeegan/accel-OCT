"""Check the validity of results from different experiments

This script takes in the .csv result file from different experiments in the Results folder
and checks for invalid models. The following conditions are checked for:
    Two models solved to optimality have different optimal objectives
    An experiment with a suboptimal solution claims to have a better solution than an optimal solution
    An upper bound for an experiment is violated by a solution from another experiment

"""

import os

import pandas as pd

# Set the option to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 2000)
pd.set_option('display.expand_frame_repr', False)

EPS = 1e-4

model = 'BendOCT'

FlowRegOCT_experiment_names = ['Baseline']
BendRegOCT_experiment_names = ['Baseline',
                               'Baseline_OLD',
                               'icCutset',
                               'cbCutset',
                               'SolutionPolishing',
                               'SPNoValCheck',
                               'PBCPBaseline_OLD',
                               'PBCPBaseline',
                               'PBCPBaselineEO',
                               'PBCPBaselineAll',
                               'MNSBaseline',
                               'EQPBaseline',
                               'EQPDABaseline',
                               'EQPGS',
                               'EnhancedCuts',
                               'EnhancedCutsV2',
                               'Ablation']

experiments = [('FlowRegOCT', FlowRegOCT_experiment_names),
               ('BendRegOCT', BendRegOCT_experiment_names)]

results_base = os.path.join('../..', 'Results')

# To run tests on script, add 'ValidateResultsTest' to model_names and modify Results/BendOCT/ValidateResultsTest/ValidateResultsTest.csv

models = []

for base_model, experiment_names in experiments:
    for exp_name in experiment_names:
        filename = os.path.join(results_base,
                                base_model,
                                exp_name,
                                f'{exp_name}.csv')

        df_model = pd.read_csv(filename)

        if exp_name == 'EnhancedCutsBROKEN':
            df_model = df_model[df_model['Benders Cuts-EC Level'] == 1]

        models.append(df_model)

df = pd.concat(models)

df['Subjob id'] = df['Subjob id'].fillna(-1).astype(int)
df['Array Job id'] = df['Array Job id'].fillna('N/A').astype(str)

def encoded_dataset_string(s):

    string_components = [s['Dataset']]

    if not pd.isna(s['Encoding']):
        if s['Encoding'] == 'Quantile Thresholds':
            string_components.append(f'_QT')
        else:
            string_components.append(f'_QB')

    if not pd.isna(s['Buckets']):
        string_components.append(f'_{int(s['Buckets'])}')

    string_components.append(f':d={int(s['depth'])}')

    string_components.append(f':l={s['lambda']}')


    return ''.join(string_components)

apply_cols = ['Dataset',
              'Encoding',
              'Buckets',
              'depth',
              'lambda']

df['Encoded Dataset'] = df[apply_cols].apply(encoded_dataset_string, axis=1)

def assess_group_validity(group):

    # Check that any optimal objectives agree

    optimal_experiments = group[group['Model Status'] == 2]
    suboptimal_experiments = group[group['Model Status'] != 2]

    experiments_valid = True
    dataset_string = group['Encoded Dataset'].iloc[0]

    # Detect if an upper bound is found which is violated by the solution from another experiment
    if group['Bound'].min() + EPS < group['Objective'].max():
        print(f'Disagreement in dataset {dataset_string}: Upper bound violated by an objective')
        experiments_valid = False

    if len(optimal_experiments) > 0:
        # Check that all experiments which claim optimality agree on the optimal objective
        if optimal_experiments['Objective'].max() - optimal_experiments['Objective'].min() > EPS:
            print(f'Disagreement in dataset {dataset_string}: optimal objective do not agree')
            experiments_valid = False

        if len(suboptimal_experiments) > 0:
            # Check that no experiments with suboptimal solutions claim better than the optimal solution
            if suboptimal_experiments['Objective'].max() > optimal_experiments['Objective'].max() + EPS:
                print(f'Disagreement in dataset {dataset_string}: suboptimal objective above claimed optimal objective')
                experiments_valid = False

    if not experiments_valid:
        print(group)
        print('\n')

    return pd.Series({'Experiments Valid': experiments_valid})

apply_cols = ['Encoded Dataset',
              'Model',
              'Model Status',
              'Objective',
              'Bound',
              'Results Directory',
              'Subjob id',
              'Array Job id']

df_grouped = df.groupby('Encoded Dataset')[apply_cols].apply(assess_group_validity)

if df_grouped['Experiments Valid'].all():
    print('\nAll experiments results are consistent with each other')
else:
    print(f'{df_grouped['Experiments Valid'].sum()} experiments are in disagreement with each other')