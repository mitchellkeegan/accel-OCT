"""Helper script for creating tables to compare experiment results

Creates a LaTeX table with aggregated per dataset results. By default tabulates number of isntances solves to optimality,
average solve time in solved instances, and average optimality gap in unsolved instances. Users must set the range of depths,
lambda values and encoding schemes to aggregate over. Per instance results can be viewed by constraining the depth and
lambda filters to one value, and choosing a single encoding scheme.

Writes tables to Results/Tables. The script creates the raw tex table inside a tablular environment
which can then be wrapped in a table environment as needed.

To use, set up a list of dictionaries each defining a table column. The following keys are available:
    Base Model: The MIP model which we search for results on
    Experiment Name : Name of the experiments (csv file should be Results/BaseModel/ExperimentName/ExperimentName.csv)
    Column Name: Name of the column in the table
    Suffix: Suffix added to results filename. Mostly used for padded EQP results
    Filters: See below

Filters are dictionaries where the keys correspond to columsn of the results csv file and the values are lists of
settings. Any rows which do not match the filters are filtered out. The script expects that the filter for each column
with take a subset of the data with one instance per dataset (two per numerical dataset). If this is not the case an error
will be thrown.

Settings used to generate tables in paper can be found in Results/Tables/Settings

"""

import os

import pandas as pd

from src.utils.data import valid_datasets

# Crude way to order the datasets in the table
dataset_ordering = {'soybean-small': 0,
                    'monk3': 1,
                    'monk1': 2,
                    'hayes-roth': 3,
                    'monk2': 4,
                    'house-votes-84': 5,
                    'spect': 6,
                    'breast-cancer': 7,
                    'balance-scale': 8,
                    'tic-tac-toe': 9,
                    'car_evaluation': 10,
                    'kr-vs-kp': 11,
                    'iris': 12,
                    'wine': 13,
                    'plrx': 14,
                    'wpbc': 15,
                    'parkinsons': 16,
                    'sonar': 17,
                    'wdbc': 18,
                    'transfusion': 19,
                    'banknote': 20,
                    'ozone-one': 21,
                    'segmentation': 22,
                    'spambase': 23,
                    'hepatitis': 24,
                    'fertility': 25,
                    'ionosphere': 26,
                    'thoracic': 27,
                    'ILPD': 28,
                    'credit': 29,
                    'biodeg': 30,
                    'seismic-bumps': 31,
                    'ann-thyroid': 32}

dataset_filter = None
encoding_filter = None  # Valid filters are 'QB_5' and 'QT_5' for numerical encodings, or 'cat' for categorical datasets
depth_filter = None
lambda_filter = None

dataset_filter = valid_datasets['categorical'] + valid_datasets['numerical'] + valid_datasets['mixed']



depth_filter = [3,4]
lambda_filter = [0.08, 0.06, 0.04, 0.02, 0.01,
           0.008, 0.006, 0.004, 0.002, 0.001,
           0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
encoding_filter = ['cat','QB_5', 'QT_5']


# # Uncomment to filter down to per-instance results
# depth_filter = [3]
# lambda_filter = [0.004]
# encoding_filter = ['cat','QB_5']

base_dir = os.getcwd()

# If doing general comparison then write the table name and directory here
tablename = f'Test Table'
table_dir = os.path.join('../../Results', 'Tables')

FlowOCT_Model = {'Base Model': 'FlowRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'FlowOCT',
                 'Filters': {}}

BendOCT_Model = {'Base Model': 'BendRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'BendOCT',
                 'Filters': {}}

models = [BendOCT_Model,
          FlowOCT_Model]

def get_encoding_name(series):
    if series['Dataset'] in valid_datasets['categorical']:
        return 'cat'
    else:
        if series['Encoding'] == 'Quantile Buckets':
            return f'QB_{int(series['Buckets'])}'
        elif series['Encoding'] == 'Quantile Thresholds':
            return f'QT_{int(series['Buckets'])}'

# For each model, load the relevant results from csv file into dataframe
# and generate sets of required combinations of hyperparameters
for model in models:
    prefix = model.get('Prefix', '')
    suffix = model.get('Suffix', '')
    extra_tag = model.get('Tag', '')
    model['Filename'] = ''.join(model['Experiment Name'].split())
    model['File Base'] = os.path.join('..', '..',
                                      'Results',
                                      model['Base Model'],
                                      model['Filename'])
    df = pd.read_csv(os.path.join(model['File Base'],
                                  model['Filename'] + suffix + '.csv'))



    # Filter based on the depth, lambda, and dataset filters
    b = pd.Series([True] * df.shape[0])
    if dataset_filter is not None:
        b &= (df['Dataset'].isin(dataset_filter))
    if depth_filter is not None:
        b &= (df['depth'].isin(depth_filter))
    if lambda_filter is not None:
        b &= (df['lambda'].isin(lambda_filter))

    if encoding_filter is not None:
        # By default set encoding scheme and number of buckets to an empty string
        if 'Encoding' in df:
            df['Encoding'] = df['Encoding'].fillna('')
        else:
            df['Encoding'] = ''

        if 'Buckets' in df:
            df['Buckets'] = df['Buckets'].fillna('')
        else:
            df['Buckets'] = ''

        df['EncodingScheme'] = df.apply(get_encoding_name, axis=1)

        b &= (df['EncodingScheme'].isin(encoding_filter))

    # Filter the dataframe based on the requested filters
    if 'Filters' in model:
        filters = model['Filters']
        for name, condition in filters.items():
            if isinstance(condition, list):
                b &= (df[name].isin(condition))
            else:
                b &= (df[name] == condition)

    df = df.loc[b].reset_index()



    # Create a new column describing the encoding scheme
    # df['EncodingScheme'] = df['Encoding'] + '_' + df['Buckets']

    model['df'] = df

def hline():
    return '\\hline\n'

def parenthesise(text):
    return '{' + f'{text}' + '}'

def multirow(depth,text):
    return '\\multirow' + parenthesise(depth) + parenthesise('*') + parenthesise(text)

def multicolumn(length,format_string,text):
    line = '\\multicolumn' + parenthesise(length) + parenthesise(format_string) + parenthesise(text)
    return line

def preamble(num_models):

    # format_string = '|c||' + 'c' * num_models + '|'
    format_string = 'c' + 'c' * num_models

    lines = ['\\begin' + parenthesise('tabular') + parenthesise(format_string)]

    lines.append('\\hline\n')

    return '\n'.join(lines)

def postamble(filename):
    lines = ['\\hline']
    lines.append('\\end' + parenthesise('tabular'))
    return '\n'.join(lines)

with open(os.path.join(table_dir, tablename + '.tex'),'w') as f:

    f.write(preamble(3 * len(models)))

    headers = [multirow(2,'Dataset')]
    col_names = [' ']

    fixed_cols = ['Solved', 'Time', 'Gap']

    for model in models:
        headers.append(multicolumn(3, 'c', model['Column Name']))
        col_names.extend(fixed_cols)

    headers[-1] += '\\\\\n'
    col_names[-1] += '\\\\\n'

    f.write(' & '.join(headers))
    f.write(' & '.join(col_names))

    # f.write(hline())
    f.write(hline())

    prev_dataset = None

    for dataset in sorted(dataset_filter,key=lambda x: dataset_ordering[x]):

        if dataset == 'car_evaluation':
            dataset_name = 'car\\_evaluation'
        else:
            dataset_name = dataset

        # Add lines between different types of datasets
        if dataset == 'iris' and prev_dataset == 'kr-vs-kp':
            f.write(hline())
        elif dataset == 'hepatitis' and prev_dataset in ['kr-vs-kp', 'spambase']:
            f.write(hline())

        line = [dataset_name]

        nums_solved, times, gaps = [], [], []

        for model in models:
            df = model['df']
            df_f = df[df['Dataset'] == dataset]

            num_rows_expected = len(depth_filter) * len(lambda_filter)

            if dataset not in valid_datasets['categorical']:
                num_rows_expected *= int(('QB_5' in encoding_filter) + ('QT_5' in encoding_filter))

            assert num_rows_expected == len(df_f)

            print(f'Results for {dataset} aggregated over {num_rows_expected} instances')

            solved_rows = df_f[df_f['Model Status'] == 2]
            unsolved_rows = df_f[df_f['Model Status'] == 9]

            nums_solved.append(len(solved_rows))

            if len(solved_rows) > 0:
                times.append(solved_rows['Solve Time'].mean())
            else:
                times.append(3600)

            if len(unsolved_rows) > 0:
                gaps.append(unsolved_rows['Gap'].mean())
            else:
                gaps.append(0.0)

        best_models = []
        best_models_num_solved = -1

        for i, num_solved in enumerate(nums_solved):
            if isinstance(num_solved, int) and num_solved >= best_models_num_solved:
                if num_solved == best_models_num_solved:
                    best_models.append(i)
                else:
                    best_models_num_solved = num_solved
                    best_models = [i]

        if len(best_models) == 1:
            best_model_idx = best_models[0]
        else:

            metric_best = float('inf')

            if best_models_num_solved == 0:
                # No model could solve any instances. Differentiate based on optimality gap
                    metric_list = gaps
            else:
                # Some models were solved, differentiate based on solve times
                    metric_list = times

            for i in best_models:
                model_metric = metric_list[i]
                if model_metric < metric_best:
                    best_model_idx = i
                    metric_best = model_metric

        current_model_idx = 0

        for num_solved, time, gap in zip(nums_solved, times, gaps):

            if current_model_idx == best_model_idx:
                line.append('$\\mathbf' + parenthesise(str(num_solved)) + '$')

                if num_solved == 0:
                    line.append('\\textbf' + parenthesise('-'))
                else:
                    line.append('$\\mathbf' + parenthesise(f'{time:.1f}') + '$')

                if num_solved == num_rows_expected:
                    line.append('\\textbf' + parenthesise('-'))
                else:
                    line.append('$\\mathbf' + parenthesise(f'{100 * gap:.2f}') + '$')

            else:
                line.append(f'${num_solved}$')

                if num_solved == 0:
                    line.append('-')
                else:
                    line.append(f'${time:.1f}$')

                if num_solved == num_rows_expected:
                    line.append('-')
                else:
                    line.append(f'${100 * gap:.2f}$')

            current_model_idx += 1

        line[-1] += '\\\\\n'
        f.write(' & '.join(line))

        prev_dataset = dataset


    f.write(postamble(tablename))

