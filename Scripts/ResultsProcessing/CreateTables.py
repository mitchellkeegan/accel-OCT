"""Helper script for creating tables to compare experiment results

Creates a LaTeX tables with per instance results for each datset. By default, tabulates number of instance solves to optimality,
average solve time in solved instances, and average optimality gap in unsolved instances. Users must set the range of depths,
lambda values and encoding schemes to aggregate over. Per instance results can be viewed by constraining the depth and
lambda filters to one value, and choosing a single encoding scheme.

Writes tables to Results/Tables. A .tex file with a tabular environment is written to Results/Tables/TexFiles for each
instance. One other .tex file is created collecting each of the individual tables which can be optionally compiled
by setting create_pdf=True

To use, set up a list of dictionaries each defining a table column. The following keys are available:
    Base Model: The MIP model which we search for results on
    Model Name Override:
    Experiment Name : Name of the experiments (csv file should be Results/BaseModel/ExperimentName/ExperimentName.csv)
    Column Name: Name of the column in the table
    Suffix: Suffix added to results filename. Mostly used for padded EQP results
    Filters: See below

Filters are dictionaries where the keys correspond to columns of the results csv file and the values are lists of
settings. Any rows which do not match the filters are filtered out. The script expects that the filter for each column
with take a subset of the data with one instance per dataset (two per numerical dataset). If this is not the case an error
will be thrown.

It is expected that the filters only allow one result per dataset (or zero in which case it will be ignored).

"""

import os

import itertools

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

def dataset_type(dataset):
    if dataset in valid_datasets['categorical']:
        return 1
    elif dataset in valid_datasets['numerical']:
        return 2
    elif dataset in valid_datasets['mixed']:
        return 3
    else:
        return None

create_pdf = False       # Set to True to automatically generate a pdf from the raw tex file
sideways_table = False   # Set to True to make table sideways in pdf (uses the rotating package). Good for wide tables
latex_filepath = 'pdflatex.exe'

# Default for global filters
global_dataset_filter = None
global_encoding_filter = None  # Valid filters are 'QB_5' and 'QT_5' for numerical encodings, or 'cat' for categorical datasets
global_depth_filter = None
global_lambda_filter = None

# Defaults for options which we search over
# Set to lists to search over every combination of elements
depth_opts = None
_lambda_opts = None
encoding_opts = None

# Can set global filters on datasets, encodings, or depth
global_dataset_filter = valid_datasets['categorical'] + valid_datasets['numerical'] + valid_datasets['mixed']
global_encoding_filter = ['cat', 'QB_5', 'QT_5']
global_depth_filter = [3,4]

depth_opts = [3,4]
_lambda_opts = [0.08, 0.06, 0.04, 0.02, 0.01,
                0.008, 0.006, 0.004, 0.002, 0.001,
                0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
encoding_opts = ['cat', 'QB_5', 'QT_5']

base_dir = os.getcwd()

# If doing general comparison then write the table name and directory here
tablename_base = f'EQP_Comparison_Table'
table_dir = os.path.join('..',
                         '..',
                         'Results',
                         'Tables')

BendRegOCT_Model = {'Base Model': 'BendRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'BendersOCT',
                 'Filters': {}}

EQP0FR_Model = {'Base Model': 'BendRegOCT',
                'Model Name Override': 'BendOCT',
               'Experiment Name': 'EQPGS',
                 'Column Name': 'FR0',
               'Suffix': '_PADDED',
               'Filters': {'EQP Initial Cuts-H Variant': ['Basic'],
                                   'EQP Initial Cuts-Group Selection': [True],
                                   'EQP Initial Cuts-Features Removed': [0]}}

EQP1FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Experiment Name': 'EQPGS',
                 'Column Name': 'FR1',
               'Suffix': '_PADDED',
               'Filters': {'EQP Initial Cuts-H Variant': ['Recursive'],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True],
                                   'EQP Initial Cuts-Features Removed': [1]}}

EQP2FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Experiment Name': 'EQPGS',
                 'Column Name': 'FR2',
               'Suffix': '_PADDED',
               'Filters': {'EQP Initial Cuts-H Variant': ['Recursive'],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True],
                                   'EQP Initial Cuts-Filter Dominated': [True],
                                   'EQP Initial Cuts-Beta Dependence': [True],
                                   'EQP Initial Cuts-Features Removed': [2]}}

models = [BendRegOCT_Model,
          EQP0FR_Model,
          EQP1FR_Model,
          EQP2FR_Model]

def get_encoding_name(series):
    if series['Dataset'] in valid_datasets['categorical']:
        return 'cat'
    else:
        if series['Encoding'] == 'Quantile Buckets':
            return f'QB_{int(series['Buckets'])}'
        elif series['Encoding'] == 'Quantile Thresholds':
            return f'QT_{int(series['Buckets'])}'

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

def postamble():
    lines = ['\\hline']
    lines.append('\\end' + parenthesise('tabular'))
    return '\n'.join(lines)

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

    # Filter based on the global depth, lambda, and dataset filters
    b = pd.Series([True] * df.shape[0])
    if global_dataset_filter is not None:
        b &= (df['Dataset'].isin(global_dataset_filter))
    if global_depth_filter is not None:
        b &= (df['depth'].isin(global_depth_filter))
    if global_lambda_filter is not None:
        b &= (df['lambda'].isin(global_lambda_filter))
    if global_encoding_filter is not None:
        b &= (df['EncodingScheme'].isin(global_encoding_filter))

    model['df'] = df

# Easier to use Results/Tables as working directory
os.chdir(table_dir)

with open(tablename_base + '.tex', 'w') as f:
    f.write('\\documentclass{article}\n\n')
    f.write('\\usepackage{multirow}\n')
    f.write('\\usepackage{fullpage}\n')

    if sideways_table:
        f.write('\\usepackage{rotating}\n')

    f.write('\\extrafloats{200}\n')

    f.write('\\begin{document}\n\n')

Captions = []
TexFiles = []

if not os.path.exists('TexFiles'):
    os.mkdir('TexFiles')

for (depth, _lambda, enc) in itertools.product(depth_opts, _lambda_opts, encoding_opts):

    for model in models:

        df = model['df'].copy()

        # Filter based on the depth, lambda, and dataset filters
        b = pd.Series([True] * df.shape[0])
        b &= (df['depth'] == depth)
        b &= (df['lambda'] == _lambda)
        b &= (df['EncodingScheme'] == enc)

        # Filter the dataframe based on the requested filters
        if 'Filters' in model:
            filters = model['Filters']
            for name, condition in filters.items():
                if isinstance(condition, list):
                    b &= (df[name].isin(condition))
                else:
                    b &= (df[name] == condition)

        df = df.loc[b].reset_index()

        model['df local'] = df

    caption = f'depth = {depth}, $\\lambda$ = {_lambda}, encoding scheme = {enc.replace('_','-')}'
    tablename = f'{tablename_base}_d={depth}_l={_lambda}_e={enc.replace('_','-')}'

    texfile = os.path.join('TexFiles', tablename + '.tex')

    Captions.append(caption)
    TexFiles.append(texfile.replace('\\','/'))

    with open(texfile ,'w') as f:

        f.write(preamble(2 * len(models)))

        headers = [multirow(2,'Dataset')]
        col_names = [' ']

        fixed_cols = ['Time', 'Gap']

        for model in models:
            headers.append(multicolumn(2, 'c', model['Column Name']))
            col_names.extend(fixed_cols)

        headers[-1] += '\\\\\n'
        col_names[-1] += '\\\\\n'

        f.write(' & '.join(headers))
        f.write(' & '.join(col_names))

        f.write(hline())

        prev_dataset = None

        for dataset in sorted(global_dataset_filter, key=lambda x: dataset_ordering[x]):

            # Add lines between different types of datasets
            if dataset_type(dataset) != dataset_type(prev_dataset):
                f.write(hline())

            dataset_name = dataset

            if dataset_name == 'car_evaluation':
                dataset_name = 'car\\_evaluation'

            line = [dataset_name]

            times, gaps, solved = [], [], []

            dataset_solved = False
            dataset_present_in_table = False

            for model in models:
                df = model['df local']
                df_f = df[df['Dataset'] == dataset]

                num_rows_expected = 1

                if len(df_f) == 0:
                    continue

                assert num_rows_expected == len(df_f)

                dataset_present_in_table = True

                # print(f'Results for {dataset} aggregated over {num_rows_expected} instances')

                instance_solved = (df_f.iloc[0]['Model Status'] == 2)

                if instance_solved:
                    dataset_solved = True

                times.append(df_f['Solve Time'].item())
                gaps.append(df_f['Gap'].item())
                solved.append(instance_solved)

            if not dataset_present_in_table:
                continue

            best_models = []

            if dataset_solved:
                # Some models were solved, differentiate based on solve times
                metric_list = times
            else:
                # No model could solve any instances. Differentiate based on optimality gap
                metric_list = gaps

            metric_best = float('inf')

            for i in range(len(models)):
                model_metric = metric_list[i]
                if model_metric < metric_best:
                    best_model_idx = i
                    metric_best = model_metric

            current_model_idx = 0

            for time, gap, instance_solved in zip(times, gaps, solved):

                if current_model_idx == best_model_idx:

                    if instance_solved:
                        line.append('$\\mathbf' + parenthesise(f'{time:.1f}') + '$')
                        line.append('\\textbf' + parenthesise('-'))

                    else:
                        line.append('\\textbf' + parenthesise('-'))
                        line.append('$\\mathbf' + parenthesise(f'{100 * gap:.2f}') + '$')

                else:

                    if instance_solved:
                        line.append(f'${time:.1f}$')
                        line.append('-')

                    else:
                        line.append('-')
                        line.append(f'${100 * gap:.2f}$')

                current_model_idx += 1

            line[-1] += '\\\\\n'
            f.write(' & '.join(line))

            prev_dataset = dataset_name

        f.write(postamble())


with open(tablename_base + '.tex', 'a') as f:

    for caption, texfile in zip(Captions, TexFiles):

        if sideways_table:
            f.write('\\begin{sidewaystable}\n')
        else:
            f.write('\\begin{table}\n')

        f.write('\t\\centering\n')
        f.write(f'\t\\input{{{texfile}}}\n')
        f.write(f'\t\\caption{{{caption}}}\n')

        if sideways_table:
            f.write('\\end{sidewaystable}\n\n')
        else:
            f.write('\\end{table}\n\n')

    f.write('\\end{document}')

if create_pdf:
    os.system(rf"{latex_filepath} -aux-directory=AuxFiles {tablename_base + '.tex'}")