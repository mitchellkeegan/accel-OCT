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
sideways_table = False  # Set to True to make table sideways in pdf (uses the rotating package)
latex_filepath = 'pdflatex.exe'

# Default for global filters
global_dataset_filter = None
global_encoding_filter = None  # Valid filters are 'QB_5' and 'QT_5' for numerical encodings, or 'cat' for categorical datasets
global_depth_filter = None
global_lambda_filter = None

# Defaults for options which we search over
#
depth_opts = None
_lambda_opts = None
encoding_opts = None

global_dataset_filter = valid_datasets['imbalanced']
global_encoding_filter = ['cat', 'QT_5']
global_depth_filter = [3,4]
# global_lambda_filter = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]

depth_opts = [3, 4]
_lambda_opts = [0.001]
# _lambda_opts = [0.08, 0.06, 0.04, 0.02, 0.01,
#                 0.008, 0.006, 0.004, 0.002, 0.001,
#                 0.0008, 0.0006, 0.0004, 0.0002, 0.0001]
encoding_opts = ['cat', 'QT_5']

base_dir = os.getcwd()

# If doing general comparison then write the table name and directory here
tablename_base = f'Table-AcceleratedComparison'
table_dir = os.path.join('..',
                         '..',
                         'Results',
                         'Tables')

BendRegOCT_Model = {'Base Model': 'BendRegOCT',
                    'Experiment Name': 'WorstCaseBaseline',
                    'Column Name': 'AccelOCT',
                    'Filters': {}}

BendWorstOCT_Model = {'Base Model': 'BendWorstOCT',
                      'Experiment Name': 'Baseline',
                      'Column Name': 'BendWorstOCT',
                      'Filters': {}}

SP_Model = {'Base Model': 'BendWorstOCT',
                      'Experiment Name': 'SolutionPolishing',
                      'Column Name': 'SP',
                      'Filters': {'Solution Polishing-Check Validity': [False],
                                  'Solution Polishing-Use Cache': [False]}}

MC_Model = {'Base Model': 'BendWorstOCT',
                          'Experiment Name': 'ModifiedCuts',
                          'Column Name': 'MC'}

EQP_Model = {'Base Model': 'BendWorstOCT',
             'Experiment Name': 'EQP',
             'Column Name': 'EQP',
             'Null Datasets': set(valid_datasets['imbalanced']) - set(valid_datasets['imbalanced eqp'])}

PBCP_Model = {'Base Model': 'BendWorstOCT',
              'Experiment Name': 'WorstPBCP',
              'Column Name': 'PBCP',
              'Filters': {'Path Bound Cutting Planes-Check Violation': [True],
                                     'Path Bound Cutting Planes-Endpoint Only': [False],
                                     'Path Bound Cutting Planes-Cut Type': ['Lazy']}}

WorstAccel_Model = {'Base Model': 'BendWorstOCT',
                    'Experiment Name': 'WorstAccel',
                    'Column Name': 'SP + EQP',
                    'Null Datasets': set(valid_datasets['imbalanced']) - set(valid_datasets['imbalanced eqp'])}

models = [BendWorstOCT_Model,
          EQP_Model,
          SP_Model,
          WorstAccel_Model]

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

    if 'Null Datasets' not in model:
        model['Null Datasets'] = set()

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


os.chdir(table_dir)

with open(tablename_base + '_table_pdf.tex', 'w') as f:
    f.write('\\documentclass{article}\n\n')
    f.write('\\usepackage{multirow}\n')
    f.write('\\usepackage{fullpage}\n')

    if sideways_table:
        f.write('\\usepackage{rotating}\n')

    f.write('\\extrafloats{200}\n')

    f.write('\\begin{document}\n\n')

Captions = []
TexFiles = []

for (depth, _lambda) in itertools.product(depth_opts, _lambda_opts):

    for model in models:

        df = model['df'].copy()

        # Filter based on the depth, lambda, and dataset filters
        b = pd.Series([True] * df.shape[0])
        b &= (df['depth'] == depth)
        b &= (df['lambda'] == _lambda)
        # b &= (df['EncodingScheme'] == enc)

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

    caption = f'depth = {depth}, $\\lambda$ = {_lambda}'
    tablename = f'{tablename_base}_d={depth}_l={_lambda}'

    texfile = os.path.join('TexFiles', tablename + '.tex')

    Captions.append(caption)
    TexFiles.append(texfile.replace('\\','/'))

    with (open(texfile ,'w') as f):

        BendRegOCT_fixed_cols = ['WC Acc', 'Acc']
        fixed_cols = ['WC Acc', 'Acc', 'Time', 'Gap']

        f.write(preamble(len(fixed_cols) * len(models)))

        headers = [multirow(2,'Dataset')]
        col_names = [' ']

        for model in models:
            headers.append(multicolumn(len(fixed_cols), 'c', model['Column Name']))
            col_names.extend(fixed_cols)

        headers[-1] += '\\\\\n'
        col_names[-1] += '\\\\\n'

        f.write(' & '.join(headers))
        f.write(' & '.join(col_names))

        f.write(hline())

        prev_dataset = None

        for dataset in sorted(global_dataset_filter, key=lambda x: dataset_ordering[x]):

            # Add lines between different types of datasets
            # if dataset_type(dataset) != dataset_type(dataset):
            #     f.write(hline())

            dataset_name = dataset

            if dataset_name == 'car_evaluation':
                dataset_name = 'car\\_evaluation'

            line = [dataset_name]

            accuracies, worst_accuracies, gaps, times, solved = [], [], [], [], []

            null_rows = []

            dataset_solved = False
            dataset_present_in_table = False

            for model in models:
                df = model['df local']
                df_f = df[df['Dataset'] == dataset]


                if (dataset == 'spect') and (len(df_f) == 2):
                    # Fix bug where spect experiments were doubled up
                    # For
                    df_f = df_f.iloc[[0]]

                num_rows_expected = 1

                if len(df_f) == 0:
                    continue

                assert num_rows_expected == len(df_f)

                if dataset in model['Null Datasets']:
                    accuracies.append(float('-inf'))
                    worst_accuracies.append(float('-inf'))
                    times.append(float('inf'))
                    gaps.append(float('inf'))
                    solved.append(False)
                    null_rows.append(True)
                    continue
                else:
                    null_rows.append(False)

                dataset_present_in_table = True

                # print(f'Results for {dataset} aggregated over {num_rows_expected} instances')

                instance_solved = (df_f.iloc[0]['Model Status'] == 2)

                if instance_solved and (model['Base Model'] != 'BendRegOCT'):
                    dataset_solved = True

                accuracies.append(round(df_f['Accuracy'].item(), ndigits=1))
                worst_accuracies.append(round(df_f['Worst Class Accuracy'].item(),ndigits=1))
                times.append(df_f['Solve Time'].item())
                gaps.append(df_f['Gap'].item())
                solved.append(instance_solved)

            if not dataset_present_in_table:
                continue

            best_models = []

            if dataset_solved:
                # Some models were solved, differentiate based on worst case accuracy, followed by solve time
                metric_list = [(wc_acc, acc, time, model_idx) for model_idx, (wc_acc, acc, time) in enumerate(zip(worst_accuracies, accuracies, times))]
                metric_list.sort(key=lambda x: (-x[0], x[2], x[3]))
            else:
                # No model could solve any instances. Differentiate based on worst case accuracy, followed by optimality gap
                metric_list = [(wc_acc, gap, model_idx) for model_idx, (wc_acc, gap) in enumerate(zip(worst_accuracies, gaps))]
                metric_list.sort(key=lambda x: (-x[0], x[1], x[2]))

            best_model_idx = metric_list[0][-1]

            for model_idx, (acc, worst_acc, time, gap, instance_solved, row_null) in enumerate(zip(accuracies,
                                                                                                   worst_accuracies,
                                                                                                   times,
                                                                                                   gaps,
                                                                                                   solved,
                                                                                                   null_rows)):

                if row_null:
                    line.append(multicolumn(len(fixed_cols), 'c', 'N/A'))
                    continue

                if model_idx == best_model_idx:

                    if instance_solved:
                        line.append('$\\mathbf' + parenthesise(f'{worst_acc:.1f}') + '$')
                        line.append('$\\mathbf' + parenthesise(f'{acc:.1f}') + '$')
                        line.append('$\\mathbf' + parenthesise(f'{time:.1f}') + '$')
                        line.append('\\textbf' + parenthesise('-'))

                    else:
                        line.append('$\\mathbf' + parenthesise(f'{worst_acc:.1f}') + '$')
                        line.append('$\\mathbf' + parenthesise(f'{acc:.1f}') + '$')
                        line.append('\\textbf' + parenthesise('-'))
                        line.append('$\\mathbf' + parenthesise(f'{100 * gap:.2f}') + '$')

                else:

                    if instance_solved:
                        line.append(f'${worst_acc:.1f}$')
                        line.append(f'${acc:.1f}$')
                        line.append(f'${time:.1f}$')
                        line.append('-')

                    else:
                        line.append(f'${worst_acc:.1f}$')
                        line.append(f'${acc:.1f}$')
                        line.append('-')
                        line.append(f'${100 * gap:.2f}$')

            line[-1] += '\\\\\n'
            f.write(' & '.join(line))

            prev_dataset = dataset_name

        f.write(postamble())

with open(tablename_base + '_table_pdf.tex', 'a') as f:
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
    os.system(rf"{latex_filepath} -aux-directory=AuxFiles {tablename_base + '_table_pdf.tex'}")