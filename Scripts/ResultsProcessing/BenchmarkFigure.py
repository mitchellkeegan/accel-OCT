"""Helper script for plotting a comparison experiment results

Plots a two-paned figure showing the number of instances solve with any given amount of time, and within any given
optimality gap. Saves plots to Results/Figures

To use, set up a list of dictionaries which define:
    1) What experiment to grab results from and
    2) Within each experiment what combinations of hyperparameters are being plotted

The script iterates over the list of dictionaries and for each one:
    1) Loads in the raw data
    2) If hyperparameters are specified, splits experiment results into the valid hyperparameter combinations
    3) Plots results for each experiment/hyperparameter combination specified

The following keys are available:
    Base Model: The MIP model which we search for results on
    Model Name Override: Override the name of the base model when plotting
    Tag: Add the Tag string to the legend in parentheses, or without parentheses if the first character is *
    Feature Name: The name of the experiment (csv file should be Results/BaseModel/FeatureName/FeatureName.csv)
    Suffix: Suffix added to results filename. Mostly used for padded EQP results
    Hyperparameters: See below
    Silent Filters: See below

Both Hyperparameters and Silent Filters are dictionaries where the keys correspond to columns and the values are lists of
settings. The silent filters silently filter out any rows which do not match the filter. The hyperparameters will filter out a subset
of results for each possible combination of hyperparameters which will be plotted separately. Note that the script keeps track of
the number of results in each subset, if there are no results it is ignored, if they are less than earlier plotted results it
is logged to console. It is the users responsibility to understand which results they expect to be plotted

Settings used to generate figures in paper can be found in Results/Figures/Settings

"""

import os
import itertools

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.data import valid_datasets
from src.utils.logging import shorthand_dict

import matplotlib.style as style

try:
    # Accessible colour palette for publication figures
    style.use('petroff10')
    petroff_palette = True
except:
    print('Warning: Publication figures use petroff10 colour palette which requires matplotlib >= 3.10\n'
          'Falling back to default tableau10 palette')
    petroff_palette = False

petroff_to_tableau = {'#ffa90e': 'tab:orange',
                      '#bd1f01': 'tab:green',
                      '#94a4a2': 'tab:red'}

############### DEFAULT SETTINGS ###############

# Filter defaults - These can be set to lists to act as filters on the plotted results
depth_filter = None
lambda_filter = None
dataset_filter = None
encoding_filter = None

eqp_features_removed = None     # If set to an integer filters the plotted datasets to those that benefit from having the max split set  setto eqp_feature_removed
lambda_cutoff = None            # If set to a float, only plot experiments with lambda <= lambda_cutoff

plot_log_scale = False  # Plot time-axis on log scale
save_figure = True      # Save figure to file
ColourMap = None        # Force lines in plot to take particular colour depending on settings (see examples in Results/Figures/Settings EQP plots)
LineStyleMap = None     # Force lines in plot to take particular line style depending on settings (see examples in Results/Figures/Settings EQP plots)
file_format = 'jpg'

############### BASELINE MODELS ###############

FlowRegOCT_Model = {'Base Model': 'FlowRegOCT',
                    'Model Name Override': 'FlowOCT',
                    'Feature Name': 'Baseline',
                    'Tag': 'Baseline'}

BendRegOCT_Model = {'Base Model': 'BendRegOCT',
                    'Model Name Override': 'BendOCT',
                    'Feature Name': 'Baseline',
                    'Tag': 'Baseline'}

############### USER SETTINGS ###############

eqp_features_removed = None

# Set up filters on optimisation parameters here
depth_filter = [3,4]
lambda_filter = [0.08, 0.06, 0.04, 0.02, 0.01,
           0.008, 0.006, 0.004, 0.002, 0.001,
           0.0008, 0.0006, 0.0004, 0.0002, 0.0001]

fig_title = ''

############### ACCELERATED MODELS ###############

models = [BendRegOCT_Model,
          FlowRegOCT_Model]

############### PROCESS SETTINGS ###############

if eqp_features_removed is not None:
    eqp_filter = valid_datasets['eqp'][eqp_features_removed]
else:
    eqp_filter = None

if lambda_cutoff is not None:
    lambda_filter = [_lambda for _lambda in lambda_filter if _lambda <= lambda_cutoff]
else:
    lambda_filter = None

filename = fig_title
file_dir = os.path.join('../../Results', 'Figures')

# Convert petroff colour specifications to default tableau if petroff palette is not available
if ColourMap is not None:
    if not petroff_palette:
        ColourMap = (ColourMap[0], {feat: petroff_to_tableau.get(col)
                                    for feat,col in ColourMap[1].items()})


def merge_dfs(model, prefix, suffixes, extra_tag):
    Filenames = [prefix + ''.join(model['Feature Name'].split()) + suffix + 'Benchmark' for suffix in suffixes]
    FileBases = [os.path.join('../..', 'Results', model['Base Model'], Filename) for Filename in Filenames]

    df_list = []

    for Filebase, Filename in zip(FileBases, Filenames):
        df_list.append(pd.read_csv(os.path.join(Filebase, Filename + '.csv')))

    return pd.concat(df_list, axis=0, join ='outer', ignore_index=True)

def get_shorthand(name,separator=None):

    if separator is not None:
        assert isinstance(separator,str)

        name_split = name.split(separator)
        shorthand_split = [shorthand_dict.get(n,n) for n in name_split]
        return separator.join(shorthand_split)

    else:
        return shorthand_dict.get(name, name)

def name_encoding_func(df):
    if df['Dataset'] in valid_datasets['categorical']:
        return df['Dataset']
    else:
        if df['Encoding'] == 'Quantile Buckets':
            return f'{df['Dataset']}_QB_{int(df['Buckets'])}'
        elif df['Encoding'] == 'Quantile Thresholds':
            return f'{df['Dataset']}_QT_{int(df['Buckets'])}'

def split_hp_names(hp_names):
    new_hp_names = []

    for hp_name in hp_names:
        hp_name_split = hp_name.split('-')
        new_hp_names.append(tuple(shorthand_dict.get(n,n) for n in hp_name_split))

    return new_hp_names

# For each model, load the relevant results from csv file into dataframe
# and generate sets of required combinations of hyperparameters
for model in models:
    prefix = model.get('Prefix','')
    suffix = model.get('Suffix','')
    extra_tag = model.get('Tag','')

    if isinstance(suffix,list):
        df = merge_dfs(model, prefix, suffix, extra_tag)
    else:
        model['Filename'] = ''.join(model['Feature Name'].split())
        model['File Base'] = os.path.join('../..', 'Results', model['Base Model'], model['Filename'])
        df = pd.read_csv(os.path.join(model['File Base'],prefix + model['Filename'] + suffix + '.csv'))

    # By default set encoding scheme and number of buckets to an empty string
    if 'Encoding' in df:
        df['Encoding'] = df['Encoding'].fillna('')
    else:
        df['Encoding'] = ''

    if 'Buckets' in df:
        df['Buckets'] = df['Buckets'].fillna('')
    else:
        df['Buckets'] = ''

    hyperparameters = model.get('Hyperparameters',{})
    model_name = model.get('Model Name Override', model['Base Model'])

    if 'Tag' in model:
        if model['Tag'][0] == '*':
            model_tag = f' {model['Tag'][1:]}'
        else:
            model_tag = f' ({model['Tag']})'
    else:
        model_tag = ''

    if len(hyperparameters) > 0:
        hp_filters = {}
        hp_names, hp_values = hyperparameters.keys(), hyperparameters.values()

        # Confirm that the hyperparameter names are valid
        for name in hp_names:
            assert name in df

        for hp_combo in itertools.product(*hp_values):

            hps = {hp_name: hp_value for hp_name, hp_value in zip(hp_names, hp_combo)}

            hp_dict = {}
            disabled_hps = set()

            for hp_name, hp_value in hps.items():
                hp_name_split = hp_name.split('-')

                if len(hp_name_split) == 1:
                    hp_name_split = hp_name_split[0]
                    hp_dict[shorthand_dict.get(hp_name_split, hp_name_split)] = shorthand_dict.get(hp_value, hp_value)

                elif len(hp_name_split) == 2:
                    hp_name_split = tuple(shorthand_dict.get(n, n) for n in hp_name_split)
                    if hp_name_split[0] not in hp_dict:
                        hp_dict[hp_name_split[0]] = []

                    if isinstance(hp_value, bool):
                        if hp_name_split[1] == 'ON':
                            if not hp_value:
                                disabled_hps.add(hp_name_split[0])

                        elif hp_value:
                            hp_dict[hp_name_split[0]].append(hp_name_split[1])

                    elif isinstance(hp_value, int):
                        hp_dict[hp_name_split[0]].append(f'{hp_name_split[1]}={hp_value}')

                    else:
                        hp_dict[hp_name_split[0]].append(f'{hp_name_split[1]}={shorthand_dict.get(hp_value,hp_value)}')

                else:
                    print(f'Invalid hyperparameter name: {hp_name}')
                    assert False

            hp_combo_string_list = []

            for feat_name, hp_settings in hp_dict.items():
                if feat_name in disabled_hps:
                    continue
                elif not isinstance(hp_settings, list):
                    hp_combo_string_list.append(f'{feat_name}={hp_settings}')

                elif len(hp_settings) == 0:
                    hp_combo_string_list.append(feat_name)
                elif (feat_name == 'Benders Cuts') and (hp_settings[0] == 'EC'):
                    hp_combo_string_list.append('EC')
                else:
                    hp_combo_string_list.append(f'{feat_name}({"+".join(setting for setting in hp_settings)})')

            hp_combo_string = model_name + ' + ' + ' + '.join(hp_combo_string_list)

            hp_combo_string += model_tag

            # if 'Tag' in model:
            #     hp_combo_string += f' ({model['Tag']})'

            hp_filters[hp_combo_string] = hps

        model['hp Filters'] = hp_filters
    else:

        model['hp Filters'] = {model_name + model_tag: {}}

        # if 'Tag' in model:
        #     model['hp Filters'] = {model_name + f' ({model['Tag']})': {}}
        # else:
        #     model['hp Filters'] = {model_name: {}}

    df['Encoded Dataset'] = df.apply(name_encoding_func, axis=1)
    # df['Encoded Dataset'] = df['Dataset'] + '_' + df['Encoding'] + '_' + df['Buckets']

    df['Instance Identifier'] = df['Dataset'] + ' ' + df['Encoding'] + ' ' + df['depth'].astype('str') + ' ' + df['lambda'].astype('str')
    model['Instance Combos'] = set(df['Instance Identifier'])

    model['df'] = df

for model in models:
    print(model['hp Filters'])

# Filter the hyperparameters and benchmarks based on what dataset + depth + lambda combinations are actually available
valid_combos = set.intersection(*[model['Instance Combos'] for model in models])

# TODO: Reimplement with group_by?
for model in models:
    model['df'] = model['df'].loc[model['df']['Instance Identifier'].isin(valid_combos)]
    model['df'].reset_index(inplace=True)

max_opt_solve_time = 0
max_opt_gap = 0

num_rows = None

# Loop over the models and generate rows for each hyperparameter combo
for model in models:
    hp_filters = model['hp Filters']

    df = model['df']

    model['Rows'] = {}

    if 'Silent Filters' in model:
        b = pd.Series([True] * df.shape[0])
        silent_filters = model['Silent Filters']
        for name, condition in silent_filters.items():
            if isinstance(condition,list):
                b &= (df[name].isin(condition))
            else:
                b &= (df[name] == condition)

        df = df.loc[b].reset_index()

    for name, filter in hp_filters.items():
        b = pd.Series([True] * df.shape[0])
        for column, condition in filter.items():
            b &= (df[column] == condition)

        if dataset_filter is not None:
            b &= (df['Dataset'].isin(dataset_filter))
        if eqp_filter is not None:
            b &= (df['Encoded Dataset'].isin(eqp_filter))
        if depth_filter is not None:
            b &= (df['depth'].isin(depth_filter))
        if lambda_filter is not None:
            b &= (df['lambda'].isin(lambda_filter))

        if encoding_filter is not None:
            if 'Encoding' in df.columns:
                b &= (df['Encoding'].isin(encoding_filter))
            else:
                print('Encoding Filter specified but dataset does not have an encoding scheme (probably categorical data)')

        # Filter rows
        rows = df.loc[b]

        if len(rows) == 0:
            print(f'Experiment {name} has zero rows. Excluding from plot')
            continue
        elif num_rows is not None:
            if len(rows) != num_rows:
                print(f'Experiment {name} has {len(rows)} rows which does not agree with previous experiments with {num_rows} rows')
        else:
            num_rows = len(rows)

        rows_time = rows.loc[rows['Model Status'] == 2].sort_values(by='Solve Time', ignore_index=True)
        rows_gap = rows.loc[rows['Model Status'] == 9].sort_values(by='Gap', ignore_index=True)

        if rows_time['Solve Time'].max() > max_opt_solve_time:
            max_opt_solve_time = rows_time['Solve Time'].max()

        if rows_gap['Gap'].max() > max_opt_gap:
            max_opt_gap = rows_gap['Gap'].max()

        model['Rows'][name] = {'Time': rows_time,
                               'Gap': rows_gap,
                               'Filter': filter}

fig_expand_mult = 1.2

fig, axs = plt.subplots(1, 2,
                        sharey=True,
                        figsize=(fig_expand_mult * 6.4, fig_expand_mult * 4.8))
fig.subplots_adjust(wspace=0)

base_num_solved = None

# Loop over each model and it's set of rows and plot on figure
for model in models:
    for name, rows_info in model['Rows'].items():
        rows_time = rows_info['Time']
        rows_gap = rows_info['Gap']
        rows_filter = rows_info['Filter']

        if base_num_solved is None:
            base_num_solved = len(rows_time)
            print(f'Base model {name} solved {base_num_solved} within time limit')

        else:
            feature_num_solved = len(rows_time)

            if feature_num_solved < base_num_solved:
                print(f'{name} solved {feature_num_solved} within time limit - less than the base model :(')
            elif feature_num_solved == base_num_solved:
                print(f'{name} solved {feature_num_solved} within time limit - the same as the base model :|')
            else:
                print(f'{name} solved {feature_num_solved} within time limit (Solved {base_num_solved} within {rows_time.iloc[base_num_solved]['Solve Time']:.2f}s)')

        linestyle = '-'
        color = None

        if ColourMap is not None:
            colour_feature, cmap = ColourMap

            if colour_feature in rows_filter:
                feat_value = rows_filter[colour_feature]

                color = cmap[feat_value]


        if LineStyleMap is not None:
            line_feature, lmap = LineStyleMap

            if line_feature in rows_filter:
                feat_value = rows_filter[line_feature]

                linestyle = lmap[feat_value]

        if plot_log_scale:
            axs[0].semilogx(rows_time['Solve Time'], rows_time.index,
                            label=name, drawstyle='steps', linestyle=linestyle, color=color)
        else:
            axs[0].plot(rows_time['Solve Time'], rows_time.index,
                        label=name, drawstyle='steps', linestyle=linestyle, color=color)


        axs[1].plot(rows_gap['Gap'] * 100, rows_gap.index + rows_time.shape[0],
                    label=name, drawstyle='steps', linestyle=linestyle, color=color)


opt_gap_xlim = min(100, max_opt_gap * 100)
# opt_gap_xlim = max_opt_gap * 100


axs[0].set_xlabel('Solve Time (s)')
axs[1].set_xlabel('Optimality Gap (%)')
axs[0].set_xlim(0.0, max_opt_solve_time)
axs[1].set_xlim(0.0, opt_gap_xlim)
axs[0].set_ylabel('Number of Instances Solved within Time')
axs[0].set_ylim(0,num_rows+1)
axs[1].yaxis.set_label_position('right')
axs[1].set_ylabel('Number of Instances Solved within Gap')
axs[1].legend(fontsize=9)
# plt.suptitle(fig_title)

if save_figure:
    plt.savefig(os.path.join(file_dir, filename + f'.{file_format}'), dpi=500)

plt.show()