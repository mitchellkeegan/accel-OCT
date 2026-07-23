############### GLOBAL HYPERPARAMETERS ###############
# Define hyperparameter splits which should apply for all models. For example, to always split by lambda or depth
# These are injected into the hyperparameters for every model, so they should be valid keys for every model
global_hyperparameters = {'lambda': [('HIGH', lambda _lambda : _lambda >= 0.009),
                                     ('LOW', lambda _lambda : _lambda < 0.0009)]}

############### ACCELERATED MODELS ###############

fig_title = f'Figure - Scalability (lambda)'
add_model_name = True

BendRegOCT_Model = {'Base Model': 'BendRegOCT',
                    'Model Name Override': 'Baseline',
                    'Feature Name': 'Baseline',
                    'Colour': '#3f90da'}

All_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'AccelOCT',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]},
                 'Colour': '#bd1f01'}

MinusPBCP_Model = {'Base Model': 'BendRegOCT',
                 'Model Name Override': 'EQPOCT',
                 'Feature Name': 'Ablation',
                  'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                     'Path Bound Cutting Planes-Enabled': [False],
                                     'Solution Polishing-Enabled':[True],
                                     'Benders Cuts-Enhanced Cuts':[True]},
                 'Colour': '#ffa90e'}

def _lambda_filter_to_linestyle(_lambda_filter):
    filter_name = _lambda_filter[0]

    if filter_name == 'HIGH':
        return '-'
    elif filter_name == 'LOW':
        return '--'

LineStyleMap = ('lambda', _lambda_filter_to_linestyle)

legend_kwargs = {'fontsize': 11,
                 'ncol': 3,
                 'loc': 'lower center',
                 'columnspacing': 1.,
                 'bbox_to_anchor': (0.5125, 0.88)}

models = [BendRegOCT_Model,
          MinusPBCP_Model,
          All_Model]