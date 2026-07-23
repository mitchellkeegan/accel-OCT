fig_title = f'Figure - All Methods Comparison'

SPNoValCheck_Model = {'Base Model': 'BendRegOCT',
                      'Model Name Override': 'BendOCT',
                      'Feature Name': 'SPNoValCheck',
                      'Hyperparameters': {'Solution Polishing-Enabled':[True]}}

PBCPBaseline_Model = {'Base Model': 'BendRegOCT',
                      'Model Name Override': 'BendOCT',
                      'Feature Name': 'PBCPBaseline',
                      'Hyperparameters': {'Path Bound Cutting Planes-Enabled': [True]},
                      'Silent Filters': {'Path Bound Cutting Planes-Endpoint Only': [False],
                                         'Path Bound Cutting Planes-Bound Negative Samples': [True],
                                         'Path Bound Cutting Planes-Bound Structure': [True],
                                         'Solution Polishing-Enabled': [False]}}

EQP2FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
                'Hyperparameters': {'EQP Initial Cuts-Enabled': [True]},
               'Silent Filters': {'EQP Initial Cuts-H Variant': ['Recursive'],
                                   'EQP Initial Cuts-Features Removed': [2],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True]}}

EC_Model = {'Base Model': 'BendRegOCT',
            'Model Name Override': 'BendOCT',
            'Feature Name': 'Enhanced Cuts',
            'Hyperparameters': {'Benders Cuts-Enhanced Cuts': {True}},
            'Silent Filters': {'Benders Cuts-EC Level': [1]}}

All_Model = {'Base Model': 'BendRegOCT',
             'Model Name Override': 'BendOCT',
             'Tag': '*All',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

# Crude way of using the petroff10 colour cycle while skipping the second colour so that the individual comparison and
# ablation result plots have matching colours.
# mpl.rcParams['axes.prop_cycle'] = cycler('color', ['#3f90da', '#bd1f01', '#94a4a2', '#832db6', '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd'])


legend_kwargs = {'fontsize': 11,
                 'ncol': 3,
                 'loc': 'lower center',
                 'bbox_to_anchor': (0.5125, 0.88)}

models = [BendRegOCT_Model,
          All_Model,
          SPNoValCheck_Model,
          EC_Model,
          EQP2FR_Model,
          PBCPBaseline_Model]