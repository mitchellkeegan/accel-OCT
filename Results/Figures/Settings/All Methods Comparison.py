fig_title = f'All Methods Comparison'

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
             'Feature Name': 'Ablation',
             'Hyperparameters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

models = [BendRegOCT_Model,
          SPNoValCheck_Model,
          EC_Model,
          EQP2FR_Model,
          PBCPBaseline_Model]