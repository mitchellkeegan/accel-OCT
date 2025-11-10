fig_title = f'Path Bound Cutting Planes Endpoint Only Comparison'

PBCPBaseline_Model = {'Base Model': 'BendRegOCT',
                      'Model Name Override': 'BendOCT',
                      'Feature Name': 'PBCPBaseline',
                      'Hyperparameters': {'Path Bound Cutting Planes-Enabled': [True]},
                      'Silent Filters': {'Path Bound Cutting Planes-Bound Negative Samples': [False],
                                         'Path Bound Cutting Planes-Bound Structure': [False],
                                         'Path Bound Cutting Planes-Endpoint Only': [False],
                                         'Solution Polishing-Enabled': [False]}}

PBCPBaselineEO_Model = {'Base Model': 'BendRegOCT',
                        'Model Name Override': 'BendOCT',
                        'Feature Name': 'PBCPBaselineEO',
                        'Hyperparameters': {'Path Bound Cutting Planes-Endpoint Only': [True]},
                        'Silent Filters': {'Path Bound Cutting Planes-Bound Negative Samples': [False],
                                           'Path Bound Cutting Planes-Bound Structure': [False],
                                           'Solution Polishing-Enabled': [False]}}

models = [BendRegOCT_Model,
          PBCPBaseline_Model,
          PBCPBaselineEO_Model]