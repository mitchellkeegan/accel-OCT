fig_title = f'Path Bound Cutting Planes Variant Comparison'

PBCPBaseline_Model = {'Base Model': 'BendRegOCT',
                      'Model Name Override': 'BendOCT',
                      'Feature Name': 'PBCPBaseline',
                      'Hyperparameters': {'Path Bound Cutting Planes-Bound Structure': [False, True],
                                          'Path Bound Cutting Planes-Bound Negative Samples': [False, True],
                                          'Solution Polishing-Enabled': [False, True]}}

PBCPBaselineAll_Model = {'Base Model': 'BendRegOCT',
                        'Model Name Override': 'BendOCT',
                         'Feature Name': 'PBCPBaselineAll',
                        'Hyperparameters': {'Path Bound Cutting Planes-Bound Negative Samples': [True],
                                            'Path Bound Cutting Planes-Bound Structure': [True],
                                            'Solution Polishing-Enabled': [True]},
                         'Silent Filters': {'Path Bound Cutting Planes-Endpoint Only': [False]}}

models = [BendRegOCT_Model,
          PBCPBaseline_Model,
          PBCPBaselineAll_Model]