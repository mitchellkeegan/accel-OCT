tablename = f'Main Results Table'

BendOCT_Model = {'Base Model': 'BendRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'BendOCT',
                 'Filters': {}}

MinusEC_Model = {'Base Model': 'BendRegOCT',
                 'Column Name': 'Accelerated BendOCT',
                 'Experiment Name': 'Ablation',
                 'Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[False]}}

MinusPBCP_Model = {'Base Model': 'BendRegOCT',
                  'Column Name': '-PBCP',
             'Experiment Name': 'Ablation',
             'Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [False],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}