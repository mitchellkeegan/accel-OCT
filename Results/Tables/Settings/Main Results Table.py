tablename = f'Table - Main Results'

BendOCT_Model = {'Base Model': 'BendRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'BendersOCT',
                 'Filters': {}}

All_Model = {'Base Model': 'BendRegOCT',
                 'Column Name': 'Accelerated BendersOCT',
                 'Experiment Name': 'Ablation',
                 'Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusPBCP_Model = {'Base Model': 'BendRegOCT',
                  'Column Name': 'Accelerated (No PBCP)',
             'Experiment Name': 'Ablation',
             'Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [False],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

models = [BendOCT_Model,
          All_Model,
          MinusPBCP_Model]