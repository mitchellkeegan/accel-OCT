fig_title = f'Ablation Test'

All_Model = {'Base Model': 'BendRegOCT',
             'Model Name Override': 'BendOCT',
             'Feature Name': 'Ablation',
             'Tag': '*+ All',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusEQP_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                  'Tag': '*- EQP',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [False],
                                 'Path Bound Cutting Planes-Enabled': [True],

                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusEC_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                 'Tag': '*- EC',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[False]}}

MinusPBCP_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                   'Tag': '*- PBCP',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [False],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusSP_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                 'Tag': '*- SP',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[False],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

models = [BendRegOCT_Model,
          MinusSP_Model,
          MinusEC_Model,
          MinusEQP_Model,
          MinusPBCP_Model,
          All_Model]