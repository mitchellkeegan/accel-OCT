fig_title = f'Figure - Ablation Test'

All_Model = {'Base Model': 'BendRegOCT',
             'Model Name Override': 'BendOCT',
             'Feature Name': 'Ablation',
             'Tag': '*All',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusEQP_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                  'Tag': '*All - EQP',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [False],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusEC_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                 'Tag': '*All - MC',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[False]}}

MinusPBCP_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                   'Tag': '*All - PBCP',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [False],
                                 'Solution Polishing-Enabled':[True],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

MinusSP_Model = {'Base Model': 'BendRegOCT',
                  'Model Name Override': 'BendOCT',
                 'Tag': '*All - SP',
             'Feature Name': 'Ablation',
             'Silent Filters': {'EQP Initial Cuts-Enabled': [True],
                                 'Path Bound Cutting Planes-Enabled': [True],
                                 'Solution Polishing-Enabled':[False],
                                 'Benders Cuts-Enhanced Cuts':[True]}}

legend_kwargs = {'fontsize': 11,
                 'ncol': 3,
                 'loc': 'lower center',
                 'bbox_to_anchor': (0.5125, 0.88)}

models = [BendRegOCT_Model,
          All_Model,
          MinusSP_Model,
          MinusEC_Model,
          MinusEQP_Model,
          MinusPBCP_Model]