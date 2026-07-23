fig_title = f'Figure - BendWorstOCT Speed Comparison'


BendWorstOCT_Model = {'Base Model': 'BendWorstOCT',
                      'Model Name Override': 'Baseline',
                      'Feature Name': 'Baseline',
                      'Tag': '*BendWorstOCT'}

EQP_Model = {'Base Model': 'BendWorstOCT',
             'Feature Name': 'EQP',
             'Tag': '*EQP'}

MC_Model = {'Base Model': 'BendWorstOCT',
            'Feature Name': 'ModifiedCuts',
            'Tag': '*MC'}

SP_HP_Model = {'Base Model': 'BendWorstOCT',
            'Feature Name': 'SolutionPolishing',
            'Hyperparameters': {'Solution Polishing-Check Validity': [True, False],
                                'Solution Polishing-Use Cache': [True, False]}}

SP_Model = {'Base Model': 'BendWorstOCT',
            'Feature Name': 'SolutionPolishing',
            'Tag': '*SP',
            'Silent Filters': {'Solution Polishing-Check Validity': [False],
                                'Solution Polishing-Use Cache': [False]}}

PBCP_HP_Model = {'Base Model': 'BendWorstOCT',
                 'Feature Name': 'WorstPBCP',
                 'Hyperparameters': {'Path Bound Cutting Planes-Check Violation': [True,False],
                                     'Path Bound Cutting Planes-Endpoint Only': [True,False],
                                     'Path Bound Cutting Planes-Cut Type': ['Lazy', 'User']}}

PBCP_Model = {'Base Model': 'BendWorstOCT',
                 'Feature Name': 'WorstPBCP',
              'Tag': '*PBCP',
                 'Silent Filters': {'Path Bound Cutting Planes-Check Violation': [True],
                                     'Path Bound Cutting Planes-Endpoint Only': [False],
                                     'Path Bound Cutting Planes-Cut Type': ['Lazy']}}

WorstAccel_Model = {'Base Model': 'BendWorstOCT',
                    'Feature Name': 'WorstAccel',
                    'Tag': '*SP + EQP'}

legend_kwargs = {'fontsize': 11,
                 'ncol': 3,
                 'loc': 'lower center',
                 'bbox_to_anchor': (0.5125, 0.88)}


models = [BendWorstOCT_Model,
          EQP_Model,
          MC_Model,
          SP_Model,
          WorstAccel_Model]