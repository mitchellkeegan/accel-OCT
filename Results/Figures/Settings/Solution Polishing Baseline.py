fig_title = f'Solution Polishing Feasibility Check Comparison'

SolutionPolishing_Model = {'Base Model': 'BendRegOCT',
                           'Model Name Override': 'BendOCT',
                           'Feature Name': 'Solution Polishing',
                           'Hyperparameters': {'Solution Polishing-Enabled': [True]},
                           'Tag': 'Feasibility Check'}

SPNoValCheck_Model = {'Base Model': 'BendRegOCT',
                      'Model Name Override': 'BendOCT',
                      'Feature Name': 'SPNoValCheck',
                      'Hyperparameters': {'Solution Polishing-Enabled': [True]}}

models = [BendRegOCT_Model,
          SolutionPolishing_Model,
          SPNoValCheck_Model]