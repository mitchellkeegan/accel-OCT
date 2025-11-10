tablename = f'Solution Polishing Table'

BendOCT_Model = {'Base Model': 'BendRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'BendOCT',
                 'Filters': {}}

SolutionPolishing_Model = {'Base Model': 'BendRegOCT',
                           'Column Name': 'Solution Polshing',
                           'Experiment Name': 'SPNoValCheck'}

models = [BendOCT_Model,
          SolutionPolishing_Model]