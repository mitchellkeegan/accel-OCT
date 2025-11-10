tablename = f'Strengthened Cuts Table'

BendOCT_Model = {'Base Model': 'BendRegOCT',
                 'Experiment Name': 'Baseline',
                 'Column Name': 'BendOCT',
                 'Filters': {}}

EC_Model = {'Base Model': 'BendRegOCT',
            'Column Name': 'Strengthened Benders Cuts',
            'Experiment Name': 'Enhanced Cuts',
            'Filters': {'Benders Cuts-Enhanced Cuts': [True],
                        'Benders Cuts-EC Level': [1]}}


models = [BendOCT_Model,
          EC_Model]
