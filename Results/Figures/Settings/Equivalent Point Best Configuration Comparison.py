fig_title = f'Equivalent Point Best Configurations Comparison'

EQP0FR_Model = {'Base Model': 'BendRegOCT',
                'Model Name Override': 'BendOCT',
               'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic'],
                                   'EQP Initial Cuts-Features Removed': [0],
                                   'EQP Initial Cuts-Group Selection': [True]}}

EQP1FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain'],
                                   'EQP Initial Cuts-Features Removed': [1],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True]}}

EQP2FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Recursive'],
                                   'EQP Initial Cuts-Features Removed': [2],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True]}}

models = [BendRegOCT_Model,
          EQP0FR_Model,
          EQP1FR_Model,
          EQP2FR_Model]


ColourMap = ('EQP Initial Cuts-H Variant', {'Basic':'#ffa90e',
                                            'Chain':'#bd1f01',
                                            'Recursive':'#94a4a2'})