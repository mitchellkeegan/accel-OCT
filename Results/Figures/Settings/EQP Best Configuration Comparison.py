fig_title = f'Figure - EQP Best Configurations Comparison'

EQP0FR_Model = {'Base Model': 'BendRegOCT',
                'Model Name Override': 'BendOCT',
               'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic'],
                                   'EQP Initial Cuts-Group Selection': [True],
                                   'EQP Initial Cuts-Features Removed': [0]}}

EQP1FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Recursive'],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True],
                                   'EQP Initial Cuts-Features Removed': [1]}}

EQP2FR_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
                'Feature Name': 'EQPGS',
               'Suffix': '_PADDED',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Recursive'],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True],
                                   'EQP Initial Cuts-Group Selection': [True],
                                   'EQP Initial Cuts-Filter Dominated': [True],
                                   'EQP Initial Cuts-Beta Dependence': [True],
                                   'EQP Initial Cuts-Features Removed': [2]}}

legend_kwargs = {'fontsize': 11,
                 'ncol': 2,
                 'loc': 'lower center',
                 'columnspacing': 0.5,
                 'bbox_to_anchor': (0.5125, 0.88)}

models = [BendRegOCT_Model,
          EQP0FR_Model,
          EQP1FR_Model,
          EQP2FR_Model]