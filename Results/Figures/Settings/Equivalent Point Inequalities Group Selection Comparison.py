fig_title = f'Equivalent Point Group Selection Comparison - {eqp_features_removed} Features Removed'


EQPBaseline_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic']},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

EQPDABaseline_Model = {'Base Model': 'BendRegOCT',
                       'Feature Name': 'EQPDABaseline',
                       'Model Name Override': 'BendOCT',
                       'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive'],
                                           'EQP Initial Cuts-Disaggregate Alpha': [True]},
                       'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}


EQPGS_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendOCT',
               'Feature Name': 'EQPGS',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic', 'Chain', 'Recursive'],
                                   'EQP Initial Cuts-Disaggregate Alpha': [True, False],
                                   'EQP Initial Cuts-Group Selection': [True]},
               'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

# Keep H variant colouring consistent even when only a subset are being plotted
ColourMap = ('EQP Initial Cuts-H Variant', {'Basic':'#ffa90e',
                                            'Chain':'#bd1f01',
                                            'Recursive':'#94a4a2'})

LineStyleMap = ('EQP Initial Cuts-Group Selection', {True: '--',
                                                     False: '-'})

models = [BendRegOCT_Model,
          EQPBaseline_Model,
          EQPDABaseline_Model,
          EQPGS_Model]