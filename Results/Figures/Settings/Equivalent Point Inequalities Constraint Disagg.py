# Note: Need to set eqp_features_removed \in [1,2] to get paper plot
fig_title = f'Equivalent Point Constraint Disaggregation Comparison - {eqp_features_removed} Features Removed'


EQPBaseline_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive']},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

EQPDABaseline_Model = {'Base Model': 'BendRegOCT',
                       'Feature Name': 'EQPDABaseline',
                       'Model Name Override': 'BendOCT',
                       'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive'],
                                           'EQP Initial Cuts-Disaggregate Alpha': [True]},
                       'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

# Keep H variant colouring consistent even when only a subset are being plotted
ColourMap = ('EQP Initial Cuts-H Variant', {'Basic':'#ffa90e',
                                            'Chain':'#bd1f01',
                                            'Recursive':'#94a4a2'})

LineStyleMap = ('EQP Initial Cuts-Disaggregate Alpha', {True: '--',
                                                        False: '-'})

models = [BendRegOCT_Model,
          EQPBaseline_Model,
          EQPDABaseline_Model]