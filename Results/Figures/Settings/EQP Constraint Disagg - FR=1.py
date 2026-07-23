eqp_features_removed = 1

fig_title = f'Figure - EQP Constraint Disagg - {eqp_features_removed} Features Removed'

EQPBaseline_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendersOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive']},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

EQPDA_Model = {'Base Model': 'BendRegOCT',
                       'Feature Name': 'EQPDA',
                       'Model Name Override': 'BendersOCT',
                       'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive'],
                                           'EQP Initial Cuts-Disaggregate Alpha': [True]},
                       'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

# Keep H variant colouring consistent even when only a subset are being plotted
ColourMap = ('EQP Initial Cuts-H Variant', {'Basic':'#ffa90e',
                                            'Chain':'#bd1f01',
                                            'Recursive':'#94a4a2'})

LineStyleMap = ('EQP Initial Cuts-Disaggregate Alpha', {True: '--',
                                                        False: '-'})

legend_kwargs = {'fontsize': 11,
                 'ncol': 3,
                 'loc': 'lower center',
                 'bbox_to_anchor': (0.5125, 0.88)}

empty_legend_entries = [1]

models = [BendRegOCT_Model,
          EQPBaseline_Model,
          EQPDA_Model]