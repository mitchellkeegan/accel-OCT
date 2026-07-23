eqp_features_removed = 1

fig_title = f'Figure - EQP Group Selection - {eqp_features_removed} Features Removed'

EQPBasic_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendersOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic']},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

EQPGSBasic_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendersOCT',
               'Feature Name': 'EQPGS',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic'],
                                   'EQP Initial Cuts-Group Selection': [True]},
               'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}


EQPDA_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendersOCT',
               'Feature Name': 'EQPDA',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive']},
               'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed],
                                  'EQP Initial Cuts-Disaggregate Alpha': [True]}}


EQPGS_Model = {'Base Model': 'BendRegOCT',
               'Model Name Override': 'BendersOCT',
               'Feature Name': 'EQPGS',
               'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Chain', 'Recursive'],
                                   'EQP Initial Cuts-Group Selection': [True]},
               'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

# Keep H variant colouring consistent even when only a subset are being plotted
ColourMap = ('EQP Initial Cuts-H Variant', {'Basic':'#ffa90e',
                                            'Chain':'#bd1f01',
                                            'Recursive':'#94a4a2'})

LineStyleMap = ('EQP Initial Cuts-Group Selection', {True: '--',
                                                     False: '-'})

legend_kwargs = {'fontsize': 11,
                 'ncol': 4,
                 'loc': 'lower center',
                 'columnspacing': 0.5,
                 'bbox_to_anchor': (0.5125, 0.88)}

legend_ordering = [0,1,2,3,4,6,5,7]
empty_legend_entries = [1]

models = [BendRegOCT_Model,
          EQPBasic_Model,
          EQPGSBasic_Model,
          EQPDA_Model,
          EQPGS_Model]