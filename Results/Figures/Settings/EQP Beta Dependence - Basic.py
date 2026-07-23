eqp_features_removed = 2

fig_title = f'Figure - EQP Beta Dependence - Basic - {eqp_features_removed} Features Removed'

EQPBaseline_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendersOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic'],
                                         'EQP Initial Cuts-Filter Dominated': [False, True]},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

ColourMap = ('EQP Initial Cuts-Filter Dominated', {False: '#ffa90e',
                                                   True: '#bd1f01'})

legend_kwargs = {'fontsize': 11,
                 'ncol': 3,
                 'loc': 'lower center',
                 'bbox_to_anchor': (0.5125, 0.88)}

# empty_legend_entries = [1]
# legend_ordering = [0, 1, 2, 4, 3, 5]


models = [BendRegOCT_Model,
          EQPBaseline_Model]