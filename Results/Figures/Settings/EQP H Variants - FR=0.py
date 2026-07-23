eqp_features_removed = 0

fig_title = f'Figure - EQP H Variants - {eqp_features_removed} Features Removed'

EQPBaseline_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendersOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic'],
                                         'EQP Initial Cuts-Filter Dominated': [False]},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

ColourMap = ('EQP Initial Cuts-H Variant', {'Basic':'#ffa90e',
                                            'Chain':'#bd1f01',
                                            'Recursive':'#94a4a2'})

legend_kwargs = {'fontsize': 11,
                 'ncol': 2,
                 'loc': 'lower center',
                 'bbox_to_anchor': (0.5125, 0.88)}

models = [BendRegOCT_Model,
          EQPBaseline_Model]