# Note: Need to set eqp_features_removed \in [0,1,2] to get paper plots
fig_title = f'Equivalent Point Inequalities H Variants - {eqp_features_removed} Features Removed'


EQPBaseline_Model = {'Base Model': 'BendRegOCT',
                     'Model Name Override': 'BendOCT',
                     'Feature Name': 'EQPBaseline',
                     'Hyperparameters': {'EQP Initial Cuts-H Variant': ['Basic', 'Chain', 'Recursive']},
                     'Silent Filters': {'EQP Initial Cuts-Features Removed': [eqp_features_removed]}}

models = [BendRegOCT_Model,
          EQPBaseline_Model]
