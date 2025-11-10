from src.models.BendOCT import BendOCT
from src.models.FlowRegOCT import FlowRegOCT
from src.models.BendRegOCT import BendRegOCT
from src.utils.data import load_instance, load_toy_instance, load_strengthened_cuts_pathological_instance
from src.utils.logging import save_optimisation_results, log_error

model = 'BendRegOCT'

callback_settings = {}
initial_cut_settings = {}

callback_settings = {'Benders Cuts': {'Enhanced Cuts': True,
                                      'Relax w': True}}

# wdbc_QB_5 d=3,l=0.02 0.859660
# thoracic_QB_5 d=3,l=0.02 0.836667

opt_params = {'Warmstart': False,
              'Polish Warmstart': False,
              'depth': 3,
              'Use Baseline': True,
              'Compare Relaxations': False,
              'lambda': 0.02,
              'Debug Mode': True,
              'Callback': callback_settings,
              'Initial Cuts': initial_cut_settings,
              'Results Directory': 'EnhancedCutsDebug'}

gurobi_params = {'LogToConsole': 1}

dataset = 'thoracic'
#
data = load_instance(dataset,
                     num_buckets=5,
                     encoding_scheme='Quantile Buckets')

# 0.7947826086956523

# data = load_strengthened_cuts_pathological_instance()

if model == 'BendOCT':
    Model = BendOCT(opt_params, gurobi_params)
elif model == 'FlowRegOCT':
    Model = FlowRegOCT(opt_params, gurobi_params)
elif model == 'BendRegOCT':
    Model = BendRegOCT(opt_params, gurobi_params)
else:
    Model = None

fit_successful = Model.fit(data)

if fit_successful is None:
    log_error(5)
elif fit_successful:
    Model.post_process_model()
    save_optimisation_results(Model)
    Model.cleanup_GRB_environment()