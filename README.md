# Acceleration Techniques for Learning Optimal Classification Trees

This repository contains code used to produce results for the paper titled "Acceleration Techniques for Learning Optimal Classification Trees with
Integer Programming"

## Directory Structure
Raw datasets and binary encodings of raw data are in the `Datasets/` directory, along with documentation. Similarly, results and documentations are available in the `Results/` directory. 
The `Scripts/` directory contains scripts which provide interfaces into the OCT model, `Scripts/ResultsProcessing/` contains a number of scripts for postprocessing of results.
`src/models/` contains base classes for OCT models and definitions of regularised version of BendOCT and FlowOCT (in BendRegOCT.py and FlowRegOCT.py respectively).
`src/utils/` contains a number of utility functions for managing datasets, logging results, error logging, tree based subroutines, and generating EQP sets.

## Basic Model Structure
Every MIP model inherits from the OCT class, which handles the construction of the model, setting up of logging, checking validity of parameters, optimisation, and post optimisation checks. 
At construction it is provided with an initial cut manager and a callback generator which mediate the relationship between the OCT model and the initial cuts and callback subroutines respectively.

Initial cuts are implemented by inheriting from the InitialCut class and overriding the add_cuts method. It also provides methods for checking for valid/useful settings and completing partial solutions
with variables added to the model by the initial cut. Callback subroutines are used for anything which would normally be implemented in a callback function, E.g. cutting planes, primal heuristic, custom logging.
They are implemented by inheriting from the CallbackSubroutine class and overriding the run_subroutine method. it also provides methods for checking for valid/useful settings and modifying the Gurobi model
prior to optimisation.

Details required for implementation are documented in the codebase.

## Model Interfaces
The simplest interface into the model is to directly call the OCT model with dictionaries specifying the model settings, initial cut settings and callback subroutine settings.
Example usages of the dictionary interface are provided below.
We also provide a command line interface in `Scripts/RunOCT.py` which wraps the dictionary interface.

## Optimisation Parameters
Optimisation parameters are passed as a dictionary when initialising the OCT model. Settings are passed in a key-value pairs, the primary settings being described below.

| Parameter        | Available Options         | Default | Description                                                               |
|------------------|---------------------------|---------|---------------------------------------------------------------------------|
| Wamrstart        | {True, False}             | False   | Enable warmstarting the solver                                            |
| Polish Warmstart | {True, False}             | 0       | Enable running solution polishing on the warmstart solution               |
| depth            | {Basic, Chain, Recursive} | Chain   | Maximum depth of decision tree                                            |
| lambda           | {True, False}             | False   | Per leaf penalty on accuracy                                              |
| Use Baseline     | {True, False}             | False   | Enable the group selection variant of the constraint set G                |
| Initial Cuts     | N/A                       | N/A     | Dictionary describing initial cut settings. See below for details         |
| Callback         | N/A                       | N/A     | Dictionary describing callback subroutine settings. See below for details |

## Gurobi Parameters
Gurobi parameters are also passed in as a dictionary when initialising the OCT model. Settings are passed in a key-value pairs, the supported settings are described below.
It is easy add support for any setting in the create_model method on the OCT class. A description of each parameter can be found at https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html.

| Parameter    | Default |
|--------------|---------|
| TimeLimit    | 3600    |
| Threads      | 1       |
| MIPGap       | 0       |
| MIPFocus     | 0       |
| Heuristics   | 0.05    |
| NodeMethod   | -1      |
| Method       | -1      |
| Seed         | 0       |
| LogToConsole | 0       |
| LogToFile    | True    |
| NodeLimit    | inf     |

## Initial Cut Options
Initial cuts are initialised in the optimisation parameters under the 'Initial Cuts' key. Each key must correspond to the 
name of an initial cut (this must correspond to the name attribute of the initial cut class).
The associated value is a dictionary of settings.

### EQP Initial Cuts
| Setting Name       | Available Settings        | Default | Description                                                                                         |
|--------------------|---------------------------|---------|-----------------------------------------------------------------------------------------------------|
| Enabled            | {True, False}             | False   | Enable the initial cuts                                                                             |
| Features Removed   | {0, 1, 2}                 | 0       | Maximum number of features allowed to be in the split set for modelled EQP sets                     |
| H Variant          | {Basic, Chain, Recursive} | Chain   | Variant of the constraint set H. Must set Features Removed > 0 when using chain and recursive variants |
| Disaggregate Alpha | {True, False}             | False   | Enable constraint disaggregation for Chain and Recursive variants of H                              |
| Group Selection    | {True, False}             | False   | Enable the group selection variant of the constraint set G                                          |

## Callback Subroutine Options
Callback subroutines are initialised in the optimisation parameters under the 'Callback' key. Each key must correspond to the 
name of callback subroutine (this must correspond to the name attribute of the callback subroutine class).
The associated value is a dictionary of settings.

### Benders Cuts
| Setting Name  | Available Settings | Default | Description                                                             |
|---------------|--------------------|---------|-------------------------------------------------------------------------|
| Enabled       | {True, False}      | True    | Enable the Benders cuts. Note that disabling them will break the model  |
| Enhanced Cuts | {True, False}      | False   | Enable strengthening of the Benders cuts                                |

### Solution Polishing
| Setting Name   | Available Settings | Default | Description                                                                                          |
|----------------|--------------------|---------|------------------------------------------------------------------------------------------------------|
| Enabled        | {True, False}      | False   | Enable solution polishing.                                                                           |
| Check Validity | {True, False}      | False   | Enable checking the validity of the MP solution w.r.t the full model before running primal heuristic |

### Path Bound Cutting Planes 
| Setting Name           | Available Settings | Default | Description                                                                  |
|------------------------|--------------------|---------|------------------------------------------------------------------------------|
| Enabled                | {True, False}      | False   | Enable the cutting planes                                                    |
| Endpoint Only          | {True, False}      | False   | Only run D2S subroutine and add cutting planes at endpoints of integral paths |
| Cut Type               | {Lazy, User}       | Lazy    | Type of callback cut user, either lazy cuts (cbLazy) or user cuts (cbCut)    |
| Bound Negative Samples | {True, False}      | False   | Modify cuts to force sample misclassified in subtree to zero                 |
| Bound Structure        | {True, False}      | False   | Add additional cuts to specify the structure of the subtree                  |

## Example Usage of Dictionary Interface
### Equivalent Point Initial Cuts

```python
from src.models.BendRegOCT import BendRegOCT
from src.utils.data import load_instance
from src.utils.logging import save_optimisation_results

callback_settings = {}
initial_cut_settings = {'EQP Initial Cuts': {'Enabled': True,
                                             'Features Removed': 2,
                                             'H Variant': 'Recursive',
                                             'Group Selection': True
                                             }
                        }

opt_params = {'depth': 4,
              'lambda': 0.006,
              'Callback': callback_settings,
              'Initial Cuts': initial_cut_settings,
              'Results Directory': 'InitialCutExample'}

gurobi_params = {'LogToConsole': 1}

data = load_instance('car_evaluation')
Model = BendRegOCT(opt_params, gurobi_params)

fit_successful = Model.fit(data)

if fit_successful:
    Model.post_process_model()
    save_optimisation_results(Model)
```

### Multiple Callback Subroutines on Numerical Dataset

```python
from src.models.BendRegOCT import BendRegOCT
from src.utils.data import load_instance
from src.utils.logging import save_optimisation_results

callback_settings = {'Path Bound Cutting Planes': {'Enabled': True,
                                                   'Bound Negative Samples': True,
                                                   'Bound Structure': False
                                                   },
                     'Solution Polishing': {'Enabled': True}
                     }

initial_cut_settings = {}

opt_params = {'depth': 3,
              'lambda': 0.006,
              'Callback': callback_settings,
              'Initial Cuts': initial_cut_settings,
              'Results Directory': 'MultiCallbackExample'}

gurobi_params = {'LogToConsole': 1,
                 'MIPGap': 0.1}

data = load_instance('thoracic',
                     encoding_scheme='Quantile Thresholds',
                     num_buckets=5)

Model = BendRegOCT(opt_params, gurobi_params)

fit_successful = Model.fit(data)

if fit_successful:
    Model.post_process_model()
    save_optimisation_results(Model)
```