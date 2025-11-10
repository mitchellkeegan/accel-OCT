import os
import csv

from gurobipy import GRB
import pandas as pd


# Provides shorthands for wordy designations
# Useful for plotting and constructing filenames
shorthand_dict = {'Encoding Scheme': 'ES',
                  'Quantile Buckets': 'QB',
                  'Quantile Thresholds': 'QT',
                  'Features Removed': 'FR',
                  'Disaggregate': 'DA',
                  'True': 'T',
                  'False': 'F',
                  'Disaggregate Alpha': 'DA',
                  'EQP Basic': 'EQPB',
                  'EQP Chain': 'EQPC',
                  'Warmstart': 'WS',
                  'Polish Warmstart': 'PolWS',
                  'Compress Data': 'CD',
                  'Solution Polishing': 'SP',
                  'Enabled': 'ON',
                  'Path Bound Cutting Planes': 'PBCP',
                  'Bound Negative Samples': 'BNS',
                  'Bound Structure': 'BSt',
                  'Subproblem LP': 'SPLP',
                  'Callback Subproblem LP': 'cbSPLP',
                  'Subproblem Dual Inspection': 'SPDI',
                  'Callback Subproblem Dual Inspection': 'cbSPDI',
                  'EQP Basic Grouped': 'EQPBG',
                  'Version': 'Vers',
                  'Blocking': 'B',
                  'Permissive': 'P',
                  'Group Selection': 'GS',
                  'Group Variant': 'GV',
                  'Alpha Version': 'AV',
                  'Basic': 'Ba',
                  'Recursive': 'Re',
                  'Chain': 'Ch',
                  'Cut Set Initial Cuts': 'icCS',
                  'Solution Method': 'SM',
                  'Dual Inspection': 'DI',
                  'Cut Set Callback': 'cbCS',
                  'Cut Type': 'CT',
                  'Check Validity': 'CV',
                  'Endpoint Only': 'EO',
                  'Minimum Node Support': 'MNS',
                  'EQP Initial Cuts': 'EQP',
                  'H Variant': 'HV',
                  'Enhanced Cuts': 'EC'}

def get_shorthand(name):
    """Wrapper for shorthand_dict which better handle booleans"""
    if isinstance(name, bool):
        return 'T' if name else 'F'
    else:
        return shorthand_dict.get(name, name)

# Maps error codes to error information
error_code_to_error = {0: 'Model Solve Failed',
                       1: 'Gurobi Unable to Find a Solution',
                       2: 'Solution Returned by Gurobi is Invalid',
                       10: 'Callback Settings Invalid',
                       11: 'Initial Cut Settings Invalid',
                       12: 'No Useful Callback or Initial Cut Settings',
                       13: 'Invalid Log Message Raised',
                       20: 'OCT Model Fit Returned None (Status Unknown)',
                       21: 'Exception Raised While Building Model',
                       22: 'Exception Raised by Callback Subroutine',
                       23: 'Gurobi Exception While Creating Gurobi Model',
                       24: 'Unknown Exception Creating Gurobi Model',
                       30: 'Backup Result Save Failed',
                       31: 'Unknown Format for Results File',
                       99: 'Generic Error',
                       100: 'Model Solve terminated by user input',
                       110: 'Callback Settings Not Useful',
                       111: 'Initial Cut Settings Not Useful',
                       112: 'Warmstart Requested but not Defined by User',
                       120: 'Solution Completer Warning',
                       130: 'Filelock Package Not Available',
                       131: 'Results File Locked',
                       132: 'Cannot Write to Results File',
                       133: 'Some Relaxation Statistics not logged by callback generator',
                       134: 'Log directory filepath is too long',
                       139: 'Generic Logging Warning',
                       140: 'Warmstart unable to return valid solution',
                       150: 'vars_to_readable method on OCT class not implemented',
                       151: 'Error in user defined vars_to_readable method on OCT class',
                       152: 'summarise_tree_info method on OCT class not implemented',
                       153: 'Error in user defined summarise_tree_info method on OCT class',
                       154: 'save_model_output method on OCT class not implemented',
                       155: 'Error in user defined save_model_output method on OCT class',
                       156: '_check_output_validity method on OCT class not implemented',
                       157: 'Error in user defined _check_output_validity method on OCT class',
                       190: 'Result successfully saved to .csv file',
                       199: 'Generic Warning'}

def log_error(error_code, notes=None):
    """

    Args:
        error_code:
        notes:

    Returns:

    """
    error_type = error_code_to_error.get(error_code, 'Unknown Error')

    error_category = 'PROGRAM ERROR' if error_code < 100 else 'PROGRAM WARNING'

    error_messages = [error_category,
                      str(error_code),
                      error_type]

    if notes is not None:
        if isinstance(notes,list):
            error_messages += notes
        else:
            error_messages.append(notes)

    log_message = ':'.join(error_messages)

    print(log_message)



class logger(object):
    """Logger class

    Basic logger which allows information to be easily logged to both the log file and the console. General idea is to
    pass sys.stdout into the logger, and then redirect sys.stdout to the logger

    Example Usage:
        console = sys.stdout
        my_logger = logger(console=console)
        LogFile = os.path.join(log_dir, 'Logfile.txt')
        my_logger.SetLogFile(LogFile)
        sys.stdout = my_logger

    """
    def __init__(self,console,mode=1):
        self.terminal = console
        self.mode = mode

    def SetLogFile(self,LogFile):
        # Set buffering=1 so that the buffer is flushed after every line
        # This ensures that when running on HPC logs are retained if PBS kills the job (E.g. due to memory limits)
        self.log = open(LogFile,mode="w",buffering=1)

    def CloseLogFile(self):
        self.log.close()

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()   # On HPC this should ensure that stdout appears in the output logs even when PBS kills the job

    def flush(self):
        pass

def _logged_results(OCTModel):
    """

    Args:
        OCTModel (OCT): Instance of OCT class which contains an optimised gurobi model

    Returns:
        Returns the dictionary logged_results which contains column names and row values
    """


    model = OCTModel.get_gurobi_model()

    # Cannot continue if gurobi model could not be found or is not trained
    if model is None:
        return

    print('\n' + '-' * 10)

    status = model.Status

    if status == 2:
        print(f'Gurobi found optimal solution with objective {model.ObjVal}')
    else:
        if model.ModelSense == 1:
            print(f'Gurobi found feasible solution with objective {model.ObjVal}, lower bound {model.ObjBound:.2f}, and gap {100 * model.MIPGap:.1f}%')
        elif model.ModelSense == -1:
            print(f'Gurobi found feasible solution with objective {model.ObjVal}, upper bound {model.ObjBound:.2f}, and gap {100 * model.MIPGap:.1f}%')
        else:
            raise Exception('Model does not have a sense??')

    print(f'Total Solve Time: {model.runTime:.1f}s' + (' (TIME LIMIT)' if (status == GRB.TIME_LIMIT) else ''))

    logged_results = {}

    # Sources that have stat logs. Call get_stats_log for each which parses stats dictionaries and formats for logging
    sources = [OCTModel, OCTModel.callback_generator, OCTModel.cut_manager]

    for source in sources:

        # For each source get a string to be print to the log and values to be written to file stored in a dict
        log_string, log_dict = source.get_stats_log()

        if log_string is not None:
            print(log_string)

        logged_results |= log_dict

    try:
        if None in OCTModel.callback_generator.relaxation_statistics.values():
            log_error(133)

        for k,v in OCTModel.callback_generator.relaxation_statistics.items():
            logged_results[k] = v
    except:
        pass

    return logged_results

def _save_to_csv(results_to_log, save_file):

    new_file = not os.path.exists(save_file)

    with open(save_file, 'a', newline='') as f:
        csv_writer = csv.writer(f)

        columns, row = [], []

        for k,v in results_to_log.items():
            columns.append(k)
            row.append(v)

        if new_file:
            csv_writer.writerow(columns)
        csv_writer.writerow(row)

    log_error(190)

def _write_results_to_file(OCTModel, results_to_log, log_dir, save_format):
    """Helper function which handles logging results to file

    The function does the following:
        Fills out the results_to_log dictionary with any other information required from OCTModel
        Creates the save file
        Calls the appropriate subroutine to save data in results_to_log based on save_format argument

    If the filelock package is available it uses a locking mechanism on the results file to prevent potential
    blocking of the results file when multiple python processes are running simultaneously. If the program fails
    to write to the results file for any reason it attempts to write to a backup .csv file in the log directory

    Args:
        OCTModel: Instance of OCT class
        results_to_log (dict): Dictionary where keys are column names and keys are associated values
        log_dir: Directory where results should be saved to
        save_format: File format for save
    """

    model = OCTModel.get_gurobi_model()
    data = model._data

    F = data['F']
    K = data['K']
    cat2bin = data['Categorical Feature Map']
    num2bin = data['Numerical Feature Map']

    results_to_log['Model'] = OCTModel.model_type
    results_to_log['Begin Opt Time'] = model._opt_start_time

    # Get statistics relating to dataset
    results_to_log['Dataset'] = data['name']
    results_to_log['Encoding'] = data['encoding']
    results_to_log['Buckets'] = data['number buckets']
    results_to_log['|F|'] = len(cat2bin) + len(num2bin)
    results_to_log['|F^C|'] = len(cat2bin)
    results_to_log['|F^N|'] = len(num2bin)
    results_to_log['|F^B|'] = len(F)
    results_to_log['|K|'] = len(K)

    # Get statistic relating to the model solve
    results_to_log['Model Status'] = model.Status
    results_to_log['Objective'] = model.ObjVal
    results_to_log['Bound'] = model.ObjBound
    results_to_log['Gap'] = model.MIPGap
    results_to_log['Solve Time'] = model.Runtime
    results_to_log['Node Count'] = model.NodeCount
    results_to_log['Fingerprint'] = model.Fingerprint
    results_to_log['Memory Usage'] = model.MaxMemUsed

    # Add optimisation parameters
    ignored_opt_params = ['Base Directory', 'Use Baseline', 'Callback', 'Initial Cuts', 'Debug Mode']
    for param_name, param_val in OCTModel.opt_params.items():
        if param_name not in ignored_opt_params:
            results_to_log[param_name] = param_val

    # Add Gurobi parameters
    ignored_gurobi_params = ['LogToConsole', 'LogToFile']
    for param_name, param_val in OCTModel.gurobi_params.items():
        if param_name not in ignored_gurobi_params:
            results_to_log[param_name] = param_val

    # Add Callback parameters
    for subr in OCTModel.callback_generator.subroutines:

        subr_name = subr.name

        for setting_name, setting_val in subr.opts.items():
            results_to_log[f'{subr_name}-{setting_name}'] = setting_val

    # Add Initial Cut parameters
    for cut in OCTModel.cut_manager.cuts:

        cut_name = cut.name

        for setting_name, setting_val in cut.opts.items():
            results_to_log[f'{cut_name}-{setting_name}'] = setting_val

    # results file should live two levels up from the log directory
    experiment_dir = os.path.dirname(os.path.dirname(log_dir))

    if save_format == 'csv':
        filename = f'{OCTModel.opt_params['Results Directory']}.csv'
        save_file = os.path.join(experiment_dir,filename)

        save_successful = True

        try:
            from filelock import Timeout, FileLock

            lock_path = save_file + '.lock'
            lock = FileLock(lock_path, timeout=60, thread_local=False)

            try:
                with lock:
                    _save_to_csv(results_to_log, save_file)

            except Timeout:
                log_error(131, f'{save_file} results file is locked, attempting save to backup in log directory')
                save_successful = False

            except Exception as err:
                log_error(132,f'{type(err)} exception encountered when writing to results file. Attempting save to backup in log directory')
                save_successful = False

        except ImportError:
            log_error(130,'May result in errors if experiments running in multiple processes')
            try:
                _save_to_csv(results_to_log, save_file)
            except:
                log_error(132, f'{type(err)} exception encountered when writing to results file. Attempting save to backup in log directory')
                save_successful = False

        if not save_successful:
            save_file = os.path.join(log_dir, filename)
            try:
                _save_to_csv(results_to_log, save_file)
            except Exception as err:
                log_error(30,f'Failed to write results to backup file with exception {type(err).__name__}')

    else:
        log_error(31,f'.{save_format} is not a supported file format')

def save_optimisation_results(OCTModel, save_format='csv'):
    """High level function to save information about the results of the optimisation

    Args:
        OCTModel (OCT): Instance of OCT class which contains an optimised gurobi model
        save_format (str): Save format. Currently only .csv files are supported

    """

    # Double check that the model was actually trained
    if not OCTModel._model_trained:
        print(f'Post processing of {OCTModel.model_type} cannot be completed because '
              f'Gurobi was unable to find a feasible solution')
        return

    log_dir = OCTModel._get_log_dir()

    if log_dir is not None:
        results_to_log = _logged_results(OCTModel)
        _write_results_to_file(OCTModel, results_to_log, log_dir, save_format)
