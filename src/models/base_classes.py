from abc import ABC, abstractmethod
import traceback
import os
import sys

import pickle
import json
import time
from datetime import datetime

from gurobipy import *

from src.utils.trees import Tree
from src.utils.logging import logger, log_error, shorthand_dict, get_shorthand

def _merge_settings(base_default_settings, default_settings, user_settings):
    """Merge dictionaries of setting together in order of precedence user_settings > default_settings > base_default_settings

    Args:
        base_default_settings (dict): Default settings as per base class
        default_settings (dict): Default settings for subclassed model
        user_settings (dict): User settings

    Returns:
        settings (dict): Merged settings of the arguments
    """

    settings = base_default_settings

    if default_settings is not None:
        settings = base_default_settings | default_settings

    if user_settings is not None:
        settings |= user_settings

    return settings

class InitialCut(ABC):
    """Base class for initial cuts

    This is a base class for implementing initial cuts. A subclass is provided to an instance of the initial cut
    manager class. Subclasses are not intended to be instantiated by the user, the initial cut manager instantiates the subclasses,
    handle the checking of the settings, adding initial cuts, and collecting run statistics

    At a high level provides the following methods:
        valid_settings: Checks if the settings provided by the user are valid for the class of initial cuts
        useful_settings: Checks if the settings provided by the user provide a benefit over the baseline
        add_cuts: Add initial cuts to the provided Gurobi model object
        gen_CompleteSolution: If the initial cut defines variables this function can fill in those variables from an incomplete solution
        update_cut_stats: Helper function called from add_cuts to track statistics about the initial cuts (E.g. runtime)


    At a minimum a subclass must implement the following:
        name: User must set the self.name attribute. This is how the initial cut will be identified.
        useful_settings: By default returns that the settings are not useful. User must override base class to specify that setting are useful
        add_cuts: To define the initial cuts

    NOTE: Must set self.name as class attribute. Without it the initial cut manager cannot associate the initial cut settings to the instance.

    gen_CompleteSolution is optional but recommended if initial cuts introduce many auxiliary variables or integer variables
    valid_settings is optional and should be used to prevent the optimisation from proceeding if the initial cut settings
    are invalid or if the dataset is not compatible with the cuts.

    """

    default_name = 'Generic Cut'

    def __init__(self, default_settings=None, user_opts=None):

        self.stats = {'Num': 0,
                      'Time': 0}

        base_default_settings = {'Enabled': False}

        try:
            self.name
        except AttributeError:
            self.name = self.default_name

        self.opts = _merge_settings(base_default_settings, default_settings, user_opts)

    def valid_settings(self, model_opts=None, data=None):
        """Default method for evaluating whether the settings for an initial cut are valid

        Overwrite if some setting combinations are invalid. By default, assumes settings are valid.
        Cut settings should be accessed by self.opts

        For example, some cuts may rely on some structure in the data which is not present in all datasets

        Args:
            model_opts (dict): opt_params or the
            data (dict): Dictionary with dataset information

        Returns:
            Returns a tuple (settings_valid, log_message). settings_valid and True is the settings are valid and
            False otherwise. log_message is None by default, if settings_valid is False then any string in log_message
            will be printed to the log

        """

        settings_valid = True
        log_message = None

        return settings_valid, log_message

    def useful_settings(self, model_opts=None, data=None):
        """Default method for evaluating whether the settings for an initial cut are useful

        Overwrite if some setting combinations are not useful. By default, assumes settings not useful.
        Defines when there is some benefit over the baseline model

        For example, some cuts may rely on some structure in the data which is not present in all datasets

        Args:
            model_opts (dict): opt_params or the
            data (dict): Dictionary with dataset information

        Returns:
            Returns a tuple (settings_useful, log_message). settings_useful and True is the settings are valid and
            False otherwise. log_message is None by default, if settings_useful is False then any string in log_message
            will be printed to the log

        """

        settings_useful = False
        log_message = None

        return settings_useful, log_message

    def gen_CompleteSolution(self):
        """Method to generate solution completer for initial cut

        If an initial cut which introduces new auxiliary variables, any warmstart or primal heuristic is unlikely
        to produce these auxiliary variables. By default, Gurobi will try to complete these partial solutions which
        can be computationally expensive, particularly if the auxiliary variables are integral.

        This method constructs and returns a function CompleteSolution() which is called whenever a partial solution
        must be completed before being suggested as a new solution to Gurobi.

        See docstring for nested function for more details

        Returns:
            Returns function CompleteSolution which can be called to complete partial solutions
        """

        def CompleteSolution(model, soln, where):
            """Complete partial solution provided in soln

            The function should fill in any auxiliary variables introduced by the initial cut class.
            Has access to self to can access initial cut options and any data stored in self while adding cuts

            There are two methods of completing solutions based on the 'where' input. If 'where' == 'Callback' then
            solution should be completed by using model.cbSetSolution(Var,Var_soln). If 'where' == 'Warm Start' then
            use Var.Start = Var_soln. For example given variable beta and values betaV for beta inferred from partial solution:

            if where == 'Callback':
                model.cbSetSolution(beta,betaV)
            elif where == 'Warm Start':
                for k,v in betaV.items():
                    beta[k].Start = v

            Args:
                model (grbModel): Gurobi model object
                soln (dict): Partial solution
                where (str): Location where function is being called from. Must be 'Callback' or 'Warm Start'

            """
            pass

        return CompleteSolution

    def update_cut_stats(self, num_added, time_added, *args):
        """Helper function to keep track of statistics for initial cut generation

        Intended to be called from callback function to keep track of statistics
        For example, how many Benders cuts have been added and in what amount of time

        Args:
            num_added (int): Number associated with statistic. Additive in nature
            time_added (float): Time taken in generating
            *args: Extra tuples which can be added to stats with format (key, value)

        """

        for arg in args:
            k, v = arg
            if k not in self.stats:
                self.stats[k] = 0
            self.stats[k] += v

        self.stats['Num'] += num_added
        self.stats['Time'] += time_added

    @abstractmethod
    def add_cuts(self, model, data):
        pass

class CallbackSubroutine(ABC):
    """Base class for callback subroutines

    This is a base class for implementing callback subroutines. This is the preferred method for implemented anything which
    must be done during the callback, E.g. cutting planes or primal heuristics. A subclass is provided to an instance of
    the callback generator class. Subclasses are not intended to be instantiated by the user, the callback generator
    instantiates the subclasses, handle the checking of the settings, calling subroutines, and collecting run statistics

    At a high level provides the following methods:
        valid_settings: Checks if the settings provided by the user are valid for the subroutine
        useful_settings: Checks if the settings provided by the user provide a benefit over the baseline
        run_subroutine: Run the subroutine
        update_model: Called before optimisation to change any model settings required for the subroutine to operatore properly
        update_cut_stats: Helper function called from add_cuts to track statistics about the subroutine (E.g. runtime)

    At a minimum a subclass must implement the following:
        name: User most set the self.name attribute. This is how the subroutine will be identified
        useful_settings: By default returns that the settings are not useful. User must override base class
        run_subroutine: To define the subroutine

    NOTE: Must set self.name. Without it the callback generator cannot associate the subroutine settings to the instance.

    valid_settings is optional and should be used to prevent the optimisation from proceeding if the subroutine settings
    are invalid or if the dataset is not compatible with the subroutine. Can also optionally set self.priority to modify
    the order in which subroutines are run. High priority subroutines will run earlier, default priority is 10

    """

    default_name = 'Generic Callback Subroutine'
    default_priority = 10

    def __init__(self, default_settings=None, user_opts=None):



        self.stats = {'Num': 0,
                      'Time': 0}

        base_default_settings = {'Enabled': False}

        try:
            self.priority
        except AttributeError:
            self.priority = self.default_priority

        try:
            self.name
        except AttributeError:
            self.name = self.default_name

        self.opts = _merge_settings(base_default_settings, default_settings, user_opts)

    def valid_settings(self, model_opts=None, data=None):
        """Return True if settings in self.opts are valid, otherwise returns false

        Overwrite if some setting combinations are invalid. By default, assumes settings are valid.
        Callback settings should be accessed by self.opts

        For example, some primal heuristic may not work properly on all instances and
        the model training prodecure should be halted.

        Args:
            model_opts (dict): opt_params or the
            data (dict): Dictionary with dataset information

        Returns:
            Returns a tuple (settings_valid, log_message). settings_valid is True if the settings are valid and
            False otherwise. log_message is None by default, if settings_valid is False then any string in log_message
            will be printed to the log

        """

        settings_valid = True
        log_message = None

        return settings_valid, log_message

    def useful_settings(self, model_opts=None, data=None):
        """Return True if settings in self.opts are useful, otherwise returns false

        Overwrite if some setting combinations are not useful. By default, assumes settings not useful.
        Defines when there is some benefit over the baseline model

        Args:
            model_opts (dict): opt_params or the
            data (dict): Dictionary with dataset information

        Returns:
            Returns a tuple (settings_valid, log_message). settings_valid is True if the settings are valid and
            False otherwise. log_message is None by default, if useful_settings is False then any string in log_message
            will be printed to the log
        """
        settings_useful = False
        log_message = None

        return settings_useful, log_message

    def update_subroutine_stats(self, num_added, time_added, *args):
        """Helper function to keep track of statistics for callback subroutines

        Intended to be called from callback function to keep track of statistics
        For example, how many Benders cuts have been added and in what amount of time

        Args:
            num_added (int): Number associated with statistic. Additive in nature
            time_added (float): Time taken in generating
            *args: Extra tuples which can be added to stats with format (key, value)

        """

        for arg in args:
            k, v = arg
            if k not in self.stats:
                self.stats[k] = 0
            self.stats[k] += v

        self.stats['Num'] += num_added
        self.stats['Time'] += time_added

    def update_model(self,model):
        """
        Overwrite to change settings or add useful information to model
        For example:
            Create a cache for storing subproblem solutions
            Set the LazyConstraints or PreCrush parameters

        Args:
            model (grbModel): Gurobi model to be updated

        """

        pass

    @abstractmethod
    def run_subroutine(self, model, where, callback_generator):
        """Run the subroutine within the callback

        This method defines the subroutine action within the callback function

        Args:
            model (grbModel)
            where (int): Callback code defining where the callback is being called from
            callback_generator (GenCallback): Instance of the callback generator from which the subroutine is called

        """

        pass

class GenCallback(ABC):
    """Base class for callback generators

    Callback generators mediate between the OCT class and the callback subroutines. Subclasses must call super.__init__ with
    the available subroutines (the classes) and user provided settings. The callback generator will instantiate each subroutine
    class with the user callback settings, and when requested by the OCT class return a callback function which implements
    the subroutines.

    At a high level provides the following methods:
        add_log_dir: Makes log directory available to subroutines. Useful if user wants to write debug information from subroutine to file
        update_model: Calls update_model for each subroutine
        valid_settings: Calls the valid_settings method for each subroutine and passes information back to OCT class
        settings_useful: Calls the settings_useful method for each subroutine and passes information back to OCT class
        get_stats_log: Collects logged statistics from subroutines from printing to log and logging in results file
        gen_callback: Constructs the callback function. Also implements functionality to log root node relaxation statistics

    NOTE: The subroutines are provided with the callback generator instance. This gives them access to shared caches and
    any helper methods defined on the callback generator

    In practice a callback generator should never be instantiated by the user, instead it is managed by the OCT class.

    Minimal example subclass:

        Class MyCallbackGenerator(GenCallback):

            name = 'My Callback Generator'

            def __init__(self, callback_settings):

                # List of subroutine classes (not instances)
                available_subroutines = [subroutine1, subroutine2]

                super().__init__(available_subroutines, callback_settings)

            def get_stats_logs(self):

                log_lines = ['\nCallback Statistics:\n']
                logged_results = {}

                for subroutine in self.subroutines:
                    subr_name, stats = subroutine.name, subroutine.stats
                    num_cuts, cut_time = stats['Num'] stats['Time']

                if subroutine.opts['Enabled']:
                    log_lines.append(f'{subr_name} - Added {num_cuts} cuts in {cut_time:.2f}s\n')

                logged_results[f'{subr_name} - Cuts Added'] = num_cuts
                logged_results[f'{subr_name} - Time'] = cut_time

            def my_shared_method(self, arg1, args2):
                # Implement some method which multiple subroutines can access as a helper function

    Attributes:
        callback_cache: Dictionary with temporary and persistent caches. The temporary cache is wiped every time the callback
                        function is called by Gurobi. The persistent cache is never wiped
        subroutines: List of subroutines provided to callback generator

    """


    def __init__(self, available_subroutines, user_callback_settings):

        self.name = 'Generic Callback'
        self.subroutines = []
        self.callback_exception_raised = False

        self.callback_cache = {'Temporary': {},
                               'Persistent': {}}

        callback_settings = {}

        for subr in available_subroutines:

            # Handle exception if the subroutine name has not been set as a class method
            # in the subclass. In this cause inherit the default name
            try:
                subr_name = subr.name
            except AttributeError:
                subr_name = subr.default_name

            subr_user_opts = user_callback_settings.get(subr_name,{})
            subr_instance = subr(subr_user_opts)

            self.subroutines.append(subr_instance)
            callback_settings[subr_name] = subr_instance.opts

        self.subroutines.sort(key = lambda x : x.priority, reverse=True)

        self.callback_settings = callback_settings

    def add_log_dir(self, log_dir):
        """Makes a log directory available to the callback subroutines
        """

        for subroutine in self.subroutines:
            subroutine.log_dir = log_dir

    def update_model(self,model):
        """Calls the update_model method for all callback subroutines

        Args:
            model (grbModel): Gurobi model to be updated

        """

        for subroutine in self.subroutines:
            if subroutine.opts['Enabled']:
                subroutine.update_model(model)

    def valid_settings(self, model_opts=None, data=None):
        """Checks that the settings for each subroutine are valid

        Args:
            model_opts (dict): opt_params or the
            data (dict): Dictionary with dataset information

        Returns:
            Returns a tuple (all_settings_valid, log_message). all_settings_valid is True if the settings for all subroutines
            are valid and False otherwise. log_messages collates messages from subroutines about invalid settings
        """

        all_settings_valid = True
        log_messages = []

        for subroutine in self.subroutines:
            if subroutine.opts['Enabled']:
                subroutine_valid, log_message = subroutine.valid_settings(model_opts=model_opts, data=data)
                if not subroutine_valid:
                    all_settings_valid = False
                    if log_message is not None:
                        if isinstance(log_message, str):
                            log_messages.append(log_message)

                        elif isinstance(log_message, list):
                            for message in log_message:
                                # Only append strings to avoid an error printing
                                if isinstance(message, str):
                                    log_messages.append(message)
                                else:
                                    log_error(13, notes=f'{subroutine.name} callback subroutine returned a non-string log message from valid_settings method')

                        else:
                            log_error(13, notes=f'{subroutine.name} callback subroutine returned a non-string log message from valid_settings method')

        return all_settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        """Checks if the settings for each subroutine provide a benefit over the baseline

        Args:
            model_opts (dict): opt_params or the
            instance (str): Name of instance (raw dataset name, not the encoded dataset name)

        Returns:
            Returns a tuple (useful_settings_found, log_message). useful_settings_found is True if at least one subroutine
            provides a benefit over the baseline. log_messages collates messages from subroutines
        """

        log_messages = []
        useful_settings_found = False

        for subroutine in self.subroutines:
            if subroutine.opts['Enabled']:
                settings_useful, log_message = subroutine.useful_settings(model_opts=model_opts, data=data)
                if settings_useful:
                    useful_settings_found = True
                else:
                    if log_message is not None:
                        if isinstance(log_message, str):
                            log_messages.append(log_message)

                        elif isinstance(log_message, list):
                            for message in log_message:
                                # Only append strings to avoid an error printing
                                if isinstance(message, str):
                                    log_messages.append(message)
                                else:
                                    log_error(13,
                                              notes=f'{subroutine.name} callback subroutine returned a non-string log message from useful_settings method')

                        else:
                            log_error(13,
                                      notes=f'{subroutine.name} callback subroutine returned a non-string log message from useful_settings method')

        return useful_settings_found, log_messages

    def get_stats_log(self):
        """Helper function which parses self.stats for logging purposes

        Should be overwritten
        Returns a string which can be printed and a dictionary with column entries to be saved to file

        Returns:
            Returns a tuple (log_printout, logged_results)
            log_printout is a string which should print out results concerning heuristic stats (return None if no heuristic used)
            logged_results is a dictionary where the keys are the names of the columns in the results save file
            and values are the corresponding entries for that column
        """

        log_printout = None
        logged_results = {}

        return log_printout, logged_results

    def gen_callback(self):
        """Callback generator for Gurobi model

        Returns:
            Returns a callback function which calls each of the subroutines in
            self.subroutines.
        """

        self.catch_next_msg = False
        self.msg_parser_failed = False

        self.root_node_accessed = False

        # Dictionary for relaxation statistics
        self.relaxation_statistics = {'Initial Root Relaxation - Obj': None,
                                      'Initial Root Relaxation - Time': None,
                                      'Initial Root Relaxation - Work': None,
                                      'Presolve Relaxation - Obj': None,
                                      'Presolve Relaxation - Time': None,
                                      'Presolve Relaxation - Work': None,
                                      'Final Root Relaxation - Obj': None,
                                      'Final Root Relaxation - Time': None,
                                      'Final Root Relaxation - Work': None}

        def callback(model, where):

            # Wipe the temporary callback cache before the next run through the subroutines
            self.callback_cache['Temporary'] = {}

            if where == GRB.Callback.PRESOLVE:
                # TODO: Grab how many columns and rows have been removed by presolve, as well as number of coefficients changed
                pass

            if where == GRB.Callback.MIPNODE:
                if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0 and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                    # If these conditions hold then a relaxation solution is available at the root node

                    if self.root_node_accessed:
                        # Keep track of relaxation statistics. Once Gurobi leaves the root node this will have recorded
                        # the root relaxation statistics
                        self.relaxation_statistics['Final Root Relaxation - Obj'] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        self.relaxation_statistics['Final Root Relaxation - Time'] = model.cbGet(GRB.Callback.RUNTIME)
                        self.relaxation_statistics['Final Root Relaxation - Work'] = model.cbGet(GRB.Callback.WORK)

                    else:
                        # If this is the first time in the root node then store the
                        self.relaxation_statistics['Presolve Relaxation - Obj'] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
                        self.relaxation_statistics['Presolve Relaxation - Time'] = model.cbGet(GRB.Callback.RUNTIME)
                        self.relaxation_statistics['Presolve Relaxation - Work'] = model.cbGet(GRB.Callback.WORK)
                        self.root_node_accessed = True

            for subroutine in self.subroutines:
                if subroutine.opts['Enabled']:
                    try:
                        subroutine.run_subroutine(model, where, self)
                    except Exception as err:
                        print(traceback.format_exc())
                        log_error(22, [f'Unknown Exception {type(err).__name__} in callback subroutine {subroutine.name}',
                                       '\\n'.join(traceback.format_exc().split('\n'))])
                        self.callback_exception_raised = True
                        model.terminate()

            if where == GRB.Callback.MESSAGE:
                if not self.msg_parser_failed:
                    try:
                        msg = model.cbGet(GRB.Callback.MSG_STRING)
                        if self.catch_next_msg:
                            self.relaxation_statistics['Initial Root Relaxation - Obj'] = float(msg.split()[2])

                        elif msg.startswith('Root relaxation'):
                            split_msg = msg.split()

                            if split_msg[2] not in ['presolved:','cutoff,']:
                                self.relaxation_statistics['Initial Root Relaxation - Obj'] = float(split_msg[3][:-1])
                                self.relaxation_statistics['Initial Root Relaxation - Time'] = float(split_msg[6])
                                self.relaxation_statistics['Initial Root Relaxation - Work'] = float(split_msg[8][1:])

                        elif msg.startswith('Barrier solved model'):
                            self.relaxation_statistics['Initial Root Relaxation - Time'] = float(msg.split()[7])
                            self.relaxation_statistics['Initial Root Relaxation - Work'] = float(msg.split()[9][1:])
                    except Exception as err:
                        print(f'{type(err)} exception encountered in callback log parser.')
                        self.msg_parser_failed = True

        return callback

class InitialCutManager(ABC):
    """Base class for callback generators

    Callback generators mediate between the OCT class and the initial cuts. Subclasses must call super.__init__ with
    the available initial cuts (the classes) and user provided settings. The initial cut manager will instantiate each
    initial cut class with the user settings, and when requested by the OCT class will call each initial cut to add cuts
    to the Gurobi model

    At a high level provides the following methods:
        add_log_dir: Makes log directory available to subroutines. Useful if user wants to write debug information from subroutine to file
        get_solution_completers: Constructs a single solution completer function from initial cut solution completers
        valid_settings: Calls the valid_settings method for each initial cut and passes information back to OCT class
        settings_useful: Calls the settings_useful method for each initial cut and passes information back to OCT class
        get_stats_log: Collects logged statistics from initial cuts from printing to log and logging in results file

    In practice an initial cut manager should never be instantiated by the user, instead it is managed by the OCT class.

    Minimal example subclass:

        Class MyInitialCutManager(InitialCutManager):

            name = 'My Initial Cut Manager'

            def __init__(self, cut_settings):

                # List of initial cut classes (not instances)
                available_cuts = [initialcut1, initialcut2]

                super().__init__(available_cuts, cut_settings)

            def get_stats_logs(self):

                log_lines = ['\nInitial Cut Statistics:\n']
                logged_results = {}

                for cut in self.cuts:
                    cut_name, stats = cut.name, cut.stats
                    num_cuts, cut_time = stats['Num'] stats['Time']

                if cut.opts['Enabled']:
                    log_lines.append(f'{cut_name} - Added {num_cuts} cuts in {cut_time:.2f}s\n')

                logged_results[f'{cut_name} - Cuts Added'] = num_cuts
                logged_results[f'{cut_name} - Time'] = cut_time

            def my_shared_method(self, arg1, args2):
                # Implement some method which multiple subroutines can access as a helper function

    Attributes:
        cuts: List of initial cut instances provided to manager

    """
    def __init__(self, available_cuts, user_cut_settings):

        self.name = 'Generic Initial Cut Manager'
        self.cuts = []

        cut_settings = {}

        for cut in available_cuts:

            # Handle exception if the cut name has not been set as a class method
            # in the subclass. In this cause inherit the default name
            try:
                cut_name = cut.name
            except AttributeError:
                cut_name = cut.default_name

            cut_user_opts = user_cut_settings.get(cut_name,{})
            cut_instance = cut(cut_user_opts)


            self.cuts.append(cut_instance)
            cut_settings[cut_name] = cut_instance.opts

        self.cut_settings = cut_settings

    def add_log_dir(self, log_dir):
        """Makes a log directory available to the initial cut implementations
        """
        for cut in self.cuts:
            cut.log_dir = log_dir

    def add_cuts(self, model):

        for cut in self.cuts:
            if cut.opts['Enabled']:
                cut.add_cuts(model)

    def get_solution_completers(self):

        self.solution_completer_error_active = {}

        def solution_completer_wrapper(solution_completer, cut_name):
            """ Decorator for solution completer functions which handles unexpected errors

            Solution completers should handle exceptions gracefully and raise a warning via log_error. For any exceptions
            which are not caught this decorator catches them, logs a warning and prevents future use of the completer

            Args:
                solution_completer (func): Solution completer function generated by initial cut instance
                cut_name (str): Name of the initial cut

            """
            def wrap_func(*args, **kwargs):
                if cut_name not in self.solution_completer_error_active:
                    self.solution_completer_error_active[cut_name] = False

                if not self.solution_completer_error_active[cut_name]:
                    try:
                        solution_completer(*args, **kwargs)
                    except Exception as err:
                        log_error(120, f'{cut_name} solution completer failed for unknown reason with Exception {type(err).__name__}')
                        self.solution_completer_error_active[cut_name] = True
            return wrap_func


        solution_completers = []

        for cut in self.cuts:
            if cut.opts['Enabled']:

                solution_completer = cut.gen_CompleteSolution()
                wrapped_solution_completer = solution_completer_wrapper(solution_completer, cut.name)
                solution_completers.append(wrapped_solution_completer)

        return solution_completers

    def valid_settings(self, model_opts=None, data=None):
        """Checks that the settings for each initial cut are valid

        Args:
            model_opts (dict):
            data (dict): Dictionary with dataset information

        Returns:
            Return True all subroutine settings are valid and False otherwise
        """

        all_settings_valid = True
        log_messages = []

        for cut in self.cuts:
            if cut.opts['Enabled']:
                cut_valid, log_message = cut.valid_settings(model_opts=model_opts, data=data)
                if not cut_valid:
                    all_settings_valid = False
                    if log_message is not None:
                        if isinstance(log_message, str):
                            log_messages.append(log_message)

                        elif isinstance(log_message, list):
                            for message in log_message:
                                # Only append strings to avoid an error printing
                                if isinstance(message, str):
                                    log_messages.append(message)
                                else:
                                    log_error(13, notes=f'{cut.name} initial cut returned a non-string log message from valid_settings method')

                        else:
                            log_error(13, notes=f'{cut.name} initial cut returned a non-string log message from valid_settings method')

        return all_settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        """Checks if the settings for each cut provide a benefit over the baseline

        Args:
            model_opts (dict): opt_params
            data (dict): Dictionary with dataset information

        Returns:
            Returns True if the setting for at least one subroutine provide a benefit over the baseline
            Otherwise return False
        """


        log_messages = []
        useful_settings_found = False

        for cut in self.cuts:
            if cut.opts['Enabled']:
                settings_useful, log_message =  cut.useful_settings(model_opts=model_opts, data=data)
                if settings_useful:
                    useful_settings_found = True
                else:
                    if log_message is not None:
                        if isinstance(log_message, str):
                            log_messages.append(log_message)

                        elif isinstance(log_message, list):
                            for message in log_message:
                                # Only append strings to avoid an error printing
                                if isinstance(message, str):
                                    log_messages.append(message)
                                else:
                                    log_error(13, notes=f'{cut.name} initial cut returned a non-string log message from useful_settings method')

                        else:
                            log_error(13, notes=f'{cut.name} initial cut returned a non-string log message from useful_settings method')

        return useful_settings_found, log_messages

    def get_stats_log(self):
        """Helper function which parses self.stats for logging purposes

        Should be overwritten by accessing self.stats and formatting it for:
            1) Printing to log file
            2) Writing to results file

        Returns:
            Returns a tuple (log_printout, logged_results)
            log_printout is a string which should print out results concerning heuristic stats (return None if no heuristic used)
            logged_results is a dictionary where the keys are the names of the columns in the results save file
            and values are the corresponding entries for that column
        """

        log_printout = None
        logged_results = {}

        return log_printout, logged_results

class OCT(ABC):
    """Base class for optimal classification tree models

    Provides various functions for settings up Gurobi models, training, logging, and post processing the model.

    For a user subclassing the base class, the following are the minimal methods which must be implemented:
        add_vars: Add decision variables to Gurobi model
        add_constraints: Add constraints to Gurobi model
        add_objective: Define the objective of the Gurobi model


    The following methods are optional but recommended:
        warm_start: Provide a warm start solution to the model
        vars_to_readable: Convert the decision variables from the optimised model to a readable format
        save_model_output: Save decision vars in readable format to file
        summarise_tree_info: Create a log output with information about the optimised solution
        _check_output_validity: Confirm the feasibility of the optimised solution


    The Gurobi model object has many attributes on it. data should be accessed by model_data. add_vars should attach
    the decision variables as model._variables, and then add_constraints and add_objective should access the variables
    via the model object. model also has an opts attribute, this is a set which can be used to specify functionality/options
    E.g. to specify constraints that heuristic solutions must follow


    The user defined settings are split into opt_params and gurobi_params. gurobi_params allows the user to set Gurobi
    parameters, see gurobi_params_base_default in __init__ method for currently implemented options


    opt_params is a dictionary with the following options:
        Warmstart (bool): Enable warmstarting the solution
        Polish Warmstart (bool): Enable solution polishing of warmstart solution
        Base Directory (str): Base directory for project. By default set relative to location of base_classes.py
        Results Directory (str): Name of results directory. This is the name of the experiment
        Compare Relaxations (bool): Enable to log various relaxation statistics (Note that this stops the optimisation after the root node)
        depth: (int): Maximum depth of tree to optimise over
        lambda (float): Regularisation parameter lambda. Set to None for variants without regularisation
        Use Baseline (bool): Ignore check on usefullness of callback subroutines/initial cuts. Must be set to True for baseline model
        Callback (dict): Dict of dicts, each key is the name of a callback subroutine with associated value being a dictionary of settings
        Initial Cuts (dict): Same structure as Callback settings dict

    The subclass __init__ should look something like:

        class OCTSubclass(OCT):
            def __init__(self, opt_params, gurobi_params):
                super().__init__(opt_params, gurobi_params, callback_generator=MyCallbackGenerator, cut_manager=MyInitialCutManager)
                self.model_type = 'OCTSubclass'

    Attributes:
        model_type: The name of the model. Must be set in __init__ AFTER call to super.__init__
        model: The Gurobi model object
        cut_manager: The initial cut manager
        callback_generator:
        GurobiLogFile: The name of the Gurobi log file
        stats: Dict storing statistics for logging. Currently only used for logging warmstart statistics
        _model_trained: False by default, only set to true after optimisation with feasible solution returned

    """

    GurobiLogFile = ''
    _model_trained = False
    stats = {}

    def __init__(self,
                 user_opt_params,
                 user_gurobi_params,
                 opt_params_defaults=None,
                 gurobi_params_defaults=None,
                 callback_generator=None,
                 cut_manager=None):

        """ Base initialisation for OCT models

        Handles parameter initialisation for Gurobi, OCT model, initial cuts and callback

        To use __init__ method when inheriting from this class, use super.__init__ and pass in
        the arguments described below in Args. self.model_type must be set AFTER calling super.__init__
        or else the default name will be taken

        Alternatively, subclass __init__ can handle functionality by itself.
        It must do the following:
            Set the name of the model in self.model_type
            Set self.opt_params to be a dictionary of optimisation params (See below defaults which must be set)
            Set self.gurobi_params to be a dictionary of gurobi params (See below defaults which must be set)
            Set self.callback_generator with an instance of GenCallback
            Set self.initial_cuts with a dict

        Args:
            user_opt_params (dict): Opt params passed by user
            user_gurobi_params (dict): Gurobi params passed by user
            opt_params_defaults (dict): Default opt params optionally set by subclass
            gurobi_params_defaults (dict): Default gurobi params optionally set by subclass
            callback_generator (GenCallback): Optional callback generator provided by subclass
            available_cuts (dict[str:InitialCut]): Optional dict of available cut.
                                                   Key should be the name of cut and value should be a subclass of InitialCut
        """


        self.model_type = 'OCTBase'

        opt_params_base_default = {'Warmstart': True,
                                   'Polish Warmstart': True,
                                   'Base Directory': os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                   'Results Directory': 'Test Folder',
                                   'Compare Relaxations': False,
                                   'depth': 3,
                                   'lambda': None,
                                   'Use Baseline': False,
                                   'Subjob id': None,
                                   'Array Job id': None,
                                   'Debug Mode': False}

        gurobi_params_base_default = {'TimeLimit': 3600,
                                      'Threads': 1,
                                      'MIPGap': 0,
                                      'MIPFocus': 0,
                                      'Heuristics': 0.05,
                                      'NodeMethod': -1,
                                      'Method': -1,
                                      'Seed': 0,
                                      'LogToConsole': 0,
                                      'LogToFile': True,
                                      'NodeLimit': float('inf')}

        # Keep track of which settings have been explicitly set by the user to create the log directory
        self.user_params = user_opt_params | user_gurobi_params

        # Create the opt_params and gurobi_params dict by merging in default and user settings
        self.opt_params = _merge_settings(opt_params_base_default, opt_params_defaults, user_opt_params)
        self.gurobi_params = _merge_settings(gurobi_params_base_default, gurobi_params_defaults, user_gurobi_params)

        # Initialise the cut manager with the initial cut settings
        if cut_manager is None:
            self.cut_manager = InitialCutManager([], {})
        else:
            cut_settings = self.opt_params.get('Initial Cuts',{})
            self.cut_manager = cut_manager(cut_settings)

        # Initialise the callback generator with the callback settings
        if callback_generator is None:
            self.callback_generator = GenCallback([],{})
        else:
            callback_settings = self.opt_params.get('Callback',{})
            self.callback_generator = callback_generator(callback_settings)

        # Get true callback settings from the callback generator
        # This should be the result of merging in user callback settings into the default settings on each subroutine
        self.opt_params['Callback'] = self.callback_generator.callback_settings

        # Get true initial cut settings from the initial cut manager
        # This should be the result of merging in user initial cut settings into the default settings on each subroutine
        self.opt_params['Initial Cuts'] = self.cut_manager.cut_settings

    def update_model_stats(self, feature_name, num_added, time_added, *args):
        """Helper function to keep track of statistics for model building

        Intended to be called from warmstart to keep track of heuristic statistics

        Args:
            feature_name (str): Name of the feature which statistics are being recorded for
            num_added (int): Number associated with statistic. Additive in nature
            time_added (float): Time taken in generating
            *args: Extra tuples which can be added to stats with format (key, value)

        """
        if feature_name not in self.stats:
            self.stats[feature_name] = {'Num': 0,
                                        'Time': 0}
            for arg in args:
                k, v = arg
                self.stats[feature_name][k] = 0

        self.stats[feature_name]['Num'] += num_added
        self.stats[feature_name]['Time'] += time_added

        for arg in args:
            k, v = arg
            self.stats[feature_name][k] += v

    def create_model(self):
        """Create Gurobi model object

        Sets Gurobi model parameters based on self.gurobi_params
        Also Sets up the Gurobi logfile from the string in self.GurobiLogFile
        By default this will be the empty string '', to use the Gurobi logfile it must be set before create_model is called
        this happens automatically if using the fit method

        Returns:
            Return the created Gurobi model object
        """
        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model entering create_model() at {time_now}')

        model = Model()

        model.Params.LogToConsole = 0

        if self.gurobi_params['LogToFile']:

            # By default, the file location will be an empty string.
            # Need to set using self._set_gurobi_logfile(), will be done automatically done if using self.fit()
            model.params.LogFile = self.GurobiLogFile

            # Clear out the logfile if it already exists
            if os.path.exists(model.params.LogFile):
                open(model.params.LogFile, 'w').close()

        model.Params.MIPGap = self.gurobi_params['MIPGap']
        model.Params.MIPFocus = self.gurobi_params['MIPFocus']
        model.Params.Heuristics = self.gurobi_params['Heuristics']
        model.Params.TimeLimit = self.gurobi_params['TimeLimit']
        model.Params.Threads = self.gurobi_params['Threads']
        model.Params.Method = self.gurobi_params['Method']
        model.Params.NodeMethod = self.gurobi_params['NodeMethod']
        model.Params.Seed = self.gurobi_params['Seed']
        model.Params.NodeLimit = self.gurobi_params['NodeLimit']
        model.Params.LogToConsole = self.gurobi_params['LogToConsole']

        model._opts = set()     # Set of options which can be set as flags to modify subroutine behaviour
        model._model_name = self.model_type

        # Do this to ensure that we can access params before we optimise the model (sometimes needed for initial cuts)
        model.update()

        return model

    def build_model(self,model):
        """Run through methods to construct model

        Args:
            model (grbModel): Gurobi model to add vars, constraints, etc... to

        """

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model entering build_model() at {time_now}')

        self.add_vars(model)
        self.add_constraints(model)
        self.add_objective(model)
        self._add_initial_cuts(model)
        self.callback_generator.update_model(model)

    def fit(self, data):
        """Fits optimal classification tree given dataset.

        Does the following:
            Creates the gurobi model object
            Sets up logging (gurobi and user logfiles)
            Checks if the settings provided are 1) valid and 2) provide a benefit compared to the baseline.
            Calls _build_model to add variables, constraints, objective, etc...
            Runs warmstart if requested
            Optimises the model
            Check that a feasible solution was found

        Args:
            data (dict): Dictionary containing dataset. At a minimum should contain 'X', 'y', 'name', 'encoded name' entries
        """

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model entering fit() at {time_now}')

        opt_params = self.opt_params
        
        self._setup_logging(data)

        # Validate the callback and initial cut settings
        callback_settings_valid, callback_valid_log_messages = self.callback_generator.valid_settings(model_opts=opt_params, data=data)
        cut_settings_valid, cut_valid_log_messages = self.cut_manager.valid_settings(model_opts=opt_params, data=data)
        
        # Check if there are any useful callback or initial cut settings
        if opt_params['Use Baseline']:
            # 'Use Baseline' disables usefulness checks. Required if running baseline models without any initial cuts/callback subroutines
            callback_settings_useful, callback_useful_log_messages = True, None
            cut_settings_useful, cut_useful_log_messages = True, None
        else:
            callback_settings_useful, callback_useful_log_messages = self.callback_generator.useful_settings(model_opts=opt_params, data=data)
            cut_settings_useful, cut_useful_log_messages = self.cut_manager.useful_settings(model_opts=opt_params,data=data)
        
        all_settings_valid = callback_settings_valid and cut_settings_valid
        some_settings_useful = cut_settings_useful or callback_settings_useful    
        
        # Raise errors and warnings
        if not callback_settings_valid:
            log_error(10, callback_valid_log_messages)
        if not cut_settings_valid:
            log_error(11, cut_valid_log_messages)
        if not callback_settings_useful:
            log_error(110, callback_useful_log_messages)
        if not cut_settings_useful:
            log_error(111, cut_useful_log_messages)
        
        if not (all_settings_valid and some_settings_useful):
            if not some_settings_useful:
                log_error(12)
            return False

        model_build_start_time = time.time()

        # Create the gurobi model. If it fails log an error and exit
        # Should pick up if a licence isn't available
        try:
            model = self.create_model()
        except GurobiError as err:
            try:
                log_error(23, err.message)
            except:
                log_error(24, 'Unknown Exception - GurobiError.message failed')
            return False
        except Exception as err:
            log_error(23, [f'Unknown Exception {type(err).__name__}', '\\n'.join(traceback.format_exc().split('\n'))])
            return False

        # Attach useful objects to Gurobi model so that they can always be accessed
        model._data = data
        model._tree = Tree(opt_params['depth'])
        model._lambda = opt_params.get('lambda', None)

        # Build the model (add variables, constraints, objective, initial cuts)
        try:
            self.build_model(model)
        except Exception as err:
            print(traceback.format_exc())
            log_error(21, [f'Unknown Exception {type(err).__name__}','\\n'.join(traceback.format_exc().split('\n'))])
            return False


        if opt_params['Warmstart']:
            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{self.model_type} model entering warm_start() at {time_now}')
            self.warm_start(model)

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model entering gen_callback() at {time_now}')

        # Generate the callback function
        callback = self.callback_generator.gen_callback()

        model_build_time = time.time() - model_build_start_time

        # Modify the behaviour of fit to only return information about the relaxation
        if opt_params['Compare Relaxations']:
            model.update()
            print('\n' + '-' * 5 + ' Relaxation Comparison Requested ' + '-' * 5)
            self.run_relaxations(model, callback=callback)

            # Set the node limit to one so that only the root node is explored
            model.Params.NodeLimit = 1

        print(f'Built Gurobi model in {model_build_time:.2f} seconds')

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model beginning optimisation at {time_now}')

        self.optimise_model(model, callback=callback)

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model finishing optimisation at {time_now}')

        self._model = model

        # Run post optimisation checks, i.e. was model solve successful, is the solution actually feasible?
        self._post_optimisation_check(model)

        return self._model_trained

    def run_relaxations(self, model, callback=None):
        """Evaluates the model relaxation, with and without presolve

        Results are logged automatically to results csv file. Generally used with NodeLimit set to one to give a
        picture of the strength of a relaxation

        Args:
            model (grbModel): Gurobi model object
            callback (func): Callback function

        """


        presolved_model = model.presolve().relax()
        relaxed_model = model.relax()

        relaxed_model.Params.Presolve = 0
        # relaxed_model.Params.Aggregate = 0
        # relaxed_model.Params.AggFill = 0

        relaxed_model.optimize(callback)
        self.update_model_stats('Relaxation Comparison (Raw Model)',
                                relaxed_model.objVal,
                                relaxed_model.Runtime)

        presolved_model.optimize(callback)
        self.update_model_stats('Relaxation Comparison (Presolved Model)',
                                presolved_model.objVal,
                                presolved_model.Runtime)

        # Reset the relaxation statistics in the callback in case they were overwritten by the raw/presolved models
        self.callback_generator.relaxation_statistics = {'Initial Root Relaxation - Obj': None,
                                      'Initial Root Relaxation - Time': None,
                                      'Initial Root Relaxation - Work': None,
                                      'Presolve Relaxation - Obj': None,
                                      'Presolve Relaxation - Time': None,
                                      'Presolve Relaxation - Work': None,
                                      'Final Root Relaxation - Obj': None,
                                      'Final Root Relaxation - Time': None,
                                      'Final Root Relaxation - Work': None}

    def optimise_model(self, model, callback=None):
        """Optimise the Gurobi model

        Args:
            model (grbModel): Gurobi model to optimise
            callback: Optional callback function.
        """

        model_opt_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f'Beginning model solve at {model_opt_start_time}')
        model._opt_start_time = model_opt_start_time

        model.optimize(callback)

    def _post_optimisation_check(self, model):
        """Check status of gurobi model

        Sets self._model_trained = True if a feasible solution has been found by Gurobi
        Otherwise print out why Gurobi did not return a feasible solution

        Args:
            model (grbModel): Gurobi model
        """

        # Check if an exception was raised in a callback subroutine
        # If it was then disregard the results
        if self.callback_generator.callback_exception_raised:
            self._model_trained = False
            return

        status = model.Status

        if status == GRB.INTERRUPTED:
            log_error(100)

        if status == GRB.INFEASIBLE:
            log_error(0,notes='Model Infeasible')
        elif status == GRB.UNBOUNDED:
            log_error(0,notes='Model Unbounded')
        elif status == GRB.NUMERIC:
            log_error(0,notes='Gurobi terminated solve due to numerical issues')
        elif status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.NODE_LIMIT]:
            log_error(1)
        else:

            try:
                output_valid, log_messages = self._check_output_validity(model)
                if output_valid:
                    self._model_trained = True

                else:
                    # If the output is invalid we log an error message and attempt to write debug information to file
                    # This includes the value of all variables in the solution, and the data dictionary

                    self._model_trained = False
                    log_error(2, log_messages)

                    variables = model._variables

                    data_to_save = {var_name: {k: var_dict[k].X for k in var_dict}
                                    for var_name, var_dict in variables.items()}
                    data_to_save['data'] = model._data

                    with open(os.path.join(self.log_dir, 'debug_output.pickle'), 'wb') as f:
                        pickle.dump(data_to_save, f)


            except Exception as err:
                log_error(157, f'Failed with Exception {type(err).__name__}')
                self._model_trained = True


    def get_gurobi_model(self):
        """Preferred method to retrieve the Gurobi model. Returns None if the model was not successfully trained

        """

        if self._model_trained:
            return self._model

    def _get_log_dir(self,data=None,model=None):
        """Helper function to retrieve the name of the logging directory

        Args:
            data (dict): Data dictionary
            model (grbModel): Gurobi model object

        Returns:

        """
        try:
            return self.log_dir
        except:
            try:
                self._create_log_dir(data)
                return self.log_dir
            except FileNotFoundError:
                log_error(134)
            except:
                try:
                    self._create_log_dir(model._data)
                    return self.log_dir
                except FileNotFoundError:
                    log_error(134)
                except:
                    print('Arguments to _get_log_dir insufficient to retrieve logging directory')

    def _create_log_dir(self,data):
        """Creates a directory for results and logging

        If enough information is provided it will create the log directory and store its location in self.log_dir

        Args:
            data (dict): Dictionary which should include name of dataset
        """

        if 'encoded name' in data:
            dataset_name = data['encoded name']
        elif 'name' in data:
            dataset_name = data['encoded name']
        else:
            dataset_name = 'UnknownDataset'
            print(f'No name provided in data supplied to fit method for {self.model_type} model')

        results_dir_base = os.path.join(self.opt_params['Base Directory'],
                                        'Results',
                                        self.model_type,
                                        self.opt_params['Results Directory'])

        # On HPC cluster store some helper information for scripts which parse the logs files
        if self.opt_params['Array Job id'] is not None:
            if self.opt_params['Subjob id'] is not None and self.opt_params['Subjob id'] == 0:

                job_info_dir = os.path.join(results_dir_base,'ExperimentInfo')
                os.makedirs(job_info_dir, exist_ok=True)

                job_id_file = os.path.join(job_info_dir, 'Job ids.txt')

                array_id_string = f'{self.opt_params['Array Job id']}'
                if os.path.exists(job_id_file):
                    array_id_string = '\n' + array_id_string

                with open(os.path.join(job_id_file),'a') as f:
                    f.write(array_id_string)

        # Give some optimisation parameters shorthand names. These are included in the log directory name
        # even if not explicitly set by the user
        log_dir_params = {'d': self.opt_params['depth']}
        if 'lambda' in self.opt_params:
            log_dir_params['l'] = self.opt_params['lambda']
        if 'Subjob id' in self.opt_params and (self.opt_params['Subjob id'] is not None):
            log_dir_params['SubID'] = self.opt_params['Subjob id']

        # Only construct log directory name from settings which the user has explicitly set
        log_dir_params |= self.user_params

        callback_setting_entries = []

        if 'Callback' in log_dir_params:
            callback_settings = log_dir_params['Callback']
            for subr_name, subr_settings in callback_settings.items():
                subr_name_short = get_shorthand(subr_name)
                if subr_name == 'Benders Cuts':
                    # Benders cuts are always enabled so we treat them as a special case where the enhanced cuts
                    # effectively take the place of the subroutine being enabled
                    subr_name_short = get_shorthand('Enhanced Cuts')
                    if 'Enhanced Cuts' not in subr_settings:
                        continue
                    elif not subr_settings['Enhanced Cuts']:
                        log_dir_params[subr_name_short] = False
                    else:
                        if 'EC Level' in subr_settings or 'Relax w' in subr_settings:
                            if 'EC Level' in subr_settings:
                                callback_setting_entries.append(f'EC-ECL={subr_settings['EC Level']}')
                            if 'Relax w' in subr_settings:
                                callback_setting_entries.append(f'EC-Rw={subr_settings['Relax w']}')
                        else:
                            log_dir_params[subr_name_short] = True

                else:
                    if 'Enabled' not in subr_settings:
                        continue
                    elif not subr_settings['Enabled']:
                        log_dir_params[subr_name_short] = False
                    else:
                        # When feature is enabled we check for other settings which have been explicitly set by user
                        subr_setting_entry = []
                        for setting_name, setting_val in subr_settings.items():
                            if setting_name != 'Enabled':
                                subr_setting_entry.append(f'{get_shorthand(setting_name)}={get_shorthand(setting_val)}')

                        if len(subr_setting_entry) > 0:
                            callback_setting_entries.append(f'{subr_name_short}-{','.join(subr_setting_entry)}')
                        else:
                            # If callback subroutine is enabled with all settings left to defaults, simply indicate that it is enabled
                            log_dir_params[subr_name_short] = True

        initial_cut_settings_entries = []

        if 'Initial Cuts' in log_dir_params:
            initial_cut_settings = log_dir_params['Initial Cuts']
            for cut_name, cut_settings in initial_cut_settings.items():
                cut_name_short = get_shorthand(cut_name)
                if 'Enabled' not in cut_settings:
                    continue
                elif not cut_settings['Enabled']:
                    log_dir_params[cut_name_short] = False
                else:
                    cut_setting_entries = []
                    for setting_name, setting_val in cut_settings.items():
                        if setting_name not in ['Enabled', 'Ignore Dataset Check']:
                            cut_setting_entries.append(f'{get_shorthand(setting_name)}={get_shorthand(setting_val)}')

                    if len(cut_setting_entries) > 0:
                        initial_cut_settings_entries.append(f'{cut_name_short}-{','.join(cut_setting_entries)}')
                    else:
                        # If initial cut is enabled with all settings left to defaults, simply indicate that it is enabled
                        log_dir_params[cut_name_short] = True

        # Set some params to be ignored in the directory name, either because we have already mapped them to a short name
        # or we never want them in the directory name
        ignored_params = ['LogToConsole', 'Use Baseline', 'Base Directory', 'Results Directory', 'LogToFile',
                          'Callback', 'Initial Cuts', 'Debug Mode', 'Compare Relaxations', 'Array Job id', 'Subjob id',
                          'lambda', 'depth']

        log_dir_name = ' '.join(f'{get_shorthand(name)}={get_shorthand(val)}'
                                   for name, val in log_dir_params.items() if name not in ignored_params)

        # Add the callback settings to the directory name
        if len(callback_setting_entries) > 0:
            log_dir_name += ' ' + ' '.join(callback_setting_entries)

        # Add the initial cut settings to the directory name
        if len(initial_cut_settings_entries) > 0:
            log_dir_name += ' ' + ' '.join(initial_cut_settings_entries)

        log_dir = os.path.join(results_dir_base,
                               dataset_name,
                               log_dir_name)

        os.makedirs(log_dir, exist_ok=True)

        self.log_dir = log_dir

    def _setup_logging(self, data):
        """Calls other methods to set up results directory and logging

        Args:
            data (dict): Dictionary which should include name of dataset
        """

        # self._create_log_dir(data)

        log_dir = self._get_log_dir(data=data)

        if log_dir is None:
            # Simply return if creation of the log directory failed
            return
        else:
            self._set_gurobi_logfile(log_dir)
            self._set_user_logfile(log_dir)
            self.cut_manager.add_log_dir(log_dir)
            self.callback_generator.add_log_dir(log_dir)


        print('Logfile initialised')

        print(f'\nLog Directory={log_dir}')
        if self.opt_params['Subjob id'] is not None:
            print(f'Job ID={self.opt_params['Subjob id']}')
        if self.opt_params['Array Job id'] is not None:
            print(f'Array Job ID={self.opt_params['Array Job id']}')

        print('')

        if 'info string' in data:
            print(data['info string'])

    def _set_gurobi_logfile(self, log_dir):
        """Sets the location of the gurobi logfile

        Intended to be called from self._setup_logging as part of logging initialisation

        Args:
            log_dir: Directory where Gurobi logs are located
        """
        self.GurobiLogFile = os.path.join(log_dir, 'Gurobi Logs.txt')

    def _set_user_logfile(self, log_dir):
        """Sets up the logfile and points stdout to the console and the logfile

        Intended to be called from self._setup_logging as part of logging initialisation

        Args:
            log_dir: Directory where user logs are to be located
        """
        console = sys.stdout
        my_logger = logger(console=console)
        LogFile = os.path.join(log_dir, 'Logfile.txt')
        my_logger.SetLogFile(LogFile)
        sys.stdout = my_logger

        self.logger = my_logger

    @abstractmethod
    def add_vars(self, model):
        """User defined method to define model variables

        Added variables must be attached to the model in a dictionary
        named model._variables where key-value pairs are variable name and variable object

        Args:
            model (grbModel): Model to add variables to
        """

    @abstractmethod
    def add_constraints(self, model):
        """User defined method to define model constraints

        Variables should be accessed through model._variables

        Args:
            model (grbModel): Model to add constraints to
        """

    @abstractmethod
    def add_objective(self, model):
        """User defined method to define model objective

        Variables should be accessed through model._variables

        Args:
            model (grbModel): Model to add objective to
        """

    def _add_initial_cuts(self, model):
        """Add initial cuts from self.available_cuts into the model

        This method also attaches the solution_completers associated with the cuts to the model object

        Args:
            model (grbModel): Gurobi model to add initial cuts to
        """

        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{self.model_type} model entering _add_initial_cuts() at {time_now}')

        self.cut_manager.add_cuts(model)
        model._solution_completers = self.cut_manager.get_solution_completers()

    def warm_start(self, model):
        """Run heuristic and add solution as warmstart

        Should be overwritten if a warmstart is required
        Method should call some heuristic solver and add solution as initial start

        Args:
            model (grbModel): Model to add warmstart solution to
        """

        log_error(112, f'Subclass model {self.model_type} has not implemented a warm_start method')

    def post_process_model(self):
        """High level method to process model results and write results to file

        Checks if the model has been trained and if so:
            Parses the solution from the Gurobi model
            Writes the results to file in human-readable format
            Writes summary of the solution to logs
            Dumps the settings used into a file in the log directory
            If debug mode is enabled
        """

        model = self.get_gurobi_model()
        debug_mode_enabled = self.opt_params['Debug Mode']

        if model is None:
            print(f'Post processing of {self.model_type} cannot be completed because '
                  f'Gurobi model does not contain a feasible solution')
            return

        log_dir = self._get_log_dir(model=model)
        if log_dir is None:
            print(f'Trained tree structure not saved for {self.model_type} model because '
                  'log directory could not be found')
            return

        self._write_params_to_file(model, log_dir)

        try:
            user_vars = self.vars_to_readable(model)
        except Exception as err:
            user_vars = 'method error'
            log_error(151, f'Failed with Exception {type(err).__name__}')

        if user_vars is None:
            log_error(150)
        elif user_vars != 'method error':
            # Assume from here that the user_vars returned by self.vars_to_readable() is correctly specified
            try:
                log_message = self.summarise_tree_info(model, user_vars)

                if log_message is None:
                    log_error(152)
                else:
                    print('\n' + '-' * 10)
                    print(log_message)

            except Exception as err:
                log_error(153, f'Failed with Exception {type(err).__name__}')

            if debug_mode_enabled:
                self._write_tree_to_file(user_vars, log_dir)

    def _check_output_validity(self, model):
        """ Check that the outputted solution is feasible

        Args:
            model (grbModel): Solved Gurobi model

        Returns:
            Return boolean output_valid which is True is the model solution is valid and False otherwise

        """

        output_valid = True
        log_message = ''

        log_error(156)

        return output_valid, log_message

    def _write_params_to_file(self, model, log_dir):
        """Writes optimisation parameters to log directory

        Should write the following into the logging directory:
            self.opt_params (including callback and initial cut settings)
            self.gurobi_params
            Useful information from model._data

        Args:
            model (grbModel): Gurobi model with tree structure in solution variables
            log_dir (str): Filepath for the logging directory
        """

        params_file = os.path.join(log_dir, 'parameters.txt')
        dump_info = {'Optimisation Params': self.opt_params,
                     'Gurobi Params': self.gurobi_params,
                     'Data Params': {'Compress Data': model._data.get('compressed',False)}}

        with open(params_file,'w') as f:
            f.write(json.dumps(dump_info, indent=4))

    def _write_tree_to_file(self,user_vars, log_dir):
        """

        Args:
            user_vars: Variables in (user defined) readable format
            log_dir (str): Filepath for the logging directory
        """

        tree_save_file = os.path.join(log_dir, 'Tree Model.pickle')
        with open(tree_save_file,'wb') as f:
            pickle.dump(user_vars,f)

        try:
            save_string = self.save_model_output(user_vars)

            if save_string == '':
                log_error(154)

            else:
                soln_var_file = os.path.join(log_dir, 'Soln Vars.txt')
                with open(soln_var_file, 'w') as f:
                    f.write(save_string)

        except Exception as err:
            log_error(155, f'Failed with Exception {type(err).__name__}')

    def save_model_output(self, user_vars):
        """Return a string which can be written to the results directory in a human-readable format

        Args:
            user_vars: Variables in (user defined) readable format

        Returns:
            save_string (str): string which should be written to file
        """

        save_string = ''

        return save_string

    def vars_to_readable(self, model):
        """Convert gurobi variables attached to model to a readable format

        Does not need to be overwritten for subclass to function. If not overwritten then
        the tree structure will not be written to file. This would also mean that the trained
        tree cannot be later reloaded from file

        Args:
            model (grbModel): Gurobi model which feasible solutions attached

        Returns:
            Tuple or dictionary of variables in human-readable format
            It should fully describe the structure and predictions of the trained decision tree
            Particular format is user defined, should match format expected by save_model_output and summarise_tree
        """

        return None

    def summarise_tree_info(self, model, user_vars):
        """

        Args:
            model (grbModel): Gurobi model object
            user_vars: Variables in (user defined) readable format

        Returns:
            save_string (str): string which should be written to file
        """

        return None

    def get_stats_log(self):
        """Helper function which parses self.stats for logging purposes

        Returns a string which can be printed and a dictionary with column entries to be saved to file
        By default only prints out stats for CART heuristic but can be modified or overwritten

        Returns:
            Returns a tuple (log_printout, logged_results)
            log_printout is a string which should print out results concerning heuristic stats (return None if no heuristic used)
            logged_results is a dictionary where the keys are the names of the columns in the results save file
            and values are the corresponding entries for that column
        """

        stats = self.stats
        logged_results = {}

        log_printout = []

        if 'CART' in stats:
            log_printout.append('\nHeuristic Statistics:')
            if 'Unpolished Obj' in stats['CART']:
                log_printout.append(f'CART - Obj = {stats['CART']['Num']} (polished from {stats['CART']['Unpolished Obj']}) in {stats['CART']['Time']:.2f}s')
            else:
                log_printout.append(f'CART - Obj = {stats['CART']['Num']} in {stats['CART']['Time']:.2f}s')

            logged_results['CART Obj'] = stats['CART']['Num']
            logged_results['CART Runtime'] = stats['CART']['Time']

        if 'Relaxation Comparison (Raw Model)' in stats or 'Relaxation Comparison (Presolved Model)' in stats:
            log_printout.append('\nRelaxation Comparison Statistics')
            if 'Relaxation Comparison (Raw Model)' in stats:
                log_printout.append(f'Raw model relaxation = {stats['Relaxation Comparison (Raw Model)']['Num']}. Solve in {stats['Relaxation Comparison (Raw Model)']['Time']:.2f}s')
                logged_results['Relaxation Comparison - Raw Objective'] = stats['Relaxation Comparison (Raw Model)']['Num']
                logged_results['Relaxation Comparison - Raw Solve Time'] = stats['Relaxation Comparison (Raw Model)']['Time']
            if 'Relaxation Comparison (Presolved Model)' in stats:
                log_printout.append(f'Presolved model relaxation = {stats['Relaxation Comparison (Presolved Model)']['Num']}. Solve in {stats['Relaxation Comparison (Presolved Model)']['Time']:.2f}s')
                logged_results['Relaxation Comparison - Presolved Objective'] = stats['Relaxation Comparison (Presolved Model)']['Num']
                logged_results['Relaxation Comparison - Presolved Solve Time'] = stats['Relaxation Comparison (Presolved Model)']['Time']


        if len(log_printout) == 0:
            return None, []
        else:
            return '\n'.join(log_printout), logged_results

    def cleanup_GRB_environment(self):
        self._model.dispose()
        mystr = disposeDefaultEnv()
