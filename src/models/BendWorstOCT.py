import os
import time

import math
import pickle
import itertools

import numpy as np
from gurobipy import *

from src.utils.generators import EQPSets
from src.utils.data import valid_datasets
from src.utils.logging import log_error
from src.utils.trees import (Custom_CART_Heuristic,
                             opt_D2_subtree_worst_case,
                             optimise_subtrees_worst_case,
                             create_recursive_tree)

from src.models.base_classes import (OCT,
                                     GenCallback,
                                     InitialCutManager,
                                     CallbackSubroutine,
                                     InitialCut)

# import line_profiler

class PathNode():
    """Class for nodes in an integral path

    """
    path = None
    f = None
    parent = None
    left_child = None
    right_child = None
    internal_node = False

    def __init__(self, n, I_mask, depth):
        self.depth = depth
        self.n = n
        self.I_mask = I_mask

class EQPInitialCut(InitialCut):
    """Class for implementing equivalent point initial cuts

    Implements the equivalent point inequalities as described in the paper. EQP sets and associated information are
    returned from instance of EQPSets class from src.utils.generators. It returns a list of tuples
    (cut_idx,rhs_bound,F_star) where cut_idx are the sample indices in the EQP sets, rhs_bound is the bound on the
    classification scores if the EQP set is not split, and F_star is the associated split set.

    Settings:
        Features Removed {0,1,2}: Maximum number of features allows in split sets. By default will only return that the
        settings are useful if the dataset/encoding actually has ADDITIONAL EQP sets for the size of the split set. That
        is to say that if no extra EQP sets are added for FR=2 vs FR=1, then the initial cuts provide no marginal benefit

        H Variant {'Basic','Chain','Recursive'}: Variant of constraint linking sample paths to classification

        Disaggregate Alpha {True,False}: Enables constraint disaggregation for chain and recursive variants

        Group Selection {True,False}: Enables group selection constraints for bounding classification scores

        Ignore Dataset Check {True, False}: If enabled do not check if the cuts are actually useful for the given dataset

    """

    name = 'EQP Initial Cuts'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Features Removed': 0,
                            'H Variant': 'Chain',
                            'Disaggregate Alpha': False,
                            'Group Selection': False,
                            'Ignore Dataset Check': False,
                            'Filter Dominated': False,
                            'Beta Dependence': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def valid_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']
        alpha_version = self.opts['H Variant']
        disaggregate_alpha = self.opts['Disaggregate Alpha']
        group_selection_enabled = self.opts['Group Selection']
        filter_dominated = self.opts['Filter Dominated']
        beta_dependence = self.opts['Beta Dependence']

        settings_valid = True
        log_messages = []

        # EQP sets with more than two features removed are not considered computationally feasible
        if features_removed not in [0, 1, 2]:
            log_messages.append(f'EQP initial cuts not valid for {features_removed} features removed. Please try a value in [0,1,2]')
            settings_valid = False
        elif features_removed < 2 and (filter_dominated or beta_dependence):
            log_messages.append(f'EQP initial cuts with "Features Removed" < 2 not compatible with domination filter or beta dependence')
            settings_valid = False

        if alpha_version not in ['Chain', 'Recursive', 'Basic']:
            log_messages.append(f'EQP initial cuts not valid for {alpha_version} version of alpha constraints. Please try a value in [\'Basic\',\'Chain\',\'Recursive\']')
            settings_valid = False
        elif alpha_version in ['Chain', 'Recursive'] and features_removed == 0:
            log_messages.append(f'EQP initial cuts Chain and Recursive alpha constraints not valid with '
                                f'"Features Removed" = 0 since it reduces to the basic alpha constraints')
            settings_valid = False
        elif alpha_version in ['Basic'] and beta_dependence:
            log_messages.append(f'EQP initial cuts basic version not valid with "Beta Dependence" set to True')


        if not isinstance(group_selection_enabled, bool):
            log_messages.append(f'EQP initial cuts not valid for {group_selection_enabled} group selection cuts. Please try a boolean value')
            settings_valid = False

        if not isinstance(disaggregate_alpha, bool):
            log_messages.append(f'EQP initial cuts not valid for {disaggregate_alpha} alpha disaggregate. Please try a boolean value')
            settings_valid = False

        if not isinstance(filter_dominated, bool):
            log_messages.append(f'EQP initial cuts not valid for {filter_dominated} filter dominance. Please try a boolean value')
            settings_valid = False

        if not isinstance(beta_dependence, bool):
            log_messages.append(f'EQP initial cuts not valid for {beta_dependence} beta dependence. Please try a boolean value')
            settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']

        settings_useful = True
        log_messages = []

        if not self.opts['Ignore Dataset Check']:

            try:
                encoded_instance_name = data['encoded name']

                # Check that instance/encoding actually have any eqp sets
                if encoded_instance_name not in valid_datasets['eqp'][features_removed]:
                    log_messages.append(f'EQP initial cuts not useful since dataset {encoded_instance_name} does not have any eqp sets with {features_removed} features removed')
                    settings_useful = False

            except KeyError as err:
                log_messages.append(f'Unable to assess validity of EQP initial cut settings due to KeyError on key {err.args[0]}')
                settings_useful = False

            except Exception as err:
                log_messages.append(f'Failed to check validity of EQP initial cut settings with Exception {type(err).__name__}')
                settings_useful = False

        return settings_useful, log_messages

    @staticmethod
    def basic_alpha_constraints(model, alpha, cut_info, b, data, tree):
        """Basic variant of H

        Adds constraints to the gurobi model for a single EQP set

        Args:
            model (grbModel): Gurobi model object
            alpha (Dict): Dictionary for decision variables to detect if split features are in the tree
            cut_info (tuple): Information about the eqp set
            b (dict): Dictionary of branch decision variables
            data (dict): Dataset information
            tree (Tree): An instance of the Tree class found in src.utils.trees

        Returns:
            Returns tuple (bounding_term, cuts added). bounding_term is the decision variable which controls the bound
            on the classification score, denoted \beta_J^G in paper. cuts_added is the number of constraints added to
            the model
        """

        cuts_added = 0

        _ , _, F_star, _ = cut_info

        # If the split set is empty then the EQP set is never split. i.e. the samples fall into the same leaf no matter
        # what and there is no decision variable required to bound
        if len(F_star) == 0:
            return 0, cuts_added

        # If F_star has been seen before then we can simply reuse the bounding term since the basic alpha constraints do
        # not care about the actual samples in the EQP set. Otherwise, create the bounding term
        if F_star not in alpha:
            alpha[F_star] = model.addVar(vtype=GRB.CONTINUOUS, ub=1)

            model.addConstr(alpha[F_star] <= quicksum(b[n,f] for n in tree.B for f in F_star))

            cuts_added += 1

        return alpha[F_star], cuts_added

    @staticmethod
    def chain_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha):
        """Chain variant of H

        Adds constraints to the gurobi model for a single EQP set

        Args:
            model (grbModel): Gurobi model object
            alpha (Dict): Dictionary for decision variables to detect if EQP sample path through split feature
            cut_info (tuple): Information about the eqp set
            b (dict): Dictionary of branch decision variables
            data (dict): Dataset information
            tree (Tree): An instance of the Tree class found in src.utils.trees
            disagg_alpha (bool): Enabled constraint disaggregation

        Returns:
            Returns tuple (bounding_term, cuts added). bounding_term is the decision variable which controls the bound
            on the classification score, denoted \beta_J^G in paper. cuts_added is the number of constraints added to
            the model
        """


        cut_idx, _, F_star, _ = cut_info

        # If the split set is empty then the EQP set is never split. i.e. the samples fall into the same leaf no matter
        # what and there is no decision variable required to bound
        if len(F_star) == 0:
            return 0, 0

        X, y = data['X'], data['y']

        F = data['F']

        cuts_added = 0

        alpha_vtype = GRB.CONTINUOUS if disagg_alpha else GRB.BINARY

        # Get some arbitrary idx from the cut, they all have the same
        i = cut_idx[0]

        # Set of features that the samples in the EQP set are identical in
        F_support = [f for f in F if f not in F_star]

        for n in tree.B:
            alpha[(F_star, cut_idx, n)] = model.addVar(vtype=alpha_vtype, ub=1)

            path = tree.ancestors(n, branch_dirs=True)
            path_left, path_right = [], []

            for n_a, d in path.items():
                if d == 0:
                    path_left.append(n_a)
                elif d == 1:
                    path_right.append(n_a)

            if disagg_alpha:
                model.addConstr(alpha[(F_star, cut_idx, n)] <= quicksum(b[n, f] for f in F_star))
                for n_a in path_left:
                    model.addConstr(alpha[(F_star, cut_idx, n)] <= quicksum(b[n_a, f] for f in F_support if X[i, f] == 0))
                for n_a in path_right:
                    model.addConstr(alpha[(F_star, cut_idx, n)] <= quicksum(b[n_a, f] for f in F_support if X[i, f] == 1))

                cuts_added += len(path) + 1

            else:
                coeff = 1 / (1 + len(path))
                rhs_sum = (quicksum(b[n, f] for f in F_star) +
                           quicksum(b[n_a, f] for n_a in path_left for f in F_support if X[i, f] == 0) +
                           quicksum(b[n_a, f] for n_a in path_right for f in F_support if X[i, f] == 1))

                model.addConstr(alpha[(F_star, cut_idx, n)] <= coeff * rhs_sum)

                cuts_added += 1

        bounding_term = quicksum(alpha[(F_star, cut_idx, n)] for n in tree.B)

        return bounding_term, cuts_added

    @staticmethod
    def recursive_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha):
        """Recursive variant of H

        Adds constraints to the gurobi model for a single EQP set

        Args:
            model (grbModel): Gurobi model object
            alpha (Dict): Dictionary for decision variables to detect if EQP sample path through split feature
            cut_info (tuple): Information about the eqp set
            b (dict): Dictionary of branch decision variables
            data (dict): Dataset information
            tree (Tree): An instance of the Tree class found in src.utils.trees
            disagg_alpha (bool): Enabled constraint disaggregation

        Returns:
            Returns tuple (bounding_term, cuts added). bounding_term is the decision variable which controls the bound
            on the classification score, denoted \beta_J^G in paper. cuts_added is the number of constraints added to
            the model
        """


        cut_idx, _, F_star, _ = cut_info

        # If the split set is empty then the EQP set is never split. i.e. the samples fall into the same leaf no matter
        # what and there is no decision variable required to bound
        if len(F_star) == 0:
            return 0, 0

        X, y = data['X'], data['y']

        F = data['F']

        cuts_added = 0
        alpha_vtype = GRB.CONTINUOUS if disagg_alpha else GRB.BINARY

        # Get some arbitrary idx from the cut
        i = cut_idx[0]

        F_support = [f for f in F if f not in F_star]

        for n in reversed(tree.B):
            alpha[cut_idx, n] = model.addVar(vtype=GRB.CONTINUOUS, ub=1)

            if n in tree.layers[-2]:
                # Special case for branch nodes which are parents of the leaf nodes
                # alpha allows to equal one at these nodes if they branch on a split feature

                model.addConstr(alpha[cut_idx, n] <= quicksum(b[n,f] for f in F_star))

                cuts_added += 1

            else:
                # Otherwise use the normal recursive constraints for alpha

                alpha_vtype = GRB.CONTINUOUS if disagg_alpha else GRB.BINARY
                alpha[cut_idx, n ,'r'] = model.addVar(vtype=alpha_vtype, ub=1)
                alpha[cut_idx, n, 'l'] = model.addVar(vtype=alpha_vtype, ub=1)

                if disagg_alpha:

                    model.addConstr(alpha[cut_idx, n, 'r'] <= quicksum(b[n, f] for f in F_support if X[i, f] == 1))
                    model.addConstr(alpha[cut_idx, n, 'r'] <= alpha[cut_idx, tree.right_child(n)])

                    model.addConstr(alpha[cut_idx, n, 'l'] <= quicksum(b[n, f] for f in F_support if X[i, f] == 0))
                    model.addConstr(alpha[cut_idx, n, 'l'] <= alpha[cut_idx, tree.left_child(n)])

                    cuts_added += 4

                else:
                    model.addConstr(2 * alpha[cut_idx, n, 'r'] <= quicksum(b[n, f] for f in F_support if X[i, f] == 1) +
                                                                  alpha[cut_idx, tree.right_child(n)])
                    model.addConstr(2 * alpha[cut_idx, n, 'l'] <= quicksum(b[n, f] for f in F_support if X[i, f] == 0) +
                                                                  alpha[cut_idx, tree.left_child(n)])

                    cuts_added += 2

                model.addConstr(alpha[cut_idx, n] <= quicksum(b[n,f] for f in F_star) + alpha[cut_idx, n, 'r'] + alpha[cut_idx, n, 'l'])

                cuts_added += 1

        bounding_term = alpha[cut_idx, 1]

        return bounding_term, cuts_added

    def add_cuts(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables

        b = variables['b']
        theta = variables['theta']

        X, y = data['X'], data['y']

        max_removed = self.opts['Features Removed']
        alpha_version = self.opts['H Variant']
        disagg_alpha = self.opts['Disaggregate Alpha']
        group_selection_enabled = self.opts['Group Selection']

        cut_start_time = time.time()

        # Generate EQP sets. EQP sets have typically been pre-generated and cached so only filtering is needed to grab
        # EQP sets with suitably small split sets.

        eqp_opts = {'Removed Features': max_removed,
                    'Method': 'Graph',
                    'Filter Dominated': self.opts['Filter Dominated'],
                    'Beta Dependence': self.opts['Beta Dependence']}

        eqp_cut_generator = EQPSets(eqp_opts, data)
        eqp_cuts = eqp_cut_generator.get_info(force_encoding=True)

        alpha = {}
        G = {}

        cuts_added = 0

        bounding_terms = {}

        for cut_info in eqp_cuts:

            cut_idx, _, _, dependents = cut_info

            if self.opts['Beta Dependence'] and (dependents is not None):
                bounding_term = quicksum(bounding_terms[dependent_cut_idx] for dependent_cut_idx in dependents)
                bounding_terms[cut_idx] = bounding_term

            else:
                # Run subroutines to define alpha variables, link them to the tree structure and construct the bounding term
                # for the constraint on theta based on the variant of H
                if alpha_version == 'Chain':
                    bounding_term, alpha_cuts_added = self.chain_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha)
                elif alpha_version == 'Recursive':
                    bounding_term, alpha_cuts_added = self.recursive_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha)
                elif alpha_version == 'Basic':
                    bounding_term, alpha_cuts_added = self.basic_alpha_constraints(model, alpha, cut_info, b, data, tree)

                cuts_added += alpha_cuts_added
                bounding_terms[cut_idx] = bounding_term

            if group_selection_enabled:
                # If group selection is enabled, use the bounding term to force the model to choose one
                # group from the EQP set to be allowed to be classified correctly

                cut_idx, _, F_star, _ = cut_info

                # Find class groups in the EQP set
                groups = {}
                for idx in cut_idx:
                    sample_label = y[idx]
                    if sample_label not in groups:
                        groups[sample_label] = []
                    groups[sample_label].append(idx)

                # Create a variable for each grouping
                for k in groups:
                    G[F_star, cut_idx, k] = model.addVar(vtype=GRB.CONTINUOUS)

                for k, group_idx in groups.items():
                    # Samples can only be correctly classified if group variable equals one or the samples are split
                    model.addConstr(quicksum(theta[i] for i in group_idx) == len(group_idx) * (G[F_star, cut_idx, k]))

                # At most one group can be active unless the samples are split
                lhs = quicksum(G[F_star, cut_idx, k] for k in groups)
                rhs = 1 + (len(groups) - 1) * bounding_term
                model.addConstr(lhs <= rhs)

                cuts_added += len(groups) + 1

            else:
                # If group selection is disabled, add a bound over the EQP set which is active whenever
                # the bounding term is forced to zero
                cut_idx, rhs_bound, _, _ = cut_info
                model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs_bound + (len(cut_idx) - rhs_bound) * bounding_term)

                cuts_added += 1

        cut_runtime = time.time() - cut_start_time

        self.update_cut_stats(cuts_added, cut_runtime, ('Auxiliary Vars', len(alpha) + len(G)))

class BendersCuts(CallbackSubroutine):
    """Class for implementing the Benders cuts

    Unlike all other callback subroutines the Benders cuts are enabled by default. Disabling them will almost certainly
    result in an invalid model

    Settings:
        Enhanced Cuts {True,False}: Enables the strengthened Benders cuts at terminal nodes
        Relax w {True, False}: Decides if the prediction variables are relaxed or not.
        EC Level {1,2}: Current invalid setting. Determines height from terminal nodes to apply strengthening.
        Setting anything but the default of 2 may break code

    """


    name = 'Benders Cuts'
    priority = 100

    def __init__(self, user_opts):

        default_settings = {'Enabled': True,
                            'Enhanced Cuts': False,
                            'Relax w': False,
                            'EC Level': 1}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def valid_settings(self, model_opts=None, data=None):

        enhanced_cuts = self.opts['Enhanced Cuts']
        relax_w = self.opts['Relax w']

        settings_valid = True
        log_messages = []

        if enhanced_cuts and relax_w:
            log_messages.append('Modified Benders cuts are not valid when relaxing w variables')

        return settings_valid, log_messages

    def update_model(self, model):
        model.Params.LazyConstraints = 1

        if self.opts['Enhanced Cuts']:
            if not self.opts['Relax w']:
                # If 'Relax w' is set to false then we force w to be binary and relax p
                w = model._variables['w']
                # p = model._variables['p']

                for k in w:
                    w[k].vtype = GRB.BINARY

                # for k in p:
                #     p[k].vtype = GRB.CONTINUOUS

    def run_subroutine(self, model, where, callback_generator):

        if where == GRB.Callback.MIPSOL:

            EPS = 1e-4  # Tolerance for determining if cut is violated

            data = model._data
            tree = model._tree
            variables = model._variables

            subroutine_start_time = time.time()

            I = data['I']
            F = data['F']
            X = data['X']
            y = data['y']

            b = variables['b']
            p = variables['p']
            w = variables['w']
            theta = variables['theta']

            bV = model.cbGetSolution(b)
            pV = model.cbGetSolution(p)
            wV = model.cbGetSolution(w)
            thetaV = model.cbGetSolution(theta)

            # Get route of samples through the tree, and branch features which will be in Benders cuts for each sample
            # See DFS method for detailed output
            DFS_result = callback_generator.DFS(1, I, bV, pV, tree, F, X, cut_vars=True)
            sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars = DFS_result

            cuts_added = 0

            for i in I:
                leaf_node = sample_node_path[i][-1] # Leaf node is the last node on the sample path

                # Check if standard cut is violated
                if wV[y[i], leaf_node] < thetaV[i] - EPS:

                    # These are the branch nodes if the leaf is an internal node. If the leaf is terminal leave as empty list
                    extra_cut_vars = F if leaf_node in tree.B else []

                    if self.opts['Enhanced Cuts']:
                        if leaf_node in tree.L:
                            parent_node = tree.parent(leaf_node)
                            sibling_node = tree.sibling(leaf_node)

                            if wV[y[i], leaf_node] + wV[y[i], sibling_node] / 2 < thetaV[i] - EPS:

                                cut_branch_vars_upper = []
                                cut_branch_vars_lower = []

                                # Separate out the branch variables related to the parent node
                                for (n,f) in cut_branch_vars[i]:
                                    if n == parent_node:
                                        cut_branch_vars_lower.append((n,f))
                                    else:
                                        cut_branch_vars_upper.append((n,f))

                                tCon = (theta[i] <= (quicksum(b[n, f] for n, f in cut_branch_vars_upper) +
                                                     quicksum(w[y[i], n] for n in sample_node_path[i]) +
                                                     quicksum(b[n, f] for n, f in cut_branch_vars_lower) / 2 +
                                                     w[y[i], sibling_node] / 2))

                            else:
                                tCon = (theta[i] <= (quicksum(b[n, f] for n, f in cut_branch_vars[i]) +
                                                     quicksum(w[y[i], n] for n in sample_node_path[i])))

                        else:
                            # Construct the standard cut
                            tCon = (theta[i] <= (quicksum(b[n, f] for n, f in cut_branch_vars[i]) +
                                                 quicksum(b[leaf_node, f] for f in extra_cut_vars) +
                                                 quicksum(w[y[i], n] for n in sample_node_path[i])))

                    else:
                        # Construct the standard cut
                        tCon = (theta[i] <= (quicksum(b[n, f] for n, f in cut_branch_vars[i]) +
                                             quicksum(b[leaf_node, f] for f in extra_cut_vars) +
                                             quicksum(w[y[i],n] for n in sample_node_path[i])))

                    model.cbLazy(tCon)
                    cuts_added += 1

            # Store whether the MP was valid. Used by solution polishing primal heuristic
            callback_generator.callback_cache['Temporary']['Valid Solution'] = (cuts_added == 0)

            # fractional_thetas_found = False
            #
            # if cuts_added == 0:
            #     for i in I:
            #         theta_val = thetaV[i]
            #         if (theta_val > 0.01) and (theta_val < 0.99):
            #             print(i, theta_val)
            #             fractional_thetas_found = True
            #
            # if fractional_thetas_found:
            #     for k, I_k in data['I_k'].items():
            #         print(k, sum(thetaV[idx] for idx in I_k) / len(I_k))
            #     callback_generator.terminate_opt = True

            subroutine_runtime = time.time() - subroutine_start_time
            self.update_subroutine_stats(cuts_added, subroutine_runtime)

    def useful_settings(self, model_opts=None, data=None):

        if self.opts['Enhanced Cuts']:
            return True, None
        else:
            return False, None

class SolutionPolishing(CallbackSubroutine):
    """Class for implementing the solution polishing primal heuristic

    Settings:
        Check Validity {True,False}: When enabled the primal heuristic only runs if the MP solutions is feasible w.r.t the full model

    """

    name = 'Solution Polishing'
    priority = 50

    def __init__(self, user_opts):

        default_settings = {'Enabled': False,
                            'Check Validity': False,
                            'Use Cache': True}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self, model):

        data = model._data

        X = data['X']
        y = data['y']

        unique_classes, unique_idx = np.unique(y, return_index=True)
        X_dummy = X[unique_idx, :]
        y_dummy = y[unique_idx]

        # Make a call to the D2S subroutine to ensure that numba functions are not compile during callback
        opt_D2_subtree_worst_case(X_dummy, y_dummy, unique_classes)

    def run_subroutine(self, model, where, callback_generator):
        if where == GRB.Callback.MIPSOL:

            if self.opts['Check Validity']:
                # Check if the solution was valid w.r.t the Benders cuts
                # and only operate on valid integral solutions
                if not callback_generator.callback_cache['Temporary']['Valid Solution']:
                    return

            # Get the current GLOBAL best solution (not the current solution)
            CurrObj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

            if CurrObj < -100:
                return

            data = model._data
            tree = model._tree
            variables = model._variables
            _lambda = model._lambda

            I = data['I']
            F = data['F']
            K = data['K']
            X = data['X']
            y = data['y']

            b = variables['b']
            p = variables['p']
            w = variables['w']
            theta = variables['theta']
            t = variables['t']

            bV = model.cbGetSolution(b)
            pV = model.cbGetSolution(p)
            wV = model.cbGetSolution(w)

            subroutine_start_time = time.time()

            # Get route of samples through the tree. See DFS method documentation for detailed output
            DFS_result = callback_generator.DFS(1, I, bV, pV, tree, F, X, cut_vars=True)
            _, samples_in_node, node_branch_feature, _ = DFS_result

            # If a cache hasn't been created, then create a persistent cache for D2S subroutine solution
            if 'D2SubtreeCache' not in callback_generator.callback_cache['Persistent']:
                callback_generator.callback_cache['Persistent']['D2SubtreeCache'] = {}

            root_node = create_recursive_tree(I, F, pV, wV, node_branch_feature, samples_in_node, tree)

            if self.opts['Use Cache']:
                cache = callback_generator.callback_cache['Persistent']['D2SubtreeCache']
            else:
                cache = None
            # Call wrapper function which finds subtree roots, runs D2S subroutine at each subtree root, and returns updated tree
            optimised_subtree = optimise_subtrees_worst_case(X, y, tree, model._opts, node_branch_feature, root_node, _lambda,
                                                             cache=cache)

            # Unpack updated solution
            b_subtrees, p_subtrees, w_subtrees, theta_polished = optimised_subtree

            soln_added = 0

            if b_subtrees is not None:
                theta_polished_np = np.asarray(theta_polished)

                min_polished_accuracy = float('inf')

                for k in K:
                    class_k_idx = (y == k)

                    class_k_polished_accuracy = theta_polished_np[class_k_idx].sum() / sum(class_k_idx)

                    if class_k_polished_accuracy < min_polished_accuracy:
                        min_polished_accuracy = class_k_polished_accuracy

                obj_polished = min_polished_accuracy - _lambda * sum(p_subtrees.values())

                # Only accept new solution if it improves on current solution by at least 0.1%
                if obj_polished > CurrObj * (1 + 0.1 / 100):
                    # Update the current incumbent
                    bV |= b_subtrees
                    wV |= w_subtrees
                    thetaV = theta_polished

                    # Theoretically should update p as well but given b and w it should be highly constrained
                    model.cbSetSolution(b, bV)
                    model.cbSetSolution(w, wV)
                    model.cbSetSolution(theta, thetaV)
                    # model.cbSetSolution(t, min_polished_accuracy)



                    # Call solution completers to complete possibly partial solution
                    for sc in model._solution_completers:
                        sc(model, {'b': bV, 'p': pV, 'w': wV, 'theta':thetaV}, 'Callback')



                    pass

                    soln_added += 1
                    print(f'**** Callback Primal Heuristic improved solution from {CurrObj} to {obj_polished} ****')

            subroutine_runtime = time.time() - subroutine_start_time
            self.update_subroutine_stats(soln_added, subroutine_runtime)

    def valid_settings(self, model_opts=None, data=None):

        settings_valid = True
        log_messages = []

        try:
            # Depth two subroutine may not work with compressed datasets
            if data['compressed']:
                log_messages.append('Solution Polishing D2S subroutine not tested with compressed datasets')
                settings_valid = False

            if model_opts['depth'] < 3:
                log_messages.append(f'Solution Polishing D2S subroutine is not useful for tree with a depth of less than 3')
                settings_valid = False

        except KeyError as err:
            log_messages.append(f'Unable to assess validity of Solution Polishing settings due to KeyError on key {err.args[0]}')
            settings_valid = False

        except Exception as err:
            log_messages.append(f'Failed to check validity of Solution Polishing settings with Exception {type(err).__name__}')
            settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class PathBoundCuttingPlanes(CallbackSubroutine):
    """Path bound cutting planes

    Settings:
        Endpoint Only {True,False}: If enabled cuts are only added at the endpoints of integral paths
        Cut Type {'Lazy','User'}: Type of cuts added. Lazy (cbLazy) or user (cbCut)
        Bound Negative Samples {True,False}: Modifies the basic cut to force misclassified samples in subtree to zero
        Bound Structure {True,False}: Additional cut which constraints the structure of the subtree to be the optimal structure
        Cut Focus {'Samples','Objective'}: Not currently implemented. Has no effect

    """

    name = 'Path Bound Cutting Planes'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Endpoint Only': False,
                            'Check Violation': True,
                            'Cut Type': 'Lazy'}

        self.path_cache = {}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self, model):
        model.Params.LazyConstraints = 1

    def add_cut(self, model, lhs, rhs):
        """Helper function to add either lazy or user cuts depending on settings

        Args:
            model (grbModel): Gurobi model object
            lhs (LinExpr): Left hand side of constraint
            rhs (LinExpr): Right hand side of constraint

        """
        if self.opts['Cut Type'] == 'Lazy':
            model.cbLazy(lhs <= rhs)
        else:
            model.cbCut(lhs <= rhs)

    # @line_profiler.profile
    def run_subroutine(self, model, where, callback_generator):

        EPS = 1e-4

        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:

            subroutine_start_time = time.time()
            cuts_added = 0

            # Load in required data
            variables = model._variables
            data = model._data
            tree = model._tree
            _lambda = model._lambda

            I = np.asarray(data['I'])   # Load in as numpy array to use efficient bitmasking operations
            I_k = data['I_k']
            F = data['F']
            K = data['K']
            X = data['X']
            y = data['y']

            b = variables['b']
            p = variables['p']
            w = variables['w']
            theta = variables['theta']
            t = variables['t']

            bR = model.cbGetNodeRel(b)
            wR = model.cbGetNodeRel(w)
            pR = model.cbGetNodeRel(p)
            thetaR = model.cbGetNodeRel(theta)
            tR = model.cbGetNodeRel(t)

            # root_node is an instance of PathNode, and by traversal root_node contains all integral paths found
            root_node = callback_generator.get_integral_paths(bR, X, I, F, tree, min_height=1)

            # # If a cache hasn't been created, then create a persistent cache for D2S subroutine solution
            # if 'D2SubtreeCache' not in callback_generator.callback_cache['Persistent']:
            #     callback_generator.callback_cache['Persistent']['D2SubtreeCache'] = {}

            # D2SubtreeCache = callback_generator.callback_cache['Persistent']['D2SubtreeCache']

            endpoint_only = self.opts['Endpoint Only']

            for node_info in callback_generator.explore_tree(root_node, yield_endpoint_only=endpoint_only):

                n, I_mask, path, _  = node_info

                if n in tree.L:
                    raise Exception('PBCP attempted to execute at a terminal node')

                # Run some basic checks for situations in which we do not want to run
                if I_mask.sum() == 0:
                    continue

                path_key = frozenset((branch_var, dir) for _, branch_var, dir in path)

                # Check if we have already seen this path
                if path_key in self.path_cache:
                    D, min_num_misclassified = self.path_cache[path_key]
                else:
                    FQ1 = np.zeros((len(K), len(F)), np.int16)
                    FQ11 = np.zeros((len(K), len(F), len(F)), np.int16)

                    y_I = y[I_mask]

                    D = np.zeros(len(K),dtype=np.int16)
                    unique_classes, D_partial = np.unique(y_I, return_counts=True)

                    if len(unique_classes) < len(K):
                        pass

                    for k, d in zip(unique_classes, D_partial):
                        D[k] = d

                    for k in K:
                        # Technically collecting the frequency counters could be done without looping over K
                        # It's beyond me how to do it with numpy though

                        # Take the subarray of X in which all samples have class y^i = k and features are filtered by F_mask
                        X_masked = X[I_mask][y[I_mask]==k]

                        # Each column of X_masked is associated with feature f, and indicates which samples have x_f^i == 1
                        # Taking the dot product of columns associated with features f_a and f_b will be equal to the number of
                        # samples for which (x_fa^i == 1 AND x_fb^i == 1)
                        FQ11[k, :, :] = X_masked.T @ X_masked

                        # The diagonal corresponds to duplicated features, i.e. x_fa^i == 1
                        FQ1[k, :] = np.diag(FQ11[k, :, :])

                    FQ0 = np.expand_dims(D, axis=1) - FQ1

                    # |K| x |F| array where entry (k,f) is the minimum number of samples with class k that MUST be misclassified if we branch on
                    # feature f in the subtree root node
                    min_num_misclassified = np.expand_dims(D, axis=1) - np.maximum(FQ0, FQ1)

                    min_num_misclassified = np.minimum(FQ0, FQ1)

                    self.path_cache[path_key] = D.tolist(), min_num_misclassified.tolist()

                subtree_nodes = tree.descendants(n)

                subtree_root = n
                opt_subtree = subtree_nodes[:3]
                downstream_subtree = subtree_nodes[3:]

                for k in K:

                    if D[k] == 0:
                        continue

                    if self.opts['Check Violation']:

                        # First construct upper bound from relaxation solution and check if it is actually violated

                        pred_in_path = D[k] * sum(wR[kk, node] for (node, _, _) in path for kk in K if kk != k)
                        pred_in_root = D[k] * sum(wR[kk, subtree_root] for kk in K if kk != k)
                        branch_in_root = sum(min_num_misclassified[k][f] * bR[subtree_root, f] for f in F)

                        # if min_num_misclassified[k, :].max() > relaxation_coeff:
                        #     print('ERROR: HERE')

                        relax_path_branch = D[k] * sum(bR[node, f] for (node, ff, _) in path for f in F if f != ff)
                        relax_downstream = D[k] * sum(pR[n_d] for n_d in downstream_subtree)

                        relax_bound = relax_path_branch + relax_downstream

                        ub = 1 - (1 / len(I_k[k])) * (pred_in_root + pred_in_path + branch_in_root - relax_bound)

                        bound_violated = (tR - ub > 1e-6)
                    else:
                        bound_violated = True

                    # Don't bother checking??
                    if bound_violated:

                        # print(f'Adding cut where t={tR} violated upper bound {ub}')

                        pred_in_path_vars = D[k] * quicksum(w[kk, node] for (node, _, _) in path for kk in K if kk != k)
                        pred_in_root_vars = D[k] * quicksum(w[kk, subtree_root] for kk in K if kk != k)
                        branch_in_root_vars = quicksum(min_num_misclassified[k][f] * b[subtree_root, f] for f in F)

                        # relaxation_coeff = D[k]

                        # relaxing_vars = (quicksum(b[node, f] for (node, ff, _) in path for f in F if f != ff) +
                        #                  quicksum(p[node] for (node, _, _) in path) +
                        #                  quicksum(p[n_d] for n_d in downstream_subtree))
                        #
                        # relaxation_bound_vars = relaxation_coeff * relaxing_vars

                        relax_path_branch_vars = D[k] * quicksum(b[node, f] for (node, ff, _) in path for f in F if f != ff)
                        relax_downstream_vars = D[k] * quicksum(p[n_d] for n_d in downstream_subtree)

                        relax_bound_vars = relax_path_branch_vars + relax_downstream_vars

                        rhs = 1 - (1 / len(I_k[k])) * (pred_in_root_vars + pred_in_path_vars + branch_in_root_vars - relax_bound_vars)

                        self.add_cut(model,
                                     t,
                                     rhs)

                        cuts_added += 1

            subroutine_runtime = time.time() - subroutine_start_time
            self.update_subroutine_stats(cuts_added, subroutine_runtime)

    def valid_settings(self, model_opts=None, data=None):
        """

        Args:
            model_opts (dict):
            data (dict): Dictionary containing

        Returns:

        """

        settings_valid = True
        log_messages = []

        try:
            if model_opts['depth'] < 2:
                log_messages.append(f'Path Bound Cutting Planes are not useful for tree with a depth of less than 3')
                settings_valid = False

            if self.opts['Cut Type'] not in ['Lazy', 'User']:
                log_messages.append(f'Path Bound Cutting Planes not valid for {self.opts['Cut Type']} cut type. Please try "Lazy" or "User"')
                settings_valid = False

        except KeyError as err:
            log_messages.append(f'Unable to assess validity of Path Bound Cutting Planes settings due to KeyError on key {err.args[0]}')
            settings_valid = False

        except Exception as err:
            log_messages.append(f'Failed to check validity of Path Bound Cutting Planes settings with Exception {type(err).__name__}')
            settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class BendersCallback(GenCallback):

    name = 'BendWorstOCT Callback'

    def __init__(self, callback_settings):

        available_subroutines = [BendersCuts,
                                 SolutionPolishing,
                                 PathBoundCuttingPlanes]

        super().__init__(available_subroutines, callback_settings)

    def DFS(self, root, I, bV, pV, tree, F, X, cut_vars=False, changed_root_branch=None):
        """Tracks the route of each sample through the tree based on integral MP solution

        Note: Does not really do a depth-first search, DFS is a poor choice of name

        This is a helper function which is made available to callback subroutines to track the path of samples through
        a tree defined by an integral MP solution.

        Args:
            root (int): Root node to begin tracing sample routes from
            I (list): subset of samples on which to run search
            bV: branch decision variables to use
            pV: Node choice decision variables (branch or leaf node)
            tree:
            F: Feature set
            X: Feature data
            cut_vars: If True, keep track of branch variables which would have sent each sample onto a different leaf
            changed_root_branch: Substituted branch variable for root node

        Returns:
            Returns a tuple (sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars) where -
                sample_node_path (dict): Keys are sample indices. Each entry is a list (in order) of the nodes on the path the sample follows
                samples_in_node (dict): Keys are nodes. Each entry is a list containing the samples which were routed into that node
                node_branch_feature (dict): Keys are nodes. Each entry if the feature branch on at said node
                cut_branch_vars (dict): Keys are sample indices. Each entry is a list of tuple (n,f) of feature f which could
                have been branched on at node n to send the sample into another leaf node. Only populated if cut_vars=True is set.
        """

        # Check if we have already cached the result
        DFS_result = self.callback_cache['Temporary'].get('DFS_result', None)

        if DFS_result is None:
            subtree_branch_nodes, subtree_leaf_nodes = tree.descendants(root, split_nodes=True)
            node_branch_feature = {}

            branch_feature = None

            # Fill in branch features at each branch node
            for n in subtree_branch_nodes:
                for f in F:
                    if bV[n, f] > 0.5:
                        node_branch_feature[n] = f

            if changed_root_branch is not None:
                node_branch_feature[root] = changed_root_branch

            sample_node_path = {i: [] for i in I}
            samples_in_node = {n: [] for n in subtree_branch_nodes + subtree_leaf_nodes}

            cut_branch_vars = {i: [] for i in I}

            # Run a DFS from the root down to the leaves for each sample
            for i in I:
                sample = X[i, :]

                current_node = root

                # When pV[current_node] == 1 then we have reached a leaf node
                while pV[current_node] < 0.5:
                    sample_node_path[i].append(current_node)
                    samples_in_node[current_node].append(i)

                    branch_feature = node_branch_feature[current_node]

                    if sample[branch_feature] == 0:
                        # Sample branches to the left
                        # Find features that would have sent the sample down the right branch if branched on
                        if cut_vars:
                            for f in F:
                                if sample[f] == 1:
                                    cut_branch_vars[i].append((current_node, f))

                        current_node = tree.left_child(current_node)

                    else:
                        # Sample branches to the right
                        # Find features that would have sent the sample down the left branch if branched on
                        if cut_vars:
                            for f in F:
                                if sample[f] == 0:
                                    cut_branch_vars[i].append((current_node, f))

                        current_node = tree.right_child(current_node)

                # Sample i now in a leaf node
                samples_in_node[current_node].append(i)
                sample_node_path[i].append(current_node)

            DFS_result = sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars
            self.callback_cache['Temporary']['DFS_result'] = DFS_result

        return DFS_result

    def get_integral_paths(self, bR, X, I, F, tree, min_height=2):


        EPS = 1e-4

        # Check if another subroutine has already found the integral paths
        root_node = self.callback_cache['Temporary'].get('Integral Paths', None)

        if root_node is None:
            root_node = PathNode(1, np.ones_like(I, dtype=bool), 0)

            to_explore = [root_node]

            while len(to_explore) > 0:
                node = to_explore.pop()

                n = node.n
                I_mask = node.I_mask

                height = tree.depth - node.depth
                if height == min_height:
                    # Stop exploring when we reach nodes with a height of min_height
                    # E.g.
                    continue

                for f in F:
                    if bR[n, f] > 1 - EPS:
                        # Node is internal to an integral path
                        node.internal_node = True

                        node.f = f

                        left_mask = (X[:, f] == 0)
                        right_mask = ~left_mask

                        node.left_child = PathNode(tree.left_child(n), I_mask & left_mask, node.depth + 1)
                        node.right_child = PathNode(tree.right_child(n), I_mask & right_mask, node.depth + 1)

                        node.left_child.parent = node
                        node.right_child.parent = node

                        to_explore.append(node.left_child)
                        to_explore.append(node.right_child)

                        break

            root_node.path = tuple()

            self.callback_cache['Temporary']['Integral Paths'] = root_node

        return root_node


    def explore_tree(self, node, yield_endpoint_only=False):
        """ Recursive generator method which runs a DFS search on the tree and returns info for each node
        Args:
            node (PathNode): Instance of PathNode class which represents a node in an integral path

        Returns:
        """

        # Setting cut_path or cut_parent to True allows the caller to do the following:
        #   cut_path - Cut the path below the current node (i.e. if it is an internal node make it an endpoint)
        #   cut_self - Cut the current node from the path, so that the parent only has one child in the set of integral paths
        #   cut_parent - Cut off the current node and it's sibling. If this is true then it was provably
        #                suboptimal for the parent to make the branch decision that it made


        cut_path, cut_self, cut_parent = False, False, False

        if not node.internal_node:
            # Always return back info for endpoint of path
            caller_sent = (yield node.n, node.I_mask, node.path, node.internal_node)

            if caller_sent is not None:
                cut_path, cut_self, cut_parent = caller_sent

            if cut_self:
                if node.n % 2 == 0:
                    node.parent.left_child = None
                else:
                    node.parent.right_child = None

            if cut_parent:
                node.parent.left_child = None
                node.parent.right_child = None
                node.parent.internal_node = False
                node.parent.f = None

        else:
            # In internal nodes always explore left and right children.
            # Do not yield the current node if the 'Endpoint Only' option is set.
            if not yield_endpoint_only:
                caller_sent = (yield node.n, node.I_mask, node.path, node.internal_node)

                if caller_sent is not None:
                    cut_path, cut_self, cut_parent = caller_sent

            if cut_parent:
                node.parent.left_child = None
                node.parent.right_child = None
                node.parent.internal_node = False
                node.parent.f = None
                return

            if cut_self:
                if node.n % 2 == 0:
                    node.parent.left_child = None
                else:
                    node.parent.right_child = None

            if cut_path:
                # If the caller send cut_path=True then we cut off the left and right children of the current node
                node.left_child = None
                node.right_child = None
                node.internal_node = False
                node.f = None

            if node.left_child is not None:
                left_child = node.left_child
                left_child.path = node.path + ((node.n, node.f, 0),)

                yield from self.explore_tree(left_child, yield_endpoint_only=yield_endpoint_only)

            if node.right_child is not None:
                right_child = node.right_child
                right_child.path = node.path + ((node.n, node.f, 1),)

                yield from self.explore_tree(right_child, yield_endpoint_only=yield_endpoint_only)

    def get_stats_log(self):
        """Parse statistics for each subroutine for logging to file and console
        """

        log_lines = ['\nCallback Statistics:\n']
        logged_results = {}

        for subroutine in self.subroutines:

            stats = subroutine.stats
            subr_name = subroutine.name

            if subr_name in ['Benders Cuts', 'Path Bound Cutting Planes']:
                num_cuts = stats['Num']
                cut_time = stats['Time']

                if subroutine.opts['Enabled']:
                    log_lines.append(f'{subr_name} - Added {num_cuts} cuts in {cut_time:.2f}s\n')

                logged_results[f'{subr_name} - Cuts Added'] = num_cuts
                logged_results[f'{subr_name} - Time'] = cut_time

            if subr_name in ['Solution Polishing']:
                num_solns = stats['Num']
                soln_time = stats['Time']

                if subroutine.opts['Enabled']:
                    log_lines.append(f'{subr_name} - Found {num_solns} improving solutions in {soln_time:.2f}s\n')

                logged_results[f'{subr_name} - Solutions Found'] = num_solns
                logged_results[f'{subr_name} - Time'] = soln_time

        if len(log_lines) == 1:
            log_printout = None
        else:
            log_printout = ''.join(log_lines)

        return log_printout, logged_results

class BendersInitialCuts(InitialCutManager):

    name = 'BendWorstOCT Cut Manager'

    def __init__(self, cut_settings):

        available_cuts = [EQPInitialCut]

        super().__init__(available_cuts, cut_settings)

    def get_stats_log(self):
        """Parse statistics for each subroutine for logging to file and console
        """

        log_lines = ['\nInitial Cut Statistics:\n']
        logged_results = {}

        for cut in self.cuts:
            stats = cut.stats
            cut_name = cut.name

            if cut_name == 'EQP Initial Cuts':
                num_added_vars = 0
                num_cuts = stats['Num']
                cut_time = stats['Time']

                if cut.opts['Enabled']:
                    num_added_vars = stats['Auxiliary Vars']
                    log_lines.append( f'{cut_name} - Added {num_cuts} cuts and {num_added_vars} variables in {cut_time:.2f}s\n')

                logged_results[f'{cut_name} - Auxiliary Vars'] = num_added_vars
                logged_results[f'{cut_name} - Cuts'] = num_cuts
                logged_results[f'{cut_name} - Time'] = cut_time

        if len(log_lines) == 1:
            log_printout = None
        else:
            log_printout = ''.join(log_lines)

        return log_printout, logged_results

class BendWorstOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params, callback_generator=BendersCallback, cut_manager=BendersInitialCuts)
        self.model_type = 'BendWorstOCT'

    def add_vars(self,model):

        data = model._data
        tree = model._tree

        I = data['I']
        F = data['F']
        K = data['K']

        b = {(n, f): model.addVar(vtype=GRB.BINARY, name=f'b_{n},{f}')
             for n in tree.B for f in F}
        p = {n: model.addVar(vtype=GRB.BINARY, name=f'p_{n}')
             for n in tree.T}
        w = {(k, n): model.addVar(vtype=GRB.BINARY, name=f'w_{k}^{n}')
             for k in K for n in tree.T}
        theta = {i: model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'theta_{i}')
                 for i in I}
        t = model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f't')

        model._variables = {'b': b,
                            'p': p,
                            'w': w,
                            'theta': theta,
                            't': t}

    def add_constraints(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree

        I_k = data['I_k']
        F = data['F']
        K = data['K']

        b = variables['b']
        p = variables['p']
        w = variables['w']
        theta = variables['theta']
        t = variables['t']

        # At each possible branch node must either branch, make a prediction, or have made a prediction at an ancestor
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) + quicksum(p[n_a] for n_a in tree.ancestors(n)) + p[n] == 1)
                           for n in tree.B}

        # Must make a prediction at exactly one node in each path through the tree
        one_prediction_per_path = {n: model.addConstr(p[n] + quicksum(p[n_a] for n_a in tree.ancestors(n)) == 1)
                                   for n in tree.L}

        # Make a single class prediction at each prediction node where p_n = 1
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == p[n])
                           for n in tree.T}

        # Bound classification score of class k by theta values for samples with true class k
        class_acc_bound = {k: model.addConstr(t <= quicksum(theta[i] for i in I_k[k]) / len(I_k[k]))
                           for k in K}

    def add_objective(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree
        _lambda = model._lambda

        p = variables['p']
        t = variables['t']

        I = data['I']

        worst_case_accuracy = t
        complexity = _lambda * quicksum(p[n] for n in tree.T)

        model.setObjective(worst_case_accuracy - complexity, GRB.MAXIMIZE)

    def warm_start(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables
        _lambda = model._lambda

        compressed = data['compressed']
        I = data['I']

        if self.opt_params['Polish Warmstart']:
            model._opts.add('CART polish solutions')
            model._opts.add('Polish Worst Case')

        if compressed:
            X, y = data['Xf'], data['yf']
        else:
            X, y = data['X'], data['y']

        b = variables['b']
        p = variables['p']
        w = variables['w']
        theta = variables['theta']

        heuristic_start_time = time.time()

        HeuristicSoln = Custom_CART_Heuristic(X, y, tree, model._opts,
                                              alpha=_lambda,
                                              cat_feature_maps=data['Categorical Feature Map'],
                                              num_feature_maps=data['Numerical Feature Map'])

        if HeuristicSoln is not None:
            for k, v in HeuristicSoln['b'].items():
                b[k].Start = v

            for k, v in HeuristicSoln['p'].items():
                p[k].Start = v

            for k, v in HeuristicSoln['w'].items():
                w[k].Start = v

            for i, v in enumerate(HeuristicSoln['theta']):
                if compressed:
                    idx_map = self.data['idxf_to_idxc']
                    j = idx_map[i]
                    theta[j].Start = v
                else:
                    theta[i].Start = v

            for sc in model._solution_completers:
                sc(model, HeuristicSoln, 'Warm Start')

            heur_obj = sum(HeuristicSoln['theta']) / len(I) - _lambda * HeuristicSoln['num leaves']
            heur_runtime = time.time() - heuristic_start_time

            if 'theta old' in HeuristicSoln:
                heur_unpolished_obj = sum(HeuristicSoln['theta old']) / len(I) - _lambda * HeuristicSoln['num leaves old']
                print(f'CART returned Heuristic Solution with {heur_obj}/{len(y)} samples classified '
                      f'(polished from {heur_unpolished_obj}) correctly in {time.time() - heuristic_start_time:.2f}s')

                self.update_model_stats('CART',
                                        heur_obj,
                                        heur_runtime,
                                        ('Unpolished Obj', heur_unpolished_obj))
            else:
                print(f'CART returned Heuristic Solution with {heur_obj}/{len(y)} samples classified '
                      f'correctly in {heur_runtime:.2f}s')

                self.update_model_stats('CART',
                                        heur_obj,
                                        heur_runtime)

        else:
            log_error(140,'CART did not return a valid heuristic solution')

    def save_model_output(self, user_vars):

        bS, pS, wS, thetaS = user_vars

        lines = []

        lines.append('\n' + '*' * 5 + ' BRANCH VARIABLES ' + '*' * 5 + '\nnode:feature')
        for node, feature in bS:
            lines.append(f'{node}:{feature}')

        lines.append('\n' + '*' * 5 + ' PREDICTION VARIABLES ' + '*' * 5 + '\nleaf:predicted class')
        for node, pred in wS:
            lines.append(f'{node}:{pred}')

        lines.append('\n' + '*' * 5 + ' CORRECTLY CLASSIFIED SAMPLES ' + '*' * 5)
        for i in thetaS:
            lines.append(f'{i}')

        save_string = '\n'.join(lines)

        return save_string

    def vars_to_readable(self,model):
        """Converts Gurobi model solution into a readable format

        Args:
            model (grbModel): Gurobi model with feasible solutions attached

        Returns:
            Returns a tuple with the following elements:
                bS (list): list of tuples (n,f) of branch nodes and branch features in the tree
                pS (list): List of nodes in the tree at which predictions are made
                wS (list): list of tuples (n,k) or leaf nodes and leaf predictions in the tree
                zS (list): List of lists of nodes traversed by each correctly classified sample
                thetaS (list): List of indices of sample which the tree correctly classifies
        """

        variables = model._variables
        data = model._data
        tree = model._tree

        b = variables['b']
        p = variables['p']
        w = variables['w']
        theta = variables['theta']

        I = data['I']
        F = data['F']
        K = data['K']

        bS = [(n, f) for n in tree.B for f in F if b[n, f].X > 0.1]
        pS = [n for n in tree.T if p[n].X > 0.1]
        wS = [(n, k) for n in tree.T for k in K if w[k, n].X > 0.2]
        thetaS = [theta[i].X for i in I]

        return bS, pS, wS, thetaS
        # return bS, pS, wS, thetaS

    def summarise_tree_info(self, model, user_vars):

        bS, pS, wS, thetaS = user_vars

        data = model._data
        tree = model._tree
        variables = model._variables

        X = data['X']
        y = data['y']
        I = data['I']
        F = data['F']
        K = data['K']

        b = variables['b']
        p = variables['p']

        # Calculate sample paths to get per-class accuracies
        DFS_result = self.callback_generator.DFS(1, I, {k: b[k].X for k in b}, {k: p[k].X for k in p}, tree, F, X)
        sample_path, _, _, _ = DFS_result

        node_prediction = {n: k for (n,k) in wS}
        classification_score_per_class = {k: {'Correct': 0,
                                              'Total': 0}
                                          for k in K}

        for sample_idx, path in sample_path.items():
            sample_class = y[sample_idx]
            leaf_node = path[-1]

            classification_score_per_class[sample_class]['Total'] += 1

            if node_prediction[leaf_node] == sample_class:
                classification_score_per_class[sample_class]['Correct'] += 1

        accuracy = 100 * sum(classification_score_per_class[k]['Correct'] for k in K) / len(I)  # Calculate "True" accuracy
        worst_class_accuracy = min(100 * classification_score_per_class[k]['Correct'] / classification_score_per_class[k]['Total']
                                   for k in K)
        complexity = model._lambda * len(pS)

        self.results_to_log['Accuracy'] = accuracy
        self.results_to_log['Worst Class Accuracy'] = worst_class_accuracy

        leaf_depths = [len(tree.ancestors(n)) for n in pS]
        avg_depth = sum(leaf_depths) / len(pS)
        max_depth = max(leaf_depths)

        self.results_to_log['Average Leaf Depth'] = avg_depth
        self.results_to_log['Max Leaf Depth'] = max_depth
        self.results_to_log['Leaves Used'] = len(pS)

        lines = []

        lines.append(f'Classified {int(sum(thetaS))}/{len(I)} samples correctly')
        lines.append(f'Achieved an accuracy of {accuracy:.2f}%')
        lines.append(f'Achieved a worst-class accuracy of {worst_class_accuracy:.2f}% with an objective of {worst_class_accuracy/100 - complexity:.3f}')
        lines.append(f'Used {len(pS)}/{len(tree.L)} possible leaf nodes')

        return '\n'.join(lines)

    def _eval_obj(self, wV, pV, thetaV, sample_paths):
        pass

    def _check_output_validity(self, model):
        """ Check that the outputted solution is feasible

        For the worst-case classification, there is some added complexity in checking the solutions. Due to the objective,
        one class will essentially "support" the objective, and changes in theta values for samples not of the support class
        will necessarily affect the objective value. Occasionally, Gurobi will supply a heuristic value where theta values
        from non-support classes take fractional values. This is presumably due to the usage of an LP solver, which picks a
        corner solution which makes the bounds on t tight for all classes. Therefore, we check that any fractional thetas
        do not correspond to the supporting class

        This method checks the following conditions hold:
            - All p variables are binary
            - All w variables are binary
            - All theta variables are binary, or can be corrected to a binary value without affecting the solution
        Args:
            model (grbModel): Solved Gurobi model

        Returns:
            Return boolean output_valid which is True is the model solution is valid and False otherwise

        """

        EPS = 1e-4

        variables = model._variables
        data = model._data
        tree = model._tree

        b = variables['b']
        p = variables['p']
        w = variables['w']
        theta = variables['theta']

        output_valid = True
        log_messages = []
        local_warning_messages = []

        for n in p:
            if (p[n].X > EPS) and (p[n].X < 1-EPS):
                output_valid = False
                log_messages.append(f'p_{n} = {p[n].X:.10f} is invalid for binary variable p')

        for (k,n) in w:
            w_equals_zero = abs(w[k,n].X) < EPS
            w_equals_one = abs(w[k,n].X - 1) < EPS
            if not (w_equals_zero or w_equals_one):
                output_valid = False
                log_messages.append(f'w_{k}^n = {w[k,n].X:.10f} is invalid for binary variable w')


        ########################################

        X = data['X']
        y = data['y']
        I = data['I']
        F = data['F']
        K = data['K']

        DFS_result = self.callback_generator.DFS(1, I, {k: b[k].X for k in b}, {k: p[k].X for k in p}, tree, F, X)
        sample_paths, _, _, _ = DFS_result

        # pV = {k: v.X for k,v in p.items()}
        # wV = {k: v.X for k,v in w.items()}

        node_prediction = {}

        for (k,n) in w:
            if w[k,n].X > 0.9:
                node_prediction[n] = k

        classification_score_per_class = {k: {'Correct': 0,
                                              'Total': 0}
                                          for k in K}

        gurobi_classification_score_per_class = {k: {'Correct': 0,
                                                     'Total': 0}
                                                 for k in K}

        for sample_idx, path in sample_paths.items():
            sample_class = y[sample_idx]
            leaf_node = path[-1]

            classification_score_per_class[sample_class]['Total'] += 1
            gurobi_classification_score_per_class[sample_class]['Total'] += 1

            if node_prediction[leaf_node] == sample_class:
                classification_score_per_class[sample_class]['Correct'] += 1

            gurobi_classification_score_per_class[sample_class]['Correct'] += theta[sample_idx].X

        true_support_class = min([(classification_score_per_class[k]['Correct'] / classification_score_per_class[k]['Total'],k)
                                  for k in K])[1]

        for i in theta:
            if (theta[i].X > 10 * EPS) and (theta[i].X < 1 - 10 * EPS):
                # If theta[i] is fractional, check if y[i] is from the support class

                if y[i] == true_support_class:
                    output_valid = False
                    log_messages.append(f'theta_{i} = {theta[i].X:.10f} is invalid for binary variable theta')
                else:
                    local_warning_messages.append(f'theta_{i} takes fractional value {theta[i].X:.3f}')

        # Patchwork way of logging that theta variables took fractional values, but output is still valid
        if len(local_warning_messages) > 0:
            log_error(199, local_warning_messages)

        return output_valid, log_messages