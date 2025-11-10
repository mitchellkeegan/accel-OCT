"""Implements classes for regularised BendOCT"""

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
                             optimise_regularised_subtrees,
                             optimise_regularised_depth2_subtree,
                             create_recursive_tree)

from src.models.base_classes import (OCT,
                                     GenCallback,
                                     InitialCutManager,
                                     CallbackSubroutine,
                                     InitialCut)

from src.models.FlowOCT import FlowOCT
from src.models.BendOCT import BendOCT

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

class CutSetMixin():
    """Mix-in class for cut-set inequalities

    This mixin can be used for initial cuts and callbacks which implement cut-set inequalities. It provides the
    solve_fractional_subproblem method which solves the Benders subproblem with fractional inputs either by
    solving the max-flow LP using Gurobi or finding minimum cuts by inspection by on the 'Solution Method' setting

    """

    def __init__(self, *args, **kwargs):
        """
        Set the tolerance and pass all inputs directly through to the __init__ method of the parent class
        """
        super().__init__(*args, **kwargs)

        # Tolerance for checking constraint violation
        self.EPS = 1e-4

    def valid_settings(self, model_opts=None, data=None):

        settings_valid = True
        log_messages = []

        solution_method = self.opts['Solution Method']

        if solution_method not in ['LP', 'Dual Inspection']:
            log_messages.append(f'Cut Set Inequalities not valid for {solution_method} Solution Method. Please try "LP" or "Dual Inspection"')
            settings_valid = False

        # When implemented in callback subroutine the cut-set inequalities can be lazy or user cuts
        if 'Cut Type' in self.opts:
            if self.opts['Cut Type'] not in ['Lazy', 'User']:
                log_messages.append(f'Cut Set cutting planes not valid for {self.opts['Cut Type']} cut type. Please try "Lazy" or "User"')
                settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        return True, None

    def LP_Solve(self, X, y, b, w, theta, bR, wR, thetaR, F, tree):
        """Solve the min-flow/min-cut subproblem associated with the i^th sample given solutions in (b,w)

        Args:
            X (ndarray): Feature values of the i^th sample
            y (int): Class of the i^th sample
            b (dict): Dictionary of Gurobi variables corresponding to branch decisions
            w: (dict): Dictionary of Gurobi variables corresponding to prediction decisions
            theta (grbVar): Gurobi variable corresponding to the classification of the i^th sample
            bR (dict): Dictionary of branch decision variable fractional solutions
            wR (dict): Dictionary of prediction decision variable fractional solutions
            thetaR (dict): Value of theta in MP solution
            F (range): Range of features indices
            tree (Tree): An instance of the Tree class found in src.utils.trees

        Returns:
            Returns temp constraint object is a cut-set inequality is violated. Otherwise, return None
        """


        LP_model = Model()
        LP_model.Params.OutputFlag = 0

        # Add flow variables for edges leaving branch nodes
        z = {(n1, n2): LP_model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{n1},{n2}')
             for n1 in tree.B for n2 in tree.children(n1)}

        # Add in variables for flow from the source node and to the sink node
        z[(tree.source, 1)] = LP_model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{tree.source},{1}')
        for n in tree.T:
            z[(n, tree.sink)] = LP_model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{n},{tree.sink}')

        # Flow in = flow out at branch nodes
        branch_flow_equality = {n: LP_model.addConstr(z[tree.parent(n), n] ==
                                                      quicksum(z[n, n_child] for n_child in tree.children(n)) + z[n, tree.sink])
                                for n in tree.B}

        # Flow in = flow out at leaf nodes
        leaf_flow_equality = {n: LP_model.addConstr(z[tree.parent(n), n] == z[n, tree.sink])
                              for n in tree.L}

        # Flow from source at most one
        LP_model.addConstr(z[tree.source, 1] <= 1)

        # Bound the left child flow capacity at each branch node
        left_child_capacity = {n: LP_model.addConstr(z[n, tree.children(n)[0]] <=
                                                  quicksum(bR[n, f] for f in F if X[f] < 0.5))
                               for n in tree.B}

        # Bound the right child flow capacity at each branch node
        right_child_capacity = {n: LP_model.addConstr(z[n, tree.children(n)[1]] <=
                                                   quicksum(bR[n, f] for f in F if X[f] > 0.5))
                                for n in tree.B}

        # Set capacity of edges from all nodes to sink node
        sink_flow_bound = {n: LP_model.addConstr(z[n, tree.sink] <= wR[y, n])
                           for n in tree.T}

        # LP_model.setObjective(quicksum(z[n,tree.sink] for n in tree.T), GRB.MAXIMIZE)
        LP_model.setObjective(z[tree.source, 1], GRB.MAXIMIZE)

        LP_model.optimize()

        status = LP_model.Status

        if status == GRB.OPTIMAL:
            if thetaR > LP_model.objVal + self.EPS:
                tcon = theta <= (quicksum(left_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[f] == 0) +
                                 quicksum(right_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[f] == 1) +
                                 quicksum(sink_flow_bound[n].Pi * w[y, n] for n in tree.T))

            else:
                tcon = None
        else:
            # On some platforms we occasionally get status == GRB.INF_OR_UNBD. This is likely a numerics issue.
            # This happens in a small minority of instances so currently we simply ignore when this happens
            # The commented out debugging section below can be used to store the inputs needed to recreate the
            # LP model and raise an exception
            tcon = None

        LP_model.dispose()

        return tcon

        # else:
        #
        #     if status == GRB.INF_OR_UNBD:
        #         # If Gurobi is unable to determine if the model is unbounded or infeasible
        #         # Rerun the model with DualReductions disabled in presolve
        #         LP_model.Params.DualReductions = 0
        #         LP_model.Params.OutputFlag = 1
        #         # LP_model.Params.FeasibilityTol = 1e-2
        #         LP_model.reset()
        #         LP_model.optimize()
        #         status = LP_model.Status
        #
        #     if self.log_dir is not None:
        #         save_info = (X, y, bR, wR, thetaR, F)
        #         debug_filepath = os.path.join(self.log_dir, 'Cutset_debug.pickle')
        #
        #         with open(debug_filepath,'wb') as f:
        #             pickle.dump(save_info,f)
        #
        #     exception_message = f'LP subproblem solve returned status code {status}'
        #     print(exception_message)
        #
        #     raise Exception(exception_message)

    def DI_Solve(self, X, y, b, w, theta, bR, wR, thetaR, F, tree):
        """Solve the min-flow/min-cut subproblem associated with the i^th sample given solutions in (b,w)

        Algorithm works from the sink node up to the source node, maintaining a minimum s-t cut which preferences cuts
        closer to the source node. Output is a minimum s-t cut which is used to generate a cut-set inequality if violated

        Args:
            X (ndarray): Feature values of the i^th sample
            y (int): Class of the i^th sample
            b (dict): Dictionary of Gurobi variables corresponding to branch decisions
            w: (dict): Dictionary of Gurobi variables corresponding to prediction decisions
            theta (grbVar): Gurobi variable corresponding to the classification of the i^th sample
            bR (dict): Dictionary of branch decision variable fractional solutions
            wR (dict): Dictionary of prediction decision variable fractional solutions
            thetaR (dict): Value of theta in MP solution
            F (range): Range of features indices
            tree (Tree): An instance of the Tree class found in src.utils.trees

        Returns:
            Returns temp constraint object is a cut-set inequality is violated. Otherwise, return None
        """

        # Fill in flow graph capacities
        cap = {}
        cap[tree.source, 1] = 1
        for n in tree.B:
            cap[n, tree.left_child(n)] = sum(bR[n, f] for f in F if X[f] == 0)
            cap[n, tree.right_child(n)] = sum(bR[n, f] for f in F if X[f] == 1)
            cap[n, tree.sink] = wR[y, n]   # Modified
        for n in tree.L:
            cap[n, tree.sink] = wR[y, n]

        node_info = {}

        # For each node initialise the cut-set with the edge to the sink node
        # for n in tree.B + tree.L:
        #     edge = (n, tree.sink)
        #     node_info[n] = [(cap[edge], edge)]

        node_info = {n: [(cap[n,tree.sink], (n, tree.sink))]
                     for n in tree.B}

        for n in tree.L:
            if cap[tree.parent(n), n] < cap[n, tree.sink] + self.EPS:
                edge = (tree.parent(n), n)
            else:
                edge = (n, tree.sink)

            node_info[n] = [(cap[edge], edge)]

        for n in reversed(tree.B):
            edge = (tree.parent(n), n)
            child_min_cuts = node_info[tree.left_child(n)] + node_info[tree.right_child(n)]
            child_cut_capacity = sum(cut[0] for cut in child_min_cuts)
            lower_cut_capacity = child_cut_capacity + cap[n,tree.sink]

            if n > 1 and cap[edge] < lower_cut_capacity + self.EPS:
                # In this case add a cut from the branch node to the parent
                node_info[n] = [(cap[edge], edge)]
            else:
                # Otherwise keep the cuts from lower down in the tree
                node_info[n].extend(child_min_cuts)

        min_cuts = node_info[1]
        min_cut_obj = sum(cut[0] for cut in min_cuts)

        # if LP_obj is not None:
        #     if not (abs(LP_obj - min_cut_obj) < self.EPS):
        #         print('Disagreement in min-cut value')
        #     return None, None

        if thetaR > min_cut_obj + self.EPS:
            # If cuts are violated, add back to MP as cutting planes
            left_edges = []
            right_edges = []
            sink_edges = []

            branch_nodes = set(tree.B)
            leaf_nodes = set(tree.L)
            for cut_capacity, edge in min_cuts:
                parent, child = edge
                if parent in leaf_nodes:
                    # Cut is from a leaf node to the sink
                    sink_edges.append((cut_capacity, parent))
                elif parent in branch_nodes:
                    # Check if cut is on a left or right edge from parent
                    if child == tree.left_child(parent):
                        left_edges.append((cut_capacity, parent))
                    elif child == tree.right_child(parent):
                        right_edges.append((cut_capacity, parent))
                    elif child == tree.sink:
                        sink_edges.append((cut_capacity, parent))
                    else:
                        raise Exception('Invalid edge in min cut set')
                else:
                    raise Exception(f'Node {parent} found in cut-set for subproblem Dual Inspection. Where did you find this node?')

            tcon = (theta <= quicksum(b[n, f] for _, n in left_edges for f in F if X[f] == 0)
                    + quicksum(b[n, f] for _, n in right_edges for f in F if X[f] == 1)
                    + quicksum(w[y, n] for _, n in sink_edges))

        else:
            tcon = None

        return tcon

    def solve_fractional_subproblem(self, *args):
        """Wrapper for solving fractional subproblems. Calls LP or dual inspection solves depending on settings

        Args:
            *args:

        Returns:
            Returns Gurobi temp constraint if a cut-set inequality was violated. Otherwise returns None
        """
        if self.opts['Solution Method'] == 'LP':
            return self.LP_Solve(*args)

        elif self.opts['Solution Method'] == 'Dual Inspection':
            return self.DI_Solve(*args)

class CutSetInitialCut(CutSetMixin,InitialCut):
    """Cut-set inequalities

    Adds the cut-set inequalities derived by solving Benders subproblem at fractional MP solutions. The initial cuts
    repeatedly solve the MP relaxation and generate cut-set inequalities before resolving until no more inequalities are
    added.

    Settings:
        Solution Method {'LP','Dual Inspection'}: Method used to solve the fractional subproblems. 'LP' by default

    """


    name = 'Cut Set Initial Cuts'

    def __init__(self, user_opts):

        default_settings = {'Enabled': False,
                            'Solution Method': 'LP'}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def add_cuts(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables

        cut_start_time = time.time()

        I = data['I']
        F = data['F']
        K = data['K']
        X = data['X']
        y = data['y']

        b = variables['b']
        w = variables['w']
        p = variables['p']
        theta = variables['theta']

        # Relax the MP
        for k in b:
            b[k].vtype = GRB.CONTINUOUS

        for k in p:
            p[k].vtype = GRB.CONTINUOUS

        num_iterations = 1
        total_cuts_added = 0

        # Save the existing settings and then turn off logging to console and file.
        LogToConsoleSetting = model.Params.LogToConsole
        LogFile = model.Params.LogFile
        model.Params.LogFile = ''
        model.Params.LogToConsole = 0

        while True:
            cuts_added = 0
            model.optimize()

            # Grab relaxation solution
            bR = {k: b[k].X
                  for k in b.keys()}
            wR = {k: w[k].X
                  for k in w.keys()}
            thetaR = {k: theta[k].X
                      for k in theta.keys()}

            # For each sample solve the fractional subproblem and get back a violated cut-set inequality if it exists
            for i in I:
                tcon = self.solve_fractional_subproblem(X[i,:], y[i],
                                                        b, w, theta[i],
                                                        bR, wR, thetaR[i],
                                                        F,
                                                        tree)

                if tcon is not None:
                    cuts_added += 1
                    model.addConstr(tcon)

            total_cuts_added += cuts_added

            if cuts_added == 0:
                print(f'Subproblem LP added {total_cuts_added} cuts in {num_iterations} iterations')
                break

            num_iterations += 1

        # Unrelax the MP
        for k in b:
            b[k].vtype = GRB.BINARY

        for k in p:
            p[k].vtype = GRB.BINARY

        # Reset the model after solving it's relaxation
        model.reset(1)

        # Set up logging to console and file again on the MIP model
        model.Params.LogFile = LogFile
        model.Params.LogToConsole = LogToConsoleSetting

        cut_runtime = time.time() - cut_start_time

        self.update_cut_stats(total_cuts_added, cut_runtime, ('Iterations', num_iterations))

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
                            'Ignore Dataset Check': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def valid_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']
        alpha_version = self.opts['H Variant']
        disaggregate_alpha = self.opts['Disaggregate Alpha']
        group_selection_enabled = self.opts['Group Selection']

        settings_valid = True
        log_messages = []

        # EQP sets with more than two features removed are not considered computationally feasible
        if features_removed not in [0, 1, 2]:
            log_messages.append(f'EQP initial cuts not valid for {features_removed} features removed. Please try a value in [0,1,2]')
            settings_valid = False

        if alpha_version not in ['Chain', 'Recursive', 'Basic']:
            log_messages.append(f'EQP initial cuts not valid for {alpha_version} version of alpha constraints. Please try a value in [\'Basic\',\'Chain\',\'Recursive\']')
            settings_valid = False
        elif alpha_version in ['Chain', 'Recursive'] and features_removed == 0:
            log_messages.append(f'EQP initial cuts Chain and Recursive alpha constraints not valid with "Features Removed" = 0 since it reduces to the basic alpha constraints')
            settings_valid = False

        if not isinstance(group_selection_enabled, bool):
            log_messages.append(f'EQP Chain initial cuts not valid for {group_selection_enabled} group selection cuts. Please try a boolean value')
            settings_valid = False

        if not isinstance(disaggregate_alpha, bool):
            log_messages.append(f'EQP Chain initial cuts not valid for {disaggregate_alpha} alpha disaggregate. Please try a boolean value')
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

        _ , _, F_star = cut_info

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


        cut_idx, _, F_star = cut_info

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


        cut_idx, _, F_star = cut_info

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
        eqp_cut_generator = EQPSets({'Features Removed': max_removed}, data)
        eqp_cuts = eqp_cut_generator.get_info()

        alpha = {}
        G = {}

        cuts_added = 0

        for cut_info in eqp_cuts:
            # Run subroutines to define alpha variables, link them to the tree structure and construct the bounding term
            # for the constraint on theta based on the variant of H
            if alpha_version == 'Chain':
                bounding_term, alpha_cuts_added = self.chain_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha)
            elif alpha_version == 'Recursive':
                bounding_term, alpha_cuts_added = self.recursive_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha)
            elif alpha_version == 'Basic':
                bounding_term, alpha_cuts_added = self.basic_alpha_constraints(model, alpha, cut_info, b, data, tree)

            cuts_added += alpha_cuts_added

            if group_selection_enabled:
                # If group selection is enabled, use the bounding term to force the model to choose one
                # group from the EQP set to be allowed to be classified correctly

                cut_idx, _, F_star = cut_info

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
                cut_idx, rhs_bound, _ = cut_info
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

    def update_model(self, model):
        model.Params.LazyConstraints = 1

        if self.opts['Enhanced Cuts']:
            if not self.opts['Relax w']:
                # If 'Relax w' is set to false then we force w to be binary and relax p
                w = model._variables['w']
                p = model._variables['p']

                for k in w:
                    w[k].vtype = GRB.BINARY

                for k in p:
                    p[k].vtype = GRB.CONTINUOUS

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
                            
                            if wV[y[i], sibling_node] < thetaV[i] - EPS:
                            #if wV[y[i], leaf_node] + wV[y[i], sibling_node] / 2 < thetaV[i] - EPS:

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


                        # Currently not functional. Do not use
                        # elif (leaf_node in tree.layers[-2]) and (self.opts['EC Level'] >= 2):
                        #     left_child, right_child = tree.children(leaf_node)
                        #
                        #     downstream_vars = ((quicksum(b[leaf_node, f] for f in F if X[i, f] == 0) + w[y[i], left_child]) / 2 +
                        #                        (quicksum(b[leaf_node, f] for f in F if X[i, f] == 1) + w[y[i], right_child]) / 2)
                        #
                        #     # for dir, child_node in enumerate(tree.children(leaf_node)):
                        #     #     if wV[y[i], child_node] < thetaV[i] - EPS:
                        #     #         # Strengthen this component of the cuts
                        #     #         downstream_vars += (quicksum(b[leaf_node, f] for f in F if X[i, f] == dir) + w[y[i], child_node]) / 2
                        #     #     else:
                        #     #         # Keep this component of the cut in the standard form
                        #     #         downstream_vars += quicksum(b[leaf_node, f] for f in F if X[i,f] == dir)
                        #
                        #
                        #     tCon = (theta[i] <= (quicksum(b[n, f] for n, f in cut_branch_vars[i]) +
                        #                          downstream_vars +
                        #                          quicksum(w[y[i], n] for n in sample_node_path[i])))

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
                            'Check Validity': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

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
            X = data['X']
            y = data['y']
            weights = data['weights']
            cat_feature_maps = data['Categorical Feature Map']
            num_feature_maps = data['Numerical Feature Map']

            b = variables['b']
            p = variables['p']
            w = variables['w']
            theta = variables['theta']

            bV = model.cbGetSolution(b)
            pV = model.cbGetSolution(p)
            wV = model.cbGetSolution(w)
            thetaV = model.cbGetSolution(theta)

            subroutine_start_time = time.time()

            # Get route of samples through the tree. See DFS method documentation for detailed output
            DFS_result = callback_generator.DFS(1, I, bV, pV, tree, F, X, cut_vars=True)
            _, samples_in_node, node_branch_feature, _ = DFS_result

            # If a cache hasn't been created, then create a persistent cache for D2S subroutine solution
            if 'D2SubtreeCache' not in callback_generator.callback_cache['Persistent']:
                callback_generator.callback_cache['Persistent']['D2SubtreeCache'] = {}

            root_node = create_recursive_tree(I, F, pV, wV, node_branch_feature, samples_in_node, tree)

            # Call wrapper function which finds subtree roots, runs D2S subroutine at each subtree root, and returns updated tree
            optimised_subtree = optimise_regularised_subtrees(X, y, tree, model._opts, node_branch_feature, root_node, _lambda,
                                                              cache=callback_generator.callback_cache['Persistent']['D2SubtreeCache'],
                                                              weights=weights)

            # Unpack updated solution
            b_subtrees, p_subtrees, w_subtrees, theta_polished = optimised_subtree

            soln_added = 0

            if b_subtrees is not None:
                PossObj = sum(theta_polished) / len(theta_polished) - _lambda * sum(p_subtrees.values())

                # Only accept new solution if it improves on current solution by at least 0.1%
                if PossObj > CurrObj * (1 + 0.1 / 100):
                    # Update the current incumbent
                    bV |= b_subtrees
                    wV |= w_subtrees
                    thetaV = theta_polished

                    # Theoretically should update p as well but given b and w it should be highly constrained
                    model.cbSetSolution(b, bV)
                    model.cbSetSolution(w, wV)
                    model.cbSetSolution(theta, thetaV)

                    # Call solution completers to complete possibly partial solution
                    for sc in model._solution_completers:
                        sc(model, {'b': bV, 'p': pV, 'w': wV, 'theta':thetaV}, 'Callback')

                    model.cbUseSolution()

                    soln_added += 1
                    print(f'**** Callback Primal Heuristic improved solution from {CurrObj} to {PossObj} ****')

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

class LeafBounds(CallbackSubroutine):
    """Experimental cutting planes which infers a bound on the number of leaves in optimal tree

    Currently not fully implemented

    """

    name = 'Leaf Bound'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'EQP Bound': False,
                            'FlowOCT Bound': False,
                            'BendOCT Bound': False}

        self.CurrObj = float('-inf')
        self.CurrObjBound = float('inf')
        self.priority = 50

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def FlowOCT_Bound(self, model):

        opt_params = {}
        gurobi_params = {}

        FlowOCT_Model = FlowOCT({}, {})

        LP_model = Model()
        LP_model.Params.OutputFlag = 1
        LP_model.Params.Method = 2

        LP_model._tree = model._tree
        LP_model._data = model._data
        # LP_model._lambda = 0.0

        FlowOCT_Model.add_vars(LP_model)
        FlowOCT_Model.add_constraints(LP_model)
        FlowOCT_Model.add_objective(LP_model)
        FlowOCT_Model._add_initial_cuts(LP_model)

        # Have to call update() on LP model here for some reason
        # Otherwise the relaxed model ends up having no variables or constraints
        LP_model.update()

        r = LP_model.relax()
        # r.update()
        r.optimize()

        if r.Status == GRB.OPTIMAL:
            print(r.objVal)
            pass
        else:
            pass

    def BendOCT_Bound(self, model):

        initial_cut_settings = {'EQP Chain': {'Enabled': True,
                                              'Features Removed': 2,
                                              'Disaggregate Alpha': True,
                                              'Group Selection': False}}

        opt_params = {'Initial Cuts': initial_cut_settings}
        gurobi_params = {}

        OCT_Model = BendOCT(opt_params, {})

        LP_model = Model()
        LP_model.Params.OutputFlag = 0
        LP_model.Params.Method = 2

        LP_model._tree = model._tree
        LP_model._data = model._data
        # LP_model._lambda = 0.0

        OCT_Model.add_vars(LP_model)
        OCT_Model.add_constraints(LP_model)
        OCT_Model.add_objective(LP_model)
        OCT_Model._add_initial_cuts(LP_model)

        # Have to call update() on LP model here for some reason
        # Otherwise the relaxed model ends up having no variables or constraints
        LP_model.update()

        r = LP_model.relax()
        r.optimize()

        if r.Status == GRB.OPTIMAL:
            return r.objVal / len(model._data['I'])
        else:
            return 1.0

    def EQP_Bound(self, model):

        data = model._data

        I = data['I']

        eqp_cut_generator = EQPSets({'Features Removed': 0}, data)
        eqp_cuts = eqp_cut_generator.get_info()

        num_misclassified = sum(len(cut_idx) - bound for (cut_idx, bound, _) in eqp_cuts)

        return (len(I) - num_misclassified) / len(I)

    def update_model(self, model):
        model.Params.LazyConstraints = 1

        accuracy_bounds = [1]

        if self.opts['EQP Bound']:
            accuracy_bounds.append(self.EQP_Bound(model))
        if self.opts['FlowOCT Bound']:
            accuracy_bounds.append(self.FlowOCT_Bound(model))
        if self.opts['BendOCT Bound']:
            accuracy_bounds.append(self.BendOCT_Bound(model))

        self.accuracy_upper_bound = min(accuracy_bounds)
        self.CurrLeafBound = 2 ** model._tree.depth

    def run_subroutine(self, model, where, callback_generator):

        cuts_added = 0

        MIPSOL = (where == GRB.Callback.MIPSOL)
        MIPNODE = (where == GRB.Callback.MIPNODE)

        if MIPSOL or MIPNODE:

            BestObj = model.cbGet(GRB.Callback.MIPSOL_OBJBST) if MIPSOL else model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            BestObjBnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND) if MIPSOL else model.cbGet(GRB.Callback.MIPNODE_OBJBND)

            _lambda = model._lambda
            tree = model._tree

            attempt_leaf_bound_update = False

            if BestObj > self.CurrObj:
                # See if the bound can be tightened

                self.CurrObj = BestObj
                attempt_leaf_bound_update = True

            if BestObjBnd < self.CurrObjBound:
                # With a tighter upper bound on the objective we might be able to derive
                # a tighter bound on the accuracy

                self.CurrObjBound = BestObjBnd

                # Check if a tighter bound on the accuracy can be inferred from the objective bound
                poss_acc_upper_bound = BestObjBnd + _lambda * (2 ** tree.depth)

                if poss_acc_upper_bound < self.accuracy_upper_bound:
                    self.accuracy_upper_bound = poss_acc_upper_bound
                    attempt_leaf_bound_update = True


            if attempt_leaf_bound_update:

                NewLeafBound = math.floor((self.accuracy_upper_bound - BestObj) / _lambda)

                if NewLeafBound < self.CurrLeafBound:
                    self.CurrLeafBound = NewLeafBound

                    p = model._variables['p']

                    model.cbLazy(quicksum(p[n] for n in p) <= NewLeafBound)
                    cuts_added += 1

                    print(f'Objective {BestObj} with accuracy bound {self.accuracy_upper_bound} '
                          f'implies a bound on the maximum number of leaves of {NewLeafBound}')

            self.update_subroutine_stats(cuts_added, 0)

    def valid_settings(self, model_opts=None, data=None):
        """

        Args:
            model_opts (dict):
            data (dict): Dictionary containing

        Returns:

        """

        settings_valid = True
        log_messages = []

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class MinimumNodeSupport(CallbackSubroutine):
    """Experimental implementation of minimum node support bounds
    """

    name = 'Minimum Node Support'
    priority = 80

    def __init__(self, user_opts):
        default_settings = {'Enabled': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self, model):
        model.Params.LazyConstraints = 1

    # @profile
    def run_subroutine(self, model, where, callback_generator):

        EPS = 1e-4

        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:

            subroutine_start_time = time.time()
            setup_start_time = time.time()

            cuts_added = 0

            # Load in required data
            variables = model._variables
            data = model._data
            tree = model._tree
            _lambda = model._lambda

            I = data['I']
            I_np = data['I_np']    # Load in as numpy array to use efficient bitmasking operations
            F = data['F']
            X = data['X']
            y = data['y']

            b = variables['b']
            p = variables['p']
            w = variables['w']

            bR = model.cbGetNodeRel(b)

            setup_time = time.time() - setup_start_time
            setup_hidden_time = 0

            root_get_start_time = time.time()

            # root_node = callback_generator.get_integral_paths_fast(bR, X, I_np, F, tree)
            root_node = callback_generator.get_integral_paths_fast(bR, X, len(I), F, tree)

            root_get_time = time.time() - root_get_start_time

            generator_callback_start_time = time.time()
            root_node.path = tuple()
            tree_explorer = callback_generator.explore_tree(root_node)
            generator_callback_time = time.time() - generator_callback_start_time

            MNS_start_time = time.time()

            try:
                node_info = next(tree_explorer)

                while True:
                    n, I_mask, path, internal_node = node_info

                    cut_path, cut_self, cut_parent = False, False, False

                    num_samples = I_mask.sum()

                    if num_samples < len(I) * _lambda:
                        # Any leaf must have at least len(I) * _lambda samples in it
                        # TODO: Instead of looking at total number of datapoints, look at number of correctly classified points
                        if n > 1:
                            model.cbLazy(quicksum(b[node, f] for (node, f, _) in path) <= len(path) - 1)
                            cuts_added += 1
                            cut_parent = True

                    else:

                        # class_values, class_counts = np.unique(y_P, return_counts=True)
                        # num_classes_present = len(class_values)

                        class_values = np.unique(y[I_mask])
                        num_classes_present = len(class_values)

                        num_samples_bound = 2 * len(I) * _lambda

                        if num_classes_present >= 2 and (num_samples < num_samples_bound):

                            # optimal_prediction = class_values[np.argmax(class_counts)]
                            class_bin_counts = np.bincount(y[I_mask])
                            optimal_prediction = np.argmax(class_bin_counts)

                            cut_self = True

                            setup_start_time = time.time()
                            wR = model.cbGetNodeRel(w)
                            setup_hidden_time += time.time() - setup_start_time

                            if wR[optimal_prediction, n] < 1 - EPS:
                                relaxing_vars = (quicksum(b[node, f] for (node, ff, _) in path for f in F if f != ff) +
                                                 quicksum(p[node] for (node, _, _) in path))

                                model.cbLazy(1 - w[optimal_prediction,n] <= relaxing_vars)
                                cuts_added += 1

                        elif num_classes_present == 1:
                            # If there is only one class left then it must predict the dominant class

                            cut_self = True

                            setup_start_time = time.time()
                            wR = model.cbGetNodeRel(w)
                            setup_hidden_time += time.time() - setup_start_time

                            if wR[class_values[0], n] < 1 - EPS:
                                relaxing_vars = (quicksum(b[node, f] for (node, ff, _) in path for f in F if f != ff) +
                                                 quicksum(p[node] for (node, _, _) in path))

                                model.cbLazy(1 - w[class_values[0], n] <= relaxing_vars)
                                # model.cbLazy(quicksum(b[n, f] for f in F) <= relaxing_vars)
                                cuts_added += 1
                                # cut_path = True


                    node_info = tree_explorer.send((cut_path, cut_self, cut_parent))

            except StopIteration:
                pass

            MNS_time = time.time() - MNS_start_time

            subroutine_runtime = time.time() - subroutine_start_time
            self.update_subroutine_stats(cuts_added,
                                         subroutine_runtime,
                                         ('Setup Time', setup_time + setup_hidden_time),
                                         ('Integral Path Time', root_get_time),
                                         ('Generator Setup Time', generator_callback_time),
                                         ('Path Filtering Time', MNS_time - setup_hidden_time))

    def valid_settings(self, model_opts=None, data=None):
        """

        Args:
            model_opts (dict):
            data (dict): Dictionary containing

        Returns:

        """

        settings_valid = True
        log_messages = []

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
                            'Cut Type': 'Lazy',
                            'Bound Negative Samples': False,
                            'Bound Structure': False,
                            'Cut Focus': 'Samples'}

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
            F = data['F']
            K = data['K']
            X = data['X']
            y = data['y']
            weights = data['weights']

            b = variables['b']
            p = variables['p']
            w = variables['w']
            theta = variables['theta']

            bR = model.cbGetNodeRel(b)
            pR = model.cbGetNodeRel(p)
            thetaR = model.cbGetNodeRel(theta)

            # root_node is an instance of PathNode, and by traversal root_node contains all integral paths found
            root_node = callback_generator.get_integral_paths(bR, X, I, F, tree)

            # If a cache hasn't been created, then create a persistent cache for D2S subroutine solution
            if 'D2SubtreeCache' not in callback_generator.callback_cache['Persistent']:
                callback_generator.callback_cache['Persistent']['D2SubtreeCache'] = {}

            D2SubtreeCache = callback_generator.callback_cache['Persistent']['D2SubtreeCache']

            # D2S subroutine operates on a classification score penalty (not accuracy)
            alpha_subroutine = len(I) * _lambda

            endpoint_only = self.opts['Endpoint Only']
            cut_focus = self.opts['Cut Focus']

            for node_info in callback_generator.explore_tree(root_node, yield_endpoint_only=endpoint_only):

                n, I_mask, path, _  = node_info

                # Run some basic checks for situations in which we do not want to run the subroutine
                if I_mask.sum() == 0:
                    continue
                if (I_mask.sum() < 2 * len(I) * _lambda) and (len(np.unique(y[I_mask]))):
                    continue

                I_P = I[I_mask]

                # Check in the cache if the path has been seen before
                path_key = frozenset((branch_var, dir) for _, branch_var, dir in path)

                if path_key in D2SubtreeCache:
                    b_subtree, w_subtree, theta_idx = D2SubtreeCache[path_key]
                else:
                    # If we have no already seen the path, then we call the subroutine with the sample which
                    # fall into the subtree
                    b_subtree, w_subtree, theta_idx = optimise_regularised_depth2_subtree(X[I_P, :], y[I_P],
                                                                                          weights=[weights[i] for i in I_P],
                                                                                          alpha=alpha_subroutine)

                    if theta_idx is None:
                        # subroutine should return None if something went wrong
                        raise Exception('Error in D2S subroutine call from Path Bound Cutting Planes Callback')
                    else:
                        # Update the cache with the calculated subtree optimal soln
                        D2SubtreeCache[path_key] = (b_subtree, w_subtree, theta_idx)

                # This is all the nodes in the subtree, not just the optimised depth 2 subtree
                subtree_nodes = tree.descendants(n)

                if cut_focus == 'Samples':
                    # Calculate the objective value for the subset of samples in I_P
                    optimal_subtree_obj = len(theta_idx) - alpha_subroutine * len(w_subtree)
                    CurrObj = sum(thetaR[i] for i in I_P) - alpha_subroutine * sum(pR[n_d] for n_d in subtree_nodes)


                elif cut_focus == 'Objective':
                    # Calculate the objective value for the subset of samples in I_P
                    optimal_subtree_obj = len(theta_idx) - alpha_subroutine * len(w_subtree)
                    CurrObj = sum(thetaR[i] for i in I_P) - alpha_subroutine * sum(pR[n_d] for n_d in subtree_nodes)

                if CurrObj > optimal_subtree_obj + EPS:
                    # If bound is violated then add cuts

                    # Decision variables which can relax the cut. The following can relax the cut:
                    #   1) If a different branch decision is taken at any node in the path
                    #   2) If a prediction is made at any node in the path
                    #   3) If a prediction is made at any node downstream of the optimised subtree
                    relaxing_vars = (quicksum(b[node,f] for (node, ff, _) in path for f in F if f != ff) +
                                     quicksum(p[node] for (node, _, _) in path) +
                                     quicksum(p[n_d] for n_d in subtree_nodes[7:]))

                    if self.opts['Bound Structure']:

                        optimised_subtree_nodes = subtree_nodes[:7]
                        parent_feature, left_feature, right_feature = b_subtree

                        # First step is to parse D2S output to get lists of the branch and leaf nodes in optimal subtree
                        # We order these the same was as the D2S outputs so that we can zip over the nodes and the node decisions

                        branch_nodes = []
                        leaf_nodes = []

                        if parent_feature is None:
                            # One leaf solution
                            leaf_nodes.append(optimised_subtree_nodes[0])

                        elif left_feature is None or right_feature is None:
                            # Two or three leaf solutions
                            branch_nodes.append(optimised_subtree_nodes[0])

                            if left_feature is None and right_feature is None:
                                # Two leaf solution
                                leaf_nodes.extend(optimised_subtree_nodes[1:3])

                            elif left_feature is None:
                                # Three leaf solution with left child as leaf node and right child as branch node

                                branch_nodes.append(optimised_subtree_nodes[2])

                                leaf_nodes.append(optimised_subtree_nodes[1])
                                leaf_nodes.extend(optimised_subtree_nodes[5:])

                            elif right_feature is None:
                                # Three leaf solution with left child as branch node and right child as leaf node

                                branch_nodes.append(optimised_subtree_nodes[1])

                                leaf_nodes.extend(optimised_subtree_nodes[2:5])

                        else:
                            # Four leaf solution
                            branch_nodes.extend(optimised_subtree_nodes[:3])
                            leaf_nodes.extend(optimised_subtree_nodes[3:])

                        # Remove empty branch decisions
                        b_subtree_filt = [f for f in b_subtree if f is not None]

                        branch_nodes_bounded_vars = (quicksum(b[n, ff] for n, f in zip(branch_nodes, b_subtree_filt) for ff in F if ff != f) +
                                                     quicksum(p[n] for n in branch_nodes))

                        leaf_nodes_bounded_vars = (quicksum(w[kk, n] for n, k in zip(leaf_nodes, w_subtree) for kk in K if kk != k) +
                                                   quicksum(b[n,f] for n in leaf_nodes if n not in tree.L for f in F))

                        subtree_size = len(branch_nodes) + len(leaf_nodes)

                        self.add_cut(model,
                                     branch_nodes_bounded_vars + leaf_nodes_bounded_vars,
                                     subtree_size * relaxing_vars)

                        cuts_added += 1

                    if self.opts['Bound Negative Samples']:
                        # Force samples misclassified in the optimal subtree to zero

                        # Find which samples were misclassified in the optimal subtree
                        theta_bounded_idx = np.ones_like(I_P, dtype=bool)
                        theta_bounded_idx[theta_idx] = False

                        theta_bounded = I_P[theta_bounded_idx]

                        lhs = quicksum(theta[i] for i in theta_bounded)
                        rhs = len(theta_bounded) * relaxing_vars

                        self.add_cut(model, lhs, rhs)
                        cuts_added += 1

                    else:
                        # Add basic cuts
                        lhs = quicksum(theta[i] for i in I_P)
                        rhs = len(theta_idx) + (len(I_P) - len(theta_idx)) * relaxing_vars

                        self.add_cut(model, lhs, rhs)
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
            # Depth two subroutine may not work with compressed datasets
            if data['compressed']:
                log_messages.append('Path Bound Cutting Planes D2S subroutine not tested with compressed datasets')
                settings_valid = False

            if model_opts['depth'] < 3:
                log_messages.append(f'Path Bound Cutting Planes are not useful for tree with a depth of less than 3')
                settings_valid = False

            if self.opts['Cut Type'] not in ['Lazy', 'User']:
                log_messages.append(f'Path Bound Cutting Planes not valid for {self.opts['Cut Type']} cut type. Please try "Lazy" or "User"')
                settings_valid = False

            if self.opts['Cut Focus'] not in ['Objective', 'Samples']:
                log_messages.append(f'Path Bound Cutting Planes not valid for {self.opts['Cut Focus']} cut type. Please try "Objective" or "Samples"')
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

class CutSetCallback(CutSetMixin,CallbackSubroutine):
    """Cut-set inequalities

    Adds the cut-set inequalities derived by solving Benders subproblem at fractional MP solutions. The subroutine is
    invoked at the root node of the branch and bound tree for fractional MP solutions.

    Settings:
        Solution Method {'LP','Dual Inspection'}: Method used to solve the fractional subproblems. 'LP' by default
        Cut Type {'Lazy','User'}: Type of cuts added. Lazy (cbLazy) or user (cbCut)

    """

    name = 'Cut Set Callback'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Solution Method': 'LP',
                            'Cut Type': 'Lazy'}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self,model):
        if self.opts['Cut Type'] == 'User':
            model.Params.PreCrush = 1

    def run_subroutine(self, model, where, callback_generator):
        # Only runs at relaxation solutions in the root node
        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) < 0.5:

                cut_type = self.opts['Cut Type']

                data = model._data
                tree = model._tree
                variables = model._variables

                subroutine_start_time = time.time()

                I = data['I']
                F = data['F']
                X = data['X']
                y = data['y']

                b = variables['b']
                w = variables['w']
                theta = variables['theta']

                bR = model.cbGetNodeRel(b)
                wR = model.cbGetNodeRel(w)
                thetaR = model.cbGetNodeRel(theta)

                cuts_added = 0

                # For each sample solve the fractional subproblem and get back a violated cut-set inequality if it exists
                for i in I:
                    tcon = self.solve_fractional_subproblem(X[i, :], y[i],
                                                            b, w, theta[i],
                                                            bR, wR, thetaR[i],
                                                            F, tree)

                    if tcon is not None:
                        cuts_added += 1
                        if cut_type == 'Lazy':
                            model.cbLazy(tcon)
                        elif cut_type == 'User':
                            model.cbCut(tcon)
                        else:
                            raise Exception('Invalid cut type requested in Cut-set Inequalities Callback')


                subroutine_runtime = time.time() - subroutine_start_time

                self.update_subroutine_stats(cuts_added,
                                             subroutine_runtime,
                                             ('Total Iterations', 1),
                                             ('Successful Iterations', 1 if (cuts_added > 0) else 0))

class BendersCallback(GenCallback):

    name = 'BendOCT Callback'

    def __init__(self, callback_settings):

        available_subroutines = [BendersCuts,
                                 SolutionPolishing,
                                 PathBoundCuttingPlanes,
                                 CutSetCallback,
                                 LeafBounds,
                                 MinimumNodeSupport]

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

    def get_integral_paths_fast(self, bR, X, len_I, F, tree):
        """Slightly optimised version of get_integral_paths
        """
        EPS = 1e-4

        # Check if another subroutine has already found the integral paths
        root_node = self.callback_cache['Temporary'].get('Integral Paths', None)

        if root_node is None:
            root_node = PathNode(1, np.ones(len_I, dtype=bool), 0)

            to_explore = [root_node]

            while len(to_explore) > 0:
                node = to_explore.pop()

                n = node.n

                height = tree.depth - node.depth
                if height == 2:
                    continue

                for f in F:
                    b_temp = bR[n,f]
                    if b_temp > 1 - EPS:
                        # Node is internal to an integral path
                        node.internal_node = True
                        node.f = f

                        I_mask = node.I_mask

                        left_mask = (X[:, f] == 0)
                        # right_mask = ~left_mask

                        node.left_child = PathNode(tree.left_child(n), I_mask & left_mask, node.depth + 1)
                        node.right_child = PathNode(tree.right_child(n), I_mask & (~left_mask), node.depth + 1)

                        node.left_child.parent = node
                        node.right_child.parent = node

                        to_explore.append(node.left_child)
                        to_explore.append(node.right_child)

                        break

                    elif b_temp > EPS:
                        break

            root_node.path = tuple()

            self.callback_cache['Temporary']['Integral Paths'] = root_node

        return root_node

    def get_integral_paths(self, bR, X, I, F, tree):


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
                if height == 2:
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


            if subr_name == 'Minimum Node Support':
                num_cuts = stats['Num']
                cut_time = stats['Time']

                int_path_time = stats.get('Integral Path Time', 0)
                path_filter_time = stats.get('Path Filtering Time', 0)
                generator_setup_time = stats.get('Generator Setup Time', 0)
                setup_time = stats.get('Setup Time', 0)


                if subroutine.opts['Enabled']:
                    log_lines.append(f'{subr_name} - Added {num_cuts} cuts in {cut_time:.2f}s\n')

                    if setup_time > 0:
                        log_lines.append(f'{subr_name} - Spent {setup_time:.2f}s on setup\n')

                    if int_path_time > 0:
                        log_lines.append(f'{subr_name} - Spent {int_path_time:.2f}s generating integral paths\n')

                    if generator_setup_time > 0:
                        log_lines.append(f'{subr_name} - Spent {generator_setup_time:.2f}s creating generator object\n')

                    if path_filter_time > 0:
                        log_lines.append(f'{subr_name} - Spent {path_filter_time:.2f}s filtering integral paths\n')

                logged_results[f'{subr_name} - Cuts Added'] = num_cuts
                logged_results[f'{subr_name} - Time'] = cut_time

            if subr_name in ['Solution Polishing']:
                num_solns = stats['Num']
                soln_time = stats['Time']

                if subroutine.opts['Enabled']:
                    log_lines.append(f'{subr_name} - Found {num_solns} improving solutions in {soln_time:.2f}s\n')

                logged_results[f'{subr_name} - Solutions Found'] = num_solns
                logged_results[f'{subr_name} - Time'] = soln_time

            if subr_name == 'Cut Set Callback':
                num_cuts = stats['Num']
                cut_time = stats['Time']

                solve_method = subroutine.opts['Solution Method']

                if subroutine.opts['Enabled']:
                    if 'Total Iterations' in stats:
                        total_iterations = stats['Total Iterations']
                        successful_iterations = stats['Successful Iterations']
                        log_lines.append(f'{subr_name} ({solve_method} solve) - Added {num_cuts} cuts in {cut_time:.2f}s (cuts added for {successful_iterations}/{total_iterations} relaxations)\n')
                    else:
                        log_lines.append(f'{subr_name} ({solve_method} solve) - Did not add any cuts (Gurobi solved before providing relaxation)\n')


                logged_results[f'{subr_name} - Cuts Added'] = num_cuts
                logged_results[f'{subr_name} - Time'] = cut_time

        if len(log_lines) == 1:
            log_printout = None
        else:
            log_printout = ''.join(log_lines)

        return log_printout, logged_results

class BendersInitialCuts(InitialCutManager):

    name = 'BendRegOCT Cut Manager'

    def __init__(self, cut_settings):

        available_cuts = [CutSetInitialCut,
                          EQPInitialCut]

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
                num_cuts = stats['Num']
                cut_time = stats['Time']

                if cut.opts['Enabled']:
                    num_added_vars = stats['Auxiliary Vars']
                    log_lines.append( f'{cut_name} - Added {num_cuts} cuts and {num_added_vars} variables in {cut_time:.2f}s\n')
                    logged_results[f'{cut_name} - Auxiliary Vars'] = num_added_vars

                logged_results[f'{cut_name} - Cuts'] = num_cuts
                logged_results[f'{cut_name} - Time'] = cut_time

            if cut_name in ['Cut Set Initial Cuts']:
                num_cuts = stats['Num']
                cut_time = stats['Time']

                if cut.opts['Enabled']:
                    num_iterations = stats['Iterations']
                    log_lines.append(f'{cut_name} - Added {num_cuts} cuts in {num_iterations} iterations in {cut_time:.2f}s\n')

                logged_results[f'{cut_name} - Cuts'] = num_cuts
                logged_results[f'{cut_name} - Time'] = cut_time

        if len(log_lines) == 1:
            log_printout = None
        else:
            log_printout = ''.join(log_lines)

        return log_printout, logged_results

class BendRegOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params, callback_generator=BendersCallback, cut_manager=BendersInitialCuts)
        self.model_type = 'BendRegOCT'

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
        w = {(k, n): model.addVar(vtype=GRB.CONTINUOUS, name=f'w_{k}^{n}')
             for k in K for n in tree.T}
        theta = {i: model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'theta_{i}')
                 for i in I}

        model._variables = {'b': b,
                            'p': p,
                            'w': w,
                            'theta': theta}

    def add_constraints(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree

        F = data['F']
        K = data['K']

        b = variables['b']
        p = variables['p']
        w = variables['w']

        # At each possible branch node must either branch, make a prediction, or have made a prediction at an ancestor
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) + quicksum(p[n_a] for n_a in tree.ancestors(n)) + p[n] == 1)
                           for n in tree.B}

        # Must make a prediction at exactly one node in each path through the tree
        one_prediction_per_path = {n: model.addConstr(p[n] + quicksum(p[n_a] for n_a in tree.ancestors(n)) == 1)
                                   for n in tree.L}

        # Make a single class prediction at each prediction node where p_n = 1
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == p[n])
                           for n in tree.T}

    def add_objective(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree
        _lambda = model._lambda

        p = variables['p']
        theta = variables['theta']

        I = data['I']

        accuracy = quicksum(theta[i] for i in I) / len(I)
        complexity = _lambda * quicksum(p[n] for n in tree.T)

        model.setObjective(accuracy - complexity, GRB.MAXIMIZE)

    def warm_start(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables
        _lambda = model._lambda

        compressed = data['compressed']
        I = data['I']

        if self.opt_params['Polish Warmstart']:
            model._opts.add('CART polish solutions')

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

        data = model._data
        tree = model._tree

        bS, pS, wS, thetaS = user_vars

        I = data['I']

        lines = []

        accuracy = 100 * sum(thetaS)/len(I)
        complexity = model._lambda * len(pS)

        lines.append(f'Classified {int(sum(thetaS))}/{len(I)} samples correctly')
        lines.append(f'Achieved an accuracy of {accuracy:.2f}% with an objective of {accuracy/100 - complexity:.3f}')
        lines.append(f'Used {len(pS)}/{len(tree.L)} possible leaf nodes')

        return '\n'.join(lines)

    def _check_output_validity(self, model):
        """ Check that the outputted solution is feasible

        This method checks the following conditions hold:
            - All p variables are binary
            - All w variables are binary
            - All theta variables are binary
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

        for n in p:
            if (p[n].X > EPS) and (p[n].X < 1-EPS):
                output_valid = False
                log_messages.append(f'p_{n} = {p[n].X:.10f} is invalid for binary variable p')

        for (k,n) in w:
            if (w[k,n].X < 0) or (w[k,n].X > 1 + EPS) or ((w[k,n].X > EPS) and (w[k,n].X < 1-EPS)):
                output_valid = False
                log_messages.append(f'w_{k}^n = {w[k,n].X:.10f} is invalid for binary variable w')

        for i in theta:
            if (theta[i].X > EPS) and (theta[i].X < 1-EPS):
                output_valid = False
                log_messages.append(f'theta_{i} = {theta[i].X:.10f} is invalid for binary variable theta')

        return output_valid, log_messages