import itertools
import time

from src.utils.data import valid_datasets
from src.utils.logging import log_error
from src.utils.trees import optimise_subtrees, optimise_depth2_subtree, CART_Heuristic
from src.utils.generators import EQPSets
from src.models.base_classes import OCT, GenCallback, InitialCutManager, CallbackSubroutine, InitialCut

import numpy as np
from gurobipy import *

class BendersCuts(CallbackSubroutine):

    name = 'Benders Cuts'
    priority = 100

    def __init__(self, user_opts):

        default_settings = {'Enabled': True,
                            'Enhanced Cuts': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self, model):
        model.Params.LazyConstraints = 1

    def run_subroutine(self, model, where, callback_generator):
        if where == GRB.Callback.MIPSOL:

            EPS = 1e-4

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

            bV = model.cbGetSolution(b)
            wV = model.cbGetSolution(w)
            thetaV = model.cbGetSolution(theta)

            DFS_result = callback_generator.DFS(1, I, bV, tree, F, X, cut_vars=True)
            sample_node_path, samples_in_node, node_branch_feature, cut_branch_vars = DFS_result

            for i in I:
                leaf_node = sample_node_path[i][-1]

                if wV[y[i], leaf_node] < thetaV[i] - EPS:

                    parent = tree.parent(leaf_node)
                    other_leaf = [n for n in tree.children(parent) if n != leaf_node][0]
                    if self.opts['Enhanced Cuts'] and wV[y[i], other_leaf] < thetaV[i] - EPS:
                        tCon = (theta[i] <= quicksum(b[n, f] for n, f in cut_branch_vars[i] if n != parent) + w[
                            y[i], leaf_node]
                                + (quicksum(b[n, f] for n, f in cut_branch_vars[i] if n == parent) + w[
                                    y[i], other_leaf]) / 2)
                    else:
                        tCon = (theta[i] <= quicksum(b[n, f] for n, f in cut_branch_vars[i]) + w[y[i], leaf_node])
                    model.cbLazy(tCon)

            subroutine_runtime = time.time() - subroutine_start_time
            self.update_subroutine_stats(1, subroutine_runtime)

class SolutionPolishing(CallbackSubroutine):

    name = 'Solution Polishing'
    priority = 50

    def __init__(self, user_opts):

        default_settings = {'Enabled': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def run_subroutine(self, model, where, callback_generator):
        if where == GRB.Callback.MIPSOL:

            CurrObj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)


            if CurrObj < -100:
                return

            data = model._data
            tree = model._tree
            variables = model._variables

            I = data['I']
            F = data['F']
            X = data['X']
            y = data['y']
            weights = data['weights']
            cat_feature_maps = data['Categorical Feature Map']
            num_feature_maps = data['Numerical Feature Map']

            b = variables['b']
            w = variables['w']
            theta = variables['theta']

            bV = model.cbGetSolution(b)
            wV = model.cbGetSolution(w)
            thetaV = model.cbGetSolution(theta)

            subroutine_start_time = time.time()

            DFS_result = callback_generator.DFS(1, I, bV, tree, F, X, cut_vars=True)
            _, samples_in_node, node_branch_feature, _ = DFS_result

            if 'D2SubtreeCache' not in callback_generator.callback_cache['Persistent']:
                callback_generator.callback_cache['Persistent']['D2SubtreeCache'] = {}

            optimised_subtree = optimise_subtrees(X, y, samples_in_node, tree, model._opts,
                                                  node_branch_feature,
                                                  cache=callback_generator.callback_cache['Persistent']['D2SubtreeCache'],
                                                  weights=weights,
                                                  cat_feature_maps=cat_feature_maps,
                                                  num_feature_maps=num_feature_maps)

            b_subtrees, w_subtrees, theta_polished = optimised_subtree

            soln_added = 0

            if b_subtrees is not None:
                if b_subtrees is not None:
                    PossObj = sum(theta_polished)

                    if PossObj > CurrObj + 0.1:
                        bV |= b_subtrees
                        wV |= w_subtrees
                        thetaV = theta_polished

                        model.cbSetSolution(b, bV)
                        model.cbSetSolution(w, wV)
                        model.cbSetSolution(theta, thetaV)

                        for sc in model._solution_completers:
                            sc(model, {'b': bV, 'theta':thetaV}, 'Callback')

                        model.cbUseSolution()

                        soln_added += 1
                        print(f'**** Callback Primal Heuristic improved solution from {CurrObj} to {PossObj} ****')

            subroutine_runtime = time.time() - subroutine_start_time
            self.update_subroutine_stats(soln_added, subroutine_runtime)

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
                log_messages.append('Solution Polishing D2S subroutine not tested with compressed datasets')
                settings_valid = False

            # Depth two subroutine does not scale for coarse discretisation
            if data['encoding'] in ['Full', 'Bucketisation'] and data['name'] in valid_datasets['numerical'] + valid_datasets['mixed']:
                log_messages.append(f'Solution Polishing D2S subroutine too slow for numerical features with {data['encoding']} encoding')
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

    name = 'Path Bound Cutting Planes'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Symmetric Cuts': False,
                            'Bound Samples': False,
                            'Bound Structure': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self, model):
        model.Params.LazyConstraints = 1

    def run_subroutine(self, model, where, callback_generator):

        EPS = 1e-4

        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:

            data = model._data
            tree = model._tree
            variables = model._variables

            subroutine_start_time = time.time()
            cuts_added = 0

            I = np.asarray(data['I'])   # Load in as numpy array to use efficient bitmasking operations
            F = data['F']
            K = data['K']
            X = data['X']
            y = data['y']

            b = variables['b']
            w = variables['w']
            theta = variables['theta']

            bR = model.cbGetNodeRel(b)
            thetaR = model.cbGetNodeRel(theta)

            integral_paths = []     # Store any integral paths in the tree
            nodes = [(1, tuple())]  # Stack of nodes to explore in format (node, path)

            while len(nodes) > 0:

                node, path = nodes.pop()

                # Check if we terminate with an integral path found
                if tree.height(node) == 2:
                    integral_paths.append(path)
                    continue

                # Check if we have any integral branch variables
                for f in F:
                    if bR[node, f] > 1 - EPS:
                        left_child, right_child = tree.children(node)
                        nodes.append((left_child, path + ((node, f, 0),)))
                        nodes.append((right_child, path + ((node, f, 1),)))

            # Set up access to the cache to store results of D2 subrouting
            if 'D2SubtreeCache' not in callback_generator.callback_cache['Persistent']:
                callback_generator.callback_cache['Persistent']['D2SubtreeCache'] = {}

            D2SubtreeCache = callback_generator.callback_cache['Persistent']['D2SubtreeCache']

            for int_path in integral_paths:

                int_path = list(int_path)

                # Construct I_mask to filter out any samples which do not follow path
                I_mask = np.ones(len(I), dtype=bool)
                for _, branch_var, dir in int_path:
                    I_mask &= (X[:, branch_var] == dir)

                I_P = I[I_mask]

                # Do not continue if the path has no samples following it
                if len(I_P) == 0:
                    continue

                path_key = frozenset((branch_var, dir) for _, branch_var, dir in int_path)

                if path_key in D2SubtreeCache:
                    b_subtree, w_subtree, theta_idx = D2SubtreeCache[path_key]
                else:
                    # Find optimal depth 2 subtree for subset I_P
                    b_subtree, w_subtree, theta_idx = optimise_depth2_subtree(X[I_P, :], y[I_P])

                    if theta_idx is None:
                        # subroutine should return None if something went wrong
                        continue
                    else:
                        # Update the cache with the calculated subtree optimal soln
                        D2SubtreeCache[path_key] = (b_subtree, w_subtree, theta_idx)

                # subroutine returns indices of I_P which were correctly classified. Take len to get optimal objective
                optimal_subtree_obj = len(theta_idx)

                # Check if the upper bound found is violated by the relaxation solution
                if sum(thetaR[i] for i in I_P) > optimal_subtree_obj + EPS:

                    # If symmetric cuts required grab all path permutations
                    if self.opts['Symmetric Cuts']:
                        paths = itertools.permutations(int_path)
                    else:
                        paths = [int_path]

                    # Add a cut for each given paths
                    # By default this is only the original path but may include all path permutations

                    for path in paths:
                        node = 1
                        path_branch_choices = []

                        for _, branch_var_list, dir in path:
                            path_branch_choices.append((node, branch_var_list))
                            if dir == 0:
                                node = tree.left_child(node)
                            elif dir == 1:
                                node = tree.right_child(node)

                        # After traversing the path the node variable will contain the subtree root
                        subtree_root = node

                        relaxing_branch_vars = quicksum(quicksum(b[n, ff] for ff in F if ff != var_list) for n, var_list in path_branch_choices)

                        if self.opts['Bound Structure']:
                            subtree_branch_nodes, subtree_leaf_nodes = tree.descendants(subtree_root, split_nodes=True)

                            bounded_branch_vars = quicksum(b[n,ff] for n,f in zip(subtree_branch_nodes, b_subtree) for ff in F if ff != f)
                            bounded_leaf_vars = quicksum(w[kk,n] for n,k in zip(subtree_leaf_nodes, w_subtree) for kk in K if kk != k)

                            subtree_size = 7

                            model.cbLazy(bounded_branch_vars + bounded_leaf_vars <= subtree_size * relaxing_branch_vars)

                        if self.opts['Bound Samples']:
                            theta_bounded_idx = np.ones_like(I_P, dtype=bool)
                            theta_bounded_idx[theta_idx] = False

                            theta_bounded = I_P[theta_bounded_idx]

                            model.cbLazy(quicksum(theta[i] for i in theta_bounded) <= len(theta_bounded) * relaxing_branch_vars)

                            cuts_added += 1

                        else:

                            rhs = optimal_subtree_obj + (len(I_P) - optimal_subtree_obj) * relaxing_branch_vars
                            model.cbLazy(quicksum(theta[i] for i in I_P) <= rhs)

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

            # Depth two subroutine does not scale for coarse discretisation
            if data['encoding'] in ['Full', 'Bucketisation'] and data['name'] in valid_datasets['numerical'] + \
                    valid_datasets['mixed']:
                log_messages.append(f'Path Bound Cutting Planes D2S subroutine too slow for numerical features with {data['encoding']} encoding')
                settings_valid = False

        except KeyError as err:
            log_messages.append(f'Unable to assess validity of Path Bound Cutting Planes settings due to KeyError on key {err.args[0]}')
            settings_valid = False

        except Exception as err:
            log_messages.append(f'Failed to check validity of Path Bound Cutting Planes settings with Exception {type(err).__name__}')
            settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):
        return True

class CbSubproblemLP(CallbackSubroutine):
    name = 'Callback Subproblem LP'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self,model):
        model.Params.PreCrush = 1

    def run_subroutine(self, model, where, callback_generator):
        # Only runs at relaxation solutions in the root node
        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) < 0.5:

                data = model._data
                tree = model._tree
                variables = model._variables

                subroutine_start_time = time.time()

                I = data['I']  # Load in as numpy array to use efficient bitmasking operations
                F = data['F']
                K = data['K']
                X = data['X']
                y = data['y']
                weights = data['weights']

                b = variables['b']
                w = variables['w']
                theta = variables['theta']

                bR = model.cbGetNodeRel(b)
                wR = model.cbGetNodeRel(w)
                thetaR = model.cbGetNodeRel(theta)

                EPS = 1e-4
                cuts_added = 0

                for i in I:
                    LP_model = Model()
                    LP_model.Params.OutputFlag = 0

                    z = {(n1, n2): LP_model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{n1}{n2}')
                         for n1 in tree.B for n2 in tree.children(n1)}

                    z[tree.source, 1] = LP_model.addVar(vtype=GRB.CONTINUOUS)
                    for n in tree.L:
                        z[n, tree.sink] = LP_model.addVar(vtype=GRB.CONTINUOUS)

                    # Flow from source at most one
                    source_flow_bound = LP_model.addConstr(z[tree.source, 1] <= 1)

                    # Flow in = flow out at branch nodes
                    branch_flow_equality = {n: LP_model.addConstr(
                        z[tree.parent(n), n] == quicksum(z[n, n_child] for n_child in tree.children(n)))
                        for n in tree.B}

                    # Flow in = flow out at leaf nodes
                    leaf_flow_equality = {n: LP_model.addConstr(z[tree.parent(n), n] == z[n, tree.sink])
                                          for n in tree.L}

                    # Bound the left child flow capacity at each branch node
                    left_child_capacity = {n: LP_model.addConstr(z[n, tree.left_child(n)] <=
                                                                 quicksum(bR[n, f] for f in F if X[i, f] < 0.5))
                                           for n in tree.B}

                    # Bound the left child flow capacity at each branch node
                    right_child_capacity = {n: LP_model.addConstr(z[n, tree.right_child(n)] <=
                                                                  quicksum(bR[n, f] for f in F if X[i, f] > 0.5))
                                            for n in tree.B}

                    # Set capacity of edges from leaves to sink node
                    sink_flow_bound = {n: LP_model.addConstr(z[n, tree.sink] <= wR[y[i], n])
                                       for n in tree.L}

                    LP_model.setObjective(weights[i] * quicksum(z[(n, tree.sink)] for n in tree.L), GRB.MAXIMIZE)

                    LP_model.optimize()

                    if thetaR[i] > LP_model.objVal + EPS:
                        cuts_added += 1

                        con = (theta[i] <= quicksum(left_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[i, f] == 0)
                                         + quicksum(right_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[i, f] == 1)
                                         + quicksum(sink_flow_bound[n].Pi * w[y[i], n] for n in tree.L))

                        model.cbCut(con)

                subroutine_runtime = time.time() - subroutine_start_time

                self.update_subroutine_stats(cuts_added,
                                             subroutine_runtime,
                                             ('Total Iterations', 1),
                                             ('Successful Iterations', 1 if (cuts_added > 0) else 0))

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class CbSubproblemDualInspection(CallbackSubroutine):
    name = 'Callback Subproblem Dual Inspection'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def update_model(self,model):
        model.Params.PreCrush = 1

    def run_subroutine(self, model, where, callback_generator):
        # Only runs at relaxation solutions in the root node
        if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) < 0.5:

                data = model._data
                tree = model._tree
                variables = model._variables

                subroutine_start_time = time.time()

                I = data['I']
                F = data['F']
                X = data['X']
                y = data['y']
                weights = data['weights']

                b = variables['b']
                w = variables['w']
                theta = variables['theta']

                bR = model.cbGetNodeRel(b)
                wR = model.cbGetNodeRel(w)
                thetaR = model.cbGetNodeRel(theta)

                EPS = 1e-4
                cuts_added = 0

                for i in I:
                    ##### Solve by inspection #####

                    # Fill in flow graph capacities
                    cap = {}
                    cap[tree.source, 1] = 1
                    for n in tree.B:
                        cap[n, tree.left_child(n)] = sum(bR[n, f] for f in F if X[i, f] == 0)
                        cap[n, tree.right_child(n)] = sum(bR[n, f] for f in F if X[i, f] == 1)
                    for n in tree.L:
                        cap[n, tree.sink] = wR[y[i], n]

                    node_info = {n: [] for n in tree.B + tree.L}

                    for n in tree.L:
                        if cap[tree.parent(n), n] < cap[n, tree.sink] + EPS:
                            edge = (tree.parent(n), n)
                        else:
                            edge = (n, tree.sink)

                        node_info[n].append((cap[edge], edge))

                        # node_info[n]['cuts'] = [edge]
                        # node_info[n]['cut capacities'] = cap[edge]

                    for n in reversed(tree.B):
                        edge = (tree.parent(n), n)
                        child_min_cuts = node_info[tree.left_child(n)] + node_info[tree.right_child(n)]
                        child_cut_capacity = sum(cut[0] for cut in child_min_cuts)
                        if n > 1 and cap[edge] < child_cut_capacity + EPS:
                            # In this case add a cut from the branch node to the parent
                            node_info[n].append((cap[edge], edge))
                        else:
                            # Otherwise keep the cuts from lower down in the tree
                            node_info[n] = child_min_cuts

                    min_cuts = node_info[1]
                    min_cut_obj = sum(cut[0] for cut in min_cuts)

                    if thetaR[i] > min_cut_obj + EPS:
                        # If cuts are violated, add back to MP as Benders cuts
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
                                else:
                                    raise Exception('Invalid edge in min cut set')
                            else:
                                raise Exception('Where did you find this node?')

                        con = (theta[i] <= quicksum(b[n, f] for _, n in left_edges for f in F if X[i, f] == 0)
                               + quicksum(b[n, f] for _, n in right_edges for f in F if X[i, f] == 1)
                               + quicksum(w[y[i], n] for _, n in sink_edges))

                        model.cbCut(con)

                        cuts_added += 1

                subroutine_runtime = time.time() - subroutine_start_time

                self.update_subroutine_stats(cuts_added,
                                             subroutine_runtime,
                                             ('Total Iterations', 1),
                                             ('Successful Iterations', 1 if (cuts_added > 0) else 0))

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class SubproblemLP(InitialCut):

    name = 'Subproblem LP'

    def __init__(self, user_opts):

        default_settings = {'Enabled': False}

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
        weights = data['weights']

        b = variables['b']
        w = variables['w']
        theta = variables['theta']

        EPS = 1e-4

        # Relax the MP
        for k in b:
            b[k].vtype = GRB.CONTINUOUS

        num_iterations = 0
        total_cuts_added = 0

        LogToConsoleSetting = model.Params.LogToConsole
        LogFile = model.Params.LogFile
        model.Params.LogFile = ''
        model.Params.LogToConsole = 0

        while True:

            cuts_added = 0
            model.optimize()
            for i in I:
                LP_model = Model()
                LP_model.Params.OutputFlag = 0

                z = {(n1, n2): LP_model.addVar(vtype=GRB.CONTINUOUS, name=f'z_{n1}{n2}')
                     for n1 in tree.B for n2 in tree.children(n1)}

                z[tree.source, 1] = LP_model.addVar(vtype=GRB.CONTINUOUS)
                for n in tree.L:
                    z[n, tree.sink] = LP_model.addVar(vtype=GRB.CONTINUOUS)

                # Flow from source at most one
                source_flow_bound = LP_model.addConstr(z[tree.source, 1] <= 1)

                # Flow in = flow out at branch nodes
                branch_flow_equality = {n: LP_model.addConstr(
                    z[tree.parent(n), n] == quicksum(z[n, n_child] for n_child in tree.children(n)))
                    for n in tree.B}

                # Flow in = flow out at leaf nodes
                leaf_flow_equality = {n: LP_model.addConstr(z[tree.parent(n), n] == z[n, tree.sink])
                                      for n in tree.L}

                # Bound the left child flow capacity at each branch node
                left_child_capacity = {n: LP_model.addConstr(z[n, tree.left_child(n)] <=
                                                             quicksum(b[n, f].X for f in F if X[i, f] < 0.5))
                                       for n in tree.B}

                # Bound the left child flow capacity at each branch node
                right_child_capacity = {n: LP_model.addConstr(z[n, tree.right_child(n)] <=
                                                              quicksum(b[n, f].X for f in F if X[i, f] > 0.5))
                                        for n in tree.B}

                # Set capacity of edges from leaves to sink node
                sink_flow_bound = {n: LP_model.addConstr(z[n, tree.sink] <= w[y[i], n].X)
                                   for n in tree.L}

                LP_model.setObjective(weights[i] * quicksum(z[(n, tree.sink)] for n in tree.L), GRB.MAXIMIZE)

                LP_model.optimize()

                if theta[i].X > LP_model.objVal + EPS:
                    # if (LP_model.objVal > 0.05) and (
                    #         max([left_child_capacity[n].Pi for n in tree.B] + [right_child_capacity[n].Pi for n in
                    #                                                            tree.B]) < 0.95):
                    cuts_added += 1
                    model.addConstr(theta[i] <= quicksum(
                        left_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[i, f] == 0)
                                    + quicksum(
                        right_child_capacity[n].Pi * b[n, f] for n in tree.B for f in F if X[i, f] == 1)
                                    + quicksum(sink_flow_bound[n].Pi * w[y[i], n] for n in tree.L))

            total_cuts_added += cuts_added

            if cuts_added == 0:
                print(f'Subproblem LP added {total_cuts_added} cuts in {num_iterations} iterations')
                break

            num_iterations += 1



        # Unrelax the MP
        for k in b:
            b[k].vtype = GRB.BINARY

        # Reset the model after solving it's relaxation
        model.reset(1)

        # Set up LogFile and LogToConsoleSettings again on the MIP model
        model.Params.LogFile = LogFile
        model.Params.LogToConsole = LogToConsoleSetting

        cut_runtime = time.time() - cut_start_time

        self.update_cut_stats(total_cuts_added, cut_runtime, ('Iterations', num_iterations))

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class SubproblemDualInspection(InitialCut):

    name = 'Subproblem Dual Inspection'

    def __init__(self, user_opts):

        default_settings = {'Enabled': False}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def add_cuts(self, model):
        data = model._data
        tree = model._tree
        variables = model._variables

        cut_start_time = time.time()

        I = data['I']  # Load in as numpy array to use efficient bitmasking operations
        F = data['F']
        K = data['K']
        X = data['X']
        y = data['y']
        weights = data['weights']

        b = variables['b']
        w = variables['w']
        theta = variables['theta']

        EPS = 1e-4

        # Relax the MP
        for k in b:
            b[k].vtype = GRB.CONTINUOUS

        num_iterations = 0
        total_cuts_added = 0

        LogToConsoleSetting = model.Params.LogToConsole
        LogFile = model.Params.LogFile
        model.Params.LogFile = ''
        model.Params.LogToConsole = 0

        while True:
            cuts_added = 0
            model.optimize()
            for i in I:
                ##### Solve by inspection #####

                # Fill in flow graph capacities
                cap = {}
                cap[tree.source, 1] = 1
                for n in tree.B:
                    cap[n, tree.left_child(n)] = sum(b[n, f].X for f in F if X[i, f] == 0)
                    cap[n, tree.right_child(n)] = sum(b[n, f].X for f in F if X[i, f] == 1)
                for n in tree.L:
                    cap[n, tree.sink] = w[y[i], n].X

                node_info = {n: [] for n in tree.B + tree.L}

                for n in tree.L:
                    if cap[tree.parent(n), n] < cap[n, tree.sink] + EPS:
                        edge = (tree.parent(n), n)
                    else:
                        edge = (n, tree.sink)

                    node_info[n].append((cap[edge], edge))

                    # node_info[n]['cuts'] = [edge]
                    # node_info[n]['cut capacities'] = cap[edge]

                for n in reversed(tree.B):
                    edge = (tree.parent(n), n)
                    child_min_cuts = node_info[tree.left_child(n)] + node_info[tree.right_child(n)]
                    child_cut_capacity = sum(cut[0] for cut in child_min_cuts)
                    if n > 1 and cap[edge] < child_cut_capacity + EPS:
                        # In this case add a cut from the branch node to the parent
                        node_info[n].append((cap[edge], edge))
                    else:
                        # Otherwise keep the cuts from lower down in the tree
                        node_info[n] = child_min_cuts

                min_cuts = node_info[1]
                min_cut_obj = sum(cut[0] for cut in min_cuts)

                if theta[i].X > min_cut_obj + EPS:
                    # If cuts are violated, add back to MP as Benders cuts
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
                            else:
                                raise Exception('Invalid edge in min cut set')
                        else:
                            raise Exception('Where did you find this node?')

                    con = (theta[i] <= quicksum(b[n, f] for _, n in left_edges for f in F if X[i, f] == 0)
                           + quicksum(b[n, f] for _, n in right_edges for f in F if X[i, f] == 1)
                           + quicksum(w[y[i], n] for _, n in sink_edges))

                    model.addConstr(con)

                    cuts_added += 1

            total_cuts_added += cuts_added

            if cuts_added == 0:
                print(f'Subproblem Dual Inspection added {total_cuts_added} cuts in {num_iterations} iterations')
                break

            num_iterations += 1

        # Unrelax the MP
        for k in b:
            b[k].vtype = GRB.BINARY

        # Reset the model after solving it's relaxation
        model.reset(1)

        # Set up LogFile and LogToConsoleSettings again on the MIP model
        model.Params.LogFile = LogFile
        model.Params.LogToConsole = LogToConsoleSetting

        cut_runtime = time.time() - cut_start_time

        self.update_cut_stats(total_cuts_added, cut_runtime, ('Iterations', num_iterations))

    def useful_settings(self, model_opts=None, data=None):
        return True, None

class EQPBasic(InitialCut):
    name = 'EQP Basic'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Features Removed': 0}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def valid_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']

        # EQP sets with more than two features removed are not considered computationally feasible
        if features_removed not in [0,1,2]:
            log_message = f'EQP Basic initial cuts not valid for {features_removed} features removed. Please try a value in [0,1,2]'
            return False, log_message

        return True, None

    def useful_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']

        settings_useful = True
        log_messages = []


        try:
            encoded_instance_name = data['encoded name']

            # Check that instance/encoding actually have any eqp sets
            if encoded_instance_name not in valid_datasets['eqp'][features_removed]:
                log_messages.append(f'EQP Basic cuts not useful since dataset {encoded_instance_name} does not have any eqp sets with {features_removed} features removed')
                settings_useful = False

        except KeyError as err:
            log_messages.append(f'Unable to assess validity of EQP Basic initial cut settings due to KeyError on key {err.args[0]}')
            settings_useful = False

        except Exception as err:
            log_messages.append(f'Failed to check validity of EQP Basic initial cut settings with Exception {type(err).__name__}')
            settings_useful = False

        return settings_useful, log_messages

    def add_cuts(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables

        b = variables['b']
        theta = variables['theta']

        y = data['y']

        F = data['F']
        K = data['K']

        max_removed = self.opts['Features Removed']

        cuts_added = 0
        cut_start_time = time.time()

        eqp_cut_generator = EQPSets({'Features Removed': max_removed}, data)
        eqp_cuts = eqp_cut_generator.get_info()


        if max_removed >= 1:
            # For practical reasons beta is indexed by tuples which allows for generalisation to betas with multiple features
            beta = {(f,): model.addVar(vtype=GRB.CONTINUOUS, name=f'beta_{f}')
                    for f in F}
            link_betas = {f: model.addConstr(beta[(f,)] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')
                          for f in F}
            model._variables['beta'] = beta

        else:
            beta = {}
            link_betas = {}

        for cut_idx, rhs_bound, removed_features in eqp_cuts:

            if len(removed_features) == 0:
                rhs = rhs_bound
            else:
                # Cover case when multiple features removed
                if removed_features not in beta:
                    beta[removed_features] = model.addVar(vtype=GRB.CONTINUOUS)
                    model.addConstr(beta[removed_features] <= quicksum(beta[(f,)] for f in removed_features))
                    cuts_added += 1

                rhs = rhs_bound + (len(cut_idx) - rhs_bound) * beta[removed_features]

            model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs)
            cuts_added += 1

        cut_runtime = time.time() - cut_start_time
        cuts_added += len(link_betas)

        self.update_cut_stats(cuts_added, cut_runtime, ('Auxiliary Vars', len(beta)))

class EQPBasicGrouped(InitialCut):
    """Class for grouped basic EQP cuts

    Supports the following options:
        Enabled - [True,False]
        Features Removed - [0,1,2]
        Version - ['Permissive','Blocking']

    """

    name = 'EQP Basic Grouped'



    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Features Removed': 0,
                            'Version': 'Permissive',
                            'Variant': 1}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def valid_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']
        version = self.opts['Version']
        variant = self.opts['Variant']

        settings_valid = True
        log_messages = []

        # EQP sets with more than two features removed are not considered computationally feasible
        if features_removed not in [0,1,2]:
            log_messages.append(f'Grouped EQP Basic initial cuts not valid for {features_removed} features removed. Please try a value in [0,1,2]')
            settings_valid = False

        if version not in ['Permissive', 'Blocking']:
            log_messages.append(f'Grouped EQP Basic initial cuts not valid for version {version}. Please try a version in [Permissive,Blocked]')
            settings_valid = False

        if variant not in [1,2]:
            log_messages.append(f'Grouped EQP Basic initial cuts not valid for variant {version}. Please try a variant in [1,2]')
            settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']

        settings_useful = True
        log_messages = []

        try:
            encoded_instance_name = data['encoded name']

            # Check that instance/encoding actually have any eqp sets
            if encoded_instance_name not in valid_datasets['eqp'][features_removed]:
                log_messages.append(
                    f'EQP Basic Grouped cuts not useful since dataset {encoded_instance_name} does not have any eqp sets with {features_removed} features removed')
                settings_useful = False

        except KeyError as err:
            log_messages.append(
                f'Unable to assess validity of EQP Basic Grouped initial cut settings due to KeyError on key {err.args[0]}')
            settings_useful = False

        except Exception as err:
            log_messages.append(
                f'Failed to check validity of EQP Basic Grouped initial cut settings with Exception {type(err).__name__}')
            settings_useful = False

        return settings_useful, log_messages

    def add_cuts(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables

        b = variables['b']
        theta = variables['theta']

        y = data['y']

        F = data['F']
        K = data['K']

        max_removed = self.opts['Features Removed']

        cuts_added = 0
        cut_start_time = time.time()

        eqp_cut_generator = EQPSets({'Features Removed': max_removed}, data)
        eqp_cuts = eqp_cut_generator.get_info()

        model._eqp_cuts = eqp_cuts

        G = {}
        link_betas = {}

        # Initialise beta variables for case when zero features are removed and fix it to zero
        # beta = {tuple(): model.addVar(vtype=GRB.CONTINUOUS,ub=1, name='beta_empty')}
        # model.addConstr(beta[tuple()] == 0)
        #
        # if max_removed >= 1:
        #     # For practical reasons beta is indexed by tuples which allows for generalisation to betas with multiple features
        #     for f in F:
        #         beta[(f,)] = model.addVar(vtype=GRB.CONTINUOUS,ub=1, name=f'beta_{f}')
        #         link_betas[(f,)] = model.addConstr(beta[(f,)] <= quicksum(b[n, f] for n in tree.B), name=f'link_beta_{f}')

        for cut_idx, rhs_bound, removed_features in eqp_cuts:

            # For case of more than one removed feature, create new beta variables and link to
            # beta variables for single features
            # if removed_features not in beta:
            #     # For practical reasons beta is indexed by tuples which allows for generalisation to betas with multiple features
            #     beta[removed_features] = model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'beta_{removed_features}')
            #     model.addConstr(beta[removed_features] <= quicksum(beta[(f,)] for f in removed_features), name=f'link_beta_{removed_features}')
            #     cuts_added += 1

            # Find groupings
            groups = {}
            for idx in cut_idx:
                sample_label = y[idx]
                if sample_label not in groups:
                    groups[sample_label] = []
                groups[sample_label].append(idx)

            # Create a variable for each grouping
            for k in groups:
                G[removed_features,cut_idx,k] = model.addVar(vtype=GRB.CONTINUOUS)


            if self.opts['Version'] == 'Permissive':
                # In this version of the cuts the G variables are permissive.
                # i.e. G_k = 1 implies that only variables in the k^th grouping can be equal to one

                if self.opts['Variant'] == 1:
                    for k, group_idx in groups.items():
                        # Samples can only be correctly classified if group variable equals one or the samples are split
                        model.addConstr(quicksum(theta[i] for i in group_idx) == len(group_idx) * (G[removed_features, cut_idx, k]))

                    # At most one group can be active unless the samples are split
                    lhs = quicksum(G[removed_features, cut_idx, k] for k in groups)
                    rhs = 1 + (len(groups) - 1) * quicksum(b[n,f] for n in tree.B for f in removed_features)
                    model.addConstr(lhs <= rhs)

                elif self.opts['Variant'] == 2:
                    for k, group_idx in groups.items():
                        # Samples can only be correctly classified if group variable equals one or the samples are split
                        model.addConstr(quicksum(theta[i] for i in group_idx) <= len(group_idx) * (G[removed_features, cut_idx, k]))

                    # At most one group can be active unless the samples are split
                    lhs = quicksum(G[removed_features, cut_idx, k] for k in groups)
                    rhs = 1 + (len(groups) - 1) * quicksum(b[n,f] for n in tree.B for f in removed_features)
                    model.addConstr(lhs == rhs)


            elif self.opts['Version'] == 'Blocking':
                # In this version of the cuts the G variables are blocking
                # i.e. G_k = 1 implies that the variables in the k^th grouping are set to zero

                # for k, group_idx in groups.items():
                #     # Samples can only be correctly classified if the group variables equals zero
                #     model.addConstr(quicksum(theta[i] for i in group_idx) <=
                #                     len(group_idx) * (1 - G[removed_features, cut_idx, k]), name=f' EQPBG Sample bounds ({cut_idx},{k})')
                #
                # # All but one group variables must be equal to one unless the samples are split
                # model.addConstr(quicksum(G[removed_features, cut_idx, k] for k in groups) ==
                #                 (len(groups) - 1) * (1 - beta[removed_features]), name=f' EQPBG beta link ({cut_idx},{k})')

                pass

            else:
                print(f'EQP Basic Grouped Cuts invalid settings of Version = {self.opts['Version']}. Please choose a valid Version')

            cuts_added += len(groups) + 1

        cut_runtime = time.time() - cut_start_time
        cuts_added += len(link_betas)

        variables['G'] = G
        # variables['beta'] = beta

        beta = {}

        self.update_cut_stats(cuts_added, cut_runtime, ('Auxiliary Vars', len(beta) + len(G)))

    def gen_CompleteSolution(self):

        self.solution_completer_warning_raised = False

        def CompleteSolution(model, soln, where):

            return None

            try:
                eqp_cuts = model._eqp_cuts
                bV = soln['b']
                thetaV = soln['theta']

            except NameError:
                log_error(120,'Requested complete solution from EQP Basic initial cuts cannot be completed because EQP sets not attached to model')
                self.solution_completer_warning_raised = True
                return None

            except KeyError:
                log_error(120,'Requested complete solution from EQP Basic initial cuts cannot be completed because branch or theta variables not provided in partial solution')
                self.solution_completer_warning_raised = True
                return None

            try:
                beta = model._variables['beta']
                G = model._variables['G']
            except:
                log_error(120,'Requested complete solution from EQP Basic initial cuts cannot be completed because auxiliary variables not attached to gurobi model')
                self.solution_completer_warning_raised = True
                return None

            try:

                data = model._data
                tree = model._tree

                y = data['y']
                F = data['F']

                max_removed = self.opts['Features Removed']

                # Set up auxiliary variable solutions
                betaV = {tuple(): 0.0}
                GV = {}

                if max_removed >= 1:
                    for f in F:
                        betaV[(f,)] = min(1, sum(bV[n, f] for n in tree.B))

                for cut_idx, _, removed_features in eqp_cuts:
                    # Cover case for two or more removed features
                    if removed_features not in betaV:
                        # For practical reasons beta is indexed by tuples which allows for generalisation to betas with multiple features
                        betaV[removed_features] = min(1, sum(betaV[(f,)] for f in removed_features))

                    # Find groupings
                    groups = {}
                    for idx in cut_idx:
                        sample_label = y[idx]
                        if sample_label not in groups:
                            groups[sample_label] = []
                        groups[sample_label].append(idx)

                    if self.opts['Version'] == 'Blocking':
                        if betaV[removed_features] > 0.5:
                            # In this case none of the groups are blocked and the bound is relaxed
                            # Set all grouping variables to zero

                            for k in groups:
                                GV[removed_features, cut_idx, k] = 0

                        else:
                            # In this case all but one group are blocked
                            # Set all but one grouping variable to 1, determined by thetaV

                            no_groups_correctly_classified = True

                            for k,group_idx in groups.items():

                                if thetaV[group_idx[0]] > 0.9:
                                    GV[removed_features, cut_idx, k] = 0
                                    no_groups_correctly_classified = False

                                else:
                                    GV[removed_features, cut_idx, k] = 1

                            if no_groups_correctly_classified:
                                # Some class not present in the EQP set was predicted in the leaf.
                                # In this case choose the non-blocked group arbitrarily to satisfy linking to beta

                                group_unblocked = list(groups.keys())[0]
                                GV[removed_features, cut_idx, group_unblocked] = 0

                    if self.opts['Version'] == 'Permissive':
                        if betaV[removed_features] > 0.5:
                            # In this case none of the groups are blocked and the bound is relaxed
                            # Set all grouping variables to zero

                            for k in groups:
                                GV[removed_features, cut_idx, k] = 1

                        else:
                            # In this case all but one group are blocked
                            # Set all but one grouping variable to 1, determined by thetaV

                            no_groups_correctly_classified = True

                            for k, group_idx in groups.items():

                                if thetaV[group_idx[0]] > 0.9:
                                    GV[removed_features, cut_idx, k] = 1
                                    no_groups_correctly_classified = False

                                else:
                                    GV[removed_features, cut_idx, k] = 0

                            if no_groups_correctly_classified:
                                # Some class not present in the EQP set was predicted in the leaf.
                                # In this case choose the non-blocked group arbitrarily to satisfy linking to beta

                                group_unblocked = list(groups.keys())[0]
                                GV[removed_features, cut_idx, group_unblocked] = 1

                if where == 'Warm Start':
                    for k,v in betaV.items():
                        beta[k].Start = v

                    for k,v in GV.items():
                        G[k].Start = v

                elif where == 'Callback':
                    model.cbSetSolution(beta, betaV)
                    model.cbSetSolution(G, GV)

            except Exception as err:
                log_error

        return CompleteSolution

class EQPChain(InitialCut):
    name = 'EQP Chain'

    def __init__(self, user_opts):
        default_settings = {'Enabled': False,
                            'Features Removed': 0,
                            'Alpha Version': 'Chain',
                            'Disaggregate Alpha': False,
                            'Group Selection': False,
                            'Group Variant': 1}

        super().__init__(default_settings=default_settings, user_opts=user_opts)

    def valid_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']
        alpha_version = self.opts['Alpha Version']
        disaggregate_alpha = self.opts['Disaggregate Alpha']
        group_selection_enabled = self.opts['Group Selection']
        group_selection_variant = self.opts['Group Variant']

        settings_valid = True
        log_messages = []

        # EQP sets with more than two features removed are not considered computationally feasible
        if features_removed not in [1,2]:
            log_messages.append(f'EQP Chain initial cuts not valid for {features_removed} features removed. Please try a value in [1,2]')
            settings_valid = False

        if alpha_version not in ['Chain', 'Recursive']:
            log_messages.append(f'EQP Chain initial cuts not valid for {alpha_version} version of alpha constraints. Please try a value in [\'Chain\',\'Recursive\']')
            settings_valid = False

        if not isinstance(group_selection_enabled, bool):
            log_messages.append(f'EQP Chain initial cuts not valid for {group_selection_enabled} group selection cuts. Please try a boolean value')
            settings_valid = False

        if not isinstance(disaggregate_alpha, bool):
            log_messages.append(f'EQP Chain initial cuts not valid for {disaggregate_alpha} alpha disaggregate. Please try a boolean value')
            settings_valid = False

        if group_selection_variant not in [1,2]:
            log_messages.append(f'EQP Chain initial cuts not valid for {group_selection_variant} variant of group selection constraints. Please try a value in [1,2]')
            settings_valid = False

        return settings_valid, log_messages

    def useful_settings(self, model_opts=None, data=None):

        features_removed = self.opts['Features Removed']

        settings_useful = True
        log_messages = []

        try:
            encoded_instance_name = data['encoded name']

            # Check that instance/encoding actually have any eqp sets
            if encoded_instance_name not in valid_datasets['eqp'][features_removed]:
                log_messages.append(
                    f'EQP Chain cuts not useful since dataset {encoded_instance_name} does not have any eqp sets with {features_removed} features removed')
                settings_useful = False

        except KeyError as err:
            log_messages.append(f'Unable to assess validity of EQP Chain initial cut settings due to KeyError on key {err.args[0]}')
            settings_useful = False

        except Exception as err:
            log_messages.append(f'Failed to check validity of EQP Chain initial cut settings with Exception {type(err).__name__}')
            settings_useful = False

        return settings_useful, log_messages

    @staticmethod
    def chain_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha):

        cut_idx, _, F_star = cut_info

        if len(F_star) == 0:
            return 0, 0

        X, y = data['X'], data['y']

        F = data['F']

        cuts_added = 0
        alpha_vtype = GRB.CONTINUOUS if disagg_alpha else GRB.BINARY

        # Get some arbitrary idx from the cut
        i = cut_idx[0]

        F_support = [f for f in F if f not in F_star]

        for n in tree.B:
            alpha[(F_star, cut_idx, n)] = model.addVar(vtype=alpha_vtype)

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

        cut_idx, _, F_star = cut_info

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
            alpha[cut_idx, n] = model.addVar(vtype=GRB.CONTINUOUS)

            if n in tree.layers[-2]:
                # Special case for branch nodes which are parents of the leaf nodes
                # alpha allows to equal one at these nodes if they branch on a split feature

                model.addConstr(alpha[cut_idx, n] <= quicksum(b[n,f] for f in F_star))

                cuts_added += 1

            else:
                # Otherwise use the normal recursive constraints for alpha

                alpha_vtype = GRB.CONTINUOUS if disagg_alpha else GRB.BINARY
                alpha[cut_idx, n ,'r'] = model.addVar(vtype=alpha_vtype)
                alpha[cut_idx, n, 'l'] = model.addVar(vtype=alpha_vtype)

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

        F = data['F']
        K = data['K']

        max_removed = self.opts['Features Removed']
        alpha_version = self.opts['Alpha Version']
        disagg_alpha = self.opts['Disaggregate Alpha']
        group_selection_enabled = self.opts['Group Selection']
        group_selection_variant = self.opts['Group Variant']

        total_cuts_added = 0
        cut_start_time = time.time()

        eqp_cut_generator = EQPSets({'Features Removed': max_removed}, data)
        eqp_cuts = eqp_cut_generator.get_info()

        alpha = {}
        G = {}

        cuts_added = 0

        for cut_info in eqp_cuts:
            # Run subroutines to define alpha variables, link them to the tree structure and construct the bounding term
            # for the constraint on theta based on the alpha_version option
            if alpha_version == 'Chain':
                bounding_term, cuts_added = self.chain_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha)
            elif alpha_version == 'Recursive':
                bounding_term, cuts_added = self.recursive_alpha_constraints(model, alpha, cut_info, b, data, tree, disagg_alpha)


            total_cuts_added += cuts_added

            if group_selection_enabled:
                # If group selection is enabled, use the bounding term to force the model to choose one
                # group from the EQP set to be allowed to be classified correctly

                cut_idx, _, F_star = cut_info

                # Find groupings
                groups = {}
                for idx in cut_idx:
                    sample_label = y[idx]
                    if sample_label not in groups:
                        groups[sample_label] = []
                    groups[sample_label].append(idx)

                # Create a variable for each grouping
                for k in groups:
                    G[F_star, cut_idx, k] = model.addVar(vtype=GRB.CONTINUOUS)

                if group_selection_variant == 1:
                    for k, group_idx in groups.items():
                        # Samples can only be correctly classified if group variable equals one or the samples are split
                        model.addConstr(quicksum(theta[i] for i in group_idx) == len(group_idx) * (G[F_star, cut_idx, k]))

                    # At most one group can be active unless the samples are split
                    lhs = quicksum(G[F_star, cut_idx, k] for k in groups)
                    rhs = 1 + (len(groups) - 1) * bounding_term
                    model.addConstr(lhs <= rhs)

                elif group_selection_variant == 2:
                    for k, group_idx in groups.items():
                        # Samples can only be correctly classified if group variable equals one or the samples are split
                        model.addConstr(quicksum(theta[i] for i in group_idx) <= len(group_idx) * (G[F_star, cut_idx, k]))

                    # At most one group can be active unless the samples are split
                    lhs = quicksum(G[F_star, cut_idx, k] for k in groups)
                    rhs = 1 + (len(groups) - 1) * bounding_term
                    model.addConstr(lhs == rhs)

                cuts_added += len(groups) + 1

            else:
                # If group selection is disabled, add a bound over the EQP set which is active whenever
                # the bounding term is forced to zero
                cut_idx, rhs_bound, _ = cut_info
                model.addConstr(quicksum(theta[i] for i in cut_idx) <= rhs_bound + (len(cut_idx) - rhs_bound) * bounding_term)

                cuts_added += 1

        cut_runtime = time.time() - cut_start_time

        self.update_cut_stats(total_cuts_added, cut_runtime, ('Auxiliary Vars', len(alpha) + len(G)))

class BendersCallback(GenCallback):

    name = 'BendOCT Callback'

    def __init__(self, callback_settings):

        available_subroutines = [BendersCuts,
                                 SolutionPolishing,
                                 CbSubproblemLP,
                                 CbSubproblemDualInspection,
                                 PathBoundCuttingPlanes]

        super().__init__(available_subroutines, callback_settings)


    def DFS(self, root, I, bV, tree, F, X, cut_vars=False, changed_root_branch=None):
        """Get DFS solution

        IMPORTANT: Assumes that DFS routine will only be run for a single input per call to the callback by Gurobi
                   If this is not the case then this should be modified to key the cache by the input (minus self)

        Args:
            root (int): Root node to begin DFS from
            I (list): subset samples on which to run search
            bV: branch decision variables to use
            tree:
            F: Feature set
            X: Feature data
            cut_vars: If True, keep track of branch variables which would have sent each sample onto a different leaf
            changed_root_branch: Substituted branch variable for root node

        Returns:

        """
        DFS_result = self.callback_cache['Temporary'].get('DFS_result', None)

        if DFS_result is None:
            subtree_branch_nodes, subtree_leaf_nodes = tree.descendants(root, split_nodes=True)
            node_branch_feature = {}

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
                while current_node in tree.B:
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

    def get_stats_log(self):
        """

        Returns:

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

            if subr_name in ['Callback Subproblem LP', 'Callback Subproblem Dual Inspection']:
                num_cuts = stats['Num']
                soln_time = stats['Time']

                if subroutine.opts['Enabled']:
                    total_iterations = stats['Total Iterations']
                    successful_iterations = stats['Successful Iterations']
                    log_lines.append(f'{subr_name[9:]} - Added {num_cuts} cuts in {cut_time:.2f}s (cuts added for {successful_iterations}/{total_iterations} relaxations)\n')

                logged_results[f'{subr_name} - Cuts Added'] = num_cuts
                logged_results[f'{subr_name} - Time'] = soln_time

        if len(log_lines) == 1:
            log_printout = None
        else:
            log_printout = ''.join(log_lines)

        return log_printout, logged_results

class BendersInitialCuts(InitialCutManager):

    name = 'BendOCT Cut Manager'

    def __init__(self, cut_settings):

        available_cuts = [SubproblemLP,
                          SubproblemDualInspection,
                          EQPBasic,
                          EQPBasicGrouped,
                          EQPChain]

        super().__init__(available_cuts, cut_settings)

    def get_stats_log(self):
        """

        Returns:

        """

        log_lines = ['\nInitial Cut Statistics:\n']
        logged_results = {}

        for cut in self.cuts:
            stats = cut.stats
            cut_name = cut.name

            if cut_name in ['EQP Basic', 'EQP Basic Grouped', 'EQP Chain']:
                num_cuts = stats['Num']
                cut_time = stats['Time']

                if cut.opts['Enabled']:
                    num_added_vars = stats['Auxiliary Vars']
                    log_lines.append( f'{cut_name} - Added {num_cuts} cuts and {num_added_vars} variables in {cut_time:.2f}s\n')
                    logged_results[f'{cut_name} - Auxiliary Vars'] = num_added_vars

                logged_results[f'{cut_name} - Cuts'] = num_cuts
                logged_results[f'{cut_name} - Time'] = cut_time

            if cut_name in ['Subproblem LP', 'Subproblem Dual Inspection']:
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

class BendOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params, callback_generator=BendersCallback, cut_manager=BendersInitialCuts)
        self.model_type = 'BendOCT'

    def add_vars(self,model):

        data = model._data
        tree = model._tree

        I = data['I']
        F = data['F']
        K = data['K']

        b = {(n, f): model.addVar(vtype=GRB.BINARY, name=f'b_{n}{f}')
                  for n in tree.B for f in F}
        w = {(k, n): model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'w_{k}^{n}')
                  for k in K for n in tree.L}
        theta = {i: model.addVar(vtype=GRB.CONTINUOUS, ub=1, name=f'theta_{i}')
                      for i in I}

        model._variables = {'b': b,
                            'w': w,
                            'theta': theta}

    def add_constraints(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree

        F = data['F']
        K = data['K']

        b = variables['b']
        w = variables['w']

        # Can only branch on one variable at each branch node
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) == 1)
                           for n in tree.B}

        # Make a single class prediction at each leaf node
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == 1)
                           for n in tree.L}

    def add_objective(self,model):
        variables = model._variables
        data = model._data

        theta = variables['theta']
        I = data['I']
        weights = data['weights']

        model.setObjective(quicksum(weights[i] * theta[i] for i in I), GRB.MAXIMIZE)

    def warm_start(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables

        compressed = data['compressed']

        if self.opt_params['Polish Warmstart']:
            model._opts.add('CART polish solutions')

        if compressed:
            X, y = data['Xf'], data['yf']
        else:
            X, y = data['X'], data['y']

        b = variables['b']
        w = variables['w']
        theta = variables['theta']

        heuristic_start_time = time.time()

        HeuristicSoln = CART_Heuristic(X, y, tree, model._opts,
                                       cat_feature_maps=data['Categorical Feature Map'],
                                       num_feature_maps=data['Numerical Feature Map'])

        if HeuristicSoln is not None:
            for k, v in HeuristicSoln['b'].items():
                b[k].Start = v

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

            heur_obj = sum(HeuristicSoln['theta'])
            heur_runtime = time.time() - heuristic_start_time

            if 'theta old' in HeuristicSoln:
                heur_unpolished_obj = sum(HeuristicSoln['theta old'])
                print(f'CART returned Heuristic Solution with {heur_obj}/{len(y)} samples classified '
                      f'(polished from {heur_unpolished_obj}) correctly in {time.time() - heuristic_start_time:.2f}s')

                self.update_model_stats('CART',
                                        heur_obj,
                                        heur_runtime,
                                        ('Unpolished Obj', heur_unpolished_obj))
            else:
                print(
                    f'CART returned Heuristic Solution with {heur_obj}/{len(y)} samples classified '
                    f'correctly in {heur_runtime:.2f}s')

                self.update_model_stats('CART',
                                        heur_obj,
                                        heur_runtime)

        else:
            print('CART did not return a valid heuristic solution')

    def save_model_output(self, user_vars):

        b, w, theta = user_vars

        lines = []

        lines.append('*' * 5 + ' BRANCH VARIABLES ' + '*' * 5 + '\nnode:feature\n')
        for node, feature in b:
            lines.append(f'{node}:{feature}\n')

        lines.append('\n' + '*' * 5 + ' PREDICTION VARIABLES ' + '*' * 5 + '\nleaf:predicted class\n')
        for node, pred in w:
            lines.append(f'{node}:{pred}\n')

        lines.append('\n' + '*' * 5 + ' CORRECTLY CLASSIFIED SAMPLES ' + '*' * 5)
        for i in theta:
            lines.append(f'{i}\n')

        save_string = ''.join(lines)

        return save_string

    def vars_to_readable(self,model):
        """Converts Gurobi model solution into a readable format

        Args:
            model (grbModel): Gurobi model with feasible solutions attached

        Returns:
            Returns a tuple with the following elements:
                bS (list): list of tuples (n,f) of branch nodes and branch features in the tree
                wS (list): list of tuples (n,k) or leaf nodes and leaf predictions in the tree
                thetaS (list): List of indices of sample which the tree correctly classifies
        """


        variables = model._variables
        data = model._data
        tree = model._tree

        b = variables['b']
        w = variables['w']
        theta = variables['theta']

        I = data['I']
        F = data['F']
        K = data['K']

        bS = [(n, f) for n in tree.B for f in F if b[n, f].X > 0.5]
        wS = [(n, k) for n in tree.L for k in K if w[k, n].X > 0.5]
        thetaS = [i for i in I if theta[i].X > 0.5]

        return bS, wS, thetaS