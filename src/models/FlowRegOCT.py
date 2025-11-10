"""Implements classes for regularised FlowOCT"""

import time

from src.utils.logging import log_error
from src.utils.trees import Custom_CART_Heuristic
from src.models.base_classes import OCT, GenCallback, InitialCutManager, CallbackSubroutine, InitialCut

from gurobipy import *

class FlowRegOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params)
        self.model_type = 'FlowRegOCT'

    def add_vars(self,model):

        data = model._data
        tree = model._tree

        I = data['I']
        F = data['F']
        K = data['K']

        b = {(n, f): model.addVar(vtype=GRB.BINARY, name=f'b_{n},{f}')
             for n in tree.B for f in F}
        p = {n: model.addVar(vtype=GRB.BINARY, name=f'p_{n}')
             for n in tree.B + tree.L}
        w = {(k, n): model.addVar(vtype=GRB.CONTINUOUS, name=f'w_{k}^{n}')
             for k in K for n in tree.L + tree.B}

        # Add flow variables for edges leaving branch nodes
        z = {(n1, n2, i): model.addVar(vtype=GRB.BINARY, name=f'z_{n1},{n2}^{i}')
             for n1 in tree.B for n2 in tree.children(n1) for i in I}

        # Add in variables for flow from the source node and to the sink node
        for i in I:
            z[(tree.source, 1, i)] = model.addVar(vtype=GRB.BINARY, name=f'z_{tree.source},{1}^{i}')
            for n in tree.T:
                z[(n, tree.sink, i)] = model.addVar(vtype=GRB.BINARY, name=f'z_{n},{tree.sink}^{i}')

        model._variables = {'b': b,
                            'p': p,
                            'w': w,
                            'z': z}

    def add_constraints(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree

        I = data['I']
        F = data['F']
        K = data['K']
        X = data['X']
        y = data['y']

        b = variables['b']
        p = variables['p']
        w = variables['w']
        z = variables['z']

        # At each possible branch node must either branch, make a prediction, or have made a prediction at an ancestor
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) + quicksum(p[n_a] for n_a in tree.ancestors(n)) + p[n] == 1)
                           for n in tree.B}

        # Must make a prediction at exactly one node in each path through the tree
        one_prediction_per_path = {n: model.addConstr(p[n] + quicksum(p[n_a] for n_a in tree.ancestors(n)) == 1)
                                   for n in tree.L}

        # Flow in = flow out at branch nodes
        branch_flow_equality = {(n, i): model.addConstr(z[tree.parent(n), n, i] ==
                                                        quicksum(z[n, n_child, i] for n_child in tree.children(n)) + z[n, tree.sink, i])
                                for n in tree.B for i in I}

        # Flow in = flow out at leaf nodes
        leaf_flow_equality = {(n, i): model.addConstr(z[tree.parent(n), n, i] == z[n, tree.sink, i])
                              for n in tree.L for i in I}

        # Flow from source at most one
        source_flow_bound = {i: model.addConstr(z[tree.source, 1, i] <= 1)
                             for i in I}

        # Bound the left child flow capacity at each branch node
        left_child_capacity = {(n, i): model.addConstr(z[n, tree.children(n)[0], i] <=
                                                       quicksum(b[n, f] for f in F if X[i, f] < 0.5))
                               for n in tree.B for i in I}

        # Bound the right child flow capacity at each branch node
        right_child_capacity = {(n, i): model.addConstr(z[n, tree.children(n)[1], i] <=
                                                        quicksum(b[n, f] for f in F if X[i, f] > 0.5))
                                for n in tree.B for i in I}

        # Set capacity of edges from all nodes to sink node
        sink_flow_bound = {(n, i): model.addConstr(z[n, tree.sink, i] <= w[y[i], n])
                           for n in tree.T for i in I}

        # Make a single class prediction at each prediction node where p_n = 1
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == p[n])
                           for n in tree.T}

    def add_objective(self,model):
        variables = model._variables
        data = model._data
        tree = model._tree
        _lambda = model._lambda

        p = variables['p']
        z = variables['z']

        I = data['I']

        accuracy = quicksum(z[(n, tree.sink, i)] for n in tree.T for i in I) / len(I)
        complexity = _lambda * quicksum(p[n] for n in tree.T)

        model.setObjective(accuracy - complexity, GRB.MAXIMIZE)

    def warm_start(self, model):

        data = model._data
        tree = model._tree
        variables = model._variables
        _lambda = model._lambda

        compressed = data['compressed']

        model._opts.add('CART flow vars')

        if self.opt_params['Polish Warmstart']:
            model._opts.add('CART polish solutions')

        if compressed:
            X, y = data['Xf'], data['yf']
        else:
            X, y = data['X'], data['y']

        b = variables['b']
        p = variables['p']
        w = variables['w']
        z = variables['z']

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

            for k, v in HeuristicSoln['z'].items():

                if compressed:
                    # Map the idx to the compressed idx
                    idx_map = data['idxf_to_idxc']
                    n1, n2, i = k
                    j = idx_map[i]
                    z[n1,n2,j].Start = v

                else:
                    z[k].Start = v

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
            log_error(140,'CART did not return a valid heuristic solution')

    def save_model_output(self, user_vars):

        bS, pS, wS, zS, thetaS = user_vars

        lines = []

        lines.append('\n' + '*' * 5 + ' BRANCH VARIABLES ' + '*' * 5 + '\nnode:feature')
        for node, feature in bS:
            lines.append(f'{node}:{feature}')

        lines.append('\n' + '*' * 5 + ' PREDICTION VARIABLES ' + '*' * 5 + '\nleaf:predicted class')
        for node, pred in wS:
            lines.append(f'{node}:{pred}')

        lines.append('\n' + '*' * 5 + ' FLOW VARIABLES ' + '*' * 5 + '\nsample:sample path')
        for i, flow_vars in enumerate(zS):
            sample_flow = '->'.join([str(n) for n in flow_vars])
            if sample_flow == '':
                sample_flow = 'misclassified'
            lines.append(f'{i}:{sample_flow}')

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
        z = variables['z']

        I = data['I']
        F = data['F']
        K = data['K']

        bS = [(n, f) for n in tree.B for f in F if b[n, f].X > 0.5]
        wS = [(n, k) for n in tree.T for k in K if w[k, n].X > 0.5]
        pS = [n for n in tree.T if p[n].X > 0.5]
        zS = [[n1 for n1 in tree.B + tree.L for n2 in tree.children(n1) + [tree.sink] if z[n1, n2, i].X > 0.5] for i in I]
        thetaS = [sum(z[n,tree.sink,i].X for n in tree.T) for i in I]

        return bS, pS, wS, zS, thetaS

    def summarise_tree_info(self, model, user_vars):

        data = model._data
        tree = model._tree

        bS, pS, wS, zS, thetaS = user_vars

        I = data['I']

        lines = []

        accuracy = 100 * sum(thetaS)/len(I)
        complexity = model._lambda * len(pS)

        lines.append(f'Classified {int(sum(thetaS))}/{len(I)} samples correctly')
        lines.append(f'Achieved an accuracy of {accuracy:.2f}% with an objective of {accuracy/100 - complexity:.3f}')
        lines.append(f'Used {len(pS)}/{len(tree.L)} possible leaf nodes')

        return '\n'.join(lines)