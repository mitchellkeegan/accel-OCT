import os
import sys
import itertools
import time
import csv

from src.utils.logging import log_error
from src.utils.trees import Custom_CART_Heuristic
from src.models.base_classes import OCT, GenCallback, InitialCutManager, CallbackSubroutine, InitialCut

from gurobipy import *

class FlowOCT(OCT):
    def __init__(self,opt_params, gurobi_params):

        super().__init__(opt_params, gurobi_params)
        self.model_type = 'FlowOCT'

    def add_vars(self,model):

        data = model._data
        tree = model._tree

        I = data['I']
        F = data['F']
        K = data['K']

        b = {(n, f): model.addVar(vtype=GRB.BINARY, name=f'b_{n}{f}')
                  for n in tree.B for f in F}
        w = {(k, n): model.addVar(vtype=GRB.CONTINUOUS, name=f'w_{k}^{n}')
                  for k in K for n in tree.L}

        # Add flow variables for edges leaving branch nodes
        z = {(n1, n2, i): model.addVar(vtype=GRB.BINARY, name=f'z_{n1}{n2}^{i}')
                  for n1 in tree.B for n2 in tree.children(n1) for i in I}

        # Add in variables for flow from the source node and to the sink node
        for i in I:
            z[(tree.source, 1, i)] = model.addVar(vtype=GRB.BINARY, name=f'z_{tree.source}{1}^{i}')
            for n in tree.L:
                z[(n, tree.sink, i)] = model.addVar(vtype=GRB.BINARY, name=f'z_{n}{tree.sink}^{i}')


        model._variables = {'b': b,
                            'w': w,
                            'z': z}

    def add_constraints(self,model):

        data = model._data
        tree = model._tree

        I = data['I']
        F = data['F']
        K = data['K']
        X = data['X']
        y = data['y']

        b = model._variables['b']
        w = model._variables['w']
        z = model._variables['z']

        # Can only branch on one variable at each branch node
        only_one_branch = {n: model.addConstr(quicksum(b[n, f] for f in F) == 1, name=f'One Branch Feature Node {n}')
                           for n in tree.B}

        # Flow in = flow out at branch nodes
        branch_flow_equality = {(n, i): model.addConstr(z[tree.parent(n), n, i] ==
                                                        quicksum(z[n, n_child, i] for n_child in tree.children(n)), name=f'Branch Flow Equality Node {n} Sample {i}')
                                for n in tree.B for i in I}

        # Flow in = flow out at leaf nodes
        leaf_flow_equality = {(n, i): model.addConstr(z[tree.parent(n), n, i] == z[n, tree.sink, i], name=f'Leaf Flow Equality Node {n} Sample {i}')
                              for n in tree.L for i in I}

        # Flow from source at most one
        source_flow_bound = {i: model.addConstr(z[tree.source, 1, i] <= 1)
                             for i in I}

        # Bound the left child flow capacity at each branch node
        left_child_capacity = {(n, i): model.addConstr(z[n, tree.left_child(n), i] <=
                                                       quicksum(b[n, f] for f in F if X[i, f] == 0), name=f'Left Child Capacity Node {n} Sample {i}')
                               for n in tree.B for i in I}

        # Bound the right child flow capacity at each branch node
        right_child_capacity = {(n, i): model.addConstr(z[n, tree.right_child(n), i] <=
                                                        quicksum(b[n, f] for f in F if X[i, f] == 1), name=f'Right Child Capacity Node {n} Sample {i}')
                                for n in tree.B for i in I}

        # Set capacity of edges from leaves to sink node
        sink_flow_bound = {(n, i): model.addConstr(z[n, tree.sink, i] <= w[y[i], n], name=f'Leaf to Sink Capacity Node {n} Sample {i}')
                           for n in tree.L for i in I}

        # Make a single class prediction at each leaf node
        leaf_prediction = {n: model.addConstr(quicksum(w[k, n] for k in K) == 1, name=f'One Prediction at Leaf {n}')
                           for n in tree.L}

        cons = None
        model._cons = cons

    def add_objective(self,model):
        data = model._data
        tree = model._tree

        z = model._variables['z']
        I = data['I']
        weights = data['weights']

        model.setObjective(quicksum(weights[i] * z[(n,tree.sink,i)] for n in tree.L for i in I), GRB.MAXIMIZE)

    def warm_start(self,model,data):
        pass

    def save_model_output(self):
        b, w, z = self.vars_to_readable()

        hp_combo_string = ','.join(f'{hp_name} = {hp_value}' for hp_name, hp_value in self.hp.items())
        soln_var_file = os.path.join(self.results_directory,
                                     self.opt_params['instance'],
                                     hp_combo_string + ' Soln Vars.txt')
        with open(soln_var_file,'w') as f:
            f.write('*'*5 + ' BRANCH VARIABLES ' + '*'*5 + '\nnode:feature\n')
            for node,feature in b:
                f.write(f'{node}:{feature}\n')

            f.write('\n' + '*' * 5 + ' PREDICTION VARIABLES ' + '*' * 5 + '\nleaf:predicted class\n')
            for node, pred in w:
                f.write(f'{node}:{pred}\n')

            f.write('\n' + '*' * 5 + ' FLOW VARIABLES ' + '*' * 5 + '\nsample:sample path\n')
            for i, flow_vars in enumerate(z):
                sample_flow = '->'.join([str(n) for n in flow_vars])
                if sample_flow == '':
                    sample_flow = 'misclassified'
                f.write(f'{i}:{sample_flow}\n')

    def vars_to_readable(self):
        # Convert model output to a readable format for saving

        b = self.b
        w = self.w
        z = self.z

        I = self.instance_data['I']
        F = self.instance_data['F']
        K = self.instance_data['K']

        tree = self.tree

        bS = [(n,f) for n in tree.B for f in F if b[n,f].X > 0.5]
        wS = [(n,k) for n in tree.L for k in K if w[k,n].X > 0.5]
        zS = [[n1 for n1 in tree.B + tree.L for n2 in tree.children(n1) if z[n1,n2,i].X > 0.5] for i in I]

        return bS, wS, zS


