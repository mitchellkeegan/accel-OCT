from abc import ABC, abstractmethod
import os

import pickle
import itertools
import time
import traceback

import numpy as np

class EQPNode():
    """
    Node class used for generating EQP sets
    """
    def __init__(self, cut_idx, cut_start_idx, node_idx, targets):
        self.cut_start_idx = cut_start_idx
        self.cut_idx = cut_idx
        self.node_idx = node_idx
        self.neighbours = {}
        self.targets = targets

class info_generator(ABC):
    """Base class for generators of information required for callbacks or initial cuts

    Information generators handle deriving information required for initial cuts or callback subroutines
    The base class implements methods for saving derived information to file and loading it back upon request
    By default information is written to Datasets/AuxFiles directory

    Subclass must at a minimum implement _generate() method. _filter() method is optional
    Canonical example for eqp sets used the _generate() method to find eqp sets with split sets with a size <= 3 (which
    are written to file to avoid recomputation) and the _filter() method to return eqp sets with adequately small split sets

    Instances initialised with the following information:

        data (dict): Contains information about dataset as returned by load_instance function
        opts (dict): Options passed to generator, structure dependent on the subclass implementation

    Typical usage example:

        eqp_generator_opts = {'Features Removed': 1}
        eqp_cut_generator = EQPSets(eqp_generator_opts, data)
        eqp_cuts = eqp_cut_generator.get_info()

    """

    default_name = 'Default Generator'

    def __init__(self, opts, data, base_dir=None):
        self.data = data
        self.opts = opts

        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__),
                                    '..',
                                    '..')


        self.info_dir = os.path.join(base_dir,
                                     'Datasets',
                                     'AuxFiles')

        try:
            self.name
        except:
            self.name = self.default_name

    @abstractmethod
    def _generate(self):
        """Generate requested information from self.data based on options specified in self.opts

        Can return information in any format. Returned information will be pickled and written to file,
        passed into self._filter() if implemented, and then returned to user

        Returns tuple (info, aux) where info can be any format and aux must be a dictionary with any auxiliary information
        to be stored. aux does not need to have any entries but must at least be an empty dictionary

        """
        pass

    def _filter(self,info):
        """Filter information returned by self._generate based on options specified in self.opts

        Args:
            info: Format is subclass specific, as provided by _generate method

        Returns:

        """
        return info

    def get_info(self, force_encoding=False):
        """Interface to get required information

        By default attempt to unpickle requested information from file. If that fails (or force_encoding is set) call
        self._generate to generate requested information and write to file.

        If information successfully loaded/generated then pass it through self._filter and return to user. Otherwise
        return None

        Args:
            force_encoding (bool): Force _generate method to be run to derive requesting information from scratch instead
                                   of loading in from file

        """
        encoded_instance_name = self.data['encoded name']
        opts_string = self._get_opts_string()
        filename = '_'.join([encoded_instance_name, self.name.replace(' ',''), opts_string]) + '.pickle'
        # filename = f'{encoded_instance_name}_{self.name.replace(' ','')}.pickle'
        file = os.path.join(self.info_dir, filename)

        if force_encoding:
            print(f'Forcing generation of info for {self.name}')
        else:
            try:
                info, aux = self._load_info(file)
            except Exception as err:
                print(f'Failed to load info for {self.name} with exception {type(err)}')
            else:
                info_valid, log_message = self._info_valid(aux)
                if info_valid:
                    return self._filter(info)
                else:
                    print(f'Saved info for {self.name} invalid with log message: \"'
                          f'{log_message}\"')

        try:
            generator_start_time = time.time()

            info, aux = self._generate()

            generation_time = time.time() - generator_start_time
            aux['Time'] = generation_time

            print(f'Successfully generated info for {self.name} in {generation_time:.5f}s')

        except Exception as err:
            print(f'Failed to generate info for {self.name} with exception {type(err)}')
            print(err)
            print(traceback.format_exc())
            return None

        try:
            self._save_info(file, (info, aux))
        except Exception as err:
            print(f'Failed to write info to file for {self.name} with exception {type(err)}')

        return self._filter(info)

    def _get_opts_string(self):
        return ''

    def _info_valid(self, aux):
        return True, ''

    def _load_info(self, file):
        with open(file, 'rb') as f:
            info = pickle.load(f)
            return info


    def _save_info(self, file, info):

        # If the save directory doesn't exist then create it up front
        dir_name = os.path.dirname(file)
        os.makedirs(dir_name, exist_ok=True)

        with open(file, 'wb') as f:
            pickle.dump(info, f)


class EQPSets(info_generator):

    name = 'EQP Sets'

    def __init__(self, user_opts, data, *args, **kwargs):
        default_opts = {'Removed Features': 2,
                        'Method': 'Graph',
                        'Filter Dominated': True,
                        'Beta Dependence': False}

        opts = default_opts | user_opts

        super().__init__(opts, data, *args, **kwargs)

    def extend_path(self, u, split_set, target_set, path, max_removed):
        base_node = self.nodes[u]

        path_extended = False
        dominated_paths = set()

        agg_target_set = target_set.union(base_node.targets)

        for neighbour, (edge_active, neighbour_split_set) in base_node.neighbours.items():
            if not edge_active:
                continue

            agg_split_set = split_set.union(neighbour_split_set)

            # path_is_dominated = False
            #
            # if self.opts['Filter Dominated']:
            #     if frozenset(agg_split_set) in self.dominating_split_sets[path[0]]:
            #         path_is_dominated = True

            if (len(agg_split_set) <= max_removed) and (frozenset(agg_split_set) not in self.dominating_split_sets[path[0]]):

                path_extended = True

                new_path = path.copy()
                new_path.append(neighbour)

                dominated_by_extended_path = self.extend_path(neighbour, agg_split_set, agg_target_set, new_path, max_removed)
                dominated_paths.update(dominated_by_extended_path)

        # path_is_dominated = False
        # if self.opts['Filter Dominated']:
        #     if tuple(path) in dominated_paths:
        #         path_is_dominated = True

        if (len(path) > 1) and (len(agg_target_set) > 1) and (tuple(path) not in dominated_paths):

            if self.opts['Filter Dominated']:
                for u in path:
                    self.dominating_split_sets[u].add(frozenset(split_set))

            total_eqp_set = set.union(*[self.nodes[u].cut_idx for u in path])

            # Quick and dirty way to get the bound implied by the eqp set
            cut_idx_tuple = tuple(sorted(total_eqp_set))
            _, counts = np.unique(self.data['y'][cut_idx_tuple,], return_counts=True)
            rhs_bound = np.max(counts)

            self.eqp_sets[frozenset(total_eqp_set)] = (frozenset(split_set), agg_target_set, rhs_bound)

            if self.opts['Filter Dominated'] and (not path_extended):
                prev_split_set_len = 0
                accumulated_split_set = set()

                # Iterate over the path and check if any subsets of the path are themselves dominated
                for node_idx in range(len(path) - 1):
                    v1,v2 = path[node_idx], path[node_idx+1]

                    from_node = self.nodes[v1]

                    edge_split_set = from_node.neighbours[v2][1]
                    accumulated_split_set.update(edge_split_set)

                    if len(accumulated_split_set) == prev_split_set_len:
                        dominated_paths.add(tuple(path[:node_idx+1]))

                    if len(accumulated_split_set) == len(split_set):
                        # Current subpath has been dominated by the path
                        dominated_paths.add(tuple(path[:node_idx+2]))

                    prev_split_set_len = len(accumulated_split_set)

        return dominated_paths

    def _generate_GRAPH(self):
        data = self.data

        X = data['X'].astype('bool')
        y = data['y']
        I = data['I']

        max_removed = self.opts['Removed Features']

        # First generate exactly equivalent points, since these will always be paired together in eqp sets
        # The nodes in the graph may then be indiviual samples or exactly equivalent points

        eqp_sets = {}
        in_compressed_idx = set()

        nodes = []
        compressed_indices = []

        for i in I:
            if i in in_compressed_idx:
                continue
            diff_mask = np.logical_xor(X[i, :], X[i + 1:, :])
            diff_mask_eqp_idx = np.flatnonzero(diff_mask.sum(axis=1) == 0)

            if len(diff_mask_eqp_idx) > 0:

                compressed_idx = diff_mask_eqp_idx + (i + 1)

                new_compressed_idx = set(compressed_idx).union({i})
                in_compressed_idx.update(new_compressed_idx)

                # Get targets in
                compressed_idx_targets = set(y[i] for i in new_compressed_idx)
                if len(compressed_idx_targets) > 1:

                    # Quick and dirty way to get the bound implied by the eqp set
                    cut_idx_tuple = tuple(sorted(new_compressed_idx))
                    _, counts = np.unique(y[cut_idx_tuple,], return_counts=True)
                    rhs_bound = np.max(counts)

                    eqp_sets[frozenset(new_compressed_idx)] = (tuple(), compressed_idx_targets, rhs_bound)

            else:
                new_compressed_idx = {i}
                compressed_idx_targets = set((y[i],))

            new_node = EQPNode(new_compressed_idx,
                               i,
                               len(nodes),
                               compressed_idx_targets)

            compressed_indices.append(i)
            nodes.append(new_node)

        # If we are only looking for exactly equivalent points then we are done
        if max_removed == 0:
            return {eqp_set: {'Removed Features': F_star,
                              'Targets': targets,
                              'Bound': rhs_bound}
                    for eqp_set, (F_star, targets, rhs_bound) in eqp_sets.items()}

        X_nodes = X[tuple(compressed_indices),]

        U = range(len(nodes))

        # Now generate the neighbour sets for the compressed nodes
        for u in U:

            # XOR sample u with all samples v > u. Creates array where entry (v,f) equals one if
            # X[u,f] != X[v,f], i.e. samples u and v would be split by feature f
            diff_mask = np.logical_xor(X_nodes[u, :], X_nodes[u + 1:, :])

            # Get indices in diff_mask for which samples u and v differ in at most
            # max_removed samples
            diff_mask_eqp_idx = np.flatnonzero(diff_mask.sum(axis=1) <= max_removed)

            # Get the node instance associated with sample u to update it's neighbour set
            u_node = nodes[u]

            for diff_idx in diff_mask_eqp_idx:

                # Get set of features which sample u and v are not equal in
                F_star = set(np.flatnonzero(diff_mask[diff_idx]))

                # Offset to index on U instead of the diff mask and then update the neighbour set
                v = diff_idx + (u + 1)
                u_node.neighbours[v] = [True, F_star]

        self.nodes = nodes
        self.eqp_sets = eqp_sets
        self.dominating_split_sets = {u: set() for u in U}

        for u in U:
            if len(nodes[u].neighbours) > 0:
                self.extend_path(u, set(), set(), [u], max_removed)

        return {eqp_set: {'Removed Features': split_set,
                          'Targets': targets,
                          'Bound': rhs_bound}
                for eqp_set, (split_set, targets, rhs_bound) in eqp_sets.items()}

    def _generate_BROKEN(self):
        """
        """
        data = self.data

        X, y = data['X'], data['y']
        I = data['I']
        F = data['F']

        n_samples, n_features = X.shape

        max_removed = self.opts['Removed Features']

        eqp_cuts = {}
        support_sets = {}

        for i, j in itertools.combinations(I, 2):
            if y[i] != y[j]:
                # Get subset of feature where x^i != x^j, i.e. if these features were removed then x^i == x^j
                F_support = tuple(np.nonzero(X[i, :] == X[j, :])[0])
                F_star = tuple(f for f in F if f not in F_support)
                if len(F_star) <= max_removed:
                    # Check if we have already seen a set of samples with identical support (support features and support feature values)
                    support_key = (F_support, tuple(X[i, F_support]))
                    if support_key in support_sets:
                        orig_cut_idx = support_sets[support_key]
                        new_cut_idx = tuple(sorted(list(set(orig_cut_idx + (i, j)))))

                        support_sets[support_key] = new_cut_idx
                        del eqp_cuts[orig_cut_idx]

                        # Determine the new bound of the cut_idx
                        classes = {}
                        for idx in new_cut_idx:
                            if y[idx] not in classes:
                                classes[y[idx]] = 1
                            else:
                                classes[y[idx]] += 1

                        bound = max(classes.values())

                        eqp_cuts[new_cut_idx] = {'Removed Features': F_star,
                                                 'Bound': bound}

                        continue

                    else:
                        eqp_cuts[i, j] = {'Removed Features': F_star,
                                          'Bound': 1}
                        support_sets[support_key] = (i, j)

        # split_sets = [(cut_idx, values['Bound'], values['Removed Features']) for cut_idx, values in eqp_cuts.items()]

        split_sets = {cut_idx: values for cut_idx, values in eqp_cuts.items()}

        return split_sets

    def _generate_COMBINED(self):
        eqp_sets_graph = self._generate_GRAPH()
        eqp_sets_broken = self._generate_BROKEN()

        return eqp_sets_graph | eqp_sets_broken

    def _get_opts_string(self):
        opts_string = self.opts['Method']

        if self.opts['Filter Dominated']:
            opts_string += '_FD'

        return opts_string

    def _generate(self):

        method = self.opts['Method']
        aux = {'Removed Features': self.opts['Removed Features']}

        if method == 'Broken':
            eqp_cuts = self._generate_BROKEN()

        elif method == 'Graph':
            eqp_cuts = self._generate_GRAPH()

        elif method == 'Combined':
            eqp_cuts = self._generate_COMBINED()

        else:
            raise ValueError(f'{method} is not a valid EQP set generation procedure')

        return eqp_cuts, aux

    def _info_valid(self, aux):
        RF_available = aux['Removed Features']
        RF_requested = self.opts['Removed Features']

        info_valid = True
        log_message = ''

        if RF_available < RF_requested:
            info_valid = False
            log_message = f'Cached EQP sets only have up to {RF_available} features in split sets, below the requested {RF_requested}'

        return info_valid, log_message

    def _filter(self, eqp_sets):

        max_removed = self.opts['Removed Features']
        beta_dependence = self.opts['Beta Dependence']

        eqp_cuts = [[tuple(sorted(cut_idx)), eqp_info['Bound'], tuple(eqp_info['Removed Features']), None]
                    for cut_idx, eqp_info in eqp_sets.items() if len(eqp_info['Removed Features']) <= max_removed]
        eqp_cuts.sort(key=lambda x: (len(x[0]), len(x[2]), x[0]))

        if beta_dependence:
            modelled = []
            dependent = []

            sample_to_eqp_set_idx = [set() for _ in self.data['I']]
            feature_to_eqp_set_idx = [set() for _ in self.data['F']]

            for eqp_set_idx, (eqp_idx, _, F_star, _) in enumerate(eqp_cuts):

                if len(F_star) > 1:

                    # Check for (sub) eqp sets of the current eqp set
                    idx_subsets = set.union(*[sample_to_eqp_set_idx[idx] for idx in eqp_idx])
                    feature_subsets = set.union(*[feature_to_eqp_set_idx[f_split] for f_split in F_star])

                    idx_subsets = {m for m in idx_subsets if set(eqp_cuts[m][0]) < set(eqp_idx)}
                    feature_subsets = {m for m in feature_subsets if set(eqp_cuts[m][2]) <= set(F_star)}

                    sub_eqp_sets = idx_subsets & feature_subsets

                    # print(eqp_idx, sub_eqp_sets)

                    covered = False
                    cover_eqp_indices = []
                    cover_split_set = set()

                    for subeqp_set_idx in sorted(sub_eqp_sets, reverse=True):
                        sub_eqp_idx, _, sub_F_star_tuple, _ = eqp_cuts[subeqp_set_idx]

                        sub_F_star_set = set(sub_F_star_tuple)

                        if len(sub_F_star_set) == len(F_star):
                            cover_eqp_indices = [sub_eqp_idx]
                            cover_split_set = F_star

                        else:
                            # Check if sub_F_star is distinct from the current cover split set.
                            # If so, add it to the cover
                            if len(sub_F_star_set & cover_split_set) == 0:
                                cover_eqp_indices.append(sub_eqp_idx)
                                cover_split_set |= sub_F_star_set

                        if len(cover_split_set) == len(F_star):
                            covered = True
                            break

                    if covered:
                        eqp_cuts[eqp_set_idx][3] = cover_eqp_indices
                        dependent.append((eqp_idx, cover_eqp_indices))

                    else:
                        modelled.append(eqp_idx)

                else:
                    modelled.append(eqp_idx)

                # Add current eqp set to our set datastructures
                for idx in eqp_idx:
                    sample_to_eqp_set_idx[idx].add(eqp_set_idx)
                for f_split in F_star:
                    feature_to_eqp_set_idx[f_split].add(eqp_set_idx)

        return eqp_cuts

        # return {tuple(sorted(eqp_indices)): set(eqp_info['Removed Features'])
        #         for eqp_indices, eqp_info in eqp_sets.items() if len(eqp_info['Removed Features']) <= max_removed}
