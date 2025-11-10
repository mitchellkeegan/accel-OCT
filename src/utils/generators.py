from abc import ABC, abstractmethod
import os

import pickle
import itertools

import numpy as np

class info_generator(ABC):
    """Base class for generators of information required for callbacks or initial cuts

    Information generators handle deriving information required for initial cuts or callback subroutines
    The base class implements methods for saving derived information to file and loading it back upon request
    By default information is written to Datasets/Auxfiles directory

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

    def __init__(self, opts, data):
        self.data = data
        self.opts = opts
        self.info_dir = os.path.join(os.path.dirname(__file__),
                                                     '..',
                                                     '..',
                                                     'Datasets',
                                                     'Auxfiles')

        try:
            self.name
        except:
            self.name = self.default_name

    @abstractmethod
    def _generate(self):
        """Generate requested information from self.data based on options specified in self.opts

        Can return information in any format. Returned information will be pickled and written to file,
        passed into self._filter() if implemented, and then returned to user

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
        filename = f'{encoded_instance_name}_{self.name.replace(' ','')}.pickle'
        file = os.path.join(self.info_dir, filename)

        if force_encoding:
            print(f'Forcing generation of info for {self.name}')
        else:
            try:
                info = self._load_info(file)
            except Exception as err:
                print(f'Failed to load info for {self.name} with exception {type(err)}')
            else:
                return self._filter(info)

        try:
            info = self._generate()
        except Exception as err:
            print(f'Failed to generate info for {self.name} with exception {type(err)}')
            return None

        try:
            self._save_info(file, info)
        except Exception as err:
            print(f'Failed to write info to file for {self.name} with exception {type(err)}')

        return self._filter(info)


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

    def _generate(self):
        """
        Generate EQP sets of size <= 3. If less is required this should be handled in self._filter

        """
        data = self.data

        X, y = data['X'], data['y']
        I = data['I']
        F = data['F']

        n_samples, n_features = X.shape

        max_removed = 3

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

        split_sets = [(cut_idx, values['Bound'], values['Removed Features']) for cut_idx, values in eqp_cuts.items()]

        return split_sets

    def _filter(self, split_sets):

        try:
            max_removed = self.opts['Features Removed']
        except:
            print('EQP Sets generator need the "Features Removed" option to be set')
            return None

        return [(cut_idx, bound, removed_features) for cut_idx, bound, removed_features in split_sets if len(removed_features) <= max_removed]