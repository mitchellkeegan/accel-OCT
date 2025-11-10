"""Utility functions relating to trees"""

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from src.utils.logging import log_error

class Node():
    """Node class for CART algorithm

    Attributes:
        n (int):
        I (ndarray): 1d array of indices of samples in the node
        F (ndarray): 1d array of classes which are in the node
        depth (int): depth of the node
        parent, left_child, right_child (Node): Pointer to parent and children nodes. Default to None
        node_type (str): Either 'branch' or 'prediction'. Defaults to None. Set during the CART algorithm
        branch_feature (int): Branch feature split on if branch node. Defaults to None
        prediction (int): Predicted class if left node. Defaults to None

    """
    def __init__(self,n,I,F,depth):
        self.n = n
        self.I = I
        self.F = F
        self.depth = depth
        self.left_child, self.right_child = None, None
        self.parent = None
        self.node_type = None
        self.branch_feature = None
        self.prediction = None

    def make_branch_node(self,f):
        self.node_type = 'branch'
        self.prediction = None

        self.branch_feature = f

    def make_leaf_node(self,y):
        self.node_type = 'prediction'
        self.branch_feature = None
        self.left_child, self.right_child = None, None

        node_y = y[self.I,]

        classes_present, counts = np.unique(node_y, return_counts=True)

        if len(classes_present) == 0:
            self.prediction = 0
            self.num_misclassified = 0
        else:
            self.prediction = classes_present[np.argmax(counts)]
            self.num_misclassified = (node_y != self.prediction).sum()

    def calculate_number_misclassified(self,y):
        """Calculates statistics around the number of samples misclassified for the node

        When called on a node it calculates the following attributes:
            leaf_num_misclassified: Number of samples misclassified if the node was a leaf node
            num_leaves: Number of leaves in the subtree
            subtree_num_misclassified: Number of samples misclassified in the subtree with the current node as the root

        The statistics are calculated recursively. Intention is that the method is called on the root node which will
        populate the attributes for all branch nodes in the tree

        Args:
            y (ndarray): 1d array of target classes

        Returns:

        """
        # Calculate the number of points misclassified for:
        # a) The leaves in the subtree (the leaf itself if the node is already a leaf)
        # b) The node if it was transformed from a branch node into a leaf node

        # Also calculate the number of leaves
        assert self.node_type is not None

        if self.node_type == 'prediction':
            return self.num_misclassified, 1
        elif self.node_type == 'branch':
            assert self.left_child is not None
            assert self.right_child is not None

            # Calculate the number of sample which would be misclassified if we turned
            # the branch node into a leaf node and cutoff the subtree below
            node_y = y[self.I,]
            classes_present, counts = np.unique(node_y, return_counts=True)

            if len(classes_present) == 0:
                prediction = 0
                self.leaf_num_misclassified = 0
            else:
                prediction = classes_present[np.argmax(counts)]
                self.leaf_num_misclassified = (node_y != prediction).sum()


            left_misclassified, left_num_leaves = self.left_child.calculate_number_misclassified(y)
            right_misclassified, right_num_leaves = self.right_child.calculate_number_misclassified(y)

            # Calculate number of samples misclassified in subtrees recursively
            self.subtree_num_misclassified = left_misclassified + right_misclassified
            self.num_leaves = left_num_leaves + right_num_leaves

            return self.subtree_num_misclassified, self.num_leaves

        else:
            assert False

    def calculate_height(self):
        """Calculate the height of the node

        The method is recursive. Intended to be called on the root node to set the height of the root node and all
        other nodes in the tree

        """

        if self.node_type == 'prediction':
            self.height = 0
            return 0
        elif self.node_type == 'branch':
            assert self.left_child is not None
            assert self.right_child is not None

            self.height = max(self.left_child.calculate_height(),
                              self.right_child.calculate_height()) + 1
            return self.height

def cost_complexity_pruning(root,X,y,alpha):
    """Run cost complexity pruning on the tree

    At each iteration explores all branch nodes and calculates an effective alpha value. If all nodes have an effective
    alpha greater than alpha then stop. Otherwise prune the node with the smallest effective alpha

    Args:
        root (Node): Root node which can be use to recursively explore the tree
        X (ndarray): 2d array of size (n_samples,n_features) with binary features
        y (ndarray): 1d array of target classes. Classes assumed to be in {0,...,|K|-1}
        alpha: Per leaf penalty on accuracy

    Does not return anything. The function modifies the root node in place

    """
    # Basic idea - calculate effective alpha for all nodes in tree and prune

    num_samples = X.shape[0]

    while True:

        # Recursively calculate for each branch node the number of samples misclassified in its subtree, the number
        # misclassified if it was turned into a leaf node, and the number of leaves in the subtree. it should populate
        # all nodes in the tree with attributes required to calculate the effective alpha value
        root.calculate_number_misclassified(y)

        to_explore = [root]
        min_alpha = float('inf')
        node_to_prune = None

        # Calculate the effective alpha at each node
        # Prune the subtree with the lowest effective alpha, or stop if all have effective_alpha > alpha
        while len(to_explore) > 0:
            node = to_explore.pop()

            if node.node_type == 'prediction':
                continue

            effective_alpha = ((node.leaf_num_misclassified - node.subtree_num_misclassified) / num_samples) / (node.num_leaves - 1)

            if effective_alpha <= min(alpha, min_alpha):
                min_alpha = effective_alpha
                node_to_prune = node

            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

        # Break when all nodes have an effective alpha > alpha
        if node_to_prune is None:
            break

        # Turn the node with the lowest effective alpha into a leaf node
        node_to_prune.make_leaf_node(y)

def node_impurity(y):
    """Calculate the Gini impurity given list of target classes for samples in a node

    Args:
        y (ndarray): 1d numpy array of target classes of samples

    Returns:
        Returns the impurity for the node
    """

    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    impurity = np.sum(p * (1-p))

    return impurity

def create_recursive_tree(I, F, pV, wV, node_branch_feature, samples_in_node, tree):
    root_node = Node(1, I, F, 0)

    to_explore = [root_node]

    while len(to_explore) > 0:
        node = to_explore.pop()

        n = node.n
        node_depth = node.depth
        features_in_node = node.F

        if pV[n] > 0.9:
            node.node_type = 'prediction'
        else:
            node.node_type = 'branch'
            left_child, right_child = tree.children(n)

            node.left_child = Node(left_child, samples_in_node[left_child], features_in_node, node_depth + 1)
            node.right_child = Node(right_child, samples_in_node[right_child], features_in_node, node_depth + 1)

            node.left_child.parent = node
            node.right_child.parent = node

            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

    return root_node

def Custom_CART_Heuristic(X,y,tree,opts,cat_feature_maps=None,num_feature_maps=None,alpha=None):
    """Implementation of CART which splits on Gini impurity and cost-complexity prunes based on accuracy

    Standard CART implementation. Split features are chosen by locally minimising the Gini impurity in the candidate
    child nodes. The tree is pruned based on alpha which is a per leaf penalty on the accuracy. If alpha=None then we
    grow a full depth tree with no pruning.

    Args:
        X (ndarray): 2d array of size (n_samples,n_features) with binary features
        y (ndarray): 1d array of target classes. Classes assumed to be in {0,...,|K|-1}
        tree: Instance of Tree describing the structure of the decision tree
        opts (set): Set of flags which can modify the behaviour of the CART implementation
        cat_feature_maps: Depreciated
        num_feature_maps: Depreciated
        alpha: Per leaf penalty on accuracy

    Returns:

    """

    assert ('No Feature Reuse' not in opts)

    num_samples, num_features = X.shape
    I = range(num_samples)
    F = range(num_features)
    K = np.unique(y).tolist()
    max_depth = tree.depth

    # Construct root node from which we can build/explore the tree recursively
    root_node = Node(1,np.asarray(I),np.asarray(F),0)
    to_explore = [root_node]

    # Explore the tree recursively. At each node we decide if we:
    #   a) Want to split the node at some feature, in which case we add the child node to to_explore
    #   b) Want to make a prediction at the leaf node
    # A node will be turned into a leaf node if it is at the maximum depth
    while len(to_explore) > 0:

        node = to_explore.pop()

        n = node.n
        samples_in_node = node.I
        features_in_node = node.F
        node_depth = node.depth

        # Become a prediction node if we have reached maximum depth
        if node_depth == max_depth:
            node.make_leaf_node(y)

        else:
            # Filter out information for the samples in the current node
            node_X = X[samples_in_node, :]
            node_y = y[samples_in_node,]

            node.node_type = 'branch'

            if len(node_y) == 0:
                # If no samples are left in the node, choose an arbitrary branch feature
                # For regularised trees this will be pruned
                # For balanced trees we grow to full size for compatibility with IP model
                best_split_feature = features_in_node[0]
            else:

                best_split_feature = None
                minimum_impurity = float('inf')

                # Find which feature minimises the Gini impurity. If there are multiple optimal features split on the
                # earlier feature
                for f in features_in_node:
                    # Get a mask array we can use to index into samples in the left and right children
                    left_child_mask = (node_X[:,f] == 0)

                    # Determine the impurity for the given split feature
                    y_left, y_right = node_y[left_child_mask], node_y[~left_child_mask]
                    impurity_left, impurity_right = node_impurity(y_left), node_impurity(y_right)
                    split_impurity = (len(y_left) * impurity_left + len(y_right) * impurity_right) / len(node_y)

                    if split_impurity < minimum_impurity:
                        best_split_feature = f
                        minimum_impurity = split_impurity

                assert (best_split_feature is not None)

            # Make the current node into a branch node
            node.make_branch_node(best_split_feature)

            left_child_mask = (node_X[:,best_split_feature] == 0)
            left_child, right_child = tree.children(n)

            node.left_child = Node(left_child, samples_in_node[left_child_mask], features_in_node, node_depth + 1)
            node.right_child = Node(right_child, samples_in_node[~left_child_mask], features_in_node, node_depth + 1)

            # Set the parent node of the newly created nodes
            node.left_child.parent = node
            node.right_child.parent = node

            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

    # Prune the tree if a complexity penalty is provided
    if alpha is not None:
        cost_complexity_pruning(root_node, X, y, alpha)

    # Set defaults for decision variables
    b = {(n,f): 0 for n in tree.B for f in F}
    p = {n: 0 for n in tree.T}
    w = {(k,n):0 for k in K for n in tree.T}
    theta = np.zeros(num_samples)

    branch_feature = {}
    to_explore = [root_node]

    # Fill in decision variables from CART tree
    while len(to_explore) > 0:
        node = to_explore.pop()

        n = node.n

        if node.node_type == 'branch':
            b[n,node.branch_feature] = 1
            branch_feature[n] = node.branch_feature
            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

        elif node.node_type == 'prediction':
            p[n] = 1
            w[node.prediction,n] = 1
            correctly_classified_samples = node.I[y[node.I,] == node.prediction]
            theta[correctly_classified_samples,] = 1
        else:
            assert False

    soln_dict = {}

    if 'CART polish solutions' in opts:

        polished_soln = optimise_regularised_subtrees(X, y, tree, opts, branch_feature, root_node, alpha)

        b_subtrees, p_subtrees, w_subtrees, theta_polished = polished_soln

        if b_subtrees is not None:

            obj_CART = sum(theta) / len(theta) - alpha * sum(p.values())
            obj_polished = sum(theta_polished) / len(theta) - alpha * sum(p_subtrees.values())

            if obj_polished > obj_CART:
                # Store some information in the solution dict which is useful for logging
                soln_dict['theta old'] = theta
                soln_dict['theta'] = theta_polished
                soln_dict['obj'] = obj_polished
                soln_dict['obj old'] = obj_CART
                soln_dict['num leaves old'] = sum(p.values())

                # Merge in the updated decision variables
                b |= b_subtrees
                w |= w_subtrees
                p |= p_subtrees
                theta = theta_polished

                # Need to update the branch_features if we later want to get the flow variables
                if 'CART flow vars' in opts:
                    branch_feature = {}
                    for n in tree.B:
                        for f in F:
                            if b[n, f] == 1:
                                branch_feature[n] = f
                                break

            elif obj_polished < obj_CART - 1e-4:
                # If the polished objective is worse than the CART objective then this probably indicates an issue
                # with the solution polishing subroutine. Log a warning if this happens
                log_error(199, f'Solution polishing returned a worse objective than CART Heuristic. objCART={obj_CART}, objPolished={obj_polished}')


    soln_dict['b'] = b
    soln_dict['w'] = w
    soln_dict['p'] = p
    soln_dict['theta'] = theta

    soln_dict['num leaves'] = sum(p.values())

    if 'CART flow vars' in opts:

        # Set all edges to zero
        z = {(n1, n2, i): 0
             for n1 in tree.B for n2 in tree.children(n1) for i in I}

        for i in I:
            z[(tree.source, 1, i)] = 0
            for n in tree.B + tree.L:
                z[(n, tree.sink, i)] = 0

        # Fill in edges which have a flow
        for i in I:
            if theta[i] > 0.5:
                z[(tree.source, 1, i)] = 1
                n = 1
                while True:
                    if p[n] == 1:
                        z[n,tree.sink,i] = 1
                        break

                    else:
                        # Branch node
                        f = branch_feature[n]
                        next_node = tree.right_child(n) if X[i,f] > 0.5 else tree.left_child(n)

                        z[n,next_node,i] = 1
                        n = next_node

        soln_dict['z'] = z

    return soln_dict

def CART_Heuristic(X,y,mytree,opts,cat_feature_maps=None,num_feature_maps=None):
    """Generate CART warmstart solution using sklearn CART implementation

    This is used to warmstart the unregaulrsied (balanced tree) FlowOCT and BendOCT models. It produces fully grown trees
    up to a maximum depth of mytree.depth

    """
    n_samples, n_features = X.shape
    I = range(n_samples)
    F = range(n_features)
    K = np.unique(y).tolist()
    max_depth = mytree.depth

    # Occasionally randomness can result in different splits which have the same local optimums. We set a random seed
    # for reproducibility.
    DecisionTree = DecisionTreeClassifier(max_depth=mytree.depth,
                                          ccp_alpha=0.0,
                                          random_state=4466)
    DecisionTree.fit(X,y)

    children_left = DecisionTree.tree_.children_left
    children_right = DecisionTree.tree_.children_right
    feature = DecisionTree.tree_.feature

    sklearn_to_bfs_map = [0] * DecisionTree.tree_.node_count
    early_leaves = []

    branch_feature = {}
    leaf_prediction = {}
    flow_var_keys = [(mytree.source,1)]
    cut_var_keys = {}

    # sklearn does not return the tree in a format which is useful for us.
    # Explore the decision tree, mapping CART tree nodes to bfs ordering, and tracking branching and predictions
    stack = [(0,1,0)]
    while len(stack) > 0:
        node, bfs_node, node_depth = stack.pop(0)
        sklearn_to_bfs_map[node] = bfs_node
        left_child = children_left[node]
        right_child = children_right[node]

        # Three cases:
        # 1) Normal branch node
        # 2) Normal leaf node
        # 3) Early leaf node, not at max depth

        # Case 1)
        if left_child != right_child:
            bfs_left_child = mytree.left_child(bfs_node)
            bfs_right_child = mytree.right_child(bfs_node)

            flow_var_keys.append((bfs_node,bfs_left_child))
            flow_var_keys.append((bfs_node,bfs_right_child))

            stack.append((left_child,bfs_left_child,node_depth+1))
            stack.append((right_child,bfs_right_child,node_depth+1))

            branch_feature[bfs_node] = feature[node]

        # Case 2)
        elif (left_child == -1) and (node_depth == max_depth):
            # Get the index of the predicted class in the current lead node and update w
            prediction_idx = np.argmax(DecisionTree.tree_.value[node, 0, :])
            predicted_class = DecisionTree.classes_[prediction_idx]

            leaf_prediction[bfs_node] = predicted_class

        # Case 3)
        elif (left_child == -1) and (node_depth < max_depth):
            prediction_idx = np.argmax(DecisionTree.tree_.value[node, 0, :])
            predicted_class = DecisionTree.classes_[prediction_idx]
            early_leaves.append((bfs_node,node_depth,predicted_class))

        else:
            print('????')

    # If CART did not return a full sized tree, manually fill out the tree by adding branch nodes which branch on feature 0
    # and predicting the class of the early leaves in all descendant leaves. It includes some logic for a now depreciated feature
    # which allows specifying feature that cannot be used based on previous branch decisions
    if len(early_leaves) > 0:
        for n_root,root_depth,root_pred in early_leaves:
            if 'No Feature Reuse' in opts:
                if cat_feature_maps is None and num_feature_maps is None:
                    print('No Feature Reuse option enabled for CART but feature maps not supplied')
                    valid_features = [0] * len(mytree.B)

                else:
                    if 'Threshold Encoding' in opts:
                        thresholded_feature_maps = num_feature_maps
                        onehot_feature_maps = cat_feature_maps
                    else:
                        thresholded_feature_maps = None
                        onehot_feature_maps = cat_feature_maps + num_feature_maps

                    invalid_features = set()
                    bin_to_cat_group = {}
                    bin_to_num_group = {}
                    if onehot_feature_maps is not None:
                        for Cf in onehot_feature_maps:
                            for f in Cf:
                                bin_to_cat_group[f] = set(Cf)
                    if thresholded_feature_maps is not None:
                        for Nf in thresholded_feature_maps:
                            for f in Nf:
                                bin_to_num_group[f] = set(Nf)

                    for n_a, dir in mytree.ancestors(n_root, branch_dirs=True).items():
                        ancestor_branch_feature = branch_feature[n_a]
                        # Ancestor branches right
                        if dir == 1:
                            for f in bin_to_cat_group.get(ancestor_branch_feature,[]):
                                invalid_features.add(f)
                            for f in bin_to_num_group.get(ancestor_branch_feature,[]):
                                if f <= ancestor_branch_feature:
                                    invalid_features.add(f)
                        elif dir == 0:
                            invalid_features.add(ancestor_branch_feature)
                            for f in bin_to_num_group.get(ancestor_branch_feature,[]):
                                if f >= ancestor_branch_feature:
                                    invalid_features.add(f)
                    valid_features = [f for f in F if f not in invalid_features]
            else:
                valid_features = [0] * len(mytree.B)

            stack = [(n_root,root_depth,valid_features)]
            while len(stack) > 0:
                node, node_depth, useable_features = stack.pop(0)

                # Turn into branch node
                if node_depth < max_depth:
                    bf = useable_features[0]
                    branch_feature[node] = bf
                    if 'No Feature Reuse' in opts:
                        if bf in bin_to_cat_group:
                            unuseable_left = [bf]
                            unuseable_right = bin_to_cat_group[bf]
                        elif bf in bin_to_num_group:
                            unuseable_left = [f for f in bin_to_num_group[bf] if f >= bf]
                            unuseable_right = [f for f in bin_to_num_group[bf] if f <= bf]
                        stack.append((mytree.left_child(node),
                                      node_depth + 1,
                                      [f for f in useable_features if f not in unuseable_left]))
                        stack.append((mytree.right_child(node),
                                      node_depth + 1,
                                      [f for f in useable_features if f not in unuseable_right]))
                    else:
                        stack.append((mytree.left_child(node),
                                      node_depth + 1,
                                      useable_features))
                        stack.append((mytree.right_child(node),
                                      node_depth + 1,
                                      useable_features))


                elif node_depth == max_depth:
                    # Otherwise we have reached a true leaf node, simple reuse the predicted class from the root
                    leaf_prediction[node] = root_pred
                else:
                    print('????')


    theta = [0] * n_samples
    theta_per_leaf = {(n,i): 0 for i in I for n in mytree.L}

    samples_in_node = {n: [] for n in mytree.B + mytree.L}

    # Follow each sample down the tree, tracking the decision path and checking if it is classified correctly
    DecisionPaths = []
    for i in I:
        node = 1
        path = []
        while node in mytree.B:
            path.append(node)
            samples_in_node[node].append(i)
            f = branch_feature[node]
            if X[i,f] == 1:
                node = mytree.right_child(node)
            else:
                node = mytree.left_child(node)

        assert node in mytree.L

        path.append(node)
        samples_in_node[node].append(i)

        if y[i] == leaf_prediction[node]:
            theta[i] = 1
            theta_per_leaf[node,i] = 1

        DecisionPaths.append(path)

    b = {(n,f): 1 if branch_feature[n] == f else 0
         for n in mytree.B for f in F}
    w = {(k,n): 1 if leaf_prediction[n] == k else 0
         for n in mytree.L for k in K}

    soln_dict = {}

    if 'CART polish solutions' in opts:
        # If requested run the solution polishing heuristic to optimise the tails
        b_subtrees,w_subtrees,theta_polished = optimise_subtrees(X,y,samples_in_node,mytree,opts,branch_feature,
                                                                 cat_feature_maps=cat_feature_maps, num_feature_maps=num_feature_maps)

        if b_subtrees is not None and sum(theta_polished) >= sum(theta):
            b |= b_subtrees
            w |= w_subtrees
            soln_dict['theta old'] = theta
            soln_dict['theta'] = theta_polished
            theta = theta_polished

            # Need to update the decision paths if we later want to get the flow variables
            if 'CART flow vars' in opts:
                branch_feature = {}
                for n in mytree.B:
                    for f in F:
                        if b[n,f] == 1:
                            branch_feature[n] = f
                            break
                # Follow each sample down the tree, tracking the decision path and checking if it is classified correctly
                DecisionPaths = []
                for i in I:
                    node = 1
                    path = []
                    while node in mytree.B:
                        path.append(node)
                        f = branch_feature[node]
                        if X[i, f] == 1:
                            node = mytree.right_child(node)
                        else:
                            node = mytree.left_child(node)

                    path.append(node)
                    DecisionPaths.append(path)


    soln_dict['b'] = b
    soln_dict['w'] = w
    soln_dict['theta'] = theta

    # For FlowOCT we also need to extract the values for the flow variables z
    if 'CART flow vars' in opts:

        # Set all edges to zero
        z = {(n1,n2,i): 0
             for n1 in mytree.B for n2 in mytree.children(n1) for i in I}

        for i in I:
            z[(mytree.source, 1, i)] = 0
            for n in mytree.L:
                z[(n, mytree.sink, i)] = 0

        # Fill in edges which have a flow
        for i in I:
            if theta[i] > 0.5:
                z[(mytree.source, 1, i)] = 1
                path = DecisionPaths[i]
                for j in range(len(path)-1):
                    z[path[j],path[j+1],i] = 1
                z[(path[-1], mytree.sink, i)] = 1

        soln_dict['z'] = z

    return soln_dict

def optimise_depth2_subtree(X,y,
                                 tree=None,
                                 weights=None,
                                 invalid_features=None,
                                 bin_to_cat_group=None,
                                 bin_to_num_group=None):
    """Numpy implementation of D2S subroutine for balanced trees

    Provides more or less the same functionality as optimise_regularised_depth2_subtree(). Has been left because the
    unregularised BendOCT model has not been updated to accommodate the newer version

    """


    # TODO: Allow for weights to be used
    if weights is not None:
        assert sum(weights) == len(weights)
    # assert bin_to_cat_group is None
    # assert bin_to_num_group is None

    # X - feature values
    # y - sample classes
    # K - Possible target classed. NOTE: may not coincide with np.unique(y) since
    # y could theoretically be a subset of the dataset with

    n_samples, n_features = X.shape
    I, F_all = range(n_samples), range(n_features)

    # Get unique classes in data and associate them to an index
    # Need k_to_class_idx for when when K != [0,1,...,|K|]
    K = np.unique(y).tolist()
    k_to_class_idx = {k: i for i, k in enumerate(K)}

    # If no tree is given create a generic depth 2 tree
    if tree is None:
        tree = Tree(2)
    elif tree.depth != 2:
        return None, None, None

    if weights is None:
        weights = [1] * n_samples

    if invalid_features is not None:
    # Given a set of invalid features, we want the subroutine to operate only on the subset of valid features
    # and then map back to the full set of features

        all_map_to_F = {}
        F_map_to_all = []   # Keep track of how our reduced feature set maps back to all features
        F_mask = []         # Used to mask the feature dimension when indexing X

        for i, f in enumerate(F_all):
            if f not in invalid_features:
                all_map_to_F[f] = len(F_map_to_all)
                F_map_to_all.append(i)
                F_mask.append(True)
            else:
                F_mask.append(False)

        F = range(len(F_map_to_all))
    else:
        F_map_to_all = F_all
        F_mask = [True] * len(F_all)
        F = F_all

    # bin_to_cat_group_mapped = {all_map_to_F[]}


    # Subroutine will fail with no data supplied
    if n_samples == 0:
        w = [0, 0, 0, 0]
        theta_idx = []

        # Still assure that output conforms to NoFeatureReuse inequalities if required
        if bin_to_cat_group is not None or bin_to_num_group is not None:
            soln_found = False
            for f in F:
                parent_feature = F_map_to_all[f]

                # Find invalid left and right children for the given candidate parent feature
                if bin_to_num_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f >= parent_feature)
                    right_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f <= parent_feature)
                elif bin_to_cat_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = bin_to_num_group[parent_feature]
                else:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = [parent_feature]

                left_child_feature, right_child_feature = None, None

                for f_left in F:
                    f_left_candidate = F_map_to_all[f_left]
                    if f_left_candidate not in left_invalid_grouping:
                        left_child_feature = f_left_candidate

                for f_right in F:
                    f_right_candidate = F_map_to_all[f_right]
                    if f_right_candidate not in right_invalid_grouping:
                        right_child_feature = f_right_candidate

                if left_child_feature is not None and right_child_feature is not None:
                    b = (parent_feature, left_child_feature, right_child_feature)
                    soln_found = True
                    break

            if not soln_found:
                # In this case there are no feature combinations which are valid according to the
                # provided invalid_feature set and feature mappings
                return None, None, None

        else:
            b = (F_map_to_all[0], F_map_to_all[0], F_map_to_all[0])

        return b, w, theta_idx

    ############### BEGIN SUBROUTINE PROPER ###############

    # # Set up frequency counters
    # FQ_ref = {'0': [[0 for d1 in F] for d0 in range(len(K))],
    #       '1': [[0 for d1 in F] for d0 in range(len(K))],
    #       '00': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '01': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '10': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '11': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))]}

    # TODO: Cut down on memory usage by using smaller ints?
    FQ1 = np.zeros((len(K), len(F)), np.int32)
    FQ11 = np.zeros((len(K), len(F), len(F)), np.int32)

    # Track number of samples of each class in the subtree
    D_ref = [0] * len(K)

    _, D = np.unique(y, return_counts=True)

    # for idx in I:
    #     k = k_to_class_idx[y[idx]]   # Offset to index into lists
    #     fv = X[idx,F_mask]     # feature vector
    #     D_ref[k] += 1
    #     for i in F:
    #         if fv[i] > 0.5:
    #             FQ_ref['1'][k][i] += weights[idx]
    #             for j in F:
    #                 if fv[j] > 0.5:
    #                     FQ_ref['11'][k][i][j] += weights[idx]

    for k in K:
        # Take the subarray of X in which all samples have class y^i = k and features are filtered by F_mask
        X_masked = X[np.ix_(y==k,F_mask)]

        k_idx = k_to_class_idx[k]

        # Each column of X_masked is associated with feature f, and indicates which samples have x_f^i == 1
        # Taking the dot product of columns associated with features f_a and f_b will be equal to the number of
        # samples for which (x_fa^i == 1 AND x_fb^i == 1)
        FQ11[k_idx,:,:] = X_masked.T @ X_masked

        # The diagonal corresponds to duplicated features, i.e. x_fa^i == 1
        FQ1[k_idx,:] = np.diag(FQ11[k_idx,:,:])


    # assert (np.sum(np.asarray(FQ_ref['1']) != FQ1) == 0)
    # assert (np.sum(np.asarray(FQ_ref['11']) != FQ11) == 0)
    #
    # # Fill out symmetry of matrix
    # for k in range(len(K)):
    #     for i in F:
    #         for j in range(i+1,len(F)):
    #             FQ_ref['11'][k][j][i] = FQ_ref['11'][k][i][j]
    #
    # for k in range(len(K)):
    #     for i in F:
    #         FQ_ref['0'][k][i] = D[k] - FQ_ref['1'][k][i]
    #         for j in F:
    #             FQ_ref['10'][k][i][j] = FQ_ref['1'][k][i] - FQ_ref['11'][k][i][j]
    #             FQ_ref['01'][k][i][j] = FQ_ref['1'][k][j] - FQ_ref['11'][k][i][j]
    #             FQ_ref['00'][k][i][j] = FQ_ref['0'][k][i] - FQ_ref['01'][k][i][j]

    FQ0 = np.expand_dims(D,axis=1) - FQ1
    FQ10 = np.expand_dims(FQ1,axis=2) - FQ11
    FQ01 = np.expand_dims(FQ1,axis=1) - FQ11
    FQ00 = np.expand_dims(FQ0,axis=2) - FQ01

    # assert (np.sum(np.asarray(FQ_ref['0']) != FQ0) == 0)
    # assert (np.sum(np.asarray(FQ_ref['10']) != FQ10) == 0)
    # assert (np.sum(np.asarray(FQ_ref['01']) != FQ01) == 0)
    # assert (np.sum(np.asarray(FQ_ref['00']) != FQ00) == 0)
    #
    # leaves = ['00', '01', '10', '11']

    # Use frequency counters to determine the optimal subtree structure

    # # Number correctly classified in each leaf for each combination of features
    # num_classified_ref = {c: [[0 for d1 in F] for d0 in F]
    #                   for c in leaves}
    # best_class = {c: [[None for d1 in F] for d0 in F]
    #                   for c in leaves}
    #
    # for i in F:
    #     for j in F:
    #         for leaf in leaves:
    #             best_class_idx = None
    #             best_class_obj = -1
    #             for k in range(len(K)):
    #                 num_in_leaf = FQ_ref[leaf][k][i][j]
    #                 if num_in_leaf > best_class_obj:
    #                     best_class_obj = num_in_leaf
    #                     best_class_idx = k
    #             num_classified_ref[leaf][i][j] = best_class_obj
    #             best_class[leaf][i][j] = K[best_class_idx]

    # For each combination of features, determine in each leaf
    # the number of samples with the majority class
    n_classified00 = np.max(FQ00,axis=0)
    n_classified01 = np.max(FQ01, axis=0)
    n_classified10 = np.max(FQ10, axis=0)
    n_classified11 = np.max(FQ11, axis=0)

    # assert (np.sum(np.asarray(num_classified_ref['00']) != n_classified00) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['01']) != n_classified01) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['10']) != n_classified10) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['11']) != n_classified11) == 0)
    #
    # left_subtree_scores_ref = [[num_classified_ref['00'][i][j] + num_classified_ref['01'][i][j] for j in F] for i in F]
    # right_subtree_scores_ref = [[num_classified_ref['10'][i][j] + num_classified_ref['11'][i][j] for j in F] for i in F]

    # Get the number of samples correctly classified in the left and right subtrees for each feature combination
    # dimension (|F|,|F|) where element (a,b) is the objective for parent feature f_a with left/right child feature f_b
    left_subtree_scores = n_classified00 + n_classified01
    right_subtree_scores = n_classified10 + n_classified11

    # assert (np.sum(np.asarray(left_subtree_scores_ref) != left_subtree_scores) == 0)
    # assert (np.sum(np.asarray(right_subtree_scores_ref) != right_subtree_scores) == 0)
    #
    # best_obj_ref = -1
    # best_parent_feature_ref = None
    #
    # for i in F:
    #     left_obj, right_obj = -1, -1
    #     left_features = []
    #     right_features = []
    #     for j in F:
    #         left_score = left_subtree_scores_ref[i][j]
    #         right_score = right_subtree_scores_ref[i][j]
    #
    #         if left_score > left_obj:
    #             left_features = [j]
    #             left_obj = left_subtree_scores_ref[i][j]
    #         elif left_score == left_obj:
    #             left_features.append(j)
    #
    #         if right_score > right_obj:
    #             right_features = [j]
    #             right_obj = right_subtree_scores_ref[i][j]
    #         elif right_score == right_obj:
    #             right_features.append(j)
    #
    #     if left_obj + right_obj > best_obj_ref:
    #         best_obj_ref = left_obj + right_obj
    #         best_parent_feature_ref = i
    #         best_left_child_features_ref = left_features
    #         best_right_child_features_ref = right_features

    # Find the combination of parent/child features that maximise the
    # number of correctly classified points

    subtree_left_maxes = np.max(left_subtree_scores, axis=1)
    subtree_right_maxes = np.max(right_subtree_scores, axis=1)

    total_scores = subtree_left_maxes + subtree_right_maxes

    # TODO: Investigate multiple potential best parent features simultaneously?
    best_parent_feature = np.argmax(total_scores)

    if bin_to_cat_group is not None or bin_to_num_group is not None:
        # If we are provided with categorical or numerical feature groupings we
        # must make sure that the solution respects them

        # Find the best objective in each subtree for the given parent feature
        best_left_subtree_objective = np.max(left_subtree_scores[best_parent_feature,:])
        best_right_subtree_objective = np.max(right_subtree_scores[best_parent_feature, :])

        # In each subtree find the set of branch features which give the maximum objective
        candidate_left_features = np.nonzero(left_subtree_scores[best_parent_feature,:] == best_left_subtree_objective)[0]
        candidate_right_features = np.nonzero(right_subtree_scores[best_parent_feature, :] == best_right_subtree_objective)[0]

        # We want the features which are made invalid in the left and right subtrees by the parent feature
        parent_feature_all = F_map_to_all[best_parent_feature]

        if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
            all_num_group = bin_to_num_group[parent_feature_all]
            F_num_group = [all_map_to_F[f] for f in all_num_group if f in all_map_to_F]

            left_invalid_grouping = [f for f in F_num_group if f >= best_parent_feature]
            right_invalid_grouping = [f for f in F_num_group if f <= best_parent_feature]

        elif bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
            all_cat_group = bin_to_cat_group[parent_feature_all]    # Binary features associated parent feature categorical variable
            F_cat_group = [all_map_to_F[f] for f in all_cat_group if f in all_map_to_F]     # Associated binary features in the REDUCED feature space

            left_invalid_grouping = [best_parent_feature]
            right_invalid_grouping = F_cat_group

        else:
            left_invalid_grouping = [best_parent_feature]
            right_invalid_grouping = [best_parent_feature]

        # Filter the feature choices which give maximum objectives in each subtree based on the supplied binary feature mappings
        candidate_left_features = candidate_left_features[~np.isin(candidate_left_features, left_invalid_grouping)]
        candidate_right_features = candidate_right_features[~np.isin(candidate_right_features, right_invalid_grouping)]

        if len(candidate_left_features) > 0 and len(candidate_right_features) > 0:
            best_left_child_feature = candidate_left_features[0]
            best_right_child_feature = candidate_right_features[0]
        else:
            print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
            best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature, :])
            best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])
    else:
        best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature,:])
        best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])




    # if bin_to_cat_group is not None or bin_to_num_group is not None:
    #     # If we are provided with categorical or numerical feature groupings we
    #     # must make sure that the solution respects them
    #     best_left_child_feature_ref = None
    #     best_right_child_feature = None
    #     for j in best_left_child_features_ref:
    #         # If invalid features were supplied, maps back to the full set of feature F_all
    #         left_feature_all = F_map_to_all[j]
    #         parent_feature_all = F_map_to_all[best_parent_feature_ref]
    #
    #         if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
    #             invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f >= parent_feature_all)
    #         else:
    #             invalid_grouping = [parent_feature_all]
    #
    #
    #         if left_feature_all not in invalid_grouping:
    #             best_left_child_feature_ref = j
    #             break
    #
    #     for j in best_right_child_features_ref:
    #         # If invalid features were supplied, maps back to the full set of feature F_all
    #         right_feature_all = F_map_to_all[j]
    #         parent_feature_all = F_map_to_all[best_parent_feature_ref]
    #
    #         if bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
    #             invalid_grouping = bin_to_cat_group[parent_feature_all]
    #         if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
    #             invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f <= parent_feature_all)
    #
    #         if right_feature_all not in invalid_grouping:
    #             best_right_child_feature = j
    #             break
    #
    #     if best_left_child_feature_ref is None:
    #         best_left_child_feature_ref = best_left_child_features_ref[0]
    #         print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
    #
    #     if best_right_child_feature is None:
    #         best_right_child_feature = best_right_child_features_ref[0]
    #         print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
    #
    # else:
    #     best_left_child_feature_ref = best_left_child_features_ref[0]
    #     best_right_child_feature_ref = best_right_child_features_ref[0]

    # assert (best_parent_feature == best_parent_feature_ref)
    # assert (best_left_child_feature == best_left_child_feature_ref)
    # assert (best_right_child_feature == best_right_child_feature_ref)
    #
    #
    # leaf_predictions = {'00': best_class['00'][best_parent_feature_ref][best_left_child_feature],
    #                     '01': best_class['01'][best_parent_feature_ref][best_left_child_feature],
    #                     '10': best_class['10'][best_parent_feature_ref][best_right_child_feature],
    #                     '11': best_class['11'][best_parent_feature_ref][best_right_child_feature]}


    # # Map from reduced feature set back to all features
    # best_parent_feature_ref = F_map_to_all[best_parent_feature_ref]
    # best_left_child_feature_ref = F_map_to_all[best_left_child_feature_ref]
    # best_right_child_feature_ref = F_map_to_all[best_right_child_feature_ref]
    #
    # b_ref = (best_parent_feature_ref, best_left_child_feature_ref, best_right_child_feature_ref)
    # w_ref = [leaf_predictions[leaf] for leaf in leaves]
    # theta = []

    w_class_idx = [np.argmax(FQ00[:, best_parent_feature, best_left_child_feature]),
                   np.argmax(FQ01[:, best_parent_feature, best_left_child_feature]),
                   np.argmax(FQ10[:, best_parent_feature, best_right_child_feature]),
                   np.argmax(FQ11[:, best_parent_feature, best_right_child_feature])]

    # Convert back to original classes, instead of the idx in the sorted list of unique classes
    w = [K[k_idx] for k_idx in w_class_idx]

    best_parent_feature = F_map_to_all[best_parent_feature]
    best_left_child_feature = F_map_to_all[best_left_child_feature]
    best_right_child_feature = F_map_to_all[best_right_child_feature]

    b = (best_parent_feature, best_left_child_feature, best_right_child_feature)

    # assert b == b_ref
    # assert w == w_ref

    # theta_ref = []
    #
    # for idx in I:
    #     sample = X[idx,:]
    #     if sample[best_parent_feature] > 0.5:
    #         # Sample goes right
    #         if sample[best_right_child_feature] > 0.5:
    #             if y[idx] == w[3]:
    #                 theta_ref.append(idx)
    #         else:
    #             if y[idx] == w[2]:
    #                 theta_ref.append(idx)
    #     else:
    #         # Sample goes left
    #         if sample[best_left_child_feature] > 0.5:
    #             if y[idx] == w[1]:
    #                 theta_ref.append(idx)
    #         else:
    #             if y[idx] == w[0]:
    #                 theta_ref.append(idx)

    theta = np.zeros(n_samples,dtype=bool)

    left_subtree_mask = (X[:,best_parent_feature] == 0)
    right_subtree_mask = (X[:, best_parent_feature] == 1)

    leaf00_mask = left_subtree_mask * (X[:,best_left_child_feature] == 0) * (y == w[0])
    leaf01_mask = left_subtree_mask * (X[:, best_left_child_feature] == 1) * (y == w[1])
    leaf10_mask = right_subtree_mask * (X[:, best_right_child_feature] == 0) * (y == w[2])
    leaf11_mask = right_subtree_mask * (X[:, best_right_child_feature] == 1) * (y == w[3])

    theta[leaf00_mask + leaf01_mask + leaf10_mask + leaf11_mask] = True
    theta = np.nonzero(theta)[0]

    # assert np.sum(np.asarray(theta_ref) != theta) == 0

    # theta[leaf00_mask] = True
    # theta[leaf01_mask] = True
    # theta[leaf10_mask] = True
    # theta[leaf11_mask] = True

    return b, w, theta

def optimise_regularised_depth2_subtree(X,y,
                                        tree=None,
                                        weights=None,
                                        alpha=None,
                                        invalid_features=None,
                                        bin_to_cat_group=None,
                                        bin_to_num_group=None):
    """Numpy based implementation of the D2S subroutine

    A pure python reference implementation, denoted by variables ending with _ref, was used to validate the implementation
    It has been left as an easier to read reference

    Args:
        X: Data matrix
        y: Target vector
        tree: Optional instance of Tree class. If not provided a generic depth 2 tree will be created
        weights: (Currently not supported) Weights of samples in objective
        alpha: Per leaf penalty on the classification score. MUST be specified in terms of number of samples
        invalid_features (list or set): Set of features which are unavailable to be split on in the optimised subtree
        bin_to_cat_group: (Current not supported)
        bin_to_num_group: (Currently not supported)

    Returns:
        Returns tuple (b,w,theta) which describes the optimal subtree solution. b is a tuple (parent, left_child, right_child)
        where each holds the branch feature at the root node, left child, and right child respectively. If a node is cut off
        (i.e. it or an ancestor is a leaf node) it takes a value of None. w is a variable length tuple with the predicted
        class in each of the leaf nodes in where nodes are in ascending bfs ordering.  The correspondence between
        prediction & node can be inferred from b. theta is a list of indices of the samples in the subtree which are
        correctly classified in the optimal solution. N.B. these indices do not correspond to the full dataset
    """

    # TODO: Allow for weight and bin_to_x_groups to be used
    if weights is not None:
        # Weights currently not supported in numpy optimise_regularised_depth2_subtree implementation.
        assert sum(weights) == len(weights)

    if bin_to_cat_group is not None or bin_to_num_group is not None:
        # Regularisation not supported with No Feature Reuse constraints
        assert (alpha is None)

    n_samples, n_features = X.shape
    I, F_all = range(n_samples), range(n_features)

    # Get unique classes in data and associate them to an index
    # Need k_to_class_idx for when K != [0,1,...,|K|-1]
    K = np.unique(y).tolist()
    k_to_class_idx = {k: i for i, k in enumerate(K)}

    # If no tree is given create a generic depth 2 tree
    if tree is None:
        tree = Tree(2)
    elif tree.depth != 2:
        return None, None, None

    if weights is None:
        weights = [1] * n_samples

    if invalid_features is not None:
    # Given a set of invalid features, we want the subroutine to operate only on the subset of valid features
    # and then map back to the full set of features

        all_map_to_F = {}
        F_map_to_all = []   # Keep track of how our reduced feature set maps back to all features
        F_mask = []         # Used to mask the feature dimension when indexing X

        for i, f in enumerate(F_all):
            if f not in invalid_features:
                all_map_to_F[f] = len(F_map_to_all)
                F_map_to_all.append(i)
                F_mask.append(True)
            else:
                F_mask.append(False)

        F = range(len(F_map_to_all))
    else:
        F_map_to_all = F_all
        F_mask = [True] * len(F_all)
        F = F_all

    # TODO: modify this to return the smallest subtree (i.e. make an arbitrary prediction in the subtree root)
    # Subroutine will fail with no data supplied
    if n_samples == 0:
        w = [0, 0, 0, 0]
        theta_idx = []

        # Still assure that output conforms to NoFeatureReuse inequalities if required
        if bin_to_cat_group is not None or bin_to_num_group is not None:
            soln_found = False
            for f in F:
                parent_feature = F_map_to_all[f]

                # Find invalid left and right children for the given candidate parent feature
                if bin_to_num_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f >= parent_feature)
                    right_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f <= parent_feature)
                elif bin_to_cat_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = bin_to_num_group[parent_feature]
                else:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = [parent_feature]

                left_child_feature, right_child_feature = None, None

                for f_left in F:
                    f_left_candidate = F_map_to_all[f_left]
                    if f_left_candidate not in left_invalid_grouping:
                        left_child_feature = f_left_candidate

                for f_right in F:
                    f_right_candidate = F_map_to_all[f_right]
                    if f_right_candidate not in right_invalid_grouping:
                        right_child_feature = f_right_candidate

                if left_child_feature is not None and right_child_feature is not None:
                    b = (parent_feature, left_child_feature, right_child_feature)
                    soln_found = True
                    break

            if not soln_found:
                # In this case there are no feature combinations which are valid according to the
                # provided invalid_feature set and feature mappings
                return None, None, None

        else:
            b = (F_map_to_all[0], F_map_to_all[0], F_map_to_all[0])

        return b, w, theta_idx

    ############### BEGIN SUBROUTINE PROPER ###############

    # # Set up frequency counters
    # FQ_ref = {'0': [[0 for d1 in F] for d0 in range(len(K))],
    #       '1': [[0 for d1 in F] for d0 in range(len(K))],
    #       '00': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '01': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '10': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
    #       '11': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))]}

    # TODO: Cut down on memory usage by using smaller ints?
    FQ1 = np.zeros((len(K), len(F)), np.int32)
    FQ11 = np.zeros((len(K), len(F), len(F)), np.int32)

    # Track number of samples of each class in the subtree
    D_ref = [0] * len(K)

    _, D = np.unique(y, return_counts=True)

    # for idx in I:
    #     k = k_to_class_idx[y[idx]]   # Offset to index into lists
    #     fv = X[idx,F_mask]     # feature vector
    #     D_ref[k] += 1
    #     for i in F:
    #         if fv[i] > 0.5:
    #             FQ_ref['1'][k][i] += weights[idx]
    #             for j in F:
    #                 if fv[j] > 0.5:
    #                     FQ_ref['11'][k][i][j] += weights[idx]

    for k in K:
        # Technically collecting the frequency counters could be done without looping over K
        # It's beyond me how to do it with numpy though

        # Take the subarray of X in which all samples have class y^i = k and features are filtered by F_mask
        X_masked = X[np.ix_(y==k,F_mask)]

        k_idx = k_to_class_idx[k]

        # Each column of X_masked is associated with feature f, and indicates which samples have x_f^i == 1
        # Taking the dot product of columns associated with features f_a and f_b will be equal to the number of
        # samples for which (x_fa^i == 1 AND x_fb^i == 1)
        FQ11[k_idx,:,:] = X_masked.T @ X_masked

        # The diagonal corresponds to duplicated features, i.e. x_fa^i == 1
        FQ1[k_idx,:] = np.diag(FQ11[k_idx,:,:])


    # assert (np.sum(np.asarray(FQ_ref['1']) != FQ1) == 0)
    # assert (np.sum(np.asarray(FQ_ref['11']) != FQ11) == 0)
    #
    # # Fill out symmetry of matrix
    # for k in range(len(K)):
    #     for i in F:
    #         for j in range(i+1,len(F)):
    #             FQ_ref['11'][k][j][i] = FQ_ref['11'][k][i][j]
    #
    # for k in range(len(K)):
    #     for i in F:
    #         FQ_ref['0'][k][i] = D[k] - FQ_ref['1'][k][i]
    #         for j in F:
    #             FQ_ref['10'][k][i][j] = FQ_ref['1'][k][i] - FQ_ref['11'][k][i][j]
    #             FQ_ref['01'][k][i][j] = FQ_ref['1'][k][j] - FQ_ref['11'][k][i][j]
    #             FQ_ref['00'][k][i][j] = FQ_ref['0'][k][i] - FQ_ref['01'][k][i][j]

    FQ0 = np.expand_dims(D,axis=1) - FQ1
    FQ10 = np.expand_dims(FQ1,axis=2) - FQ11
    FQ01 = np.expand_dims(FQ1,axis=1) - FQ11
    FQ00 = np.expand_dims(FQ0,axis=2) - FQ01

    # assert (np.sum(np.asarray(FQ_ref['0']) != FQ0) == 0)
    # assert (np.sum(np.asarray(FQ_ref['10']) != FQ10) == 0)
    # assert (np.sum(np.asarray(FQ_ref['01']) != FQ01) == 0)
    # assert (np.sum(np.asarray(FQ_ref['00']) != FQ00) == 0)
    #
    # leaves = ['00', '01', '10', '11']

    # Use frequency counters to determine the optimal subtree structure

    # # Number correctly classified in each leaf for each combination of features
    # num_classified_ref = {c: [[0 for d1 in F] for d0 in F]
    #                   for c in leaves}
    # best_class = {c: [[None for d1 in F] for d0 in F]
    #                   for c in leaves}
    #
    # for i in F:
    #     for j in F:
    #         for leaf in leaves:
    #             best_class_idx = None
    #             best_class_obj = -1
    #             for k in range(len(K)):
    #                 num_in_leaf = FQ_ref[leaf][k][i][j]
    #                 if num_in_leaf > best_class_obj:
    #                     best_class_obj = num_in_leaf
    #                     best_class_idx = k
    #             num_classified_ref[leaf][i][j] = best_class_obj
    #             best_class[leaf][i][j] = K[best_class_idx]

    # For each combination of features, determine in each leaf
    # the number of samples with the majority class
    n_classified00 = np.max(FQ00,axis=0)
    n_classified01 = np.max(FQ01, axis=0)
    n_classified10 = np.max(FQ10, axis=0)
    n_classified11 = np.max(FQ11, axis=0)

    # assert (np.sum(np.asarray(num_classified_ref['00']) != n_classified00) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['01']) != n_classified01) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['10']) != n_classified10) == 0)
    # assert (np.sum(np.asarray(num_classified_ref['11']) != n_classified11) == 0)
    #
    # left_subtree_scores_ref = [[num_classified_ref['00'][i][j] + num_classified_ref['01'][i][j] for j in F] for i in F]
    # right_subtree_scores_ref = [[num_classified_ref['10'][i][j] + num_classified_ref['11'][i][j] for j in F] for i in F]

    # Get the number of samples correctly classified in the left and right subtrees for each feature combination
    # dimension (|F|,|F|) where element (a,b) is the objective for parent feature f_a with left/right child feature f_b
    left_subtree_scores = n_classified00 + n_classified01
    right_subtree_scores = n_classified10 + n_classified11

    # assert (np.sum(np.asarray(left_subtree_scores_ref) != left_subtree_scores) == 0)
    # assert (np.sum(np.asarray(right_subtree_scores_ref) != right_subtree_scores) == 0)
    #
    # best_obj_ref = -1
    # best_parent_feature_ref = None
    #
    # for i in F:
    #     left_obj, right_obj = -1, -1
    #     left_features = []
    #     right_features = []
    #     for j in F:
    #         left_score = left_subtree_scores_ref[i][j]
    #         right_score = right_subtree_scores_ref[i][j]
    #
    #         if left_score > left_obj:
    #             left_features = [j]
    #             left_obj = left_subtree_scores_ref[i][j]
    #         elif left_score == left_obj:
    #             left_features.append(j)
    #
    #         if right_score > right_obj:
    #             right_features = [j]
    #             right_obj = right_subtree_scores_ref[i][j]
    #         elif right_score == right_obj:
    #             right_features.append(j)
    #
    #     if left_obj + right_obj > best_obj_ref:
    #         best_obj_ref = left_obj + right_obj
    #         best_parent_feature_ref = i
    #         best_left_child_features_ref = left_features
    #         best_right_child_features_ref = right_features

    # Find the combination of parent/child features that maximise the
    # number of correctly classified points

    subtree_left_maxes = np.max(left_subtree_scores, axis=1)
    subtree_right_maxes = np.max(right_subtree_scores, axis=1)

    total_scores = subtree_left_maxes + subtree_right_maxes

    if alpha is not None:
        # Run similar calculations in regularised case
        # n_classified0 = np.max(FQ0)
        # n_classified1 = np.max(FQ1)

        one_leaf_best_score = np.max(D)

        n_classified0 = np.diag(n_classified00)
        n_classified1 = np.diag(n_classified11)
        two_leaf_best_score = np.max(n_classified0 + n_classified1)

        # Score if we branch on parent and left child, leaving right child as a leaf
        three_leaf_left_scores = np.max(left_subtree_scores,axis=1) + n_classified1
        three_leaf_left_best_score = np.max(three_leaf_left_scores)

        # Score if we branch on parent and right child, leaving left child as a leaf
        three_leaf_right_scores =  np.max(right_subtree_scores,axis=1) + n_classified0
        three_leaf_right_best_score = np.max(three_leaf_right_scores)


        if three_leaf_left_best_score > three_leaf_right_best_score:
            three_leaf_goes_left = True
            three_leaf_best_score = three_leaf_left_best_score
        else:
            three_leaf_goes_left = False
            three_leaf_best_score = three_leaf_right_best_score

        four_leaf_best_score = np.max(total_scores)

        regularised_obj_vals = np.asarray([one_leaf_best_score,
                                           two_leaf_best_score,
                                           three_leaf_best_score,
                                           four_leaf_best_score]) - alpha * np.arange(1,5)

        optimal_leaf_number = np.argmax(regularised_obj_vals) + 1

        best_parent_feature = None
        best_left_child_feature = None
        best_right_child_feature = None

        if optimal_leaf_number == 2:
            best_parent_feature = np.argmax(n_classified0 + n_classified1)
        elif optimal_leaf_number == 3:
            if three_leaf_goes_left:
                # Left child is a branch node
                best_parent_feature = np.argmax(three_leaf_left_scores)
                best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature,:])
            else:
                # Right child is a branch node
                best_parent_feature = np.argmax(three_leaf_right_scores)
                best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])
        elif optimal_leaf_number == 4:
            best_parent_feature = np.argmax(total_scores)
            best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature, :])
            best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])
    else:
        optimal_leaf_number = 4

        # TODO: Investigate multiple potential best parent features simultaneously?
        best_parent_feature = np.argmax(total_scores)

        if bin_to_cat_group is not None or bin_to_num_group is not None:
            # If we are provided with categorical or numerical feature groupings we
            # must make sure that the solution respects them

            # Find the best objective in each subtree for the given parent feature
            best_left_subtree_objective = np.max(left_subtree_scores[best_parent_feature,:])
            best_right_subtree_objective = np.max(right_subtree_scores[best_parent_feature, :])

            # In each subtree find the set of branch features which give the maximum objective
            candidate_left_features = np.nonzero(left_subtree_scores[best_parent_feature,:] == best_left_subtree_objective)[0]
            candidate_right_features = np.nonzero(right_subtree_scores[best_parent_feature, :] == best_right_subtree_objective)[0]

            # We want the features which are made invalid in the left and right subtrees by the parent feature
            parent_feature_all = F_map_to_all[best_parent_feature]

            if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
                all_num_group = bin_to_num_group[parent_feature_all]
                F_num_group = [all_map_to_F[f] for f in all_num_group if f in all_map_to_F]

                left_invalid_grouping = [f for f in F_num_group if f >= best_parent_feature]
                right_invalid_grouping = [f for f in F_num_group if f <= best_parent_feature]

            elif bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
                all_cat_group = bin_to_cat_group[parent_feature_all]    # Binary features associated parent feature categorical variable
                F_cat_group = [all_map_to_F[f] for f in all_cat_group if f in all_map_to_F]     # Associated binary features in the REDUCED feature space

                left_invalid_grouping = [best_parent_feature]
                right_invalid_grouping = F_cat_group

            else:
                left_invalid_grouping = [best_parent_feature]
                right_invalid_grouping = [best_parent_feature]

            # Filter the feature choices which give maximum objectives in each subtree based on the supplied binary feature mappings
            candidate_left_features = candidate_left_features[~np.isin(candidate_left_features, left_invalid_grouping)]
            candidate_right_features = candidate_right_features[~np.isin(candidate_right_features, right_invalid_grouping)]

            if len(candidate_left_features) > 0 and len(candidate_right_features) > 0:
                best_left_child_feature = candidate_left_features[0]
                best_right_child_feature = candidate_right_features[0]
            else:
                print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
                best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature, :])
                best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])
        else:
            best_left_child_feature = np.argmax(left_subtree_scores[best_parent_feature,:])
            best_right_child_feature = np.argmax(right_subtree_scores[best_parent_feature, :])




    # if bin_to_cat_group is not None or bin_to_num_group is not None:
    #     # If we are provided with categorical or numerical feature groupings we
    #     # must make sure that the solution respects them
    #     best_left_child_feature_ref = None
    #     best_right_child_feature = None
    #     for j in best_left_child_features_ref:
    #         # If invalid features were supplied, maps back to the full set of feature F_all
    #         left_feature_all = F_map_to_all[j]
    #         parent_feature_all = F_map_to_all[best_parent_feature_ref]
    #
    #         if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
    #             invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f >= parent_feature_all)
    #         else:
    #             invalid_grouping = [parent_feature_all]
    #
    #
    #         if left_feature_all not in invalid_grouping:
    #             best_left_child_feature_ref = j
    #             break
    #
    #     for j in best_right_child_features_ref:
    #         # If invalid features were supplied, maps back to the full set of feature F_all
    #         right_feature_all = F_map_to_all[j]
    #         parent_feature_all = F_map_to_all[best_parent_feature_ref]
    #
    #         if bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
    #             invalid_grouping = bin_to_cat_group[parent_feature_all]
    #         if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
    #             invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f <= parent_feature_all)
    #
    #         if right_feature_all not in invalid_grouping:
    #             best_right_child_feature = j
    #             break
    #
    #     if best_left_child_feature_ref is None:
    #         best_left_child_feature_ref = best_left_child_features_ref[0]
    #         print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
    #
    #     if best_right_child_feature is None:
    #         best_right_child_feature = best_right_child_features_ref[0]
    #         print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')
    #
    # else:
    #     best_left_child_feature_ref = best_left_child_features_ref[0]
    #     best_right_child_feature_ref = best_right_child_features_ref[0]

    # assert (best_parent_feature == best_parent_feature_ref)
    # assert (best_left_child_feature == best_left_child_feature_ref)
    # assert (best_right_child_feature == best_right_child_feature_ref)
    #
    #
    # leaf_predictions = {'00': best_class['00'][best_parent_feature_ref][best_left_child_feature],
    #                     '01': best_class['01'][best_parent_feature_ref][best_left_child_feature],
    #                     '10': best_class['10'][best_parent_feature_ref][best_right_child_feature],
    #                     '11': best_class['11'][best_parent_feature_ref][best_right_child_feature]}


    # # Map from reduced feature set back to all features
    # best_parent_feature_ref = F_map_to_all[best_parent_feature_ref]
    # best_left_child_feature_ref = F_map_to_all[best_left_child_feature_ref]
    # best_right_child_feature_ref = F_map_to_all[best_right_child_feature_ref]
    #
    # b_ref = (best_parent_feature_ref, best_left_child_feature_ref, best_right_child_feature_ref)
    # w_ref = [leaf_predictions[leaf] for leaf in leaves]
    # theta = []

    # IMPORTANT: w is returned in order of highest nodes, followed by leftmost nodes (BFS ordering).
    if optimal_leaf_number == 1:
        w_class_idx = [np.argmax(D)]
    elif optimal_leaf_number == 2:
        w_class_idx = [np.argmax(FQ0[:, best_parent_feature]),
                       np.argmax(FQ1[:, best_parent_feature])]
    elif optimal_leaf_number == 3:
        if three_leaf_goes_left:
            w_class_idx = [np.argmax(FQ1[:, best_parent_feature]),
                           np.argmax(FQ00[:, best_parent_feature, best_left_child_feature]),
                           np.argmax(FQ01[:, best_parent_feature, best_left_child_feature])]
        else:
            w_class_idx = [np.argmax(FQ0[:, best_parent_feature]),
                           np.argmax(FQ10[:, best_parent_feature, best_right_child_feature]),
                           np.argmax(FQ11[:, best_parent_feature, best_right_child_feature])]
    elif optimal_leaf_number == 4:
        w_class_idx = [np.argmax(FQ00[:, best_parent_feature, best_left_child_feature]),
                       np.argmax(FQ01[:, best_parent_feature, best_left_child_feature]),
                       np.argmax(FQ10[:, best_parent_feature, best_right_child_feature]),
                       np.argmax(FQ11[:, best_parent_feature, best_right_child_feature])]

    # Convert back to original classes, instead of the idx in the sorted list of unique classes
    w = [K[k_idx] for k_idx in w_class_idx]

    # Map back to original feature space
    if best_parent_feature is not None:
        best_parent_feature = F_map_to_all[best_parent_feature]
    if best_left_child_feature is not None:
        best_left_child_feature = F_map_to_all[best_left_child_feature]
    if best_right_child_feature is not None:
        best_right_child_feature = F_map_to_all[best_right_child_feature]

    b = (best_parent_feature, best_left_child_feature, best_right_child_feature)

    # assert b == b_ref
    # assert w == w_ref

    # theta_ref = []
    #
    # for idx in I:
    #     sample = X[idx,:]
    #     if sample[best_parent_feature] > 0.5:
    #         # Sample goes right
    #         if sample[best_right_child_feature] > 0.5:
    #             if y[idx] == w[3]:
    #                 theta_ref.append(idx)
    #         else:
    #             if y[idx] == w[2]:
    #                 theta_ref.append(idx)
    #     else:
    #         # Sample goes left
    #         if sample[best_left_child_feature] > 0.5:
    #             if y[idx] == w[1]:
    #                 theta_ref.append(idx)
    #         else:
    #             if y[idx] == w[0]:
    #                 theta_ref.append(idx)

    theta = np.zeros(n_samples,dtype=bool)

    left_subtree_mask = (X[:,best_parent_feature] == 0)
    right_subtree_mask = (X[:, best_parent_feature] == 1)

    # Each mask indicates the samples which are correctly classified in the corresponding leaf
    # By default each mask has no effect
    leaf_root_mask = np.zeros(n_samples,dtype=bool)
    leaf0_mask = np.zeros(n_samples,dtype=bool)
    leaf1_mask = np.zeros(n_samples, dtype=bool)
    leaf00_mask = np.zeros(n_samples,dtype=bool)
    leaf01_mask = np.zeros(n_samples, dtype=bool)
    leaf10_mask = np.zeros(n_samples, dtype=bool)
    leaf11_mask = np.zeros(n_samples, dtype=bool)

    if optimal_leaf_number == 1:
        leaf_root_mask = (y == w[0])
    elif optimal_leaf_number == 2:
        leaf0_mask = left_subtree_mask * (y == w[0])
        leaf1_mask = right_subtree_mask * (y == w[1])
    elif optimal_leaf_number == 3:
        if three_leaf_goes_left:
            leaf1_mask = right_subtree_mask * (y == w[0])
            leaf00_mask = left_subtree_mask * (X[:,best_left_child_feature] == 0) * (y == w[1])
            leaf01_mask = left_subtree_mask * (X[:,best_left_child_feature] == 1) * (y == w[2])
        else:
            leaf0_mask = left_subtree_mask * (y == w[0])
            leaf10_mask = right_subtree_mask * (X[:, best_right_child_feature] == 0) * (y == w[1])
            leaf11_mask = right_subtree_mask * (X[:, best_right_child_feature] == 1) * (y == w[2])
    elif optimal_leaf_number == 4:
        leaf00_mask = left_subtree_mask * (X[:,best_left_child_feature] == 0) * (y == w[0])
        leaf01_mask = left_subtree_mask * (X[:, best_left_child_feature] == 1) * (y == w[1])
        leaf10_mask = right_subtree_mask * (X[:, best_right_child_feature] == 0) * (y == w[2])
        leaf11_mask = right_subtree_mask * (X[:, best_right_child_feature] == 1) * (y == w[3])

    theta[leaf_root_mask + leaf0_mask + leaf1_mask + leaf00_mask + leaf01_mask + leaf10_mask + leaf11_mask] = True
    theta = np.nonzero(theta)[0]

    return b, w, theta

def optimise_depth2_subtree_reference(X,y,
                            tree=None,
                            weights=None,
                            invalid_features=None,
                            bin_to_cat_group=None,
                            bin_to_num_group=None):
    """Old reference implementation for D2S subroutine for balanced trees"""

    n_samples, n_features = X.shape
    I, F_all = range(n_samples), range(n_features)

    # Get unique classes in data and associate them to an index
    K = np.unique(y).tolist()
    k_to_class_idx = {k: i for i, k in enumerate(K)}

    # If no tree is given create a generic depth 2 tree
    if tree is None:
        tree = Tree(2)
    elif tree.depth != 2:
        return None, None, None

    if weights is None:
        weights = [1] * n_samples

    if invalid_features is not None:
    # Given a set of invalid features, we want the subroutine to operate only on the subset of valid features
    # and then map back to the full set of features

        F_map_to_all = []   # Keep track of how our reduced feature set maps back to all features
        F_mask = []         # Used to mask the feature dimension when indexing X

        for i, f in enumerate(F_all):
            if f not in invalid_features:
                F_map_to_all.append(i)
                F_mask.append(True)
            else:
                F_mask.append(False)

        F = range(len(F_map_to_all))
    else:
        F_map_to_all = F_all
        F_mask = [True] * len(F_all)
        F = F_all

    # Subroutine will fail with no data supplied
    if n_samples == 0:
        w = [0, 0, 0, 0]
        theta_idx = []

        # Still assure that output conforms to NoFeatureReuse inequalities if required
        if bin_to_cat_group is not None or bin_to_num_group is not None:
            for f in F:
                parent_feature = F_map_to_all[f]

                # Find invalid left and right children for the given candidate parent feature
                if bin_to_num_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f >= parent_feature)
                    left_invalid_grouping = set(f for f in bin_to_num_group[parent_feature] if f <= parent_feature)
                elif bin_to_cat_group is not None and parent_feature in bin_to_num_group:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = bin_to_num_group[parent_feature]
                else:
                    left_invalid_grouping = [parent_feature]
                    right_invalid_grouping = [parent_feature]

                left_child_feature, right_child_feature = None, None

                for f_left in F:
                    f_left_candidate = F_map_to_all[f_left]
                    if f_left_candidate not in left_invalid_grouping:
                        left_child_feature = f_left_candidate

                for f_right in F:
                    f_right_candidate = F_map_to_all[f_right]
                    if f_right_candidate not in right_invalid_grouping:
                        right_child_feature = f_right_candidate

                if left_child_feature is not None and right_child_feature is not None:
                    b = (parent_feature, left_child_feature, right_child_feature)

        else:
            b = (F_map_to_all[0], F_map_to_all[0], F_map_to_all[0])

        return b, w, theta_idx

    ############### BEGIN SUBROUTINE PROPER ###############

    # Set up frequency counters
    FQ = {'0': [[0 for d1 in F] for d0 in range(len(K))],
          '1': [[0 for d1 in F] for d0 in range(len(K))],
          '00': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
          '01': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
          '10': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))],
          '11': [[[0 for d2 in F] for d1 in F] for d2 in range(len(K))]}


    # Track number of samples of each class in the subtree
    D = [0] * len(K)

    for idx in I:
        k = k_to_class_idx[y[idx]]   # Offset to index into lists
        fv = X[idx,F_mask]     # feature vector
        D[k] += 1
        for i in F:
            if fv[i] > 0.5:
                FQ['1'][k][i] += weights[idx]
                for j in F:
                    if fv[j] > 0.5:
                        FQ['11'][k][i][j] += weights[idx]

    # Fill out symmetry of matrix
    for k in range(len(K)):
        for i in F:
            for j in range(i+1,len(F)):
                FQ['11'][k][j][i] = FQ['11'][k][i][j]

    for k in range(len(K)):
        for i in F:
            FQ['0'][k][i] = D[k] - FQ['1'][k][i]
            for j in F:
                FQ['10'][k][i][j] = FQ['1'][k][i] - FQ['11'][k][i][j]
                FQ['01'][k][i][j] = FQ['1'][k][j] - FQ['11'][k][i][j]
                FQ['00'][k][i][j] = FQ['0'][k][i] - FQ['01'][k][i][j]
                # FQ2['00'][k][i][j] = D[k] - FQ2['1'][k][i] - FQ2['1'][k][j] + FQ2['11'][k][i][j]


    leaves = ['00', '01', '10', '11']

    # Use frequency counters to determine the optimal subtree structure

    # Number correctly classified in each leaf for each combination of features
    num_classified = {c: [[0 for d1 in F] for d0 in F]
                      for c in leaves}
    best_class = {c: [[None for d1 in F] for d0 in F]
                      for c in leaves}

    for i in F:
        for j in F:
            for leaf in leaves:
                best_class_idx = None
                best_class_obj = -1
                for k in range(len(K)):
                    num_in_leaf = FQ[leaf][k][i][j]
                    if num_in_leaf > best_class_obj:
                        best_class_obj = num_in_leaf
                        best_class_idx = k
                num_classified[leaf][i][j] = best_class_obj
                best_class[leaf][i][j] = K[best_class_idx]

    left_subtree_scores = [[num_classified['00'][i][j] + num_classified['01'][i][j] for j in F] for i in F]
    right_subtree_scores = [[num_classified['10'][i][j] + num_classified['11'][i][j] for j in F] for i in F]

    best_obj = -1
    best_parent_feature = None

    for i in F:
        left_obj, right_obj = -1, -1
        left_features = []
        right_features = []
        for j in F:
            left_score = left_subtree_scores[i][j]
            right_score = right_subtree_scores[i][j]

            if left_score > left_obj:
                left_features = [j]
                left_obj = left_subtree_scores[i][j]
            elif left_score == left_obj:
                left_features.append(j)

            if right_score > right_obj:
                right_features = [j]
                right_obj = right_subtree_scores[i][j]
            elif right_score == right_obj:
                right_features.append(j)

        if left_obj + right_obj > best_obj:
            best_obj = left_obj + right_obj
            best_parent_feature = i
            best_left_child_features = left_features
            best_right_child_features = right_features

    if bin_to_cat_group is not None or bin_to_num_group is not None:
        # If we are provided with categorical or numerical feature groupings we
        # must make sure that the solution respects them
        best_left_child_feature = None
        best_right_child_feature = None
        for j in best_left_child_features:
            # If invalid features were supplied, maps back to the full set of feature F_all
            left_feature_all = F_map_to_all[j]
            parent_feature_all = F_map_to_all[best_parent_feature]

            if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
                invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f >= parent_feature_all)
            else:
                invalid_grouping = [parent_feature_all]


            if j not in invalid_grouping:
                best_left_child_feature = j
                break

        for j in best_right_child_features:
            # If invalid features were supplied, maps back to the full set of feature F_all
            right_feature_all = F_map_to_all[j]
            parent_feature_all = F_map_to_all[best_parent_feature]

            if bin_to_cat_group is not None and parent_feature_all in bin_to_cat_group:
                invalid_grouping = bin_to_cat_group[parent_feature_all]
            if bin_to_num_group is not None and parent_feature_all in bin_to_num_group:
                invalid_grouping = set(f for f in bin_to_num_group[parent_feature_all] if f <= parent_feature_all)

            if right_feature_all not in invalid_grouping:
                best_right_child_feature = j
                break

        if best_left_child_feature is None:
            best_left_child_feature = best_left_child_features[0]
            print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')

        if best_right_child_feature is None:
            best_right_child_feature = best_right_child_features[0]
            print('Could not find optimal depth 2 subtree which satisfies \"No Feature Reuse\" option')

    else:
        best_left_child_feature = best_left_child_features[0]
        best_right_child_feature = best_right_child_features[0]


    # Update the dictionaries which hold the tree structure
    left_child, right_child = tree.children(tree.root)

    leaf_predictions = {'00': best_class['00'][best_parent_feature][best_left_child_feature],
                        '01': best_class['01'][best_parent_feature][best_left_child_feature],
                        '10': best_class['10'][best_parent_feature][best_right_child_feature],
                        '11': best_class['11'][best_parent_feature][best_right_child_feature]}

    # leaf_nodes_map = {'00': tree.left_child(left_child),
    #                   '01': tree.right_child(left_child),
    #                   '10': tree.left_child(right_child),
    #                   '11': tree.right_child(right_child)}


    # Map from reduced feature set back to all features
    best_parent_feature = F_map_to_all[best_parent_feature]
    best_left_child_feature = F_map_to_all[best_left_child_feature]
    best_right_child_feature = F_map_to_all[best_right_child_feature]

    b = (best_parent_feature,best_left_child_feature,best_right_child_feature)
    w = [leaf_predictions[leaf] for leaf in leaves]
    theta = []


    for idx in I:
        sample = X[idx,:]
        if sample[best_parent_feature] > 0.5:
            # Sample goes right
            if sample[best_right_child_feature] > 0.5:
                if y[idx] == leaf_predictions['11']:
                    theta.append(idx)
            else:
                if y[idx] == leaf_predictions['10']:
                    theta.append(idx)
        else:
            # Sample goes left
            if sample[best_left_child_feature] > 0.5:
                if y[idx] == leaf_predictions['01']:
                    theta.append(idx)
            else:
                if y[idx] == leaf_predictions['00']:
                    theta.append(idx)

    return b, w, theta

def optimise_regularised_subtrees(X,y,tree,opts,branch_features,root_node,alpha,
                                  cache=None,weights=None,cat_feature_maps=None,num_feature_maps=None):
    """Optimises the tails of a given tree

    Function which runs solution polishing on a given decision tree. Works by finding nodes which are candidate subtree
    roots for optimisation, calling the D2S subroutine on each, and then returning the updated tree

    Args:
        X (ndarray): 2d array of size (n_samples,n_features) with binary features
        y (ndarray): 1d array of target classes. Classes assumed to be in {0,...,|K|-}
        tree (Tree):
        opts (set): Flags which can be set to modify behaviour. No options are current supported
        branch_features:
        root_node (Node): Root node which recursively defines a tree. Only used to find node heights
        alpha (float): Per leaf penalty on accuracy
        cache (dict): Cache for D2S subroutine. May be modified in place
        weights (list): Weighting of samples. Not currently supported
        cat_feature_maps: Depreciated feature
        num_feature_maps: Depreciated feature

    Returns:
        Returns a tuple (b,p,w,theta) where each element is a dictionary. Together these form a PARTIAL solution only for
        the subtrees which have been optimised. These can be merged into an existing solution to update only the optimised
        subtrees while leaving the upper levels of the tree untouched
    """


    # No Feature Reuse constraints not supported
    assert ('No Feature Reuse' not in opts)

    # TODO: Mixture of tree as recursive node structure and a class is awkward.
    #  Should try and settle on one or combine them in a nice way

    n_samples, n_features = X.shape
    F = range(n_features)
    K = np.unique(y).tolist()

    # Double check that the tree is large enough
    if tree.depth < 2:
        return None, None, None, None

    # By default all samples have equal weight
    if weights is None:
        weights = [1] * n_samples


    # The calculate_height method calculates the height recursively, populating node.height for the root node
    # and all other nodes in the tree as needed to detect the correct subtree roots
    root_node.calculate_height()

    subtree_roots = []
    to_explore = [root_node]

    # Explore the tree and add nodes to the list of subtree roots if the following holds:
    # The node.height <= 2 AND (node.n == 1 OR node.parent.height > 2)
    while len(to_explore) > 0:
        node = to_explore.pop()

        if node.height <= 2:
            if node.parent is None:
                subtree_roots.append(node)
            elif node.parent.height > 2:
                subtree_roots.append(node)

        # If node is a branch node add its children to the list of nodes to explore
        if node.node_type == 'branch':
            to_explore.append(node.left_child)
            to_explore.append(node.right_child)

    # Create b, p, w dictionaries for each subtree with a root in subtree_roots
    b = {}
    p = {}
    w = {}

    # Loop over the root of each subtree to be optimised, and set all decision variables to zero in the subtree to zero.
    # Afterwards when the D2S subroutine is called we can simply update the decision variables which are non-zero
    # in the optimal solution
    for root in subtree_roots:
        # At each subtree root we want the decision variables in the depth 2 subtree
        n = root.n
        left_child, right_child = tree.left_child(n), tree.right_child(n)
        candidate_descendants = [n,
                                 left_child, right_child,
                                 tree.left_child(left_child), tree.left_child(right_child), tree.right_child(left_child), tree.right_child(right_child)]

        for n in candidate_descendants:
            if n in tree.B:
                for f in F:
                    b[n,f] = 0

                for k in K:
                    w[k,n] = 0

                p[n] = 0

            # Do not set branch variables at terminal nodes
            if n in tree.L:
                for k in K:
                    w[k,n] = 0
                p[n] = 0

    theta = [0] * n_samples

    # Alpha in the original objective is a penalty against the accuracy of the model
    # Since the subroutine only has access to the data samples at the subtree root node, it cannot
    # operate on the accuracy of the whole dataset
    # As such we convert alpha to a penalty on the number of correctly classified samples
    alpha_subroutine = X.shape[0] * alpha

    for root in subtree_roots:
        run_subroutine = True
        I_root = root.I # Samples in the subtrees

        if cache is not None:
            assert isinstance(cache,dict)

            root_ancestors = tree.ancestors(root.n, branch_dirs=True)

            path_key = frozenset((branch_features[node], dir) for node, dir in root_ancestors.items())

            # If we have already seen this subtree, reuse the results
            if path_key in cache:
                b_subtree, w_subtree, theta_subtree = cache[path_key]
                run_subroutine = False

        # Only run if we did not find a solution in the cache
        if run_subroutine:

            invalid_features = {}

            soln = optimise_regularised_depth2_subtree(X[I_root,:], y[I_root],
                                                       weights=[weights[i] for i in I_root],
                                                       alpha=alpha_subroutine,
                                                       invalid_features=invalid_features)

            # Unpack the optimal subtree solution
            b_subtree, w_subtree, theta_subtree = soln

            # Cache results of subroutine
            if cache is not None:
                cache[path_key] = (b_subtree,
                                   w_subtree,
                                   theta_subtree)

        parent_feature, left_feature, right_feature = b_subtree

        # Get bfs ordering node numbers of the subtree
        node0, node1 = tree.left_child(root.n), tree.right_child(root.n)
        node00, node01 = tree.left_child(node0), tree.right_child(node0)
        node10, node11 = tree.left_child(node1), tree.right_child(node1)

        # One leaf solution
        if parent_feature is None:
            p[root.n] = 1
            w[w_subtree[0],root.n] = 1

        # Two or three leaf solutions
        elif left_feature is None or right_feature is None:
            b[root.n,parent_feature] = 1

            if left_feature is None and right_feature is None:
                # Two leaf solution
                p[node0], p[node1] = 1, 1

                w[w_subtree[0],node0] = 1
                w[w_subtree[1],node1] = 1

            if left_feature is not None and right_feature is None:
                # Three leaf solution where left child is a branch node and right child is a leaf node
                b[node0, left_feature] = 1

                p[node1], p[node00], p[node01] = 1, 1, 1

                w[w_subtree[0], node1] = 1
                w[w_subtree[1], node00] = 1
                w[w_subtree[2], node01] = 1

            if left_feature is None and right_feature is not None:
                # Three leaf solution where left child is a leaf node and right child is a branch node
                b[node1, right_feature] = 1

                p[node0], p[node10], p[node11] = 1, 1, 1

                w[w_subtree[0], node0] = 1
                w[w_subtree[1], node10] = 1
                w[w_subtree[2], node11] = 1

        # Four leaf solutions
        else:
            b[root.n, parent_feature] = 1
            b[node0, left_feature] = 1
            b[node1, right_feature] = 1

            p[node00], p[node01], p[node10], p[node11]  = 1, 1, 1, 1

            w[w_subtree[0], node00] = 1
            w[w_subtree[1], node01] = 1
            w[w_subtree[2], node10] = 1
            w[w_subtree[3], node11] = 1

        # Cached theta_idx holds indexes of root.I, not I.
        for idx in theta_subtree:
            theta[I_root[idx]] = 1

    return b,p,w,theta

def optimise_subtrees(X,y,samples_in_node,tree,opts,branch_features,
                      cache=None,weights=None,cat_feature_maps=None,num_feature_maps=None):

    n_samples, n_features = X.shape
    F_all = range(n_features)
    K = np.unique(y).tolist()

    # Double check that the tree is large enough
    if tree.depth < 2:
        return None, None, None

    # By default all samples have equal weight
    if weights is None:
        weights = [1] * n_samples

    # Tree layers stored with 0 based indexing with leaves at position -1
    subtree_roots = tree.layers[-3]

    b = {(n,f): 0
         for n in subtree_roots + tree.layers[-2] for f in F_all}
    w = {(k,n): 0
         for k in K for n in tree.L}
    theta = [0] * n_samples

    for root in subtree_roots:
        run_subroutine = True
        if cache is not None:
            assert isinstance(cache,dict)

            root_ancestors = tree.ancestors(root, branch_dirs=True)

            path_key = frozenset(str(branch_features[node]) + str(dir) for node, dir in root_ancestors.items())

            # If we have already seen this subtree, reuse the results
            if path_key in cache:
                b_subtree, w_subtree, theta_subtree = cache[path_key]
                run_subroutine = False

        if run_subroutine:
            invalid_features = set()

            if 'No Feature Reuse' in opts:
                if cat_feature_maps is None and num_feature_maps is None:
                    print('No Feature Reuse option enabled for primal heuristic feature but feature groupings not supplied')
                else:
                    bin_to_cat_group = {}
                    bin_to_num_group = {}
                    if cat_feature_maps is not None:
                        for Cf in cat_feature_maps:
                            CF_set = set(Cf)
                            for f in Cf:
                                bin_to_cat_group[f] = CF_set

                    if num_feature_maps is not None:
                        for Nf in num_feature_maps:
                            NF_set = set(Nf)
                            for f in Nf:
                                bin_to_num_group[f] = NF_set

                    for n_a, dir in tree.ancestors(root, branch_dirs=True).items():
                        ancestor_branch_feature = branch_features[n_a]
                        # Ancestor branches right
                        if dir == 1:
                            if ancestor_branch_feature in bin_to_cat_group:
                                invalid_grouping = bin_to_cat_group[ancestor_branch_feature]
                            elif ancestor_branch_feature in bin_to_num_group:
                                invalid_grouping = [f for f in bin_to_num_group[ancestor_branch_feature] if f <= ancestor_branch_feature]

                        elif dir == 0:
                            if ancestor_branch_feature in bin_to_num_group:
                                invalid_grouping = [f for f in bin_to_num_group[ancestor_branch_feature] if f >= ancestor_branch_feature]
                            else:
                                invalid_grouping = [ancestor_branch_feature]

                        for f in invalid_grouping:
                            invalid_features.add(f)

            else:
                bin_to_cat_group, bin_to_num_group = None, None

            # subtree = tree.subtree(root, root_at_n=True)
            I_root = samples_in_node[root]

            b_subtree, w_subtree , theta_subtree = optimise_depth2_subtree(X[I_root,:], y[I_root],
                                                                           weights=[weights[i] for i in I_root],
                                                                           invalid_features=invalid_features,
                                                                           bin_to_cat_group=bin_to_cat_group,
                                                                           bin_to_num_group=bin_to_num_group)

            # Cache results of subroutine
            if cache is not None:
                cache[path_key] = (b_subtree,
                                   w_subtree,
                                   theta_subtree)

        parent_feature, left_feature, right_feature = b_subtree

        # Update the dictionaries which hold the tree structure
        left_child, right_child = tree.children(root)

        b[root, parent_feature] = 1
        b[left_child, left_feature] = 1
        b[right_child, right_feature] = 1

        _, subtree_leaf_nodes = tree.descendants(root, split_nodes=True)
        for i, n in enumerate(subtree_leaf_nodes):
            w[w_subtree[i],n] = 1

        # Cached_theta_idx holds indexes of samples_in_node[root], not I.
        I_root = samples_in_node[root]
        for idx in theta_subtree:
            theta[I_root[idx]] = 1

    return b,w,theta

class Tree():
    def __init__(self,depth,root=1):
        self.depth = depth
        self.root = root

        self.B = list(range(1, 2 ** depth))
        self.L = list(range(2 ** depth, 2 ** (depth + 1)))
        self.T = self.B + self.L
        self.source = 0
        self.sink = 2**(depth + 1)
        self.layers = [list(range(2 ** d, 2 ** (d + 1))) for d in range(depth+1)]

    def subtree(self,n,root_at_n=False):
        # Returns a tree object rooted at node
        # By default sets the root node at n=1. Maybe if it's useful I can add an option to retain the node numbering

        # TODO: Do this properly by rewriting __init__. This should work fine for now (except for self.layers, source and sink)

        if root_at_n:
            new_subtree = Tree(self.height(n), root=n)
            new_subtree.B, new_subtree.L = self.descendants(n, split_nodes=True)
        else:
            return Tree(self.height(n))

    def left_child(self,n):
        if n in self.L:
            return None
        else:
            return 2*n

    def right_child(self,n):
        if n in self.L:
            return None
        else:
            return 2*n + 1

    def children(self,n):
        left_child = self.left_child(n)

        # Return empty list if leaf node (no children)
        if left_child is None:
            return []

        right_child = self.right_child(n)

        return [left_child, right_child]

    def parent(self,n):
        # Returns the parent node of a given branch node
        # Special case at root node which has the source node as a parent
        if n==1:
            return 0

        # If even number the node is a left child of the parent.
        # If odd number node is a right child
        if n%2 == 0:
            return n//2
        else:
            return (n-1)//2

    def sibling(self,n):
        """Finds the sibling of a node n

        Returns the sibling of node n (the other child of the parent of node n)
        If n is the root node then returns None

        Args:
            n: node number

        Returns:
            Returns n_sib which is None is n==1 otherwise returns the sibling of n
        """

        if n==1:
            n_sib = None

        else:
            parent_node = self.parent(n)

            if (n % 2 == 0):
                n_sib = self.right_child(parent_node)
            else:
                n_sib = self.left_child(parent_node)

        return n_sib

    # TODO: Rewrite with recursion?
    def ancestors(self,n,branch_dirs=False):

        # Stores a dict with ancestor nodes in the keys and the branch direction (0 = left, 1 = right) of the child as a value
        # E.g. if node n is a left child then node_ancestors[parent_node(n)] = 0
        node_ancestors = {}
        left_path = []
        right_path = []

        # Work up ancestors until root node
        while n > 1:
            # Even number -> left child of parent
            # Odd number -> right child of parent
            branch = 0 if (n % 2 == 0) else 1
            n = n // 2
            node_ancestors[n] = branch

        if branch_dirs:
            return node_ancestors
        else:
            return list(node_ancestors.keys())

    #TODO: Rewrite to make less awkward
    def descendants(self,n,split_nodes=False):
        # Returns the descendants of node n, inclusive of node n
        # By defaults return branch and left nodes in one list
        # If split_nodes=True, returns tuple of (branch_nodes, leaf_nodes)


        node_descendants = []
        stack = [n]

        leaf_start_idx = -1

        while len(stack) > 0:
            current_node = stack.pop(0)

            if split_nodes and leaf_start_idx == -1 and current_node >= self.L[0]:
                leaf_start_idx = len(node_descendants)

            node_descendants.append(current_node)

            left_child = self.left_child(current_node)

            # left_child = None implies that we have reach leaf nodes which have no descendants
            if left_child is None:
                continue

            right_child = self.right_child(current_node)

            stack.append(left_child)
            stack.append(right_child)
        if split_nodes:
            return node_descendants[:leaf_start_idx], node_descendants[leaf_start_idx:]
        else:
            return node_descendants

    def descendant_leaves(self,root):
        # Return the leafs which are descendant from the given node
        # If root is a leaf node then it returns just the leaf node

        subtree_depth = self.height(root)
        return list(range(root * 2**subtree_depth,root * 2**subtree_depth + 2**subtree_depth))

    def height(self,node):

        node_depth = 0
        n = node

        while n != 1:
            node_depth += 1
            n = n // 2

        node_height = self.depth - node_depth
        return node_height