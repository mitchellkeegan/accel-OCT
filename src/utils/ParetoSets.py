import numpy as np
import numba

"""
These algorithms are from the paretoset library developed by Tommy Odland and available here: https://github.com/tommyod/paretoset
"""

@numba.jit(nopython=True, cache=True)
def any_equal_jitted(costs, cost):
    """Check if any are equal over axis 1."""

    rows, cols = costs.shape

    for i in range(rows):
        equal = True  # Assume equality
        for j in range(cols):
            if costs[i, j] != cost[j]:
                equal = False
                break  # Break out early here

        # This row in `costs` was equal to `cost`, return immediately
        if equal:
            return True

    return False


@numba.jit(nopython=True, fastmath=True, cache=True)
def dominates(a, b, length):
    """Does a dominate b?"""
    better = False
    for i in range(length):
        a_i, b_i = a[i], b[i]

        # Worse in one dimension -> does not dominate
        # This is faster than computing `at least as good` in every dimension
        if a_i > b_i:
            return False

        # Better in at least one dimension
        if a_i < b_i:
            better = True
    return better


@numba.jit(nopython=True, fastmath=True, cache=True)
def window_dominates_cost(window, cost, window_rows, window_cols):
    for i in range(window_rows):
        if dominates(window[i], cost, window_cols):
            return i
    return -1


@numba.jit(nopython=True, fastmath=True, cache=True)
def cost_dominates_window(window, cost, window_rows, window_cols):
    dominated_inds = []
    for i in range(window_rows):
        if dominates(cost, window[i], window_cols):
            dominated_inds.append(i)
    return dominated_inds


@numba.jit(nopython=True, cache=True)
def BNL(costs, distinct=True):
    """
    Block nested loops algorithm.
    """

    is_efficient = np.arange(costs.shape[0])
    n_costs, n_objectives = costs.shape
    num_efficient = 1  # Always put the first row in the window

    window_changed = True

    for i in range(1, n_costs):  # Skip the first row, since it's in the window
        # Get the cost for this row
        this_cost = costs[i]

        # If the window indices changed in the last iteration, get window again
        if window_changed:
            window = costs[is_efficient[:num_efficient]]
            window_rows, window_cols = window.shape
            window_changed = False

        # CASE 1 : DOES ANYTHING IN THE WINDOW DOMINATE THIS COST?
        # --------------------------------------------------------

        dom_index = window_dominates_cost(window, this_cost, window_rows, window_cols)
        # `dom_index` is the index of the first window element that dominates
        # the cost. If no window elements dominate the cost, -1 is returned.
        if dom_index >= 0:
            continue  # Window dominates cost, move on.

        # CASE 2 : DOES THIS COST DOMINATE ANYTHING IN THE WINDOW?
        # --------------------------------------------------------

        # Check if anything in the window is dominated by the point in question
        dominated_inds_window = cost_dominates_window(window, this_cost, window_rows, window_cols)
        # A point in the window is dominated, remove it
        if len(dominated_inds_window) != 0:
            # Get the original indices to remove
            to_removes = [is_efficient[k] for k in dominated_inds_window]
            for to_remove in to_removes:
                # Original indices of elements in the window
                for j, efficient in enumerate(is_efficient):
                    # Found a match, remove it
                    if efficient == to_remove:
                        # Move one to the left and decrement
                        is_efficient[j:num_efficient] = is_efficient[j + 1 : num_efficient + 1]
                        num_efficient -= 1
                        break  # Break out here

        # CASE 3 : ADD THE NEW COST TO THE WINDOW
        # ---------------------------------------

        # Add to window in all cases, except if `distinct` and it's already in the window
        if (not distinct) or (not any_equal_jitted(window, this_cost)):
            # Insert the index value of the point in the last position
            is_efficient[num_efficient] = i
            # Increment the number of efficient points
            num_efficient += 1
            window_changed = True

    bools = np.zeros(costs.shape[0], dtype=np.bool_)
    bools[is_efficient[:num_efficient]] = 1
    return bools

@numba.jit(nopython=True, cache=True)
def GetParetoSets(costs, max_senses=np.arange(0)):

    costs_copy = costs.copy()

    for col in max_senses:
        costs_copy[:,col] = -costs_copy[:,col]

    return BNL(costs_copy)

