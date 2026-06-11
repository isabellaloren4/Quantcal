import numpy as np
# Median of the CC estimates for each proportion
def median_cc_estimates_per_proportion(data0: dict, data1: dict, true_prevalences):
    '''Computes the median of the CC estimates for each proportion.

    Parameters:
    ---
    - data0: Dictionary with the CC estimates for class 0
    - data1: Dictionary with the CC estimates for class 1
    - true_prevalences: List of the true proportions

    Returns:
    --
    - median_estimates: Array with the computed median CC estimates
    '''
    median = {}
    for key in data0.keys():
        median_0 = np.median(data0[key])
        median_1 = np.median(data1[key])
        median[key] = [median_0, median_1]
        # print(f'Median of proportion {key}: {median_0}, {median_1}')
    true_prevalences = np.array(true_prevalences)
    median_estimates = []

    for proportion in true_prevalences:
        for (key, value) in median.items():
            if tuple(proportion) == key:
                prev_median = value
                median_estimates.append(prev_median)

    return np.array(median_estimates)