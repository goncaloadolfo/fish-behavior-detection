import numpy as np


def remove_correlated_variables(samples, correlation_thr):
    # erase variables with high correlation value
    correlation_matrix = np.corrcoef(samples.T)
    variables_to_erase = []
    
    # go trough correlation values (only half of the matrix)
    for var1_index in range(samples.shape[1]):
        for var2_index in range(var1_index + 1, samples.shape[1]):
            # too high correlation and none of the variables was already erased
            if abs(correlation_matrix[var1_index, var2_index]) >= correlation_thr \
            and var1_index not in variables_to_erase and var2_index not in variables_to_erase:
                variables_to_erase.append(var1_index)
    
    # delete the columns of the correlated variables
    return np.delete(samples, variables_to_erase, axis=1)


def z_normalization(samples):
    # normalize the received data to z score 
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    # delete variables with 0 variance
    variable_indexes = np.where(stds == 0)
    new_samples = np.delete(samples, variable_indexes, axis=1)
    new_means = np.delete(means, variable_indexes)
    new_stds = np.delete(stds, variable_indexes)
        
    return (new_samples - new_means) / new_stds
    