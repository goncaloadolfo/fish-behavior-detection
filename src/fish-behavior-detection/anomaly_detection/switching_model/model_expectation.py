import numpy as np
from scipy.stats import multivariate_normal

from anomaly_detection.switching_model.model_initialization import identify_position_node


def active_vector(grid_node, position_t1, position_t2):
    motion_vector = position_t2 - position_t1
    differences = grid_node.motion_vectors - motion_vector
    norms = np.linalg.norm(differences, axis=1)
    return np.argmin(norms)


def position_normal_prob(grid_node, active_vector, position_t1, position_t2):
    mean = position_t1 + grid_node.motion_vectors[active_vector]
    std = grid_node.motion_vectors_variances[active_vector] * np.identity(2)
    return multivariate_normal.pdf(position_t2, mean=mean, cov=std)


def joint_probability(grid, trajectory):
    position_t0 = np.array([trajectory[0][1], [trajectory[0][2]]], dtype=np.float)
    position_t1 = np.array([trajectory[1][1], [trajectory[1][2]]], dtype=np.float)
    
    node_indexes = identify_position_node(grid, position_t0)
    node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
    
    previous_field = active_vector(node, position_t0, position_t1)
    prob = node.motion_vectors_priors[previous_field]
    
    for i in range(2, len(trajectory)):
        previous_position = np.array([trajectory[i-1][1], [trajectory[i-1][2]]], dtype=np.float)
        current_position = np.array([trajectory[i][1], [trajectory[i][2]]], dtype=np.float)
        
        node_indexes = identify_position_node(grid, previous_position)
        node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
        current_field = active_vector(node, previous_position, current_position)
        
        prob *= position_normal_prob(node, current_field, previous_position, current_position)
        prob *= node.transition_matrix[previous_field][current_field]
        previous_field = current_field
        
    return prob


def complete_log_likelihood(grid, trajectories):
    complete_likelihood = 0
    
    for trajectory in trajectories:
        complete_likelihood += np.log(joint_probability(grid, trajectory))
    
    return complete_likelihood
