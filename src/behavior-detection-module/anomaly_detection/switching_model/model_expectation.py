import random

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt
import pre_processing.pre_processing_functions as ppm
from anomaly_detection.switching_model.model_initialization import create_grid, identify_position_node


def active_vector(grid_node, position_t1, position_t2):
    motion_vector = position_t2 - position_t1
    differences = grid_node.motion_vectors - motion_vector
    norms = np.linalg.norm(differences, axis=1)
    return np.argmin(norms)


def position_normal_prob(grid_node, active_vector, position_t1, position_t2):
    mean = position_t1 + grid_node.motion_vectors[active_vector]
    std = grid_node.motion_vectors_variances[active_vector] * np.identity(2)
    return multivariate_normal.pdf(position_t2, mean=mean, cov=std, allow_singular=True)


def joint_probability(grid, trajectory):
    position_t0 = np.array([trajectory[0][1], trajectory[0][2]],
                           dtype=np.float)
    position_t1 = np.array([trajectory[1][1], trajectory[1][2]],
                           dtype=np.float)

    node_indexes = identify_position_node(grid, position_t0)
    node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]

    previous_field = active_vector(node, position_t0, position_t1)
    prob = node.motion_vectors_priors[previous_field] 
    step = 1 if len(trajectory) < 50 else int(len(trajectory) / 50)
    
    for i in range(2, len(trajectory), step):
        previous_position = np.array([trajectory[i-1][1], trajectory[i-1][2]],
                                     dtype=np.float)
        current_position = np.array([trajectory[i][1], trajectory[i][2]],
                                    dtype=np.float)

        node_indexes = identify_position_node(grid, previous_position)
        node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
        current_field = active_vector(
            node, previous_position, current_position)

        position_prob = position_normal_prob(node, current_field,
                                     previous_position, current_position)
        transition_prob = node.transition_matrix[previous_field][current_field]
        
        if position_prob != 0.0 and transition_prob != 0.0:
            prob *= position_prob
            prob *= transition_prob
        else:
            prob *= 0.01
        previous_field = current_field

    return prob


def complete_log_likelihood(grid, trajectories):
    complete_likelihood = 0

    for trajectory in trajectories:
        complete_likelihood += np.log(joint_probability(grid, trajectory))

    return complete_likelihood


def normalize_vector(vector):
    return vector / np.sum(vector)


def calculate_forward_probabilities(grid, trajectory):
    forward_probabilities = np.empty((grid.nr_fields, len(trajectory) + 1))
    forward_probabilities[:, 0] = [1 / grid.nr_fields] * grid.nr_fields

    for i in range(len(trajectory)):
        t, x, y = trajectory[i]
        node_indexes = identify_position_node(grid, (x, y))
        node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
        node_transition_matrix = node.transition_matrix

        previous_probs = forward_probabilities[:, i].flatten()
        aux = np.dot(previous_probs, node_transition_matrix)
        current_probabilities = normalize_vector(aux)
        forward_probabilities[:, i+1] = current_probabilities

    return forward_probabilities[:, 1:]


def calculate_backward_probabilities(grid, trajectory):
    backward_probabilities = np.empty((grid.nr_fields, len(trajectory) + 1))
    backward_probabilities[:, -1] = [1.0] * grid.nr_fields

    for i in range(len(trajectory) - 1, -1, -1):
        t, x, y = trajectory[i]
        node_indexes = identify_position_node(grid, (x, y))
        node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
        node_transition_matrix = node.transition_matrix

        previous_probs = backward_probabilities[:, i + 1].flatten()
        aux = np.dot(previous_probs, node_transition_matrix)
        current_probabilities = normalize_vector(aux)
        backward_probabilities[:, i] = current_probabilities

    return backward_probabilities[:, :-1]


def forward_backward_procedure(grid, trajectory):
    forward_probs = calculate_forward_probabilities(grid, trajectory)
    backward_probs = calculate_backward_probabilities(grid, trajectory)
    smooth_probs = forward_probs * backward_probs

    vectors_sums = np.sum(smooth_probs, axis=0)
    return smooth_probs / vectors_sums


def calculate_transition_probability(probs_fields_t, t, field1, field2):
    prob_field1 = probs_fields_t[field1][t-1]
    prob_field2 = probs_fields_t[field2][t]
    return prob_field1 * prob_field2


def test_joint_probability(video_path, fishes_path, species_path, nr_nodes, nr_fields):
    grid, trajectories = create_grid(video_path, fishes_path, species_path,
                                     ("shark", "manta-ray"), nr_nodes, nr_fields, True)
    joint_probs = [joint_probability(grid, trajectory)
                   for trajectory in trajectories]
    plt.figure()
    plt.title("Joint Probabilities")
    plt.xlabel("counting")
    plt.ylabel("joint probability")
    plt.hist(joint_probs)
    plt.show()


def test_forward_backward_procedure(video_path, fishes_path, species_path, nr_nodes, nr_fields):
    grid, trajectories = create_grid(video_path, fishes_path, species_path,
                                     ("shark", "manta-ray"), nr_nodes, nr_fields, True)
    random_trajectory = random.choice(trajectories)
    smooth_probs = forward_backward_procedure(grid, random_trajectory)

    for i in range(len(smooth_probs)):
        plt.figure()
        plt.title(f"Smooth Probabilities on field {i}")
        plt.xlabel("t")
        plt.ylabel("smooth probability")
        plt.plot(smooth_probs[i])
    plt.show()
    

def main():
    # tests
    test_joint_probability("resources/videos/v29.m4v",
                           "resources/detections/v29-fishes.json",
                           "resources/classification/species-gt-v29.csv",
                           36, 4)
    
    # test_forward_backward_procedure("resources/videos/v29.m4v",
    #                                 "resources/detections/v29-fishes.json",
    #                                 "resources/classification/species-gt-v29.csv",
    #                                 36, 4)


if __name__ == "__main__":
    main()
