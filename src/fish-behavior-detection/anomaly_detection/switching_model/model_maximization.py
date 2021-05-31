from collections import defaultdict

import numpy as np

from anomaly_detection.switching_model.model_initialization import identify_position_node


def calculate_field_variance(grid, trajectories, smooth_probs, field):
    nodes_variance_results = defaultdict(lambda: [0.0, 0.0])

    for i in range(len(trajectories)):
        current_trajectory = trajectories[i]
        for j in range(1, len(current_trajectory)):
            previous_position = np.array(
                [current_trajectory[j-1][1],
                 current_trajectory[j-1][2]], dtype=np.float
            )
            current_position = np.array(
                [current_trajectory[j][1],
                 current_trajectory[j][2]], dtype=np.float
            )

            node_indexes = identify_position_node(grid, previous_position)
            node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]

            real_motion_vector = current_position - previous_position
            node_motion_vector = node.motion_vectors[field]
            smooth_prob = smooth_probs[i][j]

            nodes_variance_results[node_indexes][0] += np.linalg.norm(
                real_motion_vector - node_motion_vector)**2
            nodes_variance_results[node_indexes][1] += smooth_prob


def update_nodes_variance(nodes_variance_results):
    for node_indexes in nodes_variance_results:
        x, y = node_indexes
        total_weight_variance = nodes_variance_results[node_indexes][0]
        smooth_probs_sum = nodes_variance_results[node_indexes][1]
        grid.grid_matrix[y][x].motion_vectors_variance[field] = total_weight_variance / smooth_probs_sum


def update_variances(grid, trajectories, smooth_probs_list):
    for field in range(grid.nr_fields):
        field_smooth_probs = [trajectory_smooth_probs[field]
                              for trajectory_smooth_probs in smooth_probs_list]
        field_results = calculate_field_variance(grid, trajectories,
                                                 field_smooth_probs, field)
        update_nodes_variance(field_results)


def is_neighbor(node1_indexes, node2_indexes):
    x_difference = abs(node1_indexes[0] - node2_indexes[0])
    y_difference = abs(node1_indexes[1] - node2_indexes[1])
    return x_difference == 1 and y_difference == 0 or x_difference == 0 and y_difference == 1


def velocity_diff_matrix(grid, field):
    # todo
    raise NotImplementedError()
