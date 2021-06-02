from collections import defaultdict

import numpy as np

from anomaly_detection.switching_model.model_initialization import identify_position_node
from anomaly_detection.switching_model.model_expectation import calculate_transition_probability, complete_log_likelihood, forward_backward_procedure, normalize_vector


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

            nodes_variance_results[node_indexes][0] += smooth_prob * np.linalg.norm(
                real_motion_vector - node_motion_vector)**2
            nodes_variance_results[node_indexes][1] += smooth_prob

    return nodes_variance_results


def update_nodes_variance(grid, nodes_variance_results, field):
    for node_indexes in nodes_variance_results:
        x, y = node_indexes
        total_weight_variance = nodes_variance_results[node_indexes][0]
        smooth_probs_sum = nodes_variance_results[node_indexes][1]
        grid.grid_matrix[y][x].motion_vectors_variances[field] = total_weight_variance / smooth_probs_sum


def update_variances(grid, trajectories, smooth_probs_list):
    for field in range(grid.nr_fields):
        field_smooth_probs = [trajectory_smooth_probs[field]
                              for trajectory_smooth_probs in smooth_probs_list]
        field_results = calculate_field_variance(grid, trajectories,
                                                 field_smooth_probs, field)
        update_nodes_variance(grid, field_results, field)


def is_neighbor(node1_indexes, node2_indexes):
    x_difference = abs(node1_indexes[0] - node2_indexes[0])
    y_difference = abs(node1_indexes[1] - node2_indexes[1])
    return x_difference == 1 and y_difference == 0 or x_difference == 0 and y_difference == 1


def velocity_diff_matrix(grid, field):
    neighbor_diff_matrix = np.zeros((grid.nr_nodes * 2, grid.nr_nodes * 2))
    all_nodes = grid.grid_matrix.flatten()

    for i in range(len(all_nodes)):
        for j in range(len(all_nodes)):
            node1 = all_nodes[i]
            node2 = all_nodes[j]

            node1_indexes = identify_position_node(grid, node1.centroid)
            node2_indexes = identify_position_node(grid, node2.centroid)

            if is_neighbor(node1_indexes, node2_indexes):
                node1_vector = node1.motion_vectors[field]
                node2_vector = node2.motion_vectors[field]
                field_diff = node1_vector - node2_vector

                neighbor_diff_matrix[i*2][j*2] = field_diff[0]
                neighbor_diff_matrix[i*2+1][j*2+1] = field_diff[1]

    return neighbor_diff_matrix


def calculate_interpolation_coefs(delta, position, all_nodes):
    interpolation_nodes = []
    diff_areas = []

    for i in range(len(all_nodes)):
        node = all_nodes[i]
        x_diff = abs(position[0] - node.centroid[0])
        y_diff = abs(position[1] - node.centroid[1])

        if max(x_diff, y_diff) < delta:
            # coef = x_diff * y_diff
            # coef = coef if coef != 0 else 1
            coef = delta**-2 * x_diff * y_diff
            interpolation_nodes.append(i)
            diff_areas.append(coef)

    diff_areas = np.array(diff_areas)
    # interpolation_coefs = (1 / diff_areas) / np.sum(diff_areas)
    interpolation_coefs = np.array(diff_areas)
    return interpolation_nodes, interpolation_coefs


def calculate_interpolation_coefs2(delta, position, all_nodes):
    interpolation_nodes = []
    diff_areas = []

    for i in range(len(all_nodes)):
        node = all_nodes[i]
        x_diff = abs(position[0] - node.centroid[0])
        y_diff = abs(position[1] - node.centroid[1])

        if max(x_diff, y_diff) < delta:
            coef = x_diff * y_diff
            coef = coef if coef != 0 else 1
            interpolation_nodes.append(i)
            diff_areas.append(coef)

    diff_areas = np.array(diff_areas)
    interpolation_coefs = (1 / diff_areas) / np.sum(diff_areas)
    return interpolation_nodes, interpolation_coefs


def bilinear_interpolation_basis_function(delta, position, all_nodes):
    interpolation_nodes, interpolation_coefs = calculate_interpolation_coefs(
        delta, position, all_nodes)
    interpolation_coefs_matrix = np.zeros((2, 2*len(all_nodes)))

    for i in range(len(interpolation_nodes)):
        node_index = interpolation_nodes[i]
        coef = interpolation_coefs[i]
        interpolation_coefs_matrix[0][node_index*2] = coef
        interpolation_coefs_matrix[1][node_index*2+1] = coef

    return interpolation_coefs_matrix


def calculate_rks(grid, trajectories, field, smooth_probs_list, delta):
    nodes_rks = {}
    all_nodes = grid.grid_matrix.flatten()

    for i in range(len(trajectories)):
        trajectory = trajectories[i]
        trajectory_smooth_probs = smooth_probs_list[i]

        for j in range(1, len(trajectory)):
            previous_position = np.array(
                [trajectory[j-1][1], trajectory[j-1][2]], dtype=np.float)
            current_position = np.array(
                [trajectory[j][1], trajectory[j][2]], dtype=np.float)

            node_indexes = identify_position_node(grid, previous_position)
            node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]

            field_prob = trajectory_smooth_probs[j]
            motion_variance = node.motion_vectors_variances[field]
            interpolation_coefs = bilinear_interpolation_basis_function(
                delta, previous_position, all_nodes)

            Rk = field_prob/motion_variance * \
                np.dot(interpolation_coefs.T, interpolation_coefs)
            rk = field_prob/motion_variance * \
                np.dot(interpolation_coefs.T,
                       current_position - previous_position)

            if node_indexes in nodes_rks:
                nodes_rks[node_indexes][0] += Rk
                nodes_rks[node_indexes][1] += rk

            else:
                nodes_rks[node_indexes] = [Rk, rk]

    return nodes_rks


def calculate_vector_coefs(nodes_rks, neighbor_diff_matrix, alpha):
    nodes_vector_coefs = {}

    for node_indexes in nodes_rks:
        Rk = nodes_rks[node_indexes][0]
        rk = nodes_rks[node_indexes][1]
        aux = Rk + np.dot(neighbor_diff_matrix.T,
                          neighbor_diff_matrix) / alpha**2
        nodes_vector_coefs[node_indexes] = np.dot(np.linalg.pinv(aux), rk)

    return nodes_vector_coefs


def update_field_motion_vectors(grid, vector_coefs, field, delta):
    all_nodes = grid.grid_matrix.flatten()
    for node_indexes in vector_coefs:
        node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
        interpolation_coefs = bilinear_interpolation_basis_function(
            delta, node.centroid, all_nodes)
        node.motion_vectors[field] = np.dot(
            interpolation_coefs, vector_coefs[node_indexes])


def update_motion_vectors(grid, trajectories, trajectories_smooth_probs, delta, alpha):
    for field in range(grid.nr_fields):
        field_smooth_probs = [trajectory_probs[field]
                              for trajectory_probs in trajectories_smooth_probs]
        neighbor_matrix = velocity_diff_matrix(grid, field)

        nodes_rks = calculate_rks(
            grid, trajectories, field, field_smooth_probs, delta)
        nodes_vector_coefs = calculate_vector_coefs(
            nodes_rks, neighbor_matrix, alpha)
        update_field_motion_vectors(grid, nodes_vector_coefs, field, delta)


def calculate_matrix_gradients(trajectories, grid, smooth_probs, interpolation_node, all_nodes, delta):
    gradients = np.zeros((grid.nr_fields, grid.nr_fields))

    for field1 in range(grid.nr_fields):
        for field2 in range(grid.nr_fields):
            gradient = 0.0

            for i in range(len(trajectories)):
                trajectory = trajectories[i]
                trajectory_smooth_probs = smooth_probs[i]

                for j in range(1, len(trajectory)):
                    previous_position = np.array(
                        [trajectory[j][1], trajectory[j][2]], dtype=np.float)
                    node_indexes = identify_position_node(
                        grid, previous_position)
                    node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]

                    transition_prob = calculate_transition_probability(
                        trajectory_smooth_probs, j, field1, field2)
                    current_transition_prob = node.transition_matrix[field1][field2]
                    interpolation_nodes, interpolation_coefs = calculate_interpolation_coefs2(
                        delta, previous_position, all_nodes)
                    interpolation_coef = interpolation_coefs[interpolation_nodes.index(interpolation_node)] \
                        if interpolation_node in interpolation_nodes else 0.0
                    gradient += transition_prob * interpolation_coef / current_transition_prob

            gradients[field1][field2] = gradient

    return gradients


def update_transition_matrices(grid, trajectories, smooth_probs, delta):
    all_nodes = grid.grid_matrix.flatten()
    transition_matrices = []

    for i in range(len(all_nodes)):
        print(f"calculating gradients node {i+1}/{len(all_nodes)}...")
        gradients = calculate_matrix_gradients(
            trajectories, grid, smooth_probs, i, all_nodes, delta)
        transition_matrices.append(all_nodes[i].transition_matrix + gradients)

    for i in range(len(all_nodes)):
        node = all_nodes[i]
        new_matrix = np.zeros(node.transition_matrix.shape)
        for m in range(len(transition_matrices)):
            interpolation_nodes, interpolation_coefs = calculate_interpolation_coefs2(delta, node.centroid, all_nodes)
            interpolation_coef = interpolation_coefs[interpolation_nodes.index(m)] if m in interpolation_nodes else 0
            new_matrix += transition_matrices[m] * interpolation_coef
        
        for i in range(len(new_matrix)):
            new_matrix[i] = normalize_vector(new_matrix[i])
        node.transition_matrix = new_matrix


def fit_switching_model(grid, trajectories, delta, alpha, nr_iterations):
    complete_log_likelihoods = []
    
    for i in range(nr_iterations):
        smooth_probs = [forward_backward_procedure(grid, trajectory)
                    for trajectory in trajectories]
        
        print(f"\niteration {i+1}/{nr_iterations}")
        print("updating variances...")
        update_variances(grid, trajectories, smooth_probs)
        print("updating motion vectors...")
        update_motion_vectors(grid, trajectories, smooth_probs, delta, alpha)
        print("updating matrices...")
        update_transition_matrices(grid, trajectories, smooth_probs, delta)

        print("calculating likelihood...")
        iteration_clh = complete_log_likelihood(grid, trajectories)
        complete_log_likelihoods.append(iteration_clh)
        print("complete log likelihood: ", iteration_clh)

    return complete_log_likelihoods
