import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import (exponential_weights,
                                                 smooth_positions)
from trajectory_reader.trajectories_reader import read_fishes

N = 4


# region model initialization
def initialize_parameters(grid, k):
    kmeans = KMeans(n_clusters=k)

    for node in grid:
        node_vectors = np.array(node.all_motion_vectors)

        if len(node_vectors) > 0:
            kmeans.fit(np.array(node_vectors))
            node.clusters = kmeans.labels_

            labels, counts = np.unique(node.clusters, return_counts=True)
            total_node_vectors = np.sum(counts)
            label_count = list(zip(labels, counts))
            label_count.sort(key=lambda x: x[1], reverse=True)

            core_vectors = []
            initial_probs = []
            transition_matrix = []
            vectors_variance = []
            for label, count in label_count:
                core_vectors.append(kmeans.cluster_centers_[label])
                initial_prob = count / total_node_vectors
                initial_probs.append(initial_prob)
                transition_matrix.append([initial_prob] * k)

                label_samples = node_vectors[node.clusters == label]
                vector_variance = np.var(label_samples, axis=0)
                vectors_variance.append([
                    [vector_variance[0], 0],
                    [0, vector_variance[1]]
                ])

            node.core_vectors = core_vectors
            node.initial_probs = np.array(initial_probs)
            node.transition_matrix = np.array(transition_matrix).T
            node.vectors_variance = np.array(vectors_variance)


def set_motion_vectors(fishes_file, grid):
    fishes = read_fishes(fishes_file)

    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish, False)
        smooth_positions(fish, exponential_weights(24, 0.01))
        save_motion_vectors(grid, fish.trajectory)


def save_motion_vectors(grid, trajectory):
    for i in range(1, len(trajectory)):
        motion_vector = [trajectory[i][1] - trajectory[i - 1][1],
                         trajectory[i][2] - trajectory[i - 1][2]]
        for node in grid:
            if (trajectory[i][1], trajectory[i][2]) in node:
                node.all_motion_vectors.append(motion_vector)
                break


class GridNode:

    def __init__(self, node_id, x_limits, y_limits):
        self.__node_id = node_id
        self.__x_limits = x_limits
        self.__y_limits = y_limits

        self.__all_motion_vectors = []
        self.__clusters = []

        self.__core_vectors = []
        self.__initial_probs = []
        self.__transition_matrix = None
        self.__vectors_variance = None

        self.__Rks = []
        self.__rks = []

    @property
    def node_id(self):
        return self.__node_id

    @property
    def x_limits(self):
        return self.__x_limits

    @property
    def y_limits(self):
        return self.__y_limits

    @property
    def all_motion_vectors(self):
        return self.__all_motion_vectors

    @property
    def clusters(self):
        return self.__clusters

    @property
    def core_vectors(self):
        return self.__core_vectors

    @property
    def initial_probs(self):
        return self.__initial_probs

    @property
    def transition_matrix(self):
        return self.__transition_matrix

    @property
    def vectors_variance(self):
        return self.__vectors_variance

    @property
    def Rks(self):
        return self.__Rks

    @property
    def rks(self):
        return self.__rks

    @core_vectors.setter
    def core_vectors(self, value):
        self.__core_vectors = value

    @initial_probs.setter
    def initial_probs(self, value):
        self.__initial_probs = value

    @clusters.setter
    def clusters(self, value):
        self.__clusters = value

    @transition_matrix.setter
    def transition_matrix(self, value):
        self.__transition_matrix = value

    @vectors_variance.setter
    def vectors_variance(self, value):
        self.__vectors_variance = value

    @Rks.setter
    def Rks(self, value):
        self.__Rks = value

    @rks.setter
    def rks(self, value):
        self.__rks = value

    def __contains__(self, value):
        return self.__x_limits[0] <= value[0] <= self.__x_limits[1] \
               and self.__y_limits[0] <= value[1] <= self.__y_limits[1]

    def __str__(self):
        result = ""
        result += f"X Limits: {self.__x_limits}\n"
        result += f"Y Limits: {self.__y_limits}\n"
        result += f"Core Vectors:\n{self.__core_vectors}\n"
        result += f"Transition Matrix:\n{self.__transition_matrix}\n"
        result += f"Vectors Variance:\n{self.__vectors_variance}"
        return result


def create_grid(nr_nodes, frame_shape):
    n = int(nr_nodes ** 0.5)
    y_pace = frame_shape[0] / n
    x_pace = frame_shape[1] / n

    nodes = []
    for i in range(n):
        for j in range(n):
            nodes.append(GridNode(
                i + j,
                (x_pace * i, x_pace * (i + 1)),
                (y_pace * j, y_pace * (j + 1))
            ))

    return nodes


# endregion


# region expectation phase
def trajectory_expectation(grid, trajectory):
    active_vectors = get_active_vectors(grid, trajectory)
    k = len(active_vectors[0][0].core_vectors)

    f_probs = forward_probs(active_vectors, k)
    b_probs = backward_probs(active_vectors, k)
    smoothed_probs = f_probs * b_probs

    vectors_sums = np.array([np.sum(smoothed_probs, axis=1)]).T
    return smoothed_probs / vectors_sums


def forward_probs(active_vectors, k):
    fprobs_t = np.empty((1 + len(active_vectors), k))
    fprobs_t[0] = active_vectors[0][0].initial_probs

    for i in range(len(active_vectors)):
        if i != 0 and active_vectors[i - 1][0].node_id != active_vectors[i][0].node_id:
            node_initial_probs = active_vectors[i][0].initial_probs
            t_probs = node_initial_probs * fprobs_t[i]

        else:
            t_probs = np.dot(active_vectors[i][0].transition_matrix, fprobs_t[i:np.newaxis].T)

        normalized_probs = t_probs / np.sum(t_probs)
        fprobs_t[i + 1] = normalized_probs

    return fprobs_t[1:]


def backward_probs(active_vectors, k):
    bprobs_t = np.empty((1 + len(active_vectors), k))
    bprobs_t[-1] = np.array([1.0] * k)

    for i in range(len(active_vectors) - 1, -1, -1):
        if i != len(active_vectors) - 1 and active_vectors[i][0].node_id != active_vectors[i + 1][0].node_id:
            node_initial_probs = active_vectors[i][0].initial_probs
            t_probs = node_initial_probs * bprobs_t[i + 1]

        else:
            t_probs = np.dot(active_vectors[i][0].transition_matrix, bprobs_t[i + 1:np.newaxis].T)

        normalized_probs = t_probs / np.sum(t_probs)
        bprobs_t[i] = normalized_probs

    return bprobs_t[:-1]


def get_active_vectors(grid, trajectory):
    active_vectors = []
    for i in range(1, len(trajectory)):
        motion_vector = [trajectory[i][1] - trajectory[i - 1][1],
                         trajectory[i][2] - trajectory[i - 1][2]]
        for node in grid:
            if (trajectory[i][1], trajectory[i][2]) in node:
                active_vectors.append(
                    (node, most_likely_vector(node, motion_vector))
                )
                break

    return active_vectors


def most_likely_vector(node, motion_vector):
    diffs = np.array(motion_vector) - node.core_vectors
    norms = np.linalg.norm(diffs, axis=1)
    return np.argmin(norms)


# endregion

# region maximization phase
def update_variances(grid, trajectories, smooth_probs):
    nr_vectors = smooth_probs[0].shape[1]

    for node in grid:
        for k in nr_vectors:
            numerator = 0
            denominator = 0

            for i in range(len(trajectories)):
                current_trajectory = trajectories[i]
                trajectory_probs = smooth_probs[i]

                for j in range(1, len(current_trajectory)):
                    last_pos = np.array([current_trajectory[j - 1][1], current_trajectory[j - 1][2]])
                    current_pos = np.array([current_trajectory[j][1], current_trajectory[j][2]])

                    if current_pos not in node:
                        continue

                    motion_diff_norm = np.linalg.norm(current_pos - last_pos - node.core_vectors[k])
                    numerator += trajectory_probs[j][k] * motion_diff_norm ** 2
                    denominator += trajectory_probs[j][k]

            if denominator != 0:
                node.vectors_variance[k][k] = numerator / denominator


def update_vectors(grid, trajectories, smooth_probs, alpha):
    nr_vectors = smooth_probs[0].shape[1]
    calculate_rs(grid, trajectories, smooth_probs)

    for node in grid:
        node_center = ((node.x_limits[0] + node.x_limits[1]) / 2,
                       (node.y_limits[0] + node.y_limits[1]) / 2)
        vel_diff = calculate_neighbors_velocity_diffs(grid, node, nr_vectors)
        aux = np.dot(vel_diff.T, vel_diff) / alpha ** 2

        for k in nr_vectors:
            tk = node.rks[k] / (node.Rks + aux)
            # idk what x to pass to the basis functions
            node.core_vectors[k] = np.dot(ts_apply_basis_functions(node_center), tk)


def update_transition_matrices(grid, trajectories, smooth_probs, alpha):
    # todo
    raise NotImplementedError()


def calculate_rs(grid, trajectories, smooth_probs):
    nr_vectors = smooth_probs[0].shape[1]

    for node in grid:
        Rks_list = []
        rks_list = []

        for k in nr_vectors:
            Rks = np.zeros((2 * N, 2 * N))
            rks = np.zeros((2 * N, 1))

            for i in range(len(trajectories)):
                current_trajectory = trajectories[i]
                trajectory_probs = smooth_probs[i]

                for j in range(1, len(current_trajectory)):
                    last_pos = np.array([current_trajectory[j - 1][1], current_trajectory[j - 1][2]])
                    current_pos = np.array([current_trajectory[j][1], current_trajectory[j][2]])

                    if current_pos not in node:
                        continue

                    aux = ts_apply_basis_functions(last_pos)
                    Rks += trajectory_probs[j][k] / node.vectors_variance[k][k] * np.dot(aux.T, aux)
                    rks += trajectory_probs[j][k] / node.vectors_variance[k][k] * np.dot(aux.T, current_pos - last_pos)

            Rks_list.append(Rks)
            rks_list.append(rks)

        node.Rks = np.array(Rks_list)
        node.rks = np.array(rks_list)


def ts_apply_basis_functions(position):
    # todo
    raise NotImplementedError()


def b_spline_basic_functions(position):
    # todo
    raise NotImplementedError()


def calculate_neighbors_velocity_diffs(grid, node, k):
    neighbors = get_neighbors_nodes(grid, node.node_id)
    diffs_matrix = np.empty((len(neighbors), 2))
    for i in range(len(neighbors)):
        diffs_matrix[i] = node.core_vectors[k] - neighbors[i].core_vectors[k]
    return diffs_matrix


def get_neighbors_nodes(grid, node_id):
    length = len(grid) ** 0.5
    neighbors_ids = [node_id - length, node_id - 1, node_id + 1, node_id + length]
    return [node for node in grid if node.node_id in neighbors_ids]


# endregion


# region visualization
def draw_motion_vectors(grid, frame_shape, k):
    frames = [np.full((frame_shape[0], frame_shape[1], 3), 255, dtype=np.uint8)
              for _ in range(k)]

    for node in grid:
        core_vectors = node.core_vectors

        if len(core_vectors) > 0:
            node_center = (int(node.x_limits[0] + (node.x_limits[1] - node.x_limits[0]) / 2),
                           int(node.y_limits[0] + (node.y_limits[1] - node.y_limits[0]) / 2))

            for i in range(k):
                final_position = (int(node_center[0] + core_vectors[i][0] * 10),
                                  int(node_center[1] + core_vectors[i][1] * 10))
                cv2.arrowedLine(frames[i], node_center, final_position,
                                (0, 255, 0), thickness=2)

    return frames


def visualize_clusters(gridnode):
    all_vectors = gridnode.all_motion_vectors
    clusters = gridnode.clusters
    core_vectors = gridnode.core_vectors
    colors = generate_colors(len(core_vectors), clusters)

    plt.figure()
    plt.title(
        f"Motion Vectors of Node x({gridnode.x_limits[0]:.0f}, {gridnode.x_limits[1]:.0f})" +
        f" and y({gridnode.y_limits[0]:.0f}, {gridnode.y_limits[1]:.0f})"
    )
    plt.xlabel("x")
    plt.ylabel("y")

    xs = [vector[0] for vector in all_vectors]
    ys = [vector[1] for vector in all_vectors]
    plt.scatter(xs, ys, c=colors)

    core_xs = [core_vector[0] for core_vector in core_vectors]
    core_ys = [core_vector[1] for core_vector in core_vectors]
    plt.scatter(core_xs, core_ys, c="red")
    plt.grid()


def generate_colors(k, clusters):
    colors = np.random.random(size=(k, 3))
    return colors[clusters.astype(int)]


# endregion


def main():
    fishes = "../resources/detections/v29-fishes.json"
    k = 4
    frame_shape = (480, 720)
    grid = create_grid(49, frame_shape)

    set_motion_vectors(fishes, grid)
    initialize_parameters(grid, k)

    vector_frames = draw_motion_vectors(grid, frame_shape, k)
    for i in range(k):
        cv2.imshow(f"motion frame {i}", vector_frames[i])

    example_node = random.choice(grid)
    visualize_clusters(example_node)
    print(example_node)

    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
