import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import (exponential_weights,
                                                 smooth_positions)
from sklearn.cluster import KMeans
from trajectory_reader.trajectories_reader import read_fishes


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
            transition_matrix = []
            vectors_variance = []
            for label, count in label_count:
                core_vectors.append(kmeans.cluster_centers_[label])
                initial_prob = count / total_node_vectors
                transition_matrix.append([initial_prob] * k)

                label_samples = node_vectors[node.clusters == label]
                vector_variance = np.var(label_samples, axis=0)
                vectors_variance.append([
                    [vector_variance[0], 0],
                    [0, vector_variance[1]]
                ])

            node.core_vectors = core_vectors
            node.transition_matrix = np.array(transition_matrix).T
            node.vectors_variance = np.array(vectors_variance)


def active_vectors(grid, trajectory):
    active_vectors = []
    for i in range(1, len(trajectory)):
        motion_vector = [trajectory[i][1] - trajectory[i-1][1],
                         trajectory[i][2] - trajectory[i-1][2]]
        for node in grid:
            if (trajectory[i][1], trajectory[i][2]) in node:
                active_vectors.append(
                    (node.node_id, most_likely_vector(node, motion_vector))
                )
                break

    return active_vectors


def most_likely_vector(node, motion_vector):
    diffs = np.array(motion_vector) - node.core_vectors
    norms = np.linalg.norm(diffs, axis=1)
    return np.argmin(norms)


def set_motion_vectors(fishes_file, grid):
    fishes = read_fishes(fishes_file)

    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish, False)
        smooth_positions(fish, exponential_weights(24, 0.01))
        save_motion_vectors(grid, fish.trajectory)


def save_motion_vectors(grid, trajectory):
    for i in range(1, len(trajectory)):
        motion_vector = [trajectory[i][1] - trajectory[i-1][1],
                         trajectory[i][2] - trajectory[i-1][2]]
        for node in grid:
            if (trajectory[i][1], trajectory[i][2]) in node:
                node.all_motion_vectors.append(motion_vector)
                break


def draw_motion_vectors(grid, frame_shape, k):
    frames = [np.full((frame_shape[0], frame_shape[1], 3), 255, dtype=np.uint8)
              for _ in range(k)]

    for node in grid:
        core_vectors = node.core_vectors

        if len(core_vectors) > 0:
            node_center = (int(node.x_limits[0] + (node.x_limits[1] - node.x_limits[0])/2),
                           int(node.y_limits[0] + (node.y_limits[1] - node.y_limits[0])/2))

            for i in range(k):
                final_position = (int(node_center[0] + core_vectors[i][0]*10),
                                  int(node_center[1] + core_vectors[i][1]*10))
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


class GridNode:

    def __init__(self, node_id, x_limits, y_limits):
        self.__node_id = node_id
        self.__x_limits = x_limits
        self.__y_limits = y_limits

        self.__all_motion_vectors = []
        self.__clusters = []
        self.__core_vectors = []
        self.__transition_matrix = None
        self.__vectors_variance = None

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
    def transition_matrix(self):
        return self.__transition_matrix

    @property
    def vectors_variance(self):
        return self.__vectors_variance

    @core_vectors.setter
    def core_vectors(self, value):
        self.__core_vectors = value

    @clusters.setter
    def clusters(self, value):
        self.__clusters = value

    @transition_matrix.setter
    def transition_matrix(self, value):
        self.__transition_matrix = value

    @vectors_variance.setter
    def vectors_variance(self, value):
        self.__vectors_variance = value

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
                (x_pace*i, x_pace*(i+1)),
                (y_pace*j, y_pace*(j+1))
            ))

    return nodes


if __name__ == "__main__":
    fishes = "resources/detections/v29-fishes.json"
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
