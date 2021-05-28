import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import cv2


def init_transition_matrix(nr_fields):
    uniform_prob = 1 / nr_fields
    return np.full((nr_fields, nr_fields), uniform_prob)


def init_covariance_matrix(nr_fields):
    return np.zeros((nr_fields, nr_fields))


class GridNode:

    def __init__(self, x_limits, y_limits, nr_fields):
        self.__centroid = ((x_limits[1] + x_limits[0]) / 2,
                           (y_limits[1] + y_limits[0]) / 2)
        self.__x_limits = x_limits
        self.__y_limits = y_limits
        self.__motion_vectors = np.empty((nr_fields, 2))
        self.__transition_matrix = init_transition_matrix(nr_fields)
        self.__covariance_matrix = init_covariance_matrix(nr_fields)

    @property
    def centroid(self):
        return self.__centroid

    @property
    def x_limits(self):
        return self.__x_limits

    @property
    def y_limits(self):
        return self.__y_limits

    @property
    def motion_vectors(self):
        return self.__motion_vectors

    @property
    def transition_matrix(self):
        return self.__transition_matrix

    @property
    def covariance_matrix(self):
        return self.__covariance_matrix


def create_grid_matrix(width, height, side, nr_fields):
    x_bins = np.linspace(0, width, side + 1)
    y_bins = np.linspace(0, height, side + 1)

    grid = np.empty((side, side))
    for i in range(side):
        for j in range(side):
            x_limits = (x_bins[i], x_bins[i+1])
            y_limits = (y_bins[j], y_bins[j+1])
            grid[i][j] = GridNode(x_limits, y_limits, nr_fields)

    return grid


class SquareGrid:

    def __init__(self, width, height, nr_nodes, nr_fields):
        self.__width = width
        self.__height = height
        self.__side = nr_nodes ** 0.5
        self.__x_interval = width / self.__side
        self.__y_interval = height / self.__side
        self.__grid_matrix = create_grid_matrix(width, height,
                                                self.__side, nr_fields)
        self.__nr_fields = nr_fields

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def side(self):
        return self.__side

    @property
    def x_interval(self):
        return self.__x_interval

    @property
    def y_interval(self):
        return self.__y_interval

    @property
    def grid_matrix(self):
        return self.__grid

    @property
    def nr_fields(self):
        return self.__nr_fields


def identify_position_node(grid, pos):
    x_index = pos[0] / grid.x_interval
    y_index = pos[1] / grid.y_interval
    return (x_index, y_index)


def collect_motion_vectors(grid, trajectories):
    motion_vectors = defaultdict(list)

    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            current_position = np.array([trajectory[i][1], trajectory[i][2]])
            next_position = np.array([trajectory[i+1][1], trajectory[i+1][2]])
            motion_vector = next_position - current_position

            node_indexes = identify_position_node(grid, next_position)
            motion_vectors[node_indexes].append(motion_vector)

    return motion_vectors


def initialize_fields(grid, trajectories):
    kmeans_instance = KMeans(grid.nr_fields)
    motion_vectors = collect_motion_vectors(grid, trajectories)

    for node_indexes, node_vectors in motion_vectors.items():
        kmeans_instance.fit(np.array(node_vectors))
        grid[node_indexes[1]][node_indexes[0]
                              ].motion_vectors = kmeans_instance.cluster_centers_


def draw_grid(frame, grid):
    for i in range(grid.side):
        for j in range(grid.side):
            node = grid.grid_matrix[i][j]
            cv2.line(frame, (node.x_limits[1], node.y_limits[0]),
                     (node.x_limits[1], node.y_limits[1]), (0, 255, 0), 3)
            cv2.line(frame, (node.x_limits[0], node.y_limits[1]),
                     (node.x_limits[1], node.y_limits[1]), (0, 255, 0), 3)

            if i == 0:
                cv2.line(frame, (node.x_limits[0], node.y_limits[0]),
                         (node.x_limits[1], node.y_limits[0]), (0, 255, 0), 3)

            if j == 0:
                cv2.line(frame, (node.x_limits[0], node.y_limits[0]),
                         (node.x_limits[0], node.y_limits[1]), (0, 255, 0), 3)


def visualize_fields(grid):
    # todo
    raise NotImplementedError()


def pretty_print_parameters(grid):
    # todo
    raise NotImplementedError()


def main():
    # todo
    raise NotImplementedError()


if __name__ == "__main__":
    main()
