from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

import pre_processing.pre_processing_functions as ppm
from trajectory_reader.trajectories_reader import read_fishes_filter
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import smooth_positions
from trajectory_features.trajectory_feature_extraction import exponential_weights
from copy import deepcopy
import json


def init_transition_matrix(nr_fields):
    uniform_prob = 1 / nr_fields
    return np.full((nr_fields, nr_fields), uniform_prob)


def init_variances(nr_fields):
    return np.zeros((nr_fields,))


class GridNode:

    def __init__(self, x_limits, y_limits, nr_fields):
        self.__centroid = np.array([(x_limits[1] + x_limits[0]) / 2,
                                    (y_limits[1] + y_limits[0]) / 2])
        self.__x_limits = x_limits
        self.__y_limits = y_limits
        self.__motion_vectors = np.empty((nr_fields, 2))
        self.__transition_matrix = init_transition_matrix(nr_fields)
        self.__motion_vectors_variances = init_variances(nr_fields)
        self.__motion_vectors_priors = np.empty((nr_fields,))
        self.__init_position_prior = 0.0

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
    def motion_vectors_variances(self):
        return self.__motion_vectors_variances

    @property
    def motion_vectors_priors(self):
        return self.__motion_vectors_priors

    @property
    def init_position_prior(self):
        return self.__init_position_prior

    @motion_vectors.setter
    def motion_vectors(self, value):
        self.__motion_vectors = value

    @transition_matrix.setter
    def transition_matrix(self, value):
        self.__transition_matrix = value

    @motion_vectors_priors.setter
    def motion_vectors_priors(self, value):
        self.__motion_vectors_priors = value

    @init_position_prior.setter
    def init_position_prior(self, value):
        self.__init_position_prior = value

    def encode(self):
        return {
            "centroid": self.__centroid.tolist(),
            "x_limits": self.__x_limits,
            "y_limits": self.y_limits,
            "motion_vectors": self.__motion_vectors.tolist(),
            "transition_matrix": self.__transition_matrix.tolist(),
            "motion_vectors_variance": self.__motion_vectors_variances.tolist(),
            "motion_vectors_priors": self.__motion_vectors_priors.tolist(),
            "init_position_prior": self.__init_position_prior
        }

    def decode(self, grid_node_dict):
        self.__centroid = np.array(grid_node_dict["centroid"], dtype=np.float)
        self.__x_limits = grid_node_dict["x_limits"]
        self.__y_limits = grid_node_dict["y_limits"]
        self.__motion_vectors = np.array(grid_node_dict["motion_vectors"], dtype=np.float)
        self.__transition_matrix = np.array(grid_node_dict["transition_matrix"], dtype=np.float)
        self.__motion_vectors_variances = np.array(grid_node_dict["motion_vectors_variance"], dtype=np.float)
        self.__motion_vectors_priors = np.array(grid_node_dict["motion_vectors_priors"], dtype=np.float)
        self.__init_position_prior = grid_node_dict["init_position_prior"]

    def pretty_print(self):
        for attribute_description, attribute_value in self.__dict__.items():
            print(f"\n{attribute_description}:\n{attribute_value}")


def create_grid_matrix(width, height, side, nr_fields):
    x_bins = np.linspace(0, width, side + 1)
    y_bins = np.linspace(0, height, side + 1)

    grid = np.empty((side, side), dtype=object)
    for i in range(side):
        for j in range(side):
            x_limits = (int(x_bins[i]), int(x_bins[i+1]))
            y_limits = (int(y_bins[j]), int(y_bins[j+1]))
            grid[i][j] = GridNode(x_limits, y_limits, nr_fields)

    return grid


class SquareGrid:

    def __init__(self, width, height, nr_nodes, nr_fields):
        self.__width = width
        self.__height = height
        self.__nr_nodes = nr_nodes
        self.__side = int(nr_nodes ** 0.5)
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
    def nr_nodes(self):
        return self.__nr_nodes

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
        return self.__grid_matrix

    @property
    def nr_fields(self):
        return self.__nr_fields

    def encode(self):
        encoded_nodes = deepcopy(self.__grid_matrix)
        for i in range(len(encoded_nodes)):
            for j in range(len(encoded_nodes[i])):
                encoded_nodes[i][j] = encoded_nodes[i][j].encode()

        return {
            "width": self.__width,
            "height": self.__height,
            "nr_nodes": self.__nr_nodes,
            "side": self.__side,
            "x_interval": self.__x_interval,
            "y_interval": self.__y_interval,
            "nr_fields": self.__nr_fields,
            "grid_matrix": encoded_nodes.tolist()
        }

    def decode(self, grid_dict):
        self.__width = grid_dict["width"]
        self.__height = grid_dict["height"]
        self.__nr_nodes = grid_dict["nr_nodes"]
        self.__side = grid_dict["side"]
        self.__x_interval = grid_dict["x_interval"]
        self.__y_interval = grid_dict["y_interval"]
        self.__nr_fields = grid_dict["nr_fields"]

        encoded_matrix = grid_dict["grid_matrix"]
        self.__grid_matrix = create_grid_matrix(self.__width, self.__height,
                                                self.__side, self.__nr_fields)

        for i in range(self.__side):
            for j in range(self.__side):
                self.__grid_matrix[i][j].decode(encoded_matrix[i][j])


def identify_position_node(grid, pos):
    x_index = int(pos[0] / grid.x_interval)
    y_index = int(pos[1] / grid.y_interval)
    x_index = x_index if x_index < grid.side else grid.side - 1
    y_index = y_index if y_index < grid.side else grid.side - 1
    return (x_index, y_index)


def collect_motion_vectors(grid, trajectories):
    motion_vectors = defaultdict(list)

    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            current_position = np.array(
                [trajectory[i][1], trajectory[i][2]], dtype=np.float)
            next_position = np.array(
                [trajectory[i+1][1], trajectory[i+1][2]], dtype=np.float)
            motion_vector = next_position - current_position

            node_indexes = identify_position_node(grid, next_position)
            motion_vectors[node_indexes].append(motion_vector)

    return motion_vectors


def initialize_init_position_priors(grid, trajectories):
    initial_position_counter = defaultdict(lambda: 0)

    for trajectory in trajectories:
        initial_pos = trajectory[0]
        node_indexes = identify_position_node(
            grid, (initial_pos[1], initial_pos[2]))
        initial_position_counter[node_indexes] += 1

    for node_indexes, counter in initial_position_counter.items():
        grid.grid_matrix[node_indexes[1]][node_indexes[0]
                                          ].init_position_prior = counter / len(trajectories)


def initialize_fields(grid, trajectories):
    kmeans_instance = KMeans(grid.nr_fields)
    motion_vectors = collect_motion_vectors(grid, trajectories)

    for node_indexes, node_vectors in motion_vectors.items():
        data = np.array(node_vectors)

        if len(data) >= grid.nr_fields:
            kmeans_instance.fit(data)
            labels = kmeans_instance.labels_
            centroid_vectors = kmeans_instance.cluster_centers_

            label, counts = np.unique(labels, return_counts=True)
            label_count_tuples = list(zip(label, counts))
            label_count_tuples.sort(key=lambda x: x[1], reverse=True)

            node = grid.grid_matrix[node_indexes[1]][node_indexes[0]]
            for i in range(grid.nr_fields):
                node.motion_vectors[i] = kmeans_instance.cluster_centers_[
                    label_count_tuples[i][0]]
                node.motion_vectors_priors[i] = label_count_tuples[i][1] / \
                    len(data)
                node.transition_matrix[:, i] = [
                    node.motion_vectors_priors[i]] * grid.nr_fields

        else:
            random_vectors = 2 * \
                np.random.random_sample((grid.nr_fields, 2)) - 1
            node.motion_vectors = random_vectors
            node.motion_vectors_priors = np.full((grid.nr_fields,),
                                                 1 / grid.nr_fields)


def create_grid(video_path, fishes_path, species_path,
                species_of_interest, nr_nodes, nr_fields, return_trajectories=False):
    resolution = ppm.video_resolution(video_path)
    grid = SquareGrid(resolution[1], resolution[0], nr_nodes, nr_fields)
    fishes = read_fishes_filter(fishes_path, species_path,
                                species_of_interest)

    trajectories = []
    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish, False)
        smooth_positions(fish, exponential_weights(24, 0.01))
        trajectories.append(fish.trajectory)

    initialize_init_position_priors(grid, trajectories)
    initialize_fields(grid, trajectories)
    return grid if not return_trajectories else grid, trajectories


def save_model_to_file(grid, file_path):
    encoded_grid = grid.encode()
    with open(file_path, "w") as file:
        json.dump(encoded_grid, file, indent=4)


def load_model_from_file(file_path):
    with open(file_path, "r") as file:
        encoded_model = json.load(file)
        square_grid = SquareGrid(encoded_model["width"], encoded_model["height"],
                                 encoded_model["nr_nodes"], encoded_model["nr_fields"])
        square_grid.decode(encoded_model)
        return square_grid


def draw_grid(frame, grid):
    frame_copy = frame.copy()

    for i in range(grid.side):
        for j in range(grid.side):
            node = grid.grid_matrix[i][j]
            cv2.putText(frame_copy, f"({i}, {j})",
                        tuple(
                            (node.centroid - np.array([grid.x_interval * 0.25, 0])).astype(np.int)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.line(frame_copy, (node.x_limits[1], node.y_limits[0]),
                     (node.x_limits[1], node.y_limits[1]), (0, 0, 255), 2)
            cv2.line(frame_copy, (node.x_limits[0], node.y_limits[1]),
                     (node.x_limits[1], node.y_limits[1]), (0, 0, 255), 2)
            cv2.line(frame_copy, (node.x_limits[0], node.y_limits[0]),
                     (node.x_limits[1], node.y_limits[0]), (0, 0, 255), 2)
            cv2.line(frame_copy, (node.x_limits[0], node.y_limits[0]),
                     (node.x_limits[0], node.y_limits[1]), (0, 0, 255), 2)

    return frame_copy


def visualize_fields(frame, grid, scale):
    frames = []

    for n in range(grid.nr_fields):
        frame_copy = frame.copy()
        for i in range(grid.side):
            for j in range(grid.side):
                node = grid.grid_matrix[i][j]
                cv2.arrowedLine(frame_copy, tuple(node.centroid.astype(np.int)),
                                tuple(
                                    (node.centroid +
                                     node.motion_vectors[n] * scale).astype(np.int)),
                                (0, 0, 255), 2)
        frames.append(frame_copy)

    return frames


def pretty_print_parameters(grid):
    for i in range(grid.side):
        for j in range(grid.side):
            print(f"\n\n####### Node ({i}, {j}) #######")
            grid.grid_matrix[i][j].pretty_print()


def test_initialization(video_path, fishes_path, species_path, nr_nodes, nr_fields):
    grid = create_grid(video_path, fishes_path, species_path,
                       ("shark", "manta-ray"), nr_nodes, nr_fields, return_trajectories=True)[0]
    background_image = ppm.background_image_estimation(video_path, 300)

    grid_frame = draw_grid(background_image, grid)
    fields_images = visualize_fields(background_image, grid, 10)
    pretty_print_parameters(grid)

    cv2.imshow("grid", grid_frame)
    for i in range(len(fields_images)):
        cv2.imshow(f"field {i}", fields_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_model_saving(video_path, fishes_path, species_path, nr_nodes, nr_fields):
    grid = create_grid(video_path, fishes_path, species_path,
                       ("shark", "manta-ray"), nr_nodes, nr_fields, return_trajectories=True)[0]
    save_model_to_file(grid, "resources/models/test-model.model")
    model = load_model_from_file("resources/models/test-model.model")
    pretty_print_parameters(model)


if __name__ == "__main__":
    # test_initialization("resources/videos/v29.m4v",
    #                     "resources/detections/v29-fishes.json",
    #                     "resources/classification/species-gt-v29.csv",
    #                     16, 4)

    test_model_saving("resources/videos/v29.m4v",
                      "resources/detections/v29-fishes.json",
                      "resources/classification/species-gt-v29.csv",
                      16, 4)
