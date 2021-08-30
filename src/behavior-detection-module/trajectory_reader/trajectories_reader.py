"""
- definition of the Fish entity
- read detections file
- read/save trajectories to file
"""

import json
import os
import random
from collections import namedtuple

import cv2

import labeling.trajectory_labeling as labeling_module
from trajectory_reader.visualization import draw_trajectory

BoundingBox = namedtuple("BoundingBox", ["width", "height"])


class Fish:
    """
    Entity - Fish
    """

    def __init__(self, fish_id, trajectory=None, bounding_boxes=None, positions=None):
        # initialization
        self.__fish_id = fish_id
        self.__trajectory = [] if trajectory is None else trajectory
        self.__bounding_boxes_size = {} if bounding_boxes is None else bounding_boxes
        self.__positions = {} if positions is None else positions

    def add_position(self, data_point, width, height):
        # add new information from a frame
        self.__trajectory.append(data_point)
        self.__bounding_boxes_size[data_point[0]] = BoundingBox(width, height)
        self.__positions[data_point[0]] = [data_point[1], data_point[2]]
        return self

    def get_position(self, t):
        if t in self.__positions:
            return self.__positions[t]

    def encode(self):
        # convert fish values to a dict
        return {
            "trajectory": self.__trajectory,
            "bounding-boxes": {t: (bounding_box.width, bounding_box.height)
                               for t, bounding_box in self.__bounding_boxes_size.items()}
        }

    def decode(self, fish_dict):
        trajectory = fish_dict['trajectory']
        trajectory_list = []
        trajectory_positions_dict = {}

        for data_point in trajectory:
            t = int(data_point[0])
            x = int(data_point[1])
            y = int(data_point[2])
            trajectory_list.append([t, x, y])
            trajectory_positions_dict[t] = [x, y]

        self.__positions = trajectory_positions_dict
        self.__trajectory = trajectory_list

        # bounding boxes
        self.__bounding_boxes_size = {int(t): BoundingBox(int(bb_tuple[0]), int(bb_tuple[1]))
                                      for t, bb_tuple in fish_dict["bounding-boxes"].items()}

    @property
    def fish_id(self):
        return self.__fish_id

    @property
    def trajectory(self):
        return self.__trajectory

    @property
    def bounding_boxes(self):
        return self.__bounding_boxes_size

    @property
    def positions(self):
        return self.__positions


def read_detections(detections_file_path):
    """
    Read the detections ground truth from a file.

    Args:
        detections_file_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    # read file content
    trajectories = {}

    with open(detections_file_path, 'r') as f:
        detections = f.readlines()

    for line in detections:
        fields = list(map(int, line.split(",")))
        frame = fields[0]
        nr_fish = fields[1]

        # add the new detected positions to the fish
        for i in range(int(nr_fish)):
            detection = fields[2 + (i * 5):2 + (i * 5) + 5]

            # centroid
            data_point = [frame,
                          int((detection[0] + detection[2]) / 2),
                          int((detection[1] + detection[3]) / 2)]

            fish_id = detection[-1]
            if fish_id in trajectories.keys():
                trajectories[fish_id].add_position(
                    data_point,
                    abs(detection[0] - detection[2]),
                    abs(detection[1] - detection[3])
                )

            # new fish detected
            else:
                trajectories[fish_id] = Fish(fish_id).add_position(
                    data_point,
                    abs(detection[0] - detection[2]),
                    abs(detection[1] - detection[3])
                )

    return trajectories


def save_fishes(fishes, output_path, starting_id):
    """
    Save a set of fishes information to a file using JSON format.
    The ID will start counting from starting_id.

    Args:
        fishes ([type]): [description]
        output_path ([type]): [description]
        starting_id ([type]): [description]
    """
    # encode all the fishes
    fishes_encode = {}
    fish_id = starting_id
    for fish in fishes:
        fishes_encode[fish_id] = fish.encode()
        fish_id += 1

    # append to file
    if os.path.exists(output_path) and os.path.isfile(output_path):
        with open(output_path, 'r+') as f:
            current_fishes = json.load(f)
            current_fishes.update(fishes_encode)
            f.seek(0)  # reset file pointer to position 0
            json.dump(current_fishes, f, indent=4)

    # create a new one
    else:
        with open(output_path, 'w') as f:
            json.dump(fishes_encode, f, indent=4)


def read_fishes(fishes_file_path):
    """
    Read fishes contained in the file with the received path.

    Args:
        fishes_file_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    with open(fishes_file_path) as f:
        # dict with all fishes
        fishes = json.load(f)
        fishes_set = set()

        # decode each one
        for fish_id, fish_dict in fishes.items():
            fish_instance = Fish(int(fish_id))
            fish_instance.decode(fish_dict)
            fishes_set.add(fish_instance)
    return fishes_set


def read_fishes_filter(fishes_file_path, species_file_path, species_of_interest):
    fishes = read_fishes(fishes_file_path)
    species = labeling_module.read_species_gt(species_file_path)
    return [fish for fish in fishes if species[fish.fish_id] in species_of_interest]


def union_gt(*detection_file_paths, output_path):
    """
    Read fishes from different detection files
    and write them to the same file.

    Args:
        output_path ([type]): [description]
    """
    all_fishes = []

    # read the fishes in all received paths
    for detection_file_path in detection_file_paths:
        all_fishes += list(read_detections(detection_file_path).values())

    # save all to the same file
    save_fishes(all_fishes, output_path, 0)


def get_random_fish(fishes_file_path, seed=None):
    if seed is not None:
        random.seed(seed)
    fishes = list(read_fishes(fishes_file_path))
    fishes.sort(key=lambda x: x.fish_id)
    return random.choice(fishes)


def read_trajectory_test():
    trajectories = read_detections(
        "resources/detections/detections-v29-sharks-mantas.txt"
    )
    example_fish = list(trajectories.values())[0]
    # print(example_trajectory)
    trajectory_frame = draw_trajectory(
        example_fish.trajectory, (480, 720), (0, 0, 0), path=False
    )
    cv2.imshow("example trajectory", trajectory_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_fishes_test():
    fishes = read_fishes("../resources/detections/v29-fishes.json")
    print("number of fishes: ", len(fishes))
    example_fish = fishes.pop()
    print("example fish: ", example_fish.fish_id)
    print("first data points: ", example_fish.trajectory[:10])
    print("trajectory length: ", len(example_fish.trajectory))


if __name__ == "__main__":
    read_trajectory_test()
    # read_fishes_test()
    # union_gt("../resources/detections/detections-joao-v29.txt",
    #          "../resources/detections/detections-v29-sharks-mantas.txt",
    #          output_path="../resources/detections/v29-fishes.json")
