"""
Fish entity; capability to read detections file.
"""

import numpy as np
import cv2

from collections import namedtuple
from visualization import draw_trajectory


BoundingBox = namedtuple("BoundingBox", ["width", "height"])


class Fish:
    """
    Fish entity
    """

    def __init__(self, fish_id):
        self.__fish_id = fish_id
        self.__trajectory = []
        self.__bounding_boxes_size = {}

    def add_position(self, data_point, width, height):
        self.__trajectory.append(data_point)
        self.__bounding_boxes_size[data_point[0]] = BoundingBox(width, height)
        return self

    @property
    def fish_id(self):
        return self.__fish_id

    @property
    def trajectory(self):
        return self.__trajectory

    @property
    def bounding_boxes(self):
        return self.__bounding_boxes_size

    def get_bounding_box_size(self, t):
        return self.__bounding_boxes_size[t]

    def get_position(self, t):
        data_point = list(filter(lambda x: x[0] == t, self.__trajectory))
        return data_point[0] if len(data_point) != 0 else None


def produce_trajectories(detections_file):
    trajectories = {}
    with open(detections_file, 'r') as f:
        for line in f:
            fields = list(map(int, line.split(",")))
            frame = fields[0]
            nr_fish = fields[1]
            # add the new detected positions to the fish
            for i in range(int(nr_fish)):
                detection = fields[2+(i*5):2+(i*5)+5]
                # centroid
                data_point = (frame,
                              int((detection[0] + detection[2])/2),
                              int((detection[1] + detection[3])/2))
                fish_id = detection[-1]
                if fish_id in trajectories:
                    trajectories[fish_id].add_position(
                        data_point, abs(detection[0]-detection[2]), abs(detection[1]-detection[3]))
                # new fish detected
                else:
                    trajectories[fish_id] = Fish(
                        fish_id).add_position(data_point, abs(detection[0]-detection[2]), abs(detection[1]-detection[3]))
    return trajectories
