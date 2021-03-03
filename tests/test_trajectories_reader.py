"""
Tests to trajectories reader module
"""

import cv2

from trajectories_reader import produce_trajectories
from visualization import draw_trajectory


def read_trajectory_test():
    trajectories = produce_trajectories("../data/Dsc 0029-lowres_gt.txt")
    example_trajectory = list(trajectories.values())[0].trajectory
    print(example_trajectory)
    draw_trajectory(example_trajectory, (480, 720), (0, 0, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    read_trajectory_test()
