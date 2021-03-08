"""
Set of tests to the trajectory labeling module
"""

import logging

from trajectory_labeling import TrajectoryLabeling, read_episodes, trajectory_labeling_logger 
from trajectories_reader import produce_trajectories


def trajectory_labeling_behavior_test():
    # test the general behavior
    trajectory_labeling_logger.setLevel(logging.DEBUG)
    fishes = list(produce_trajectories("../data/v29-sharks-mantas-gt.txt").values())
    TrajectoryLabeling(fishes, "../data/Dsc 0029-lowres.m4v", "../data/v29-gt-episodes-test.txt").start()
    
    
def episodes_reading_test():
    # check if the episodes are being well read
    episodes = read_episodes("../data/v29-gt-episodes-test.txt")
    for episode in episodes:
        print(repr(episode))


if __name__ == "__main__":
    # trajectory_labeling_behavior_test()
    episodes_reading_test()
    