"""
Tests to trajectory feature extraction module
"""

from trajectories_reader import produce_trajectories
from trajectory_feature_extraction import analyze_trajectory


def analyze_trajectory_test():
    fishes = list(produce_trajectories("../data/Dsc 0037-lowres_gt.txt").values())
    analyze_trajectory("../data/Dsc 0037-lowres.m4v",
                       "conf/regions-example.json", 
                       fishes[0])


if __name__ == "__main__":
    analyze_trajectory_test()
