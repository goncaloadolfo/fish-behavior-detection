"""
Tests to trajectory feature extraction module
"""

from trajectories_reader import produce_trajectories
from trajectory_feature_extraction import analyze_trajectory, frequency_analysis
from regions_selector import read_regions


def analyze_trajectory_test(frequency):
    fishes = list(produce_trajectories("../data/Dsc 0037-lowres_gt.txt").values())
    regions = read_regions("conf/regions-example.json")
    analyze_trajectory("../data/Dsc 0037-lowres.m4v",
                       regions, 
                       fishes[0], 
                       frequency)
    
    
def frequency_analysis_test(frequencies):
    fishes = list(produce_trajectories("../data/Dsc 0037-lowres_gt.txt").values())
    regions = read_regions("conf/regions-example.json")
    frequency_analysis(fishes[0], regions, frequencies)


if __name__ == "__main__":
    analyze_trajectory_test(frequency=15)
    # frequency_analysis_test([1, 12, 24])
    