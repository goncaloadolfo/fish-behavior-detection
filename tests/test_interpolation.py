"""
Set of tests to the interpolation module.
"""

import cv2
import matplotlib.pyplot as plt

from trajectories_reader import read_detections
from interpolation import simulate_gap, fill_gaps_linear, fill_gaps_newton, \
    equidistant_interpolation_points, near_interpolation_points, linear_performance, \
        newton_performance
from visualization import draw_position_plots, draw_trajectory, show_trajectory


def linear_interpolation_test():
    # read trajectories and choose an example one
    trajectories = read_detections("../data/Dsc 0029-lowres_gt.txt")
    example_fish = list(trajectories.values())[0]
    # generate a gap
    trajectory_with_gap, gap_interval = simulate_gap(example_fish, 20)
    draw_trajectory(example_fish.trajectory, (480, 720), (0, 0, 0))
    # fill gaps with linear interpolation
    fill_gaps_linear(trajectory_with_gap)
    # visualize result
    draw_position_plots(trajectory_with_gap, gap_interval,
                        None, with_gap=False)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def newton_interpolation_test():
    # read trajectories and choose an example one
    trajectories = read_detections("../data/Dsc 0029-lowres_gt.txt")
    example_fish = list(trajectories.values())[0]
    # generate a gap
    trajectory_with_gap, gap_interval = simulate_gap(example_fish, 20)
    draw_trajectory(example_fish.trajectory, (480, 720), (0, 0, 0))
    # fill gaps with newton interpolation
    interpolation_points = fill_gaps_newton(
        trajectory_with_gap, 5, equidistant_interpolation_points)
    # visualize result
    draw_position_plots(trajectory_with_gap, gap_interval,
                        interpolation_points, with_gap=False)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_estimation_test():
    # read trajectories and choose an example one
    trajectories = read_detections("../data/Dsc 0029-lowres_gt.txt")
    example_fish = list(trajectories.values())[0]
    # generate a gap
    trajectory_with_gap, gap_interval = simulate_gap(
        example_fish, 30)
    draw_trajectory(example_fish.trajectory, (480, 720), (0, 0, 0))
    # fill gaps with linear interpolation
    fill_gaps_linear(trajectory_with_gap)
    # visualize video, showing interpolation results
    show_trajectory("../data/Dsc 0029-lowres.m4v",
                    example_fish, trajectory_with_gap, [gap_interval])  #, "../data/interpolation-example1.mp4")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def evaluation_test():
    # evaluation results of the different methods
    linear_performance("../data/Dsc 0029-lowres_gt.txt", range(1, 50, 3))
    newton_performance("../data/Dsc 0029-lowres_gt.txt", equidistant_interpolation_points,
                       [4, 5, 6], range(1, 10, 2))
    newton_performance("../data/Dsc 0029-lowres_gt.txt", near_interpolation_points,
                       [4, 5, 6], range(1, 10, 1))
    plt.show()


if __name__ == "__main__":
    linear_interpolation_test()
    # newton_interpolation_test()
    # draw_estimation_test()
    # evaluation_test()
