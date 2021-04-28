"""
Pre-process trajectories, fill gaps using interpolation.
Implements linear and newton interpolation.
"""

from collections import defaultdict
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from trajectory_reader.trajectories_reader import BoundingBox, read_detections
from trajectory_reader.visualization import (draw_position_plots,
                                             draw_trajectory, show_trajectory,
                                             simple_line_plot)


# region interpolation methodologies
class NewtonInterpolation:
    """
    Object that allows the trajectory interpolation using a Newton polynomial.
    """

    def __init__(self, example_points):
        """
        Assigns sample points and calculates newton coefficients independently for x and y positions.

        Args:
            example_points (list of tuples (t, x, y)): list of trajectory' sample points to interpolate
        """
        self.__example_points = example_points
        self.__calculate_coefs()

    def __calculate_coefs(self):
        ts, xs, ys = [], [], []
        for example_point in self.__example_points:
            ts.append(example_point[0])
            xs.append(example_point[1])
            ys.append(example_point[2])
        # calculate coefficients to interpolate x and y position values
        self.__xcoefs = NewtonInterpolation.calculate_coefs(
            np.array(ts), np.array(xs))
        self.__ycoefs = NewtonInterpolation.calculate_coefs(
            np.array(ts), np.array(ys))
        self.__ts = ts

    def reset(self, example_points):
        """
        Assigns new sample points and recalculates newton coefficients independently for x and y positions.

        Args:
            example_points (list of tuples (t, x, y)): list of trajectory' sample points to interpolate
        """
        self.__init__(example_points)

    def predict(self, t):
        """
        Predicts a new position for a instant time t, using the newton polynomial.

        Args:
            t (int): time instant

        Returns:
            tuple (t, x, y): predicted position
        """
        return (t,
                NewtonInterpolation.interpolate(
                    self.__xcoefs, self.__ts, t),
                NewtonInterpolation.interpolate(self.__ycoefs, self.__ts, t))

    @staticmethod
    def interpolate(coefs, xs, x):
        """
        Predict a new y value given the value of x, the newton coefficients, and the x values related to each coefficient.

        Args:
            coefs (float): list of newton coefficients
            xs (int): list of x values related to each coefficient
            x (int): x value of the prediction

        Returns:
            int: y value of the prediction
        """
        y = 0
        for i in range(1, len(coefs)):
            aux = coefs[i]  # coefficient of i degree
            # newton basis polynomial
            for j in range(i):
                aux *= (x - xs[j])
            y += aux
        return coefs[0] + y

    @staticmethod
    def calculate_coefs(x, y):
        """
        Newton coefficients calculation. These are the divided differences' values.

        Args:
            x (int): list of x values for the sample points
            y (int): list of y values for the sample points

        Returns:
            list of floats: list of newton coefficients (number of sample points-1 values)
        """
        n = len(x)
        divided_differences = [
            [None for _ in range(n)] for _ in range(n)]  # table nxn
        # initialize first column
        for i in range(n):
            divided_differences[i][0] = y[i]
        # calculate divided diffs
        for j in range(1, n):
            for i in range(n-j):  # only half of the table is needed
                divided_differences[i][j] = (divided_differences[i+1][j-1] - divided_differences[i][j-1]) \
                    / (x[i+j]-x[i])
        return divided_differences[0]


def linear_interpolation(starting_point, ending_point, discretization=True):
    """
    Returns the predicted position points of the gap, applying a linear interpolation.

    Args:
        starting_point (tuple (t, x, y)): starting point of the gap (left edge)
        ending_point (tuple (t, x, y)): ending point of the gap (right edge)

    Returns:
        list of tuples (t, x, y): predicted gap positions
    """
    nr_missing_points = (ending_point[0] - starting_point[0]) - 1
    x_step = (ending_point[1] - starting_point[1]) / (nr_missing_points + 1)
    y_step = (ending_point[2] - starting_point[2]) / (nr_missing_points + 1)

    gap_points = []
    for i in range(1, nr_missing_points+1):
        x_value = int(starting_point[1] + (i*x_step)) if discretization \
            else starting_point[1] + (i*x_step)
        y_value = int(starting_point[2] + (i*y_step)) if discretization \
            else starting_point[2] + (i*y_step)
        gap_points.append([starting_point[0]+i, x_value, y_value])

    return gap_points
# endregion


# region ways to choose sample points
def near_interpolation_points(trajectory, gap_starting_index, n):
    """
    The returned sample points using this method are the nearest points to the gap,
    taking into account that it can has another gaps close to it. The idea is
    to keep the sample points center (according to time) close to the gap center.
    It starts by adding the edges of the gap, and iteratively adds the point on the right
    or the point on the left, according to the proposed idea, until it gets n+1 points.

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        gap_starting_index (int): index of the trajectory list where it was detected a gap
        n (int): polynomial degree where these points will be used

    Returns:
        list of tuples (t, x, y): list with the chosen data points
    """
    points = []
    current_temporal_sum = 0
    gap_center = (trajectory[gap_starting_index+1][0] +
                  trajectory[gap_starting_index][0]) / 2
    # add both edge points
    points.extend([trajectory[gap_starting_index],
                   trajectory[gap_starting_index+1]])
    current_temporal_sum += (trajectory[gap_starting_index]
                             [0] + trajectory[gap_starting_index+1][0])
    edge_indexes = [gap_starting_index-1, gap_starting_index+2]
    for _ in range(n-1):
        # no more edge points to add
        if edge_indexes[0] < 0 and edge_indexes[1] >= len(trajectory):
            break
        # no more edge points before the gap
        elif edge_indexes[0] < 0:
            points.append(trajectory[edge_indexes[1]])
            edge_indexes[1] += 1
        # no more edge points after the gap
        elif edge_indexes[1] >= len(trajectory):
            points.insert(0, trajectory[edge_indexes[0]])
            edge_indexes[0] -= 1
        # keep the gap center near the example points center
        else:
            previous_edge_point = trajectory[edge_indexes[0]]
            posterior_edge_point = trajectory[edge_indexes[1]]
            new_average1 = (current_temporal_sum +
                            previous_edge_point[0]) / (len(points)+1)
            new_average2 = (current_temporal_sum +
                            posterior_edge_point[0]) / (len(points)+1)
            if abs(new_average1-gap_center) < abs(new_average2-gap_center):
                points.insert(0, trajectory[edge_indexes[0]])
                edge_indexes[0] -= 1
                current_temporal_sum += previous_edge_point[0]
            else:
                points.append(trajectory[edge_indexes[1]])
                edge_indexes[1] += 1
                current_temporal_sum += posterior_edge_point[0]
    return points


def equidistant_interpolation_points(trajectory, n):
    """
    The returned sample points using this method are equidistant from each other,
    according to its index and not taking into account possible gaps in the trajectory.

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        n (int): polynomial degree where these points will be used

    Returns:
        list of tuples (t, x, y): list with the chosen data points
    """
    sampling_step = int(len(trajectory) / n)
    points = [trajectory[i] for i in range(0, len(trajectory), sampling_step)]
    return points
# endregion


# region generate and fill gaps
def fill_gaps_linear(trajectory, fish, discretization=True):
    """
    Detects gaps on the trajectory and fills them using a linear interpolation (on place).

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
    """
    i = 0
    while(i < len(trajectory)):
        if i == 0:
            i += 1
            continue

        current_point = trajectory[i]
        previous_point = trajectory[i-1]
        t1 = previous_point[0]
        t2 = current_point[0]

        if t2 - t1 > 1:  # gap
            predicted_points = linear_interpolation(
                previous_point, current_point, discretization
            )
            trajectory[i:i] = predicted_points
            i += len(predicted_points) + 1

            if fish is not None:
                for predicted_point in predicted_points:
                    fish.positions[predicted_point[0]] = (
                        predicted_point[1], predicted_point[2]
                    )

                current_bbs = fish.bounding_boxes
                predicted_bbs = linear_interpolation(
                    (t1, current_bbs[t1].height, current_bbs[t1].width),
                    (t2, current_bbs[t2].height, current_bbs[t2].width),
                    discretization
                )
                for t, height, width in predicted_bbs:
                    current_bbs[t] = BoundingBox(width, height)

        else:
            i += 1


def fill_gaps_newton(trajectory, n, example_points_methodology):
    """
    Detects gaps on the trajectory and fills them using a newton interpolation (on place).

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        n (int): polynomial degree
        example_points_methodology (function): function that receives as input
            the same two args of this function (and in the same order), and returns
            a list of sample points to be used in the interpolation

    Returns:
        list of tuples (t, x, y): list of example points used in the interpolation
    """
    i = 0
    example_points = None
    while(i < len(trajectory)):
        if i == 0:
            i += 1
            continue
        current_point = trajectory[i]
        previous_point = trajectory[i-1]
        if current_point[0] - previous_point[0] > 1:  # gap
            if example_points_methodology.__name__ == near_interpolation_points.__name__:
                example_points = example_points_methodology(trajectory, i-1, n)
            else:
                example_points = example_points_methodology(trajectory, n)
            newton_method = NewtonInterpolation(example_points)
            # obtain near example data points and calculate newton polynomial
            predicted_points = [newton_method.predict(t) for t in range(
                previous_point[0]+1, current_point[0])]
            trajectory[i:i] = predicted_points
            i += len(predicted_points) + 1
        else:
            i += 1
    return example_points


def simulate_gap(fish, gap_size, gap_interval=None):
    """
    Generates a gap in the received trajectory. The starting index is randomly chosen.
    A new trajectory is returned. It is not done in place.

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        gap_size (int): size of the gap

    Returns:
        list of tuples (t, x, y), int, int:
            trajectory with a fake gap, initial instant of the gap, final instant of the gap
    """
    trajectory = fish.trajectory
    trajectory_copy = trajectory.copy()

    # specific interval
    if gap_interval is not None:
        start = trajectory_copy.index(
            [gap_interval[0]+1] + fish.get_position(gap_interval[0]+1)
        )
        end = trajectory_copy.index(
            [gap_interval[1]] + fish.get_position(gap_interval[1])
        )
        del trajectory_copy[start:end]
        return trajectory_copy, gap_interval

    # random initial point
    else:
        while(True):
            gap_start_index = randint(1, len(trajectory_copy)-gap_size-2)

            # no gaps around
            if trajectory[gap_start_index][0] - trajectory[gap_start_index-1][0] == 1 \
                    and trajectory[gap_start_index+gap_size][0] - trajectory[gap_start_index+gap_size-1][0] == 1:
                del trajectory_copy[gap_start_index:gap_start_index+gap_size]
                break
        return trajectory_copy, (trajectory[gap_start_index-1][0], trajectory[gap_start_index+gap_size][0])
# endregion


# region performance
def mse(true_trajectory, output_trajectory, gaps):
    square_errors = 0
    nr_gap_points = 0
    for t_initial, t_final in gaps:
        # get the portion of the trajectory relative to this gap
        true_gap_values = list(filter(
            lambda x: x[0] >= t_initial and x[0] <= t_final, true_trajectory))
        output_values = list(filter(
            lambda x: x[0] >= t_initial and x[0] <= t_final, output_trajectory))
        gap_size = len(true_gap_values)
        # calculate the square error for each of the missing positions
        for i in range(gap_size):
            square_errors += (true_gap_values[i][1] - output_values[i][1])**2 + (
                true_gap_values[i][2] - output_values[i][2])**2
        nr_gap_points += gap_size
    return 0 if nr_gap_points == 0 else square_errors / nr_gap_points


def newton_performance(trajectories_file_path, sample_points_method, degrees, gap_sizes):
    fishes = read_detections(trajectories_file_path)
    plt.figure()
    results = defaultdict(lambda: [])
    for gap_size in gap_sizes:
        # reuse generated gap to allow comparasion
        gap_info = defaultdict(lambda: 0)
        for degree in degrees:
            total_mse = 0
            for fish in fishes.values():
                trajectory = fish.trajectory
                # simulate gap and fill the gaps with newton interpolation
                if not gap_info[fish.fish_id]:
                    trajectory_with_gap, gap_interval = simulate_gap(
                        fish, gap_size)
                    gap_info[fish.fish_id] = gap_interval
                else:
                    trajectory_with_gap, gap_interval = simulate_gap(
                        fish, gap_size, gap_info[fish.fish_id])
                fill_gaps_newton(trajectory_with_gap, degree,
                                 sample_points_method)
                # calculate error
                total_mse += mse(trajectory,
                                 trajectory_with_gap, [gap_interval])
            results[degree].append(total_mse / len(fishes))
    # plot it
    for degree, mses in results.items():
        simple_line_plot(plt.gca(), gap_sizes, mses,
                         f"Newton Interpolation Performance\n{sample_points_method.__name__}", "mean MSE", "gap size", marker="o:", label=f"degree={degree}")
    plt.gca().legend()


def linear_performance(trajectories_file_path, gap_sizes):
    fishes = read_detections(trajectories_file_path)
    results = []

    for gap_size in gap_sizes:
        total_mse = 0
        for fish in fishes.values():
            trajectory = fish.trajectory

            # simulate gap and fill the gaps with linear interpolation
            trajectory_with_gap, gap_interval = simulate_gap(fish, gap_size)
            fill_gaps_linear(trajectory_with_gap, None)

            # calculate error
            total_mse += mse(trajectory, trajectory_with_gap, [gap_interval])

        results.append(total_mse / len(fishes))

    # plot results
    plt.figure()
    simple_line_plot(plt.gca(), gap_sizes, results,
                     "Linear Interpolation Performance", "mean MSE", "gap size", "o:r")
# endregion


def linear_interpolation_test():
    # read trajectories and choose an example one
    trajectories = read_detections(
        "resources/detections/detections-v29-sharks-mantas.txt"
    )
    example_fish = list(trajectories.values())[0]

    # generate a gap
    trajectory_with_gap, gap_interval = simulate_gap(example_fish, 20)
    frame = draw_trajectory(example_fish.trajectory, (480, 720), (0, 0, 0))

    # fill gaps with linear interpolation
    fill_gaps_linear(trajectory_with_gap, None)

    # visualize result
    draw_position_plots(trajectory_with_gap, gap_interval,
                        None, with_gap=False)
    cv2.imshow("trajectory", frame)
    plt.show()
    cv2.destroyAllWindows()


def newton_interpolation_test():
    # read trajectories and choose an example one
    trajectories = read_detections(
        "resources/detections/detections-v29-sharks-mantas.txt"
    )
    example_fish = list(trajectories.values())[0]

    # generate a gap
    trajectory_with_gap, gap_interval = simulate_gap(example_fish, 20)
    frame = draw_trajectory(example_fish.trajectory, (480, 720), (0, 0, 0))

    # fill gaps with newton interpolation
    interpolation_points = fill_gaps_newton(
        trajectory_with_gap, 5, equidistant_interpolation_points
    )

    # visualize result
    draw_position_plots(trajectory_with_gap, gap_interval,
                        interpolation_points, with_gap=False)
    cv2.imshow("trajectory", frame)
    plt.show()
    cv2.destroyAllWindows()


def draw_estimation_test():
    # read trajectories and choose an example one
    trajectories = read_detections(
        "resources/detections/detections-v29-sharks-mantas.txt"
    )
    example_fish = list(trajectories.values())[0]

    # generate a gap
    trajectory_with_gap, gap_interval = simulate_gap(
        example_fish, 30)
    frame = draw_trajectory(example_fish.trajectory, (480, 720), (0, 0, 0))
    cv2.imshow("trajectory", frame)

    # fill gaps with linear interpolation
    fill_gaps_linear(trajectory_with_gap, None)

    # visualize video, showing interpolation results
    show_trajectory("resources/videos/v29.m4v",
                    example_fish, trajectory_with_gap, [gap_interval])  # , "interpolation-example1.mp4")
    plt.show()
    cv2.destroyAllWindows()


def evaluation_test():
    # evaluation results of the different methods
    linear_performance(
        "resources/detections/detections-v29-sharks-mantas.txt",
        range(1, 30, 3)
    )
    newton_performance("resources/detections/detections-v29-sharks-mantas.txt",
                       equidistant_interpolation_points,
                       [4, 5, 6], range(1, 10, 2))
    newton_performance("resources/detections/detections-v29-sharks-mantas.txt",
                       near_interpolation_points,
                       [4, 5, 6], range(1, 10, 1))
    plt.show()


if __name__ == "__main__":
    linear_interpolation_test()
    # newton_interpolation_test()
    # draw_estimation_test()
    # evaluation_test()
