"""
Pre-process trajectories, fill gaps using interpolation.
Implements linear and newton interpolation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from trajectories_reader import produce_trajectories
from visualization import draw_trajectory, draw_position_plots, simple_line_plot


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
        self.__init__(self, example_points)

    def predict(self, t):
        """
        Predicts a new position for a instant time t, using the newton polynomial. 

        Args:
            t (int): time instant

        Returns:
            tuple (t, x, y): predicted position
        """
        return (t,
                int(NewtonInterpolation.interpolate(
                    self.__xcoefs, self.__ts, t)),
                int(NewtonInterpolation.interpolate(self.__ycoefs, self.__ts, t)))

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


def linear_interpolation(starting_point, ending_point):
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
    gap_points = [(starting_point[0]+i,
                   int(starting_point[1] + (i*x_step)),
                   int(starting_point[2] + (i*y_step)))
                  for i in range(1, nr_missing_points+1)]
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
    gap_center = trajectory[gap_starting_index+1][0] - \
        trajectory[gap_starting_index][0] / 2
    # add both edge points
    points.extend([trajectory[gap_starting_index],
                   trajectory[gap_starting_index+1]])
    edge_indexes = [gap_starting_index-1, gap_starting_index+2]
    for _ in range(2, n+2):
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
            aux1 = points[-1][0] - previous_edge_point[0]
            aux2 = posterior_edge_point[0] - points[0][0]
            if abs(aux1-gap_center) < abs(aux2-gap_center):
                points.insert(0, trajectory[edge_indexes[0]])
                edge_indexes[0] -= 1
            else:
                points.append(trajectory[edge_indexes[1]])
                edge_indexes[1] += 1
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
def fill_gaps_linear(trajectory):
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
        if current_point[0] - previous_point[0] > 1:  # gap
            predicted_points = linear_interpolation(
                previous_point, current_point)
            trajectory[i:i] = predicted_points
            i += len(predicted_points) + 1
        else:
            i += 1


def fill_gaps_newton(trajectory, n):
    """
    Detects gaps on the trajectory and fills them using a newton interpolation (on place).

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        n (int): polynomial degree
    """
    example_points = equidistant_interpolation_points(trajectory, n)
    newton_method = NewtonInterpolation(example_points)
    i = 0
    while(i < len(trajectory)):
        if i == 0:
            i += 1
            continue
        current_point = trajectory[i]
        previous_point = trajectory[i-1]
        if current_point[0] - previous_point[0] > 1:  # gap
            # obtain near example data points and calculate newton polynomial
            predicted_points = [newton_method.predict(t) for t in range(
                previous_point[0]+1, current_point[0])]
            trajectory[i:i] = predicted_points
            i += len(predicted_points) + 1
        else:
            i += 1


def simulate_gap(trajectory, gap_size):
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
    trajectory_copy = trajectory.copy()
    # choose randomly the initial point of the gap
    gap_start_index = randint(1, len(trajectory_copy)-gap_size-2)
    del trajectory_copy[gap_start_index:gap_start_index+gap_size]
    return trajectory_copy, (trajectory[gap_start_index-1][0], trajectory[gap_start_index+gap_size][0])
# endregion


# region test cases
def __linear_interpolation_example(gap_size):
    trajectories = produce_trajectories("../data/Dsc 0029-lowres_gt.txt")
    example_trajectory = list(trajectories.values())[0].trajectory
    trajectory_with_gap, gap_interval = simulate_gap(
        example_trajectory, gap_size)
    draw_trajectory(example_trajectory, (480, 720), (0, 0, 0))
    fill_gaps_linear(trajectory_with_gap)
    draw_position_plots(trajectory_with_gap, gap_interval, with_gap=False)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def __newton_interpolation_example(n, gap_size):
    trajectories = produce_trajectories("../data/Dsc 0029-lowres_gt.txt")
    example_trajectory = list(trajectories.values())[0].trajectory
    trajectory_with_gap, gap_interval = simulate_gap(example_trajectory, 20)
    draw_trajectory(example_trajectory, (480, 720), (0, 0, 0))
    fill_gaps_newton(trajectory_with_gap, 5)
    draw_position_plots(trajectory_with_gap, gap_interval, with_gap=False)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# endregion


if __name__ == "__main__":
    # __linear_interpolation_example(20)
    __newton_interpolation_example(5, 20)
