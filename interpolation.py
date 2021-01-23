"""
Pre-process trajectories, fill gaps using interpolation.
"""     

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from trajectories_reader import produce_trajectories
from visualization import draw_trajectory, draw_position_plots, simple_line_plot


class NewtonInterpolation:
    
    def __calculate_coefs(self):
        ts, xs, ys = [], [], []
        for example_point in self.__example_points:
            ts.append(example_point[0])
            xs.append(example_point[1])
            ys.append(example_point[2])
        # calculate coefficients to interpolate x and y position values
        self.__xcoefs = NewtonInterpolation.calculate_coefs(np.array(ts), np.array(xs))
        self.__ycoefs = NewtonInterpolation.calculate_coefs(np.array(ts), np.array(ys))
        self.__ts = ts 
    
    def predict(self, t):
        return (t, 
                int(NewtonInterpolation.interpolate(self.__xcoefs, self.__ts, t)), 
                int(NewtonInterpolation.interpolate(self.__ycoefs, self.__ts, t)))
        
    def initialize(self, example_points):
        self.__example_points = example_points
        self.__calculate_coefs()
    
    @staticmethod
    def interpolate(coefs, xs, x):
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
        n = len(x)
        divided_differences = [[None for _ in range(n)] for _ in range(n)]  # table nxn    
        # initialize first column
        for i in range(n):
            divided_differences[i][0] = y[i]
        # calculate divided diffs
        for j in range(1, n):
            for i in range(n-j):  # only half of the table is needed
                divided_differences[i][j] = (divided_differences[i+1][j-1] - divided_differences[i][j-1]) \
                                            / (x[i+j]-x[i])
        return divided_differences[0]
    
        
def near_interpolation_points(trajectory, gap_starting_index, n): 
    points = []
    gap_center = trajectory[gap_starting_index+1][0]-trajectory[gap_starting_index][0] / 2
    # add both edge points
    points.extend([trajectory[gap_starting_index], trajectory[gap_starting_index+1]])
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
    sampling_step = int(len(trajectory) / n) 
    points = [trajectory[i] for i in range(0, len(trajectory), sampling_step)]
    return points


def linear_interpolation(starting_point, ending_point):
    nr_missing_points = (ending_point[0] - starting_point[0]) - 1 
    x_step = (ending_point[1] - starting_point[1]) / (nr_missing_points + 1)
    y_step = (ending_point[2] - starting_point[2]) / (nr_missing_points + 1)
    gap_points = [(starting_point[0]+i, 
                    int(starting_point[1] + (i*x_step)), 
                    int(starting_point[2] + (i*y_step)))
                  for i in range(1, nr_missing_points+1)]
    return gap_points


def fill_gaps_linear(trajectory):
    i = 0
    while(i < len(trajectory)):
        if i == 0:
            i += 1
            continue
        current_point = trajectory[i]
        previous_point = trajectory[i-1]
        if current_point[0] - previous_point[0] > 1:  # gap
            predicted_points = linear_interpolation(previous_point, current_point)
            trajectory[i:i] = predicted_points
            i += len(predicted_points) + 1
        else:
            i += 1


def fill_gaps_newton(trajectory, n):
    newton_method = NewtonInterpolation()
    example_points = equidistant_interpolation_points(trajectory, n)
    newton_method.initialize(example_points)
    i = 0
    while(i < len(trajectory)):
        if i == 0:
            i += 1
            continue
        current_point = trajectory[i]
        previous_point = trajectory[i-1]
        if current_point[0] - previous_point[0] > 1:  # gap
            # obtain near example data points and calculate newton polynomial  
            predicted_points = [newton_method.predict(t) for t in range(previous_point[0]+1, current_point[0])]  
            trajectory[i:i] = predicted_points
            i += len(predicted_points) + 1
        else:
            i += 1

        
def simulate_gap(trajectory, gap_size):
    trajectory_copy = trajectory.copy()
    # choose randomly the initial point of the gap
    gap_start_index = randint(1, len(trajectory_copy)-gap_size-2)
    del trajectory_copy[gap_start_index:gap_start_index+gap_size]
    return trajectory_copy, (trajectory[gap_start_index-1][0], trajectory[gap_start_index+gap_size][0])


if __name__ == "__main__":
    # read trajectories and get one of them
    trajectories = produce_trajectories("../data/Dsc 0029-lowres_gt.txt")
    example_trajectory = list(trajectories.values())[0].trajectory
    # generate a gap in the trajectory
    trajectory_wgap, gap_interval = simulate_gap(example_trajectory, 20)
    # visualize
    draw_trajectory(example_trajectory, (480, 720), (0, 0, 0))
    fill_gaps_newton(trajectory_wgap, 5)
    #fill_gaps_linear(trajectory_wgap)
    draw_position_plots(trajectory_wgap, gap_interval, with_gap=False)
    # show plots
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    