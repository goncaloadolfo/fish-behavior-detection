"""
High-level information visualization methods. 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def simple_line_plot(ax, xs, ys, title, ylabel, xlabel):
    """
    Draws a line plot on the received axes.

    Args:
        ax ([matplotlib.axes]): entity with figure elements
        xs ([list of numbers]): x values
        ys ([list of numbers]): y values
        title ([str]): title of the plot
        ylabel ([str]): label of the y axis
        xlabel ([str]): label of the x axis
    """
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.plot(xs, ys)


def draw_trajectory(trajectory, frame_size, color):
    """
    Draws the trajectory on a given frame.

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        frame_size (tuple (height, width)): frame resolution
        color (tuple (b, g, r)): color of the trajectory points in rgb format
    """
    frame = np.full((frame_size[0], frame_size[1], 3), 255, dtype=np.uint8)
    for data_point in trajectory:
        cv2.circle(frame, (data_point[1], data_point[2]),
                   radius=2, color=color, thickness=-1)
    cv2.imshow("trajectory", frame)


def draw_position_plots(trajectory, gap_interval, with_gap=True):
    """
    Draws the x and y position variation over time (in different figures).

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        gap_interval (tuple (t_initial, t_final)): initial and final instant of the gap
        with_gap (bool, optional): specifies if the received trajectory still has the 
            missing points of the gap. Defaults to True.

    Returns:
        matplotlib.axes, matplotlib.axes: axes of the x variation plot and axes of the y variation plot
    """
    def plot_variation(xs, ys, title, ylabel, xlabel):
        plt.figure()
        ax = plt.gca()
        simple_line_plot(ax, xs, ys, title, ylabel, xlabel)
        # circles in the gap edges
        ax.scatter(gap_interval[0], ys[xs.index(
            gap_interval[0])], s=10, c="r", marker="o")
        ax.scatter(gap_interval[1], ys[xs.index(
            gap_interval[1])], s=10, c="r", marker="o")
        return ax
    ts, xs, ys = [], [], []
    for data_point in trajectory:
        ts.append(data_point[0])
        xs.append(data_point[1])
        ys.append(data_point[2])
    # to avoid the line interligating both sides of the gap
    if with_gap:
        ts.insert(gap_interval[0]+1, np.nan)
        xs.insert(gap_interval[0]+1, np.nan)
        ys.insert(gap_interval[0]+1, np.nan)
    return plot_variation(ts, xs, "X Position", "x value", "frame"), \
        plot_variation(ts, ys, "Y Position", "y value", "frame")
