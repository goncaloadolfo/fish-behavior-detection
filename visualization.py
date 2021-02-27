"""
High-level information visualization methods.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def simple_line_plot(ax, xs, ys, title, ylabel, xlabel, marker='-', label=None):
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
    ax.plot(xs, ys, marker, label=label)


def simple_bar_chart(ax, xs, ys, title, ylabel, xlabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.bar(xs, ys, align="center")
    

def draw_trajectory(trajectory, frame_size, color, regions=None):
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
    if regions is not None:
        for region in regions:
            region.draw(frame)
    cv2.imshow("trajectory", frame)


def draw_position_plots(trajectory, gap_interval, interpolation_points, with_gap=True):
    """
    Draws the x and y position variation over time (in different figures).

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        gap_interval (tuple (t_initial, t_final)): initial and final instant of the gap
        interpolation_points(list of tuples (t, x, y)): list of example points used in the interpolation
        with_gap (bool, optional): specifies if the received trajectory still has the
            missing points of the gap. Defaults to True.

    Returns:
        matplotlib.axes, matplotlib.axes: axes of the x variation plot and axes of the y variation plot
    """
    def plot_variation(xs, ys, title, ylabel, xlabel, interpolation_points):
        plt.figure()
        ax = plt.gca()
        simple_line_plot(ax, xs, ys, title, ylabel, xlabel)
        # example points highliting
        if interpolation_points is not None:
            ts, values = [], []
            for data_point in interpolation_points:
                value_index = 1 if title == "X Position" else 2
                ts.append(data_point[0])
                values.append(data_point[value_index])
            ax.scatter(ts, values, s=10, c="g", marker="o",
                       label="interpolation point")
        # circles in the gap edges
        ax.scatter(gap_interval[0], ys[xs.index(
            gap_interval[0])], s=10, c="r", marker="o", label="gap edge")
        ax.scatter(gap_interval[1], ys[xs.index(
            gap_interval[1])], s=10, c="r", marker="o")
        ax.legend()
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
    return plot_variation(ts, xs, "X Position", "x value", "frame", interpolation_points), \
        plot_variation(ts, ys, "Y Position", "y value",
                       "frame", interpolation_points)


def show_trajectory(video_path, fish, estimated_trajectory, simulated_gaps, 
                    record=None, frame_name="gap estimation"):
    # record and read settings
    cap = cv2.VideoCapture(video_path)
    waitkey_value = 75
    out = None
    if record is not None:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(record, apiPreference=cv2.CAP_FFMPEG, fourcc=codec, fps=15,
                              frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        waitkey_value = 1
        print("recording...")
    # set next frame to be read to the initial timestamp index of the trajectory
    t = estimated_trajectory[0][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    while(t < estimated_trajectory[-1][0]):
        _, frame = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # check if it is part of a generated gap
        is_simulated_gap = False
        for gap in simulated_gaps:
            if t >= gap[0] and t <= gap[1]:
                is_simulated_gap = True
                break
        # and if it exists in the true trajectory
        true_data_point = fish.get_position(t)
        if not is_simulated_gap and true_data_point is not None:
            centroid = (true_data_point[1], true_data_point[2])
            bounding_box_size = fish.get_bounding_box_size(t)
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            cv2.rectangle(frame,
                          (int(centroid[0] - (bounding_box_size.width/2)),
                           int(centroid[1] - bounding_box_size.height/2)),
                          (int(centroid[0] + (bounding_box_size.width/2)),
                           int(centroid[1] + bounding_box_size.height/2)),
                          (0, 255, 0), 2)
        else:
            data_point = filter(lambda x: x[0] == t, estimated_trajectory)
            if len(data_point) > 0:
                data_point = list(data_point)
                cv2.circle(
                    frame, (data_point[0][1], data_point[0][2]), 5, (0, 0, 255), -1)
        # show or write frame
        if record is None:
            cv2.imshow(frame_name, frame)
        else:
            out.write(frame)
        if cv2.waitKey(waitkey_value) & 0xFF == ord('q'):
            break
    cap.release()
    if out is not None:
        print("done!")
        out.release()


def draw_fishes(frame, fishes, t):
    for fish in fishes:
        bounding_box_size = fish.get_bounding_box_size(t)
        centroid = fish.get_position(t)
        centroid = (centroid[1], centroid[2])
        cv2.rectangle(frame,
                      (int(centroid[0] - (bounding_box_size.width/2)),
                       int(centroid[1] - bounding_box_size.height/2)),
                      (int(centroid[0] + (bounding_box_size.width/2)),
                       int(centroid[1] + bounding_box_size.height/2)),
                      (0, 255, 0), 2)
    return frame
