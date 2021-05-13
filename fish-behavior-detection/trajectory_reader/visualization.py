"""
High-level information visualization methods.
"""

import time

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np


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


def simple_bar_chart(ax, xs, ys, title, ylabel, xlabel, label=None, width=0.8):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.bar(xs, ys, align="center", label=label, width=width)


def histogram(ax, values, title, ylabel, xlabel, density=False, cumulative=False):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    return ax.hist(values, density=density, cumulative=cumulative)


def histogram2d(ax, values, values2, title, ylabel, xlabel, colormap="Blues", cmin=0, with_text=False, frame=None):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    counts, binsx, binsy, quad_image = ax.hist2d(
        values, values2, cmap=colormap, cmin=cmin
    )

    cmap = matplotlib.cm.get_cmap("YlOrBr")
    min_value = np.min(counts)
    max_value = np.max(counts)

    if with_text:
        x_interval = binsx[1] - binsx[0]
        y_interval = binsy[1] - binsy[0]

        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                text_position = (int(binsx[row] + x_interval / 2), int(binsy[col] + y_interval / 2))
                cell_count = int(counts[row, col])
                ax.text(text_position[0], text_position[1], cell_count, color="black", ha="center", va="center")

                if frame is not None:
                    color = cmap((cell_count - min_value) / (max_value - min_value))
                    bgr = (int(color[2]*255), int(color[1]*255.0), int(color[0]*255.0))
                    cv2.putText(frame, str(cell_count), text_position, cv2.FONT_HERSHEY_COMPLEX, 0.5, bgr, thickness=2)

    return counts, binsx, binsy, quad_image, frame


def draw_trajectory(trajectory, frame_size, color,
                    regions=None, frame=None, path=True,
                    interval=24, identifier=None):
    """
    Draws the trajectory on a given frame.

    Args:
        trajectory (list of tuples (t, x, y)): list of positions
        frame_size (tuple (height, width)): frame resolution
        color (tuple (b, g, r)): color of the trajectory points in rgb format
    """
    # declare frame matrix if not already set
    if frame is None:
        frame = np.full((frame_size[0], frame_size[1], 3), 255, dtype=np.uint8)

    step = 1 if path else interval
    for i in range(0, len(trajectory), step):
        # velocity vector
        if i + step < len(trajectory):
            cv2.arrowedLine(frame, (int(trajectory[i][1]), int(trajectory[i][2])),
                            (int(trajectory[i + step][1]),
                             int(trajectory[i + step][2])),
                            color, thickness=2)

    # draw bounding boxes of each region
    if regions is not None:
        for region in regions:
            region.draw(frame)

    if identifier is not None:
        cv2.putText(frame, str(identifier),
                    (int(trajectory[0][1]), int(trajectory[0][2] - 7)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, color, thickness=1)

    return frame


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
        xs.append(int(data_point[1]))
        ys.append(int(data_point[2]))

    # to avoid the line interligating both sides of the gap
    if with_gap:
        ts.insert(gap_interval[0] + 1, np.nan)
        xs.insert(gap_interval[0] + 1, np.nan)
        ys.insert(gap_interval[0] + 1, np.nan)

    return plot_variation(ts, xs, "X Position", "x value", "frame", interpolation_points), \
           plot_variation(ts, ys, "Y Position", "y value",
                          "frame", interpolation_points)


def show_trajectory(video_path, fish, estimated_trajectory, simulated_gaps,
                    record=None, frame_name="gap estimation"):
    # record and read settings
    cap = cv2.VideoCapture(video_path)
    out = None

    if record is not None:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(record, apiPreference=cv2.CAP_FFMPEG, fourcc=codec, fps=15,
                              frameSize=(
                              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        print("recording...")

    # set next frame to be read to the initial timestamp index of the trajectory
    t = estimated_trajectory[0][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)

    while (t < estimated_trajectory[-1][0]):
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
            centroid = (int(true_data_point[0]), int(true_data_point[1]))
            bounding_box_size = fish.bounding_boxes[t]
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
            cv2.rectangle(frame,
                          (int(centroid[0] - (bounding_box_size.width / 2)),
                           int(centroid[1] - bounding_box_size.height / 2)),
                          (int(centroid[0] + (bounding_box_size.width / 2)),
                           int(centroid[1] + bounding_box_size.height / 2)),
                          (0, 255, 0), 2)

        else:
            data_point = list(
                filter(lambda x: x[0] == t, estimated_trajectory)
            )
            if len(data_point) > 0:
                cv2.circle(
                    frame,
                    (int(data_point[0][1]), int(data_point[0][2])),
                    5, (0, 0, 255), -1
                )

        # show or write frame
        if record is None:
            cv2.imshow(frame_name, frame)
        elif record is not None:
            out.write(frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        print("done!")
        out.release()


def show_fish_trajectory(frame_description, path_video, fish, episodes):
    video_capture = cv2.VideoCapture(path_video)
    trajectory = fish.trajectory
    fish_episodes = [episode for episode in episodes
                     if episode.fish_id == fish.fish_id]
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, trajectory[0][0])

    while True:
        _, frame = video_capture.read()
        t = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        is_between_interesting = False
        for episode in fish_episodes:
            if t >= episode.t_initial and t <= episode.t_final:
                is_between_interesting = True
                break
        color = (0, 0, 255) if is_between_interesting else (0, 255, 0)

        bb = fish.bounding_boxes[t]
        centroid = fish.get_position(t)
        cv2.rectangle(frame,
                      (int(centroid[0] - bb.width / 2),
                       int(centroid[1] - bb.height / 2)),
                      (int(centroid[0] + bb.width / 2),
                       int(centroid[1] + bb.height / 2)),
                      color, thickness=2)
        cv2.imshow(frame_description, frame)

        if t == trajectory[-1][0]:
            key = cv2.waitKey(0)
            if key == ord('a'):
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, trajectory[0][0])
            elif key == ord('q'):
                break

        if cv2.waitKey(30) == ord('q'):
            break


def draw_fishes(frame, fishes, t):
    for fish in fishes:
        bounding_box_size = fish.bounding_boxes[t]
        centroid = fish.get_position(t)
        cv2.rectangle(frame,
                      (int(centroid[0] - (bounding_box_size.width / 2)),
                       int(centroid[1] - bounding_box_size.height / 2)),
                      (int(centroid[0] + (bounding_box_size.width / 2)),
                       int(centroid[1] + bounding_box_size.height / 2)),
                      (0, 255, 0), 2)
    return frame
