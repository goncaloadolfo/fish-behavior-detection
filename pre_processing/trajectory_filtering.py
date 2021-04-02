import random

import cv2
from labeling.regions_selector import read_regions
import matplotlib.pyplot as plt
import numpy as np
from trajectory_features.trajectory_feature_extraction import TrajectoryFeatureExtraction, extract_features
from trajectory_reader.trajectories_reader import read_fishes
from trajectory_reader.visualization import draw_trajectory, simple_line_plot

from pre_processing.interpolation import fill_gaps_linear


def smooth_positions(fish, weights):
    half_window = int((len(weights) - 1) / 2)
    trajectory = fish.trajectory
    positions = fish.positions

    for i, data_point in enumerate(trajectory):
        t = data_point[0]
        xs, ys = _get_edge_positions(fish, t, half_window)

        new_x = int(np.average(xs, weights=weights))
        new_y = int(np.average(ys, weights=weights))

        fish.positions[t] = (new_x, new_y)
        trajectory[i][1] = new_x
        trajectory[i][2] = new_y


def exponential_weights(window_size, alpha):
    half_window_weights = [
        alpha * (1 - alpha)**n for n in range(0, int(window_size/2))
    ]
    reversed_weights = half_window_weights.copy()
    reversed_weights.reverse()
    return reversed_weights + [0] + half_window_weights


def _get_edge_positions(fish, t, half_window):
    xs = []
    ys = []
    initial_position = fish.get_position(fish.trajectory[0][0])
    final_position = fish.get_position(fish.trajectory[-1][0])

    for i in range(t-half_window, t+half_window+1):
        edge_point = fish.get_position(i)
        if edge_point is None:
            edge_point = initial_position if i < t else final_position
        xs.append(edge_point[0])
        ys.append(edge_point[1])

    return xs, ys


def _plot_positions(axs, trajectory, ts, filtered):
    title_label = "original" if not filtered else "filtered"
    row = 0 if not filtered else 1

    xs = [data_point[1] for data_point in trajectory]
    ys = [data_point[2] for data_point in trajectory]

    simple_line_plot(axs[row][0], ts, xs,
                     f"{title_label.capitalize()} x values", "x position", "t")
    simple_line_plot(axs[row][1], ts, ys,
                     f"{title_label.capitalize()} y values", "y position", "t")


def _plot_time_series(fe_objs, features_of_interest):
    # todo: create a figure (subplots), each time series of interest will have 2 plots
    # todo: left side will be the features gathered from original trajectory
    # todo: and right side the features gathered from the filtered one
    raise NotImplementedError()


def positions_filtering_test(window_size, alpha, features_of_interest):
    # todo: refractor - function to get a random fish
    fishes = list(read_fishes("resources/detections/v29-fishes.json"))
    fishes.sort(lambda x: x.fish_id)
    example_fish = random.choice(fishes)
    regions = read_regions("resources/regions-example.json")
    fill_gaps_linear(example_fish.trajectory, example_fish)

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(
        f"Positions filtering window_size={window_size} and alpha={alpha}"
    )

    ts = [data_point[0] for data_point in example_fish.trajectory]
    fe_obj_original = extract_features(example_fish, regions)
    frame_original_trajectory = draw_trajectory(
        example_fish.trajectory, frame_size=(480, 720), color=(0, 255, 0)
    )
    _plot_positions(axs, example_fish.trajectory, ts, filtered=False)

    if window_size % 2 != 0:
        window_size += 1
    weights = exponential_weights(window_size, alpha)
    smooth_positions(example_fish, weights)
    fe_obj_filtered = extract_features(example_fish, regions)
    frame_filtered_trajectory = draw_trajectory(
        example_fish.trajectory, frame_size=(480, 720), color=(0, 255, 0)
    )
    _plot_positions(axs, example_fish.trajectory, ts, filtered=True)

    _plot_time_series((fe_obj_original, fe_obj_filtered))
    cv2.imshow("Original trajectory", frame_filtered_trajectory)
    cv2.imshow("Filtered trajectory", frame_filtered_trajectory)


if __name__ == '__main__':
    positions_filtering_test(window_size=12, alpha=0.1,
                             features_of_interest=[
                                 TrajectoryFeatureExtraction.SPEEDS_ATR_NAME,
                                 TrajectoryFeatureExtraction.TAS_ATR_NAME
                             ]
                             )
    plt.show()
    cv2.destroyAllWindows()
