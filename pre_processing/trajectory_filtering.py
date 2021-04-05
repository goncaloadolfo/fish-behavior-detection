import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trajectory_features.trajectory_feature_extraction as fe_module
from labeling.regions_selector import read_regions
from trajectory_reader.trajectories_reader import get_random_fish
from trajectory_reader.visualization import draw_trajectory, simple_line_plot

from pre_processing.interpolation import fill_gaps_linear


def identify_descontinuity_points(fish, geo_regions, distance_thr, speed_thr, angle_thr):
    fe_obj = fe_module.extract_features(fish, geo_regions, 1, 6, 0.5)
    speed_time_series = getattr(
        fe_obj, fe_module.TrajectoryFeatureExtraction.SPEEDS_ATR_NAME
    )
    angle_time_series = getattr(
        fe_obj, fe_module.TrajectoryFeatureExtraction.TAS_ATR_NAME
    )

    speed_time_series = _set_timestamps(
        fish.trajectory[0][0], len(fish.trajectory), speed_time_series
    )
    angle_time_series = _set_timestamps(
        fish.trajectory[0][0], len(fish.trajectory), angle_time_series
    )

    descontinuity_ts = douglass_peucker(fish.trajectory, distance_thr, "distance") + \
        douglass_peucker(speed_time_series, speed_thr, "speed") + \
        douglass_peucker(angle_time_series, angle_thr, "direction")
    _sort_moments(descontinuity_ts)

    return descontinuity_ts


def _sort_moments(moments):
    moments.sort(key=lambda x: x[0])
    i = 1
    while(i != len(moments)):
        if moments[i][0] == moments[i-1][0]:
            del moments[i]
        else:
            i += 1


def _set_timestamps(initial_t, trajectory_length, time_series):
    time_series_copy = time_series.copy()
    len_diff = trajectory_length - len(time_series)
    return enumerate(time_series_copy, start=initial_t + len_diff)


def douglass_peucker(sequence, epsilon, tag=None):
    initial_point = _data_point_array(sequence[0])
    final_point = _data_point_array(sequence[-1])

    data_points_array = np.array(
        [_data_point_array(point) for point in sequence]
    )
    distances = np.linalg.norm(
        np.cross(final_point - initial_point,
                 initial_point - data_points_array)
    ) / np.linalg.norm(final_point - initial_point)

    max_distance = np.max(distances)
    max_distance_index = np.argmax(distances)

    if max_distance > epsilon:
        results_segment1 = douglass_peucker(
            sequence[1:max_distance_index], epsilon, tag
        )
        results_segment2 = douglass_peucker(
            sequence[max_distance_index:], epsilon, tag
        )
        return results_segment1 + results_segment2

    else:
        return [(sequence[0][0], tag), (sequence[-1][0], tag)]


def _data_point_array(data_point):
    return np.array([data_point[0], data_point[1]]) \
        if len(data_point) == 2 \
        else np.array([data_point[1], data_point[2]])


def smooth_positions(fish, weights):
    fish_copy = copy.deepcopy(fish)
    half_window = int((len(weights) - 1) / 2)
    trajectory = fish.trajectory
    positions = fish.positions

    for i, data_point in enumerate(trajectory):
        t = data_point[0]
        xs, ys = _get_edge_positions(fish_copy, t, half_window)

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


def _plot_time_series(fe_objs, descriptions, features_of_interest, trajectory_ts):
    n_rows = len(features_of_interest) / 2 \
        if len(features_of_interest) % 2 == 0 \
        else len(features_of_interest) / 2 + 1
    n_cols = 2 if len(features_of_interest) > 1 else len(features_of_interest)
    fig, axs = plt.subplots(ncols=n_cols, nrows=int(n_rows))

    for i, feature in enumerate(features_of_interest):
        row = int(i/2)
        col = int(i % 2)
        ax = axs[row, col] if n_rows > 1 else axs[col]
        for j, fe_obj in enumerate(fe_objs):
            time_series = getattr(fe_obj, feature)

            if len(trajectory_ts) > len(time_series):
                trajectory_ts = trajectory_ts[
                    len(trajectory_ts) - len(time_series):
                ]

            simple_line_plot(ax, trajectory_ts, time_series,
                             f"{feature}", "value", "t", label=descriptions[j])
        ax.legend()


def positions_filtering_test(window_size, alpha, features_of_interest):
    example_fish = get_random_fish("resources/detections/v29-fishes.json")
    regions = read_regions("resources/regions-example.json")
    fill_gaps_linear(example_fish.trajectory, example_fish)

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(
        f"Positions filtering window_size={window_size} and alpha={alpha}"
    )

    ts = [data_point[0] for data_point in example_fish.trajectory]
    fe_obj_original = fe_module.extract_features(
        example_fish, regions, 1, 12, 0.01
    )
    frame_original_trajectory = draw_trajectory(
        example_fish.trajectory, frame_size=(480, 720), color=(0, 255, 0)
    )
    _plot_positions(axs, example_fish.trajectory, ts, filtered=False)

    if window_size % 2 != 0:
        window_size += 1
    weights = exponential_weights(window_size, alpha)
    smooth_positions(example_fish, weights)
    fe_obj_filtered = fe_module.extract_features(
        example_fish, regions, 1, 12, 0.01
    )
    frame_filtered_trajectory = draw_trajectory(
        example_fish.trajectory, frame_size=(480, 720), color=(0, 255, 0)
    )
    _plot_positions(axs, example_fish.trajectory, ts, filtered=True)

    _plot_time_series((fe_obj_original, fe_obj_filtered),
                      ("original trajectory", "filtered trajectory"),
                      features_of_interest,
                      ts)
    cv2.imshow("Original trajectory", frame_filtered_trajectory)
    cv2.imshow("Filtered trajectory", frame_filtered_trajectory)


if __name__ == '__main__':
    positions_filtering_test(window_size=12, alpha=0.01,
                             features_of_interest=(
                                 fe_module.TrajectoryFeatureExtraction.SPEEDS_ATR_NAME,
                                 fe_module.TrajectoryFeatureExtraction.CURVATURES_ATR_NAME,
                                 fe_module.TrajectoryFeatureExtraction.TAS_ATR_NAME
                             )
                             )
    plt.show()
    cv2.destroyAllWindows()
