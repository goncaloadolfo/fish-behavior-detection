import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import trajectory_features.trajectory_feature_extraction as fe_module
import trajectory_reader.trajectories_reader as tr
from labeling.regions_selector import read_regions
from trajectory_reader.visualization import draw_trajectory, simple_line_plot

from pre_processing.interpolation import fill_gaps_linear

DISTANCE_TAG = "distance"
SPEED_TAG = "speed"
ANGLE_TAG = "angle"


def segment_trajectory(fish, descontinuity_points):
    original_trajectory = fish.trajectory
    original_bbs = copy.deepcopy(original_bbs)
    original_positions = fish.positions

    if len(descontinuity_points > 2):
        fishes = []
        for i in range(len(descontinuity_points)-1):
            segment_t0 = descontinuity_points[i][0]
            segment_tf = descontinuity_points[i+1][0]

            segment_trajectory = [data_point for data_point in original_trajectory
                                  if data_point[0] >= segment_t0 and data_point[0] < segment_tf]
            segment_bbs = {t: bb for t, bb in original_bbs.items()
                           if t >= segment_t0 and t < segment_tf}
            segment_positions = {t: position for t, position in original_bbs.items()
                                 if t >= segment_t0 and t < segment_tf}

            if i == len(descontinuity_points) - 1:
                segment_trajectory.append(original_trajectory[-1])
                segment_bbs[segment_tf] = original_bbs[segment_tf]
                segment_positions[segment_tf] = original_positions[segment_tf]

            fishes.append(tr.Fish(fish.fish_id), segment_trajectory,
                          segment_bbs, segment_positions)
        return fishes

    else:
        return fish


def play_trajectory_segments(video_path, fish, descontinuity_points, write_path=None):
    video_capture = cv2.VideoCapture(video_path)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, fish.trajectory[0][0])

    if write_path is not None:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(write_path,
                                 apiPreference=cv2.CAP_FFMPEG,
                                 fourcc=codec, fps=15,
                                 frameSize=(
                                     int(video_capture.get(
                                         cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(video_capture.get(
                                         cv2.CAP_PROP_FRAME_HEIGHT))
                                 )
                                 )

    color = np.random.randint(0, 256, 3)
    color = (int(color[0]), int(color[1]), int(color[2]))
    colors = [color]

    current_tag = descontinuity_points[0][1]
    descontinuities = dict(descontinuity_points)

    for i in range(len(fish.trajectory)):
        _, frame = video_capture.read()
        t = fish.trajectory[i][0]
        previous_point = None if i == 0 \
            else (int(fish.trajectory[i-1][1]), int(fish.trajectory[i-1][2]))
        current_point = (
            int(fish.trajectory[i][1]), int(fish.trajectory[i][2])
        )

        if t in descontinuities:
            color = np.random.randint(0, 256, 3)
            color = (int(color[0]), int(color[1]), int(color[2]))
            colors.append(color)
            current_tag = descontinuities[t]

        cv2.circle(frame, current_point, 5, color, -1)
        if t in fish.bounding_boxes:
            bb = fish.bounding_boxes[t]
            half_width = int(bb.width / 2)
            half_height = int(bb.height / 2)
            cv2.rectangle(frame,
                          (current_point[0] - half_width,
                           current_point[1] - half_height),
                          (current_point[0] + half_width,
                           current_point[1] + half_height),
                          color, 2)
        cv2.putText(frame, current_tag,
                    (current_point[0], current_point[1] - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, color)

        draw_path(frame, fish.trajectory, t, descontinuities, colors)
        cv2.imshow("trajectory segments", frame)
        if write_path is not None:
            writer.write(frame)
        cv2.waitKey(36)
    video_capture.release()
    if write_path is not None:
        writer.release()


def draw_path(frame, trajectory, current_t, descontinuities, colors):
    segment_index = 0
    color = colors[segment_index]

    for i in range(len(trajectory)):
        if i == 0:
            continue
        elif trajectory[i-1][0] == current_t:
            break

        if trajectory[i][0] in descontinuities:
            segment_index += 1
            color = colors[segment_index]

        cv2.line(frame, (int(trajectory[i-1][1]), int(trajectory[i-1][2])),
                 (int(trajectory[i][1]), int(trajectory[i][2])), color, 2)


def smooth_positions_dp(fish, geo_regions, distance_thr, speed_thr, angle_thr):
    descontinuity_points, _, _ = identify_descontinuity_points(
        fish, geo_regions, distance_thr, speed_thr, angle_thr
    )
    _sort_moments(descontinuity_points)
    resample_trajectory(fish, descontinuity_points)
    return descontinuity_points


def resample_trajectory(fish, descontinuity_points):
    descontinuity_ts = [t for t, _ in descontinuity_points]
    i = 0
    while(i < len(fish.trajectory)):
        if fish.trajectory[i][0] not in descontinuity_ts:
            del fish.positions[fish.trajectory[i][0]]
            del fish.trajectory[i]
        else:
            i += 1
    fill_gaps_linear(fish.trajectory, fish)


def identify_descontinuity_points(fish, geo_regions, distance_thr, speed_thr, angle_thr):
    fe_obj = fe_module.extract_features(fish, geo_regions, 1, 24, 0.01)
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

    return (douglass_peucker(fish.trajectory, distance_thr, DISTANCE_TAG) +
            douglass_peucker(speed_time_series, speed_thr, SPEED_TAG) +
            douglass_peucker(angle_time_series, angle_thr, ANGLE_TAG),
            speed_time_series, angle_time_series)


def _sort_moments(moments):
    moments.sort(key=lambda x: x[0])
    i = 1
    while(i != len(moments)):
        if moments[i][0] == moments[i-1][0]:
            del moments[i]
        else:
            i += 1


def _set_timestamps(initial_t, trajectory_length, time_series):
    start_t = initial_t + (trajectory_length - len(time_series))
    return [(start_t + i, time_series[i]) for i in range(len(time_series))]


def douglass_peucker(sequence, epsilon, tag=None):
    if len(sequence) == 0:
        return []

    initial_point = _data_point_array(sequence[0])
    final_point = _data_point_array(sequence[-1])

    data_points_array = np.array(
        [_data_point_array(point) for point in sequence]
    )
    distances = np.abs(np.cross(final_point - initial_point, initial_point - data_points_array)
                       / np.linalg.norm(final_point - initial_point))

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

        new_x = np.average(xs, weights=weights)
        new_y = np.average(ys, weights=weights)

        fish.positions[t] = (new_x, new_y)
        trajectory[i][1] = new_x
        trajectory[i][2] = new_y


def exponential_weights(window_size, alpha, forward_only=False):
    if forward_only:
        return [alpha * (1 - alpha)**n for n in range(0, window_size)]

    else:
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


def _plot_descontinuities(time_series, time_series_description, descontinuity_points_list, thresholds):
    ts = [t for t, _ in time_series]
    time_series_values = [value for _, value in time_series]

    plt.figure()
    simple_line_plot(plt.gca(), ts, time_series_values,
                     time_series_description, "value", "t", label="original")

    for i in range(len(descontinuity_points_list)):
        descontinuity_points_list[i].sort(key=lambda x: x[0])
        descontinuity_ts = [t for t, _ in descontinuity_points_list[i]]
        key_values = [
            time_series_values[ts.index(t)] for t in descontinuity_ts
        ]

        simple_line_plot(plt.gca(), descontinuity_ts, key_values,
                         time_series_description, "value", "t",
                         label=f"douglass peucker output ep={thresholds[i]}")
        print(
            f"Number of key points ({time_series_description}, ep={thresholds[i]}): {len(descontinuity_ts)}"
        )

    plt.gca().legend()


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


def douglass_peucker_tuning(distance_thrs, speed_thrs, angle_thrs, seed):
    example_fish = tr.get_random_fish(
        "resources/detections/v29-fishes.json", seed)
    regions = read_regions("resources/regions-example.json")
    fill_gaps_linear(example_fish.trajectory, example_fish)

    _dp_feature_tuning(example_fish, regions, distance_thrs, DISTANCE_TAG)
    _dp_feature_tuning(example_fish, regions, speed_thrs, SPEED_TAG)
    _dp_feature_tuning(example_fish, regions, angle_thrs, ANGLE_TAG)


def _dp_feature_tuning(fish, regions, thrs, feature_tag):
    descontinuity_points_list = []

    for thr in thrs:
        distance_thr = 0 if feature_tag != DISTANCE_TAG else thr
        speed_thr = 0 if feature_tag != SPEED_TAG else thr
        angle_thr = 0 if feature_tag != ANGLE_TAG else thr

        descontinuity_points, speed_time_series, angle_time_series = identify_descontinuity_points(
            fish, regions, distance_thr, speed_thr, angle_thr
        )
        descontinuity_points_list.append(
            [point for point in descontinuity_points if point[1] == feature_tag]
        )

    if feature_tag == DISTANCE_TAG:
        x_time_series = [(t, x) for t, x, y in fish.trajectory]
        y_time_series = [(t, y) for t, x, y in fish.trajectory]
        _plot_descontinuities(x_time_series, "x position",
                              descontinuity_points_list, thrs)
        _plot_descontinuities(y_time_series, "y position",
                              descontinuity_points_list, thrs)

    else:
        time_series = speed_time_series if feature_tag == SPEED_TAG else angle_time_series
        _plot_descontinuities(time_series, feature_tag,
                              descontinuity_points_list, thrs)


def smooth_positions_dp_test(fish, regions, distance_thr, speed_thr, angle_thr):
    original_trajectory = draw_trajectory(
        fish.trajectory, (480, 720), (0, 255, 0)
    )
    descontinuity_points = smooth_positions_dp(
        fish, regions, distance_thr, speed_thr, angle_thr
    )
    filtered_trajectory = draw_trajectory(
        fish.trajectory, (480, 720), (0, 255, 0)
    )
    print(
        f"Trajectory points: {len(descontinuity_points)}/{len(fish.trajectory)}")
    cv2.imshow("original path", original_trajectory)
    cv2.imshow("filtered path", filtered_trajectory)
    return descontinuity_points


def douglass_peucker_test(distance_thr, speed_thr, angle_thr, seed):
    example_fish = tr.get_random_fish(
        "resources/detections/v29-fishes.json", seed)
    regions = read_regions("resources/regions-example.json")
    fill_gaps_linear(example_fish.trajectory, example_fish)

    _dp_feature_tuning(example_fish, regions, [distance_thr], DISTANCE_TAG)
    _dp_feature_tuning(example_fish, regions, [speed_thr], SPEED_TAG)
    _dp_feature_tuning(example_fish, regions, [angle_thr], ANGLE_TAG)

    descontinuity_points = smooth_positions_dp_test(example_fish, regions,
                                                    distance_thr, speed_thr, angle_thr)
    play_trajectory_segments("resources/videos/v29.m4v",
                             example_fish, descontinuity_points, "resources/segmentation-example.mp4")


def positions_filtering_test(window_size, alpha, features_of_interest, seed):
    example_fish = tr.get_random_fish(
        "resources/detections/v29-fishes.json", seed)
    regions = read_regions("resources/regions-example.json")
    fill_gaps_linear(example_fish.trajectory, example_fish)

    fig, axs = plt.subplots(ncols=2, nrows=2)
    fig.suptitle(
        f"Positions filtering window_size={window_size} and alpha={alpha}"
    )

    ts = [data_point[0] for data_point in example_fish.trajectory]
    fe_obj_original = fe_module.extract_features(
        example_fish, regions, 1, 24, 0.01
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
        example_fish, regions, 1, 24, 0.01
    )
    frame_filtered_trajectory = draw_trajectory(
        example_fish.trajectory, frame_size=(480, 720), color=(0, 255, 0)
    )
    _plot_positions(axs, example_fish.trajectory, ts, filtered=True)

    _plot_time_series((fe_obj_original, fe_obj_filtered),
                      ("original trajectory", "filtered trajectory"),
                      features_of_interest,
                      ts)
    cv2.imshow("Original trajectory", frame_original_trajectory)
    cv2.imshow("Filtered trajectory", frame_filtered_trajectory)


if __name__ == '__main__':
    # positions_filtering_test(window_size=24, alpha=0.01,
    #                          features_of_interest=(
    #                              fe_module.TrajectoryFeatureExtraction.SPEEDS_ATR_NAME,
    #                              fe_module.TrajectoryFeatureExtraction.CURVATURES_ATR_NAME,
    #                              fe_module.TrajectoryFeatureExtraction.TAS_ATR_NAME,
    #                              fe_module.TrajectoryFeatureExtraction.CDS_ATR_NAME
    #                          ), seed=1
    #                          )

    douglass_peucker_test(10, 0.75, 20, 1)

    # douglass_peucker_tuning([10, 20, 30], [0.5, 1, 2], [10, 30, 50], 1)

    plt.show()
    cv2.destroyAllWindows()
