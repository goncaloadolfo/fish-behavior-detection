import sys

import cv2
import seaborn
import numpy as np
import matplotlib.pyplot as plt

from trajectory_features.trajectory_feature_extraction import (exponential_weights,
                                                               TrajectoryFeatureExtraction)
from feeding.utils import ErrorTracker, extract_feeding_warnings, _get_predicted_class, _get_true_class


def active_pixels(frame_t1, frame_t2, motion_thr, return_frame=False, region=None):
    # convert to gray scale
    frame_t1_gray_scale = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
    frame_t2_gray_scale = cv2.cvtColor(frame_t2, cv2.COLOR_BGR2GRAY)

    # calculate motion frame
    diff_frame = np.abs(frame_t2_gray_scale - frame_t1_gray_scale)
    _, motion_frame = cv2.threshold(diff_frame, motion_thr,
                                    255.0, cv2.THRESH_BINARY)

    # number of active pixels
    if region is not None:
        x_limits, y_limits = region
        nap = np.sum(motion_frame[y_limits[0]: y_limits[1],
                                  x_limits[0]: x_limits[1]] == 255)
    else:
        nap = np.sum(motion_frame == 255)

    # motion frame personalization
    if region is not None and return_frame:
        x_limits, y_limits = region
        frame_copy = frame_t1.copy()

        # convert motion frame to the right 3D shape
        motion_frame_3d = np.zeros((motion_frame.shape[0],
                                    motion_frame.shape[1],
                                    3), dtype=np.uint8)
        for i in range(motion_frame.shape[0]):
            for j in range(motion_frame.shape[1]):
                value = motion_frame[i][j]
                if value != 0:
                    motion_frame_3d[i, j, :] = [value, value, value]

        # replace motion region with motion frame results
        frame_copy[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1], :
                   ] = motion_frame_3d[y_limits[0]:y_limits[1], x_limits[0]:x_limits[1], :]

        # red box in this area
        cv2.rectangle(frame_copy, (x_limits[0], y_limits[0]),
                      (x_limits[1]-1, y_limits[1]-1), (0, 0, 255), 5)
        motion_frame = frame_copy

    return nap if not return_frame else nap, motion_frame


def calculate_motion_time_series(video_capture, motion_thr, region=None):
    time_series = []
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _, previous_frame = video_capture.read()
    t = 0

    # calculate number of active pixels at each frame
    while True:
        print(f"processing frame {t}/{total_frames}")
        _, current_frame = video_capture.read()
        if current_frame is None:
            break

        nr_active_pixels, _ = active_pixels(previous_frame, current_frame,
                                            motion_thr, return_frame=False, region=region)
        time_series.append(nr_active_pixels)

        previous_frame = current_frame
        t += 1

    return time_series


def tune_motion_thr(video_capture, motion_thrs):
    max_min_diffs = []

    for motion_thr in motion_thrs:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        motion_time_series = calculate_motion_time_series(video_capture,
                                                          motion_thr)
        motion_time_series = motion_time_series[::30]
        TrajectoryFeatureExtraction.exponential_sliding_average(motion_time_series, 24,
                                                                exponential_weights(24, 0.1,
                                                                                    forward_only=True))
        max_min_diffs.append(np.max(motion_time_series) -
                             np.min(motion_time_series))

    plt.figure()
    plt.title("Min-Max Difference Per Motion Threshold")
    plt.xlabel("motion threshold")
    plt.ylabel("min-max diff")
    plt.plot(motion_thrs, max_min_diffs, "-o")


def get_example_motion_frames(video_capture, motion_thr, feeding_warnings, region=None):
    # if no feeding moments were detected
    if len(feeding_warnings) == 0:
        return _calculate_diff_frame(video_capture, motion_thr, 0, region)

    # get timestamps for each of the phases
    first_feeding_warning = feeding_warnings[0]
    t_normal = int(first_feeding_warning[0]/2 * 30)
    t_feeding = int(
        (first_feeding_warning[1] + first_feeding_warning[0])/2 * 30)

    # motion frames
    return (_calculate_diff_frame(video_capture, motion_thr, t_normal, region),
            _calculate_diff_frame(video_capture, motion_thr, t_feeding, region))


def evaluate_motion_method(video_capture, motion_thr, feeding_thr, duration, ground_truth, region=None):
    confusion_matrix = np.zeros((2, 2), dtype=np.int)
    motion_time_series = calculate_motion_time_series(video_capture,
                                                      motion_thr, region)
    y = motion_time_series
    smoothed_y = y.copy()
    TrajectoryFeatureExtraction.exponential_sliding_average(smoothed_y, 30,
                                                            exponential_weights(30, 0.1,
                                                                                forward_only=True))
    feeding_warnings = extract_feeding_warnings(smoothed_y, feeding_thr,
                                                duration*30)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    error_tracker = ErrorTracker()
    for t in range(len(smoothed_y)):
        predicted_class = _get_predicted_class(t, feeding_warnings)
        true_class = _get_true_class(t, ground_truth)
        confusion_matrix[predicted_class][true_class] += 1

        if true_class != predicted_class:
            error_tracker.append_new_timestamp(t+1)

    return confusion_matrix, error_tracker


def analyze_training_results(video_capture, motion_thr, feeding_thr, duration,
                             show_frames=False, show_feeding_results=False, region=None):
    # motion timeseries
    motion_time_series = calculate_motion_time_series(video_capture,
                                                      motion_thr, region)

    # smooth timeseries
    y = motion_time_series[::30]
    smoothed_y = y.copy()
    TrajectoryFeatureExtraction.exponential_sliding_average(smoothed_y, 30,
                                                            exponential_weights(30, 0.1,
                                                                                forward_only=True))

    # extract feeding moments
    feeding_warnings = extract_feeding_warnings(smoothed_y, feeding_thr,
                                                duration)

    # plot motion timeseries - original and smoothed
    x = np.arange(1, video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) / 30.0
    plt.figure()
    plt.title("Motion Variance")
    plt.xlabel("t(s)")
    plt.ylabel("motion level")
    x = x[::30]

    plt.plot(x, y, label="original")
    plt.plot(x, smoothed_y, label="smooth")
    plt.legend()

    plt.figure()
    plt.title("Number of Active Pixels")
    plt.ylabel("counts")
    plt.xlabel("#active pixels")
    plt.hist(smoothed_y)

    # plot feeding results
    if show_feeding_results:
        plt.figure()
        plt.title("Feeding Periods")
        plt.ylabel("motion level")
        plt.xlabel("t(s)")
        plt.plot(x, smoothed_y)

        for t_initial, t_final in feeding_warnings:
            start_index = int(t_initial)
            end_index = int(t_final)
            plt.plot(x[start_index: end_index+1],
                     smoothed_y[start_index: end_index+1],
                     label="feeding period")
        plt.legend()

    # example motion frames
    if show_frames:
        try:
            normal_frame, feeding_frame = get_example_motion_frames(video_capture, motion_thr,
                                                                    feeding_warnings, region)
            cv2.imshow("normal motion frame", normal_frame)
            cv2.imshow("feeding motion frame", feeding_frame)
        # could not unpack
        except (TypeError, ValueError):
            normal_frame = get_example_motion_frames(video_capture, motion_thr,
                                                     feeding_warnings, region)
            cv2.imshow("normal motion frame", normal_frame)


def _calculate_diff_frame(video_capture, motion_thr, t, region):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, t)
    _, frame_t1 = video_capture.read()
    _, frame_t2 = video_capture.read()
    frame = active_pixels(frame_t1, frame_t2, motion_thr,
                          return_frame=True, region=region)[1]
    return cv2.resize(frame, (1280, 720))


# region imperative/tests code
def motion_thr_tunning_test():
    # impact of motion threshold on timeseries
    video_capture = cv2.VideoCapture("resources/videos/feeding-v1-trim.mp4")
    tune_motion_thr(video_capture, [40, 50, 60, 70])
    video_capture.release()
    plt.show()


def baseline_results_test(region, feeding_thr, show_results=True):
    # baseline results - timeseries + frame examples
    video_capture = cv2.VideoCapture("resources/videos/feeding-v1-trim.mp4")
    analyze_training_results(video_capture, 40, feeding_thr, 30,
                             show_frames=True, show_feeding_results=show_results,
                             region=region)
    video_capture.release()
    plt.show()


def evaluation_results_test(region, feeding_thr):
    # test videos
    video_test1 = cv2.VideoCapture("resources/videos/feeding-v1-trim2.mp4")
    video_test2 = cv2.VideoCapture("resources/videos/feeding-v2.mp4")

    # evaluate feeding results
    results, error_tracker = evaluate_motion_method(video_test1, 40,
                                                    feeding_thr, 20, [], region)
    results2, error_tracker2 = evaluate_motion_method(video_test2, 40, feeding_thr, 20,
                                                      [(0, 16500)], region)

    # release resources
    video_test1.release()
    video_test2.release()

    # plot results
    plt.figure()
    plt.title("Results Test Video 1")
    plt.xlabel("true class")
    plt.ylabel("predicted class")
    seaborn.heatmap(results.astype(np.int), annot=True, cmap="YlGnBu", fmt='d')
    error_tracker.draw_errors_timeline(1, np.sum(results),
                                       "Test Video 1 Errors Timeline")

    plt.figure()
    plt.title("Results Test Video 2")
    plt.xlabel("true class")
    plt.ylabel("predicted class")
    seaborn.heatmap(results2.astype(np.int),
                    annot=True, cmap="YlGnBu", fmt='d')
    error_tracker2.draw_errors_timeline(1, np.sum(results2),
                                        "Test Video 2 Errors Timeline", 16500)

    plt.show()
# endregion


if __name__ == "__main__":
    # motion_thr_tunning_test()
    # baseline_results_test(None, 625_000)
    baseline_results_test([(0, 1920), (440, 1080)], 394_000,
                          show_results=True)
    # evaluation_results_test(None, 625_000)
    # evaluation_results_test([(0, 1920), (440, 1080)], 394_000)
