import sys

import cv2
import seaborn
import numpy as np
import matplotlib.pyplot as plt

from trajectory_features.trajectory_feature_extraction import (exponential_weights,
                                                               TrajectoryFeatureExtraction)


def active_pixels(frame_t1, frame_t2, motion_thr, return_frame=False):
    frame_t1_gray_scale = cv2.cvtColor(frame_t1, cv2.COLOR_BGR2GRAY)
    frame_t2_gray_scale = cv2.cvtColor(frame_t2, cv2.COLOR_BGR2GRAY)
    diff_frame = np.abs(frame_t2_gray_scale - frame_t1_gray_scale)
    _, motion_frame = cv2.threshold(diff_frame, motion_thr,
                                    255.0, cv2.THRESH_BINARY)
    return np.sum(motion_frame == 255) if not return_frame else np.sum(motion_frame == 255), motion_frame


def calculate_motion_time_series(video_capture, motion_thr):
    time_series = []
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    _, previous_frame = video_capture.read()
    t = 0

    while True:
        print(f"processing frame {t}/{total_frames}")
        _, current_frame = video_capture.read()
        if current_frame is None:
            break

        nr_active_pixels, _ = active_pixels(previous_frame,
                                            current_frame, motion_thr)
        time_series.append(nr_active_pixels)

        previous_frame = current_frame
        t += 1

    return time_series


def extract_feeding_warnings(motion_time_series, feeding_thr, duration):
    feeding_warnings = []
    feeding_flag = False
    feeding_duration = 0

    for t in range(len(motion_time_series)):
        nr_active_pixels = motion_time_series[t]

        if nr_active_pixels >= feeding_thr:
            feeding_duration += 1
            feeding_flag = True

        elif feeding_flag and feeding_duration >= duration:
            feeding_warnings.append((t - 1 - feeding_duration, t - 1))
            feeding_duration = 0
            feeding_flag = False

        else:
            feeding_duration = 0
            feeding_flag = False

        if t == len(motion_time_series) - 1 and feeding_flag:
            feeding_warnings.append((t - feeding_duration, t))

    return feeding_warnings


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


def get_example_motion_frames(video_capture, motion_thr, feeding_warnings):
    first_feeding_warning = feeding_warnings[0]
    t_normal = int(first_feeding_warning[0]/2 * 30)
    t_feeding = int(
        (first_feeding_warning[1] + first_feeding_warning[0])/2 * 30)
    return (_calculate_diff_frame(video_capture, motion_thr, t_normal),
            _calculate_diff_frame(video_capture, motion_thr, t_feeding))


def evaluate_motion_method(video_capture, motion_thr, feeding_thr, duration, ground_truth):
    confusion_matrix = np.zeros((2, 2), dtype=np.int)
    motion_time_series = calculate_motion_time_series(video_capture,
                                                      motion_thr)
    y = motion_time_series
    smoothed_y = y.copy()
    TrajectoryFeatureExtraction.exponential_sliding_average(smoothed_y, 30,
                                                            exponential_weights(30, 0.1,
                                                                                forward_only=True))
    feeding_warnings = extract_feeding_warnings(smoothed_y, feeding_thr,
                                                duration*30)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    for t in range(len(smoothed_y)):
        predicted_class = _get_predicted_class(t, feeding_warnings)
        true_class = _get_true_class(t, ground_truth)
        confusion_matrix[predicted_class][true_class] += 1

    return confusion_matrix


def analyze_training_results(video_capture, motion_thr, feeding_thr, duration,
                             show_frames=False, show_feeding_results=False):
    motion_time_series = calculate_motion_time_series(video_capture,
                                                      motion_thr)
    y = motion_time_series[::30]
    smoothed_y = y.copy()
    TrajectoryFeatureExtraction.exponential_sliding_average(smoothed_y, 30,
                                                            exponential_weights(30, 0.1,
                                                                                forward_only=True))
    feeding_warnings = extract_feeding_warnings(smoothed_y, feeding_thr,
                                                duration)

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

    if show_frames:
        normal_motion_frame, feeding_motion_frame = get_example_motion_frames(video_capture,
                                                                              motion_thr, feeding_warnings)
        cv2.imshow("normal motion frame", normal_motion_frame)
        cv2.imshow("feeding motion frame", feeding_motion_frame)


def _get_predicted_class(frame, predicted_warnings):
    for t_initial, t_final in predicted_warnings:
        if t_initial <= frame <= t_final:
            return 1

        if frame < t_initial:
            break

    return 0


def _get_true_class(frame, true_feeding_period):
    for t_initial, t_final in true_feeding_period:
        if t_initial <= frame <= t_final:
            return 1

        if frame < t_initial:
            break

    return 0


def _calculate_diff_frame(video_capture, motion_thr, t):
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, t)
    _, frame_t1 = video_capture.read()
    _, frame_t2 = video_capture.read()
    frame = active_pixels(frame_t1, frame_t2, motion_thr, return_frame=True)[1]
    return cv2.resize(frame, (1280, 720))


if __name__ == "__main__":
    video_capture = cv2.VideoCapture("resources/videos/feeding-v1-trim.mp4")
    video_test1 = cv2.VideoCapture("resources/videos/feeding-v1-trim2.mp4")
    video_test2 = cv2.VideoCapture("resources/videos/feeding-v2.mp4")

    # tune_motion_thr(video_capture, [40, 50, 60, 70])

    # video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # analyze_training_results(video_capture, 40, 625_000, 20,
    #                          show_frames=True, show_feeding_results=True)

    results = evaluate_motion_method(video_test1, 40, 650_000, 20, [])
    results2 = evaluate_motion_method(video_test2, 40, 650_000, 20,
                                      [(0, 12550)])

    plt.figure()
    plt.title("Results Test Video 1")
    plt.xlabel("true class")
    plt.ylabel("predicted class")
    seaborn.heatmap(results, annot=True, cmap="YlGnBu")

    plt.figure()
    plt.title("Results Test Video 2")
    plt.xlabel("true class")
    plt.ylabel("predicted class")
    seaborn.heatmap(results2, annot=True, cmap="YlGnBu")

    plt.show()
    video_capture.release()
    video_test1.release()
    video_test2.release()
