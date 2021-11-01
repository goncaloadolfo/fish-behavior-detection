import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from trajectory_features.trajectory_feature_extraction import TrajectoryFeatureExtraction, exponential_weights
from feeding.utils import ErrorTracker, extract_feeding_warnings, _get_predicted_class, _get_true_class


def define_optical_points(region, nr_points):
    # grid coordinates
    x_values = np.linspace(region[0][0], region[0][1], nr_points[0] + 2)[1:-1]
    y_values = np.linspace(region[1][0], region[1][1], nr_points[1] + 2)[1:-1]

    # put in the right shape
    points_coords = [[[x, y]] for x in x_values for y in y_values]
    return np.array(points_coords).astype(np.float32)


def calculate_motion_vectors(frame1, frame2, coordinates, resolution):
    # resize
    frame1_resized = cv2.resize(frame1, resolution)
    frame2_resized = cv2.resize(frame2, resolution)

    # gray frames
    frame1_gray = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)

    # optical flow
    predicted_points, found, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray,
                                                            coordinates, None)

    # calculate motion vectors
    vectors = np.zeros((len(coordinates), 2))
    for i in range(len(coordinates)):
        if found[i] == 1:
            p0 = coordinates[i]
            p1 = predicted_points[i]
            vectors[i] = p1 - p0
        else:
            vectors[i] = [np.nan, np.nan]

    return vectors, found


def calculate_average_magnitude(vectors):
    # calculate magnitudes of each vector
    magnitudes = np.linalg.norm(vectors, axis=1)
    magnitudes_rmv_nans = magnitudes[np.logical_not(np.isnan(magnitudes))]
    return magnitudes, np.average(magnitudes_rmv_nans)


def draw_vectors_frame(frame, resolution, vectors, coordinates, found, scale):
    resized_frame = cv2.resize(frame, resolution)

    for i in range(len(coordinates)):
        # if it wasnt possible to predict point
        if not found[i]:
            continue

        # start and end points
        p0 = coordinates[i].ravel().astype(np.int)
        p1 = (p0 + vectors[i].ravel()*scale).astype(np.int)

        # draw vector
        resized_frame = cv2.arrowedLine(resized_frame, tuple(p0),
                                        tuple(p1), (0, 255, 0), 2)

    return resized_frame


def average_magnitude_timeseries(video_capture, region, nr_points, resolution):
    # initial state
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    optical_points = define_optical_points(region, nr_points)
    _, previous_frame = video_capture.read()
    time_series = np.empty((total_frames-1,))

    # process video
    for i in itertools.count():
        print("processing frame ", i)
        # read frame
        _, current_frame = video_capture.read()
        if current_frame is None:
            break

        # optical flow
        vectors, _ = calculate_motion_vectors(previous_frame, current_frame,
                                              optical_points, resolution)
        _, average_magnitude = calculate_average_magnitude(vectors)

        # update status
        time_series[i] = average_magnitude
        previous_frame = current_frame

    return time_series


def plot_magnitudes_histogram(magnitudes):
    # plot the histogram of the magnitudes
    plt.figure()
    plt.title("Magnitudes Histogram")
    plt.xlabel("magnitude value")
    plt.ylabel("#samples")
    plt.hist(magnitudes)


def plot_magnitudes_time_series(magnitudes_time_series, feeding_period=None,
                                smooth=False):
    magnitudes_time_series = magnitudes_time_series[::30]
    magnitudes_time_series = magnitudes_time_series.tolist()
    if smooth:
        TrajectoryFeatureExtraction.exponential_sliding_average(magnitudes_time_series, 30,
                                                                exponential_weights(30, 0.1,
                                                                                    forward_only=True))
    # plot time series
    plt.figure()
    plt.title("Average Magnitude")
    plt.xlabel("t")
    plt.ylabel("average magnitude")
    plt.plot(magnitudes_time_series, label="normal")

    # feeding period
    start_index = int(feeding_period[0]/30)
    end_index = int(feeding_period[1]/30)
    plt.plot(range(start_index, end_index),
             magnitudes_time_series[start_index: end_index],
             label="feeding")
    plt.legend()


def optical_flow_video(src_video, resolution, optical_points):
    # input and output sources
    input_video = cv2.VideoCapture(src_video)
    input_fps = int(input_video.get(cv2.CAP_PROP_FPS))
    output_path = src_video.split('.')[0] + "-of.mp4"
    output_video = cv2.VideoWriter(output_path, apiPreference=cv2.CAP_FFMPEG,
                                   fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps=input_fps, frameSize=resolution)

    _, previous_frame = input_video.read()
    for i in itertools.count():
        # read frame
        _, current_frame = input_video.read()
        if current_frame is None:
            break
        print("processing frame ", i)

        # optical flow
        motion_vectors, found = calculate_motion_vectors(previous_frame, current_frame,
                                                         optical_points, resolution)
        vectors_frame = draw_vectors_frame(current_frame, resolution,
                                           motion_vectors, optical_points, found, 7)
        output_video.write(vectors_frame)
        previous_frame = current_frame

    # release resources
    input_video.release()
    output_video.release()


def evaluate_optical_flow(video_capture, resolution, region, nr_points,
                          feeding_threshold, min_duration, ground_truth):
    # calculate optical flow timeseries
    average_magnitude_ts = average_magnitude_timeseries(video_capture, region,
                                                        nr_points, resolution)

    # subsample and smooth
    average_magnitude_ts = average_magnitude_ts.tolist()
    TrajectoryFeatureExtraction.exponential_sliding_average(average_magnitude_ts, 30,
                                                            exponential_weights(30, 0.1,
                                                                                forward_only=True))

    # find feeding moments
    feeding_moments = extract_feeding_warnings(average_magnitude_ts,
                                               feeding_threshold, min_duration)

    # evaluation
    error_tracker = ErrorTracker()
    confusion_matrix = np.zeros((2, 2))

    for t in range(len(average_magnitude_ts)):
        predicted_class = _get_predicted_class(t, feeding_moments)
        true_class = _get_true_class(t, ground_truth)
        confusion_matrix[predicted_class][true_class] += 1

        if predicted_class != true_class:
            error_tracker.append_new_timestamp(t)

    return confusion_matrix, error_tracker


def two_frames_test(resolution, region):
    # general settings
    optical_points = define_optical_points(region, (30, 15))
    print("optical points")
    print(optical_points)
    print("shape: ", optical_points.shape)

    # read frames
    video_capture = cv2.VideoCapture("resources/videos/feeding-v1-trim.mp4")
    _, frame1 = video_capture.read()
    _, frame2 = video_capture.read()

    # optical flow
    motion_vectors, found = calculate_motion_vectors(
        frame1, frame2, optical_points, resolution
    )
    magnitudes, average_magnitude = calculate_average_magnitude(motion_vectors)

    print("---")
    print("motion vectors:")
    print(motion_vectors)

    print("---")
    print("magnitudes: ", magnitudes)
    print("average magnitude: ", average_magnitude)

    # show vectors frame
    frame = draw_vectors_frame(frame1, resolution, motion_vectors,
                               optical_points, found, 7)
    cv2.rectangle(frame, (region[0][0], region[1][0]),
                  (region[0][1]-1, region[1][1]-1), (0, 0, 255), 5)
    plot_magnitudes_histogram(magnitudes)
    cv2.imshow("optical flow", frame)
    plt.show()


# region imperative/tests code
def time_series_test(resolution, region):
    # settings
    video_capture = cv2.VideoCapture("resources/videos/feeding-v1-trim.mp4")
    nr_points = (30, 15)

    # time series
    magnitudes_time_series = average_magnitude_timeseries(video_capture, region,
                                                          nr_points, resolution)
    plot_magnitudes_time_series(
        magnitudes_time_series, (5400, 9067), smooth=True)
    plt.show()


def optical_flow_video_test(resolution, region):
    # general settings
    input_video_path = "resources/videos/feeding-v1-trim.mp4"
    optical_points = define_optical_points(
        region,
        (30, 15)
    )

    # save optical flow video
    optical_flow_video(input_video_path, resolution, optical_points)


def optical_flow_evaluation_test(resolution, region):
    # general settings
    test_video1 = cv2.VideoCapture("resources/videos/feeding-v1-trim2.mp4")
    test_video2 = cv2.VideoCapture("resources/videos/feeding-v2.mp4")

    gt_video_1 = []
    gt_video_2 = [(0, 16500)]
    nr_points = (30, 15)

    # results
    confusion_matrix, error_tracker = evaluate_optical_flow(test_video1, resolution,
                                                            region, nr_points, 2.8, 30, gt_video_1)
    confusion_matrix2, error_tracker2 = evaluate_optical_flow(test_video2, resolution,
                                                              region, nr_points, 2.8, 30, gt_video_2)

    # visualization
    plt.figure()
    plt.title("Test video 1 results")
    seaborn.heatmap(confusion_matrix.astype(np.int),
                    annot=True, cmap="YlGnBu", fmt='d')
    error_tracker.draw_errors_timeline(0, int(test_video1.get(cv2.CAP_PROP_POS_FRAMES)),
                                       "Video test 1 errors")

    plt.figure()
    plt.title("Test video 2 results")
    seaborn.heatmap(confusion_matrix2.astype(np.int),
                    annot=True, cmap="YlGnBu", fmt='d')
    error_tracker2.draw_errors_timeline(0, int(test_video2.get(cv2.CAP_PROP_POS_FRAMES)),
                                        "Video test 2 errors", 16500)

    test_video1.release()
    test_video2.release()
    plt.show()
# endregion


def main():
    two_frames_test((1280, 720), [(0, 1280), (0, 720)])
    # two_frames_test((1280, 720), [(0, 1280), (290, 720)])
    # time_series_test((1280, 720), [(0, 1280), (0, 720)])
    # time_series_test((1280, 720), [(0, 1280), (290, 720)])
    # optical_flow_video_test((1280, 720), [(0, 1280), (0, 720)])
    # optical_flow_evaluation_test((1280, 720), [(0, 1280), (0, 720)])
    # optical_flow_evaluation_test((1280, 720), [(0, 1280), (290, 720)])


if __name__ == "__main__":
    main()
