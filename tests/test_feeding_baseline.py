"""
Set of tests to the feeding baseline module
"""

import random
import matplotlib.pyplot as plt
import logging
import cv2
import time

from feeding_baseline import feeding_baseline_logger, FeedingBaseline, detections_at_t, analyze_fiffb
from trajectories_reader import produce_trajectories
from interpolation import fill_gaps_linear
from visualization import draw_fishes

random.seed(0)
test_feeding_baseline_logger = logging.getLogger(__name__)
test_feeding_baseline_logger.setLevel(logging.DEBUG)
test_feeding_baseline_logger.addHandler(logging.StreamHandler())


def delaunay_test(vertical_range, logging_level=logging.DEBUG, show=True):
    feeding_baseline_logger.setLevel(logging_level)
    # generate positions
    positions = []
    for _ in range(7):
        positions.append((random.randint(1, 720), random.randint(
            vertical_range[0],
            vertical_range[1]))
        )
    feeding_baseline_obj = FeedingBaseline(40)
    feeding_baseline_obj.set_positions(positions)
    # calculate flocking index
    feeding_baseline_obj.predict()
    if show:
        test_feeding_baseline_logger.debug(
            f"positions:\n {feeding_baseline_obj.feeding_positions}"
        )
        test_feeding_baseline_logger.debug(
            f"outliers:\n {feeding_baseline_obj.outlier_positions}"
        )
        test_feeding_baseline_logger.debug(
            f"flocking index: {feeding_baseline_obj.flocking_index}"
        )
        # draw frame
        cv2.imshow(f"triangulation results {vertical_range}",
                   feeding_baseline_obj.results_frame(720, 480))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return feeding_baseline_obj.flocking_index


def delaunay_real_data_test():
    random.seed(4)
    feeding_baseline_logger.setLevel(logging.DEBUG)
    # read GT and video file
    fishes = produce_trajectories("../data/Dsc 0037-lowres_gt.txt").values()
    feeding_baseline_logger.debug(f"number of trajectories: {len(fishes)}")
    video_capture = cv2.VideoCapture("../data/Dsc 0037-lowres.m4v")
    # pre process trajectories
    for fish in fishes:
        fill_gaps_linear(fish.trajectory)
    # get a frame at a random timestamp
    random_t = random.randint(
        0,
        int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)-1)
    )
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, random_t)
    _, frame = video_capture.read()
    # get fishes in that frame
    fishes_at_t, positions = detections_at_t(fishes, random_t)
    # apply feeding logic
    feeding_baseline_obj = FeedingBaseline(mesh_thr=1)
    feeding_baseline_obj.set_positions(positions)
    feeding_baseline_obj.predict()
    # draw results
    test_feeding_baseline_logger.debug(
        f"positions:\n {feeding_baseline_obj.feeding_positions}"
    )
    test_feeding_baseline_logger.debug(
        f"outliers:\n {feeding_baseline_obj.outlier_positions}"
    )
    test_feeding_baseline_logger.debug(
        f"flocking index: {feeding_baseline_obj.flocking_index}"
    )
    cv2.imshow("mesh result", feeding_baseline_obj.results_frame(720, 480))
    cv2.imshow("test frame", draw_fishes(frame, fishes_at_t, random_t))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fiffb_analysis_test():
    feeding_baseline_logger.setLevel(logging.INFO)
    analyze_fiffb("../data/Dsc 0037-lowres_gt.txt", 0, 6400)
    plt.show()


def mesh_calculation_errors_test():
    # initial settings
    feeding_baseline_logger.setLevel(logging.INFO)
    error_counter = 0
    n_times = 100_000
    test_feeding_baseline_logger.debug(f"Number of iterations: {n_times}")
    start_time = time.process_time()
    # delaunay test n times
    for iteration in range(n_times):
        result = delaunay_test((1, 480),
                               logging_level=logging.INFO,
                               show=False)
        if result == -1:
            error_counter += 1
        # verbose
        if iteration % 5_000 == 0:
            test_feeding_baseline_logger.info(
                f"iteration {iteration}/{n_times}"
            )
    # print results
    test_feeding_baseline_logger.info(
        f"duration time (seconds): {time.process_time() - start_time}"
    )
    test_feeding_baseline_logger.info(
        f"failed to calculate mesh in {error_counter}/{n_times} iterations"
    )


delaunay_test((1, 480))  # triangular mesh
# delaunay_test((200, 240))  # line
# delaunay_real_data_test()
# fiffb_analysis_test()
# mesh_calculation_errors_test()
