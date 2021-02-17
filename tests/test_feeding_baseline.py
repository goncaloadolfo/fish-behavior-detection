"""
Set of tests to the feeding baseline module
"""

import random
import matplotlib.pyplot as plt

from feeding_baseline import *
from trajectories_reader import *
from interpolation import *
from visualization import *


def delaunay_test(vertical_range):
    # generate positions
    positions = []
    for _ in range(7):
        positions.append((randint(1, 720), randint(
            vertical_range[0],
            vertical_range[1]))
        )
    feeding_baseline_obj = FeedingBaseline(70)
    feeding_baseline_obj.set_positions(positions)
    # calculate flocking index
    feeding_baseline_obj.predict()
    logger.debug(f"positions:\n {feeding_baseline_obj.feeding_positions}")
    logger.debug(f"outliers:\n {feeding_baseline_obj.outlier_positions}")
    logger.debug(f"flocking index: {feeding_baseline_obj.flocking_index}")
    # draw frame
    cv2.imshow(f"triangulation results {vertical_range}",
               feeding_baseline_obj.results_frame(720, 480))


def delaunay_real_data_test():
    # read GT and video file
    fishes = produce_trajectories("../data/Dsc 0037-lowres_gt.txt").values()
    logger.debug(f"number of trajectories: {len(fishes)}")
    video_capture = cv2.VideoCapture("../data/Dsc 0037-lowres.m4v")
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
    feeding_baseline_obj = FeedingBaseline(mesh_thr=50)
    feeding_baseline_obj.set_positions(positions)
    feeding_baseline_obj.predict()
    # draw results
    logger.debug(f"positions:\n {feeding_baseline_obj.feeding_positions}")
    logger.debug(f"outliers:\n {feeding_baseline_obj.outlier_positions}")
    logger.debug(f"flocking index: {feeding_baseline_obj.flocking_index}")
    cv2.imshow("mesh result", feeding_baseline_obj.results_frame(720, 480))
    cv2.imshow("test frame", draw_fishes(frame, fishes_at_t, random_t))


def fiffb_analysis_test():
    analyze_fiffb("../data/Dsc 0037-lowres_gt.txt", 0, 6400)
    plt.show()


logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
delaunay_test((1, 480))  # triangular mesh
# delaunay_test((200, 270))  # line mesh
# delaunay_real_data_test()
cv2.waitKey(0)
cv2.destroyAllWindows()
# logger.setLevel(logging.INFO)
# fiffb_analysis_test()
