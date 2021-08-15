"""
Module that implements the ability to detect feeding periods and fish with lack of interest.
(baseline version) 
"""

import logging
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

from pre_processing.interpolation import fill_gaps_linear
from trajectory_reader.trajectories_reader import read_detections
from trajectory_reader.visualization import draw_fishes, simple_line_plot
from trajectory_features.trajectory_feature_extraction import TrajectoryFeatureExtraction, exponential_weights

feeding_baseline_logger = logging.getLogger(__name__)
feeding_baseline_logger.addHandler(logging.StreamHandler())


class Triangle:

    def __init__(self, p1, p2, p3):
        self.__points = [p1, p2, p3]

    @property
    def perimeter(self):
        edge_lengths = list(
            map(lambda x: np.linalg.norm(x[0] - x[1]), self.edges)
        )
        return np.sum(edge_lengths)

    @property
    def points(self):
        return self.__points

    @property
    def edges(self):
        edges = []
        for i in range(3):
            for j in range(i + 1, 3):
                edges.append([self.__points[i], self.__points[j]])
        return edges


class FeedingBaseline:

    def __init__(self, mesh_thr):
        self.reset()
        self.__mesh_thr = mesh_thr

    def set_positions(self, positions):
        self.__positions = np.array(positions)

    def reset(self):
        self.__positions = []
        self.__outliers = []
        self.__center_of_mass = []
        self.__mesh = []
        self.__flocking_index = -1

    def predict(self):
        self.__detect_outliers()
        self.__calculate_flocking_index()

    def results_frame(self, frame_width, frame_height):
        frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)
        # center of mass
        cv2.circle(frame,
                   (int(self.__center_of_mass[0]),
                    int(self.__center_of_mass[1])),
                   radius=8,
                   color=(0, 255, 0),
                   thickness=-1)
        # draw outliers
        for outlier in self.__outliers:
            cv2.circle(frame,
                       (outlier[0], outlier[1]),
                       radius=4,
                       color=(0, 0, 255),
                       thickness=-1)
        # triangular mesh
        if isinstance(self.__mesh[0], Triangle):
            for triangle in self.__mesh:
                for point in triangle.points:
                    cv2.circle(frame,
                               (point[0], point[1]),
                               radius=6,
                               color=(255, 0, 0),
                               thickness=-1)
                for edge in triangle.edges:
                    cv2.line(frame,
                             pt1=(edge[0][0], edge[0][1]),
                             pt2=(edge[1][0], edge[1][1]),
                             color=(175, 0, 0),
                             thickness=4)
        # line mesh
        else:
            for i in range(len(self.__mesh) - 1):
                cv2.line(frame,
                         pt1=(self.__mesh[i][0], self.__mesh[i][1]),
                         pt2=(self.__mesh[i + 1][0], self.__mesh[i + 1][1]),
                         color=(175, 0, 0),
                         thickness=4)
        return frame

    @property
    def feeding_positions(self):
        return self.__positions

    @property
    def outlier_positions(self):
        return self.__outliers

    @property
    def flocking_index(self):
        return self.__flocking_index

    @property
    def mesh(self):
        return self.__mesh

    def __detect_outliers(self):
        # median centroid (feeding center of mass)
        self.__center_of_mass = np.median(self.__positions, axis=0)
        # distance from every point to the center of mass
        distances = [np.linalg.norm(position - self.__center_of_mass)
                     for position in self.__positions]
        
        if len(distances) > 0:
            # distances third quartile
            q3 = np.percentile(distances, 75)
            # detect outlier positions
            non_outlier_mask = np.ones((len(self.__positions),), dtype=bool)
            for i in range(len(self.__positions)):
                if distances[i] > 1.5 * q3:
                    self.__outliers.append(self.__positions[i])
                    non_outlier_mask[i] = False
            self.__positions = self.__positions[non_outlier_mask]

    def __triangular_mesh(self):
        # calculate triangles
        delaunay_result = Delaunay(
            self.__positions,
            qhull_options="QJ Qc")
        # get triangles and calculate flocking index
        self.__flocking_index = 0
        feeding_baseline_logger.debug(
            f"delaunay mesh:\n {delaunay_result.simplices}")
        for p1_i, p2_i, p3_i in delaunay_result.simplices:
            triangle = Triangle(
                self.__positions[p1_i],
                self.__positions[p2_i],
                self.__positions[p3_i])
            self.__mesh.append(triangle)
            self.__flocking_index += triangle.perimeter

    def __line_mesh(self):
        sorted_positions = sorted(self.__positions, key=lambda x: x[0])
        self.__mesh.append(sorted_positions[0])
        self.__flocking_index = 0
        for i in range(1, len(sorted_positions)):
            self.__mesh.append(sorted_positions[i])
            self.__flocking_index += np.linalg.norm(
                sorted_positions[i - 1] - sorted_positions[i]
            )

    def __calculate_flocking_index(self):
        if len(self.__positions) > 0:
            min_y = self.__positions.min(axis=0)[1]
            max_y = self.__positions.max(axis=0)[1]
            vertical_distance = max_y - min_y

            if vertical_distance < self.__mesh_thr or len(self.__positions) - len(self.__outliers) <= 3:
                self.__line_mesh()
            else:
                self.__triangular_mesh()


def detections_at_t(fishes, t):
    fishes_at_t = []
    positions = []
    for fish in fishes:
        position = fish.get_position(t)
        if position is not None:
            fishes_at_t.append(fish)
            positions.append(position)
    return fishes_at_t, positions


def analyze_fiffb(gt_path, initial_t, final_t):
    fishes = read_detections(gt_path).values()
    ts = range(initial_t, final_t + 1)
    fiffbs = []
    feeding_baseline_obj = FeedingBaseline(mesh_thr=50)
    for t in ts:
        # get positions of the fishes at that instant
        _, positions = detections_at_t(fishes, t)
        # calculate aggregation index
        feeding_baseline_obj.reset()
        feeding_baseline_obj.set_positions(positions)
        feeding_baseline_obj.predict()
        fiffbs.append(feeding_baseline_obj.flocking_index
                      if feeding_baseline_obj.flocking_index != -1
                      else np.nan
                      )
        if t % 500 == 0:
            feeding_baseline_logger.info(
                f"Calculating FIFFB frame {t}/{final_t}")
    feeding_baseline_logger.info(
        f"Calculating FIFFB frame {final_t}/{final_t}")
    # plot results
    plt.figure()
    fiffbs = fiffbs[::30]
    ts = ts[::30]
    TrajectoryFeatureExtraction.exponential_sliding_average(fiffbs, 24,
                                                            exponential_weights(24, 0.1,
                                                                                forward_only=True))
    simple_line_plot(plt.gca(), ts, fiffbs, "Aggregation Index", "FIFFB", "t")


def fiffb_time_series(fishes, initial_t, final_t):
    fiffbs = {}
    meshes = {}
    outliers = {}

    feeding_baseline_obj = FeedingBaseline(mesh_thr=0)
    ts = range(initial_t, final_t + 1)

    for t in ts:
        _, positions = detections_at_t(fishes, t)

        if len(positions) >= 3:
            feeding_baseline_obj.reset()
            feeding_baseline_obj.set_positions(positions)
            feeding_baseline_obj.predict()

            if feeding_baseline_obj.flocking_index != -1:
                fiffbs[t] = feeding_baseline_obj.flocking_index
                outliers[t] = np.array(
                    feeding_baseline_obj.outlier_positions).tolist()

                if isinstance(feeding_baseline_obj.mesh[0], Triangle):
                    meshes[t] = np.array([edge for triangle in feeding_baseline_obj.mesh
                                          for edge in triangle.edges]).tolist()
                else:
                    meshes[t] = np.array(feeding_baseline_obj.mesh).tolist()

    return fiffbs, meshes, outliers


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
        print(
            f"positions:\n {feeding_baseline_obj.feeding_positions}"
        )
        print(
            f"outliers:\n {feeding_baseline_obj.outlier_positions}"
        )
        print(
            f"flocking index: {feeding_baseline_obj.flocking_index}"
        )

        # draw frame
        cv2.imshow(f"triangulation results {vertical_range}",
                   feeding_baseline_obj.results_frame(720, 480))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return feeding_baseline_obj.flocking_index


def delaunay_real_data_test(video_path, detections_path, resolution):
    # random.seed(4)
    feeding_baseline_logger.setLevel(logging.DEBUG)

    # read GT and video file
    fishes = read_detections(detections_path).values()
    feeding_baseline_logger.debug(f"number of trajectories: {len(fishes)}")
    video_capture = cv2.VideoCapture(video_path)

    # pre process trajectories
    for fish in fishes:
        fill_gaps_linear(fish.trajectory, None)

    # get a frame at a random timestamp
    random_t = random.randint(
        0,
        int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
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
    print(
        f"positions:\n {feeding_baseline_obj.feeding_positions}"
    )
    print(
        f"outliers:\n {feeding_baseline_obj.outlier_positions}"
    )
    print(
        f"flocking index: {feeding_baseline_obj.flocking_index}"
    )
    
    mesh_frame = feeding_baseline_obj.results_frame(resolution[0], resolution[1])
    test_frame = draw_fishes(frame, fishes_at_t, random_t)
    
    resized_mesh_frame = cv2.resize(mesh_frame, (720, 480))
    resized_test_frame = cv2.resize(test_frame, (720, 480))
    
    cv2.imshow("mesh result", resized_mesh_frame)
    cv2.imshow("test frame", resized_test_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fiffb_analysis_test(detection_path):
    feeding_baseline_logger.setLevel(logging.INFO)
    analyze_fiffb(detection_path, 0, 6400)
    plt.show()


def mesh_calculation_errors_test():
    # initial settings
    feeding_baseline_logger.setLevel(logging.INFO)
    error_counter = 0
    n_times = 100_000
    print(f"Number of iterations: {n_times}")
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
            print(
                f"iteration {iteration}/{n_times}"
            )

    # print results
    print(
        f"duration time (seconds): {time.process_time() - start_time}"
    )
    print(
        f"failed to calculate mesh in {error_counter}/{n_times} iterations"
    )


def main():
    # random.seed(0)
    delaunay_test((1, 480))  # triangular mesh
    # delaunay_test((200, 240))  # line
    # delaunay_real_data_test("resources/videos/v37.m4v",
    #                         "resources/detections/detections-v37.txt", (720, 480))
    # delaunay_real_data_test("resources/videos/GP011844_Trim.mp4",
    #                         "resources/detections/GP011844_Trim_gt.txt", (1920, 1440))
    # delaunay_real_data_test("resources/videos/v29.m4v",
    #                         "resources/detections/detections-v29-sharks-mantas.txt", (720, 480))
    # fiffb_analysis_test("resources/detections/detections-v37.txt")
    # fiffb_analysis_test("resources/detections/GP011844_Trim_gt.txt")
    # fiffb_analysis_test("resources/detections/detections-v29-sharks-mantas.txt")
    # mesh_calculation_errors_test()


if __name__ == "__main__":
    main()    
