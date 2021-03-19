"""
Module that implements the ability to detect feeding periods and fish with lack of interest.
(baseline version) 
"""

import logging
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

from trajectory_reader.trajectories_reader import read_detections
from trajectory_reader.visualization import simple_line_plot

feeding_baseline_logger = logging.getLogger(__name__)
feeding_baseline_logger.addHandler(logging.StreamHandler())


class Triangle():

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
            for j in range(i+1, 3):
                edges.append([self.__points[i], self.__points[j]])
        return edges


class FeedingBaseline():

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
                   radius=4,
                   color=(0, 255, 0),
                   thickness=-1)
        # draw outliers
        for outlier in self.__outliers:
            cv2.circle(frame,
                       (outlier[0], outlier[1]),
                       radius=2,
                       color=(0, 0, 255),
                       thickness=-1)
        # triangular mesh
        if isinstance(self.__mesh[0], Triangle):
            for triangle in self.__mesh:
                for point in triangle.points:
                    cv2.circle(frame,
                               (point[0], point[1]),
                               radius=3,
                               color=(255, 0, 0),
                               thickness=-1)
                for edge in triangle.edges:
                    cv2.line(frame,
                             pt1=(edge[0][0], edge[0][1]),
                             pt2=(edge[1][0], edge[1][1]),
                             color=(175, 0, 0),
                             thickness=2)
        # line mesh
        else:
            for i in range(len(self.__mesh) - 1):
                cv2.line(frame,
                         pt1=(self.__mesh[i][0], self.__mesh[i][1]),
                         pt2=(self.__mesh[i+1][0], self.__mesh[i+1][1]),
                         color=(175, 0, 0),
                         thickness=2)
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

    def __detect_outliers(self):
        # median centroid (feeding center of mass)
        self.__center_of_mass = np.median(self.__positions, axis=0)
        # distance from every point to the center of mass
        distances = [np.linalg.norm(position - self.__center_of_mass)
                     for position in self.__positions]
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
        try:
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
        except QhullError as e:
            feeding_baseline_logger.warning(
                f"not able to calculate delaunay triangulation: {e}")

    def __line_mesh(self):
        sorted_positions = sorted(self.__positions, key=lambda x: x[0])
        self.__mesh.append(sorted_positions[0])
        self.__flocking_index = 0
        for i in range(1, len(sorted_positions)):
            self.__mesh.append(sorted_positions[i])
            self.__flocking_index += np.linalg.norm(
                sorted_positions[i-1] - sorted_positions[i]
            )

    def __calculate_flocking_index(self):
        min_y = self.__positions.min(axis=0)[1]
        max_y = self.__positions.max(axis=0)[1]
        vertical_distance = max_y - min_y
        if vertical_distance < self.__mesh_thr:
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
            positions.append((position[1], position[2]))
    return fishes_at_t, positions


def analyze_fiffb(gt_path, initial_t, final_t):
    fishes = read_detections(gt_path).values()
    ts = range(initial_t, final_t+1)
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
    simple_line_plot(plt.gca(), ts, fiffbs, "Aggregation Index", "FIFFB", "t")
