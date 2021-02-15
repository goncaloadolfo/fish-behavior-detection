"""
Module that implements the ability to detect feeding periods and fish with lack of interest.
(baseline version) 
"""

import logging
import numpy as np
import cv2
from random import randint
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.reset()

    def set_positions(self, positions):
        self.__positions = np.array(positions)

    def reset(self):
        self.__positions = []
        self.__outliers = []
        self.__triangles = []
        self.__flocking_index = -1

    def predict(self):
        # self.__detect_outliers()
        self.__calculate_flocking_index()

    def results_frame(self, frame_width, frame_height):
        frame = np.full((frame_width, frame_height, 3), 255, dtype=np.uint8)
        # draw outliers
        for outlier in self.__outliers:
            cv2.circle(frame,
                       (outlier[0], outlier[1]),
                       radius=2,
                       color=(0, 0, 255),
                       thickness=-1)
        # draw triangles:
        for triangle in self.__triangles:
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
        median_centroid = np.median(self.__positions, axis=0)
        # distance from every point to the center of mass
        distances = [np.linalg.norm(position - median_centroid)
                     for position in self.__positions]
        # first and third quartiles
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        # detect outlier positions
        non_outlier_mask = np.ones((len(self.__positions),), dtype=int)
        for i in range(len(self.__positions)):
            if distances[i] > 1.5 * q3:
                self.__outliers.append(self.__positions[i])
                non_outlier_mask[i] = 0
        self.__positions = self.__positions[non_outlier_mask]

    def __calculate_flocking_index(self):
        try:
            # calculate triangles
            delaunay_result = Delaunay(
                self.__positions,
                qhull_options="QJ Qc")
            # get triangles and calculate flocking index
            self.__flocking_index = 0
            for p1_i, p2_i, p3_i in delaunay_result.simplices:
                triangle = Triangle(
                    self.__positions[p1_i],
                    self.__positions[p2_i],
                    self.__positions[p3_i])
                self.__triangles.append(triangle)
                self.__flocking_index += triangle.perimeter
        except QhullError as e:
            logger.warning(
                f"not able to calculate delaunay triangulation: {e}")


def delaunay_test():
    # generate positions
    positions = []
    for _ in range(7):
        positions.append((randint(1, 720), randint(1, 480)))
    feeding_baseline_obj = FeedingBaseline()
    feeding_baseline_obj.set_positions(positions)
    # calculate flocking index
    feeding_baseline_obj.predict()
    logger.debug(f"positions: {positions}")
    logger.debug(f"outliers: {feeding_baseline_obj.outlier_positions}")
    logger.debug(f"flocking index: {feeding_baseline_obj.flocking_index}")
    # draw frame
    cv2.imshow("triangulation results",
               feeding_baseline_obj.results_frame(480, 720))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    delaunay_test()
