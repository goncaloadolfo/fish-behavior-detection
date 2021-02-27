"""
Feature extraction from trajectory:
    - speed
    - acceleration
    - turning angle
    - curvature
    - centered distance function
    - pass by 
    - normalized bounding box
"""

from itertools import permutations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from regions_selector import read_regions
from visualization import draw_trajectory, show_trajectory, simple_line_plot


FEATURES_ORDER = ["speed", "acceleration", "turning-angle",
                  "curvature", "centered-distance", "normalized-bb"]


class TrajectoryFeatureExtraction():

    def __init__(self, regions):
        self.__regions = regions
        self.reset()

    def reset(self):
        region_ids = [region.region_id for region in self.__regions]
        self.__pass_by_description = list(permutations(region_ids, 1)) \
            + list(permutations(region_ids, 2))
        # results
        self.__speeds = []
        self.__accelerations = []
        self.__turning_angles = []
        self.__curvatures = []
        self.__centered_distances = []
        self.__pass_by = {k: 0 for k in self.__pass_by_description}
        self.__normalized_bounding_boxes = []

        # state
        self.__trajectory_centroid = None
        self.__last_region = None

    def set_trajectory(self, trajectory, bounding_boxes):
        self.__bounding_boxes = bounding_boxes
        self.__trajectory = trajectory
        self.__trajectory_centroid = np.mean(trajectory, axis=0)[1:]

    def extract_features(self):
        for i in range(len(self.__trajectory)):
            if i > 1:
                position = np.array(self.__trajectory[i][1:])
                # speed, acceleration, turning angle and curvature
                self.__motion_features(i)
                # geographic transitions
                self.__pass_by_features(position)
                # distance from trajectory center
                self.__centered_distance.append(
                    np.linalg.norm(position - self.__trajectory_centroid)
                )
                # normalized bounding box
                bounding_box = self.__bounding_boxes[i]
                self.__normalized_bounding_boxes.append(
                    bounding_box.width / bounding_box.height
                )

    def get_feature_vector(self):
        # list with all time series
        time_series_list = [self.__speeds, self.__accelerations, self.__turning_angles,
                            self.__curvatures, self.__centered_distances, self.__normalized_bounding_boxes]
        # results
        vector_description = []
        vector = []
        # extract statistics from each time series
        for i in range(len(FEATURES_ORDER)):
            description, statistical_features = TrajectoryFeatureExtraction.statistical_features(
                time_series_list, FEATURES_ORDER[i]
            )
            vector_description.extend(description)
            vector.extend(statistical_features)
        # add pass by features
        for regions_key, pass_by_value in self.__pass_by.items():
            description = f"region({regions_key[0]})" if len(regions_key) == 1 \
                else f"transition({regions_key[0]}-{regions_key[1]})"
            vector_description.append(description)
            vector.append(pass_by_value)
        return vector_description, vector

    @property
    def speed_time_series(self):
        return self.__speeds

    @property
    def acceleration_time_series(self):
        return self.__accelerations

    @property
    def turning_angle_time_series(self):
        return self.__turning_angles

    @property
    def curvature_time_series(self):
        return self.__curvatures

    @property
    def centered_distances(self):
        return self.__centered_distances

    @property
    def geographic_transitions(self):
        return self.__pass_by

    @property
    def normalized_bounding_boxes(self):
        return self.__normalized_bounding_boxes

    @static_method
    def statistical_features(time_series, feature_label):
        return [
            f"mean-{feature_label}",
            f"median-{feature_label}",
            f"std-{feature_label}",
            f"min-{feature_label}",
            f"max-{feature_label}",
            f"quartile1-{feature_label}",
            f"quartile3-{feature_label}",
            f"autocorr-{feature_label}",
        ],
        [
            np.mean(time_series),
            np.median(time_series),
            np.std(time_series),
            np.min(time_series),
            np.max(time_series),
            np.percentile(time_series, 25),
            np.percentile(time_series, 75),
            np.median([1] + [np.corrcoef(time_series[:-i], time_series[i:])
                             for i in range(1, len(time_series))]
                      )
        ]

    def __motion_features(self, i):
        p1 = self.__trajectory[i-2]
        p2 = self.__trajectory[i-1]
        current_point = self.__trajectory[i]
        # derivative in x
        dx_dt = [
            (p2[1] - p1[1]) / (p2[0] - p1[0]),
            (current_point[1] - p2[1]) / (current_point[0] - p2[0])
        ]
        # derivative in y
        dy_dt = [
            (p2[2] - p1[2]) / (p2[0] - p1[0]),
            (current_point[2] - p2[2]) / (current_point[0] - p2[0])
        ]
        # second derivatives
        dx2_dt2 = (dx_dt[1] - dx_dt[0]) / (current_point[0] - p2[0])
        dy2_dt2 = (dy_dt[1] - dy_dt[0]) / (current_point[0] - p2[0])
        # features
        self.__speeds.append(np.linalg.norm([dx_dt[1], dy_dt[1]]))
        self.__accelerations.append(np.linalg.norm([dx2_dt2, dy2_dt2]))
        self.__curvatures.append(
            abs(dx2_dt2 * dy_dt[1] - dx_dt[1] * dy2_dt2) /
            (dx_dt[1] ** 2 + dy_dt[1] ** 2)**1.5
        )
        # turning angle
        motion_vector = [dx_dt[1], dy_dt[1]]
        self.__turning_angles.append(
            np.rad2deg(np.arctan2(motion_vector[1], motion_vector[0]))
        )

    def __pass_by_features(self, position):
        # check in which region is the given position
        # and update pass by status
        for region in self.__regions:
            if position in region:
                if self.__last_region is None or self.__last_region == region.region_id:
                    self.__pass_by[(region.region_id,)] += 1
                else:
                    self.__pass_by[(self.__last_region, region.region_id)] += 1
                self.__last_region = region.region_id


def analyze_trajectory(video_path, regions_config_path, fish):
    # draw trajectory with the geographic regions
    regions = read_regions(regions_config_path)
    video_capture = cv2.VideoCapture(video_path)
    draw_trajectory(fish.trajectory,
                    (video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                     video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    (0, 0, 0),
                    regions)
    # calculate and draw feature plots
    features_extractor_obj = TrajectoryFeatureExtraction(regions)
    features_extractor_obj.set_trajectory(fish.trajectory, fish.bounding_boxes)
    features_extractor_obj.extract_features()
    time_series_list = [features_extractor_obj.speed_time_series,
                        features_extractor_obj.acceleration_time_series,
                        features_extractor_obj.turning_angle_time_series,
                        features_extractor_obj.curvature_time_series,
                        features_extractor_obj.centered_distances,
                        features_extractor_obj.normalized_bounding_boxes]
    for i in range(len(time_series_list)):
        plt.figure()
        simple_line_plot(plt.gca(),
                         range(len(fish.trajectory)),
                         time_series_list[i],
                         FEATURES_ORDER[i],
                         "value",
                         "t")
    plt.show()
    cv2.waitKey(1)
    cv2.destroyAllWindows()


def draw_time_series(*args, descriptions):
    # todo
    raise NotImplementedError()


def draw_region_transitions_information(pass_by_info):
    # todo
    raise NotImplementedError()


def trajectory_repeated_reading(video_path, fish):
    # todo
    raise NotImplementedError()
