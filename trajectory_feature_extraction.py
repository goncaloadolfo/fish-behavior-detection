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
from multiprocessing import Process

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from regions_selector import read_regions
from visualization import draw_trajectory, show_trajectory, simple_line_plot, \
    simple_bar_chart


FEATURES_ORDER = ["speed", "acceleration", "turning-angle",
                  "curvature", "centered-distance", "normalized-bb"]
fe_logger = logging.getLogger(__name__)
fe_logger.addHandler(logging.StreamHandler())
fe_logger.setLevel(logging.INFO)


class TrajectoryFeatureExtraction():

    def __init__(self, regions, calculation_frequency):
        self.__regions = regions
        self.__calculation_frequency = calculation_frequency
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
        self.__calculation_positions = []

    def set_trajectory(self, trajectory, bounding_boxes):
        self.__bounding_boxes = bounding_boxes
        self.__trajectory = trajectory
        self.__trajectory_centroid = np.mean(trajectory, axis=0)[1:]

    def extract_features(self):
        for i in range(len(self.__trajectory)):
            current_position = self.__trajectory[i]
            if len(self.__calculation_positions) == 0 or \
                current_position[0] - self.__calculation_positions[-1][0] >= self.__calculation_frequency:
                    self.__calculation_positions.append(current_position) 
            if len(self.__calculation_positions) > 2:
                position = np.array(self.__trajectory[i][1:])
                # speed, acceleration, turning angle and curvature
                self.__motion_features()
                # geographic transitions
                self.__pass_by_features(position)
                # distance from trajectory center
                self.__centered_distances.append(
                    np.linalg.norm(position - self.__trajectory_centroid)
                )
                # normalized bounding box
                try:
                    bounding_box = self.__bounding_boxes[self.__trajectory[i][0]]
                    self.__normalized_bounding_boxes.append(
                        bounding_box.width / bounding_box.height
                    )
                except KeyError:
                    continue
                # update state
            
                

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
                time_series_list[i], FEATURES_ORDER[i]
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

    @property
    def pass_by_info(self):
        return self.__pass_by

    @staticmethod
    def statistical_features(time_series, feature_label):
        return ([
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
            np.median([1.0] + [np.corrcoef(time_series[:-i], time_series[i:])[0, 1]
                               for i in range(1, len(time_series) - 1)
                               if not np.isnan(np.corrcoef(time_series[:-i], time_series[i:])[0, 1])
                               ]
                      )
        ])

    def __motion_features(self):
        p1 = self.__calculation_positions[-3]
        p2 = self.__calculation_positions[-2]
        current_point = self.__calculation_positions[-1]
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
        self.__accelerations.append(
            np.linalg.norm([dx_dt[1], dy_dt[1]]) -
            np.linalg.norm([dx_dt[0], dy_dt[0]])
        )
        curvature_numerator = abs(dx2_dt2 * dy_dt[1] - dx_dt[1] * dy2_dt2)
        curvature_denominator = (dx_dt[1] ** 2 + dy_dt[1] ** 2)**1.5
        self.__curvatures.append(
            curvature_numerator / curvature_denominator if curvature_denominator != 0 else 0
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


def analyze_trajectory(video_path, regions, fish, frequency):
    # calculate and draw feature plots
    features_extractor_obj = TrajectoryFeatureExtraction(regions, frequency)
    features_extractor_obj.set_trajectory(fish.trajectory, fish.bounding_boxes)
    features_extractor_obj.extract_features()
    time_series_list = [features_extractor_obj.speed_time_series,
                        features_extractor_obj.acceleration_time_series,
                        features_extractor_obj.turning_angle_time_series,
                        features_extractor_obj.curvature_time_series,
                        features_extractor_obj.centered_distances,
                        features_extractor_obj.normalized_bounding_boxes]
    draw_time_series(*time_series_list, descriptions=FEATURES_ORDER)
    draw_region_transitions_information(features_extractor_obj.pass_by_info)
    # draw trajectory and regions
    trajectory_repeated_reading(video_path, regions, fish)
    # features vector
    description, vector = features_extractor_obj.get_feature_vector()
    fe_logger.info(f"\ndescription: \n{description}")
    fe_logger.info(f"\nvector: \n{vector}")
    fe_logger.info(f"Dimensions: {len(vector)}")
    plt.show()
    cv2.destroyAllWindows()


def draw_time_series(*args, descriptions):
    # draw each of time series received as argument
    for i in range(len(args)):
        plt.figure()
        simple_line_plot(plt.gca(),
                         range(len(args[i])),
                         args[i],
                         descriptions[i],
                         "value",
                         "t")


def draw_region_transitions_information(pass_by_info):
    regions = list(pass_by_info.keys())
    values = list(pass_by_info.values())
    plt.figure() 
    # regions histogram 
    simple_bar_chart(plt.gca(), range(len(values)), values,
                     "Region Transitions", "Number frames", "Region")
    plt.gca().set_xticks(range(len(values)))
    plt.gca().set_xticklabels(regions)


def trajectory_repeated_reading(video_path, regions, fish):
    def repeating_callback(event, x, y, flags, param):
        nonlocal visualization_process
        if event == cv2.EVENT_LBUTTONDOWN:
            visualization_process.terminate()
            visualization_process = Process(target=show_trajectory,
                                            args=(video_path, fish, fish.trajectory, [], None, "trajectory"))
            visualization_process.start()
    # trajectory and regions frame
    video_capture = cv2.VideoCapture(video_path)
    draw_trajectory(fish.trajectory,
                    (int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                    (0, 0, 0),
                    regions)
    # repeated visualization
    visualization_process = Process(target=show_trajectory,
                                    args=(video_path, fish, fish.trajectory, [], None, "trajectory"))
    cv2.namedWindow("trajectory")
    cv2.setMouseCallback("trajectory", repeating_callback)
    visualization_process.start()


def frequency_analysis(fish, regions, frequencies):
    # for each frequency
    for frequency in frequencies:
        # calculate features
        fe_obj = TrajectoryFeatureExtraction(regions, frequency)
        fe_obj.set_trajectory(fish.trajectory, fish.bounding_boxes)        
        fe_obj.extract_features()
        # draw speed
        draw_time_series(fe_obj.speed_time_series, descriptions=[f"Speed frequency={frequency}"])
    plt.show()
    