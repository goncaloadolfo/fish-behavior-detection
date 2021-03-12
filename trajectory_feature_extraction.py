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
from functools import reduce

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt

from regions_selector import read_regions
from visualization import draw_trajectory, show_trajectory, simple_line_plot, \
    simple_bar_chart
from trajectories_reader import read_detections, read_fishes
from trajectory_labeling import read_species_gt
from interpolation import fill_gaps_linear


FEATURES_ORDER = ["speed", "acceleration", "turning-angle",
                  "curvature", "centered-distance", "normalized-bb"]
fe_logger = logging.getLogger(__name__)
fe_logger.addHandler(logging.StreamHandler())
fe_logger.setLevel(logging.INFO)


class TrajectoryFeatureExtraction():

    def __init__(self, regions, calculation_period, sliding_window, alpha):
        self.__regions = regions
        self.__calculation_frequency = calculation_period
        self.__sliding_window = sliding_window
        self.__sliding_weights = [alpha * (1 -alpha)**n for n in range(0, sliding_window)]
        self.reset()

    def reset(self):
        region_ids = [region.region_id for region in self.__regions]
        self.__pass_by_description = list(permutations(region_ids, 1)) + list(permutations(region_ids, 2))
        
        # results
        self.__speeds = []
        self.__accelerations = []
        self.__turning_angles = []
        self.__curvatures = []
        self.__centered_distances = []
        self.__pass_by = {k: 0 for k in self.__pass_by_description}
        self.__normalized_bounding_boxes = []
        self.__time_series_list = [self.__speeds, self.__accelerations, self.__turning_angles,
                            self.__curvatures, self.__centered_distances, self.__normalized_bounding_boxes]
        

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
        
        # smooth time series
        if self.__sliding_window > 1:
            for time_series in self.__time_series_list:
                TrajectoryFeatureExtraction.exponencial_sliding_average(time_series, 
                                                                        self.__sliding_window, 
                                                                        self.__sliding_weights
                                                                        )

    def get_feature_vector(self):
        # results
        vector_description = []
        vector = []
        
        # extract statistics from each time series
        for i in range(len(FEATURES_ORDER)):
            description, statistical_features = TrajectoryFeatureExtraction.statistical_features(
                self.__time_series_list[i], FEATURES_ORDER[i]
            )
            vector_description.extend(description)
            vector.extend(statistical_features)
        
        # add pass by features
        total_calculated_frames = len(self.__speeds)
        total_transitions = 0
        for key, value in self.__pass_by.items():
            if len(key) == 2:  # transition
                total_transitions += value
                                
        for regions_key, pass_by_value in self.__pass_by.items():
            # region
            if len(regions_key) == 1:
                vector_description.append(f"region({regions_key[0]})")
                normalized_value = pass_by_value/total_calculated_frames
                vector.append(normalized_value)
                self.__pass_by[regions_key] = normalized_value
                
            # transition
            else:
                vector_description.append(
                    f"transition({regions_key[0]}-{regions_key[1]})")
                transitions_ratio = 0 if total_transitions == 0 else pass_by_value / total_transitions
                vector.append(transitions_ratio)
                
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
    
    @staticmethod
    def exponencial_sliding_average(time_series, sliding_window, weights):
        # expand edge
        time_series_copy = time_series + [time_series[-1]] * sliding_window  

        # calculate new values
        for i in range(len(time_series)): 
            time_series[i] = np.average(time_series_copy[i+1 : i+1+sliding_window], weights=weights)           

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


def analyze_trajectory(video_path, regions, fish, calculation_period, sliding_window, alpha):
    # calculate and draw feature plots
    features_extractor_obj = TrajectoryFeatureExtraction(regions, calculation_period, sliding_window, alpha)
    features_extractor_obj.set_trajectory(fish.trajectory, fish.bounding_boxes)
    features_extractor_obj.extract_features()
    time_series_list = [features_extractor_obj.speed_time_series,
                        features_extractor_obj.acceleration_time_series,
                        features_extractor_obj.turning_angle_time_series,
                        features_extractor_obj.curvature_time_series,
                        features_extractor_obj.centered_distances,
                        features_extractor_obj.normalized_bounding_boxes]
    description, vector = features_extractor_obj.get_feature_vector()
    draw_time_series(*time_series_list, descriptions=FEATURES_ORDER)
    draw_regions_information(features_extractor_obj.pass_by_info)
    
    # draw trajectory and regions
    trajectory_repeated_reading(video_path, regions, fish)
    
    # features vector
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


def draw_regions_information(pass_by_info):
    regions = []
    values = []
    for region, value in pass_by_info.items():
        # time spent in each region
        if len(region) == 1:
            regions.append(region)
            values.append(value)
        # transitions
        else:
            fe_logger.info(f"Number of transitions {region}: {value}")
    plt.figure()
    # regions histogram
    simple_bar_chart(plt.gca(), range(len(values)), values,
                     "Regions", "Time (%)", "Region")
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


def frequency_analysis(fish, regions, calculation_periods):
    # for each frequency
    for frequency in calculation_periods:
        # calculate features
        fe_obj = TrajectoryFeatureExtraction(regions, frequency, sliding_window=1, alpha=None)
        fe_obj.set_trajectory(fish.trajectory, fish.bounding_boxes)
        fe_obj.extract_features()
        
        # draw speed
        draw_time_series(fe_obj.speed_time_series, descriptions=[
                         f"Speed frequency={frequency}"])
    plt.show()


def build_dataset(fishes_file_path, species_gt_path, regions_path, output_path=None):
    """
    - Feature extraction from each trajectory
    - Write dataset to file if output_path is defined      
    """
    # read detections, regions, and species gt
    fishes = read_fishes(fishes_file_path)
    regions = read_regions(regions_path)
    species_gt = read_species_gt(species_gt_path)

    # dataset 
    samples = []
    gt = []
    
    for fish in fishes:
        # pre-processing
        fill_gaps_linear(fish.trajectory)

        # feature extraction
        fe_obj = TrajectoryFeatureExtraction(regions, calculation_period=1, sliding_window=12, alpha=0.6)
        fe_obj.set_trajectory(fish.trajectory, fish.bounding_boxes)
        fe_obj.extract_features()
        features_description, sample = fe_obj.get_feature_vector()

        # update data structures
        samples.append(sample)
        gt.append(species_gt[fish.fish_id])

    # save to file
    if output_path is not None and len(samples) > 0:
        with open(output_path, 'w') as f:
            f.write(','.join(features_description) + ",species\n")
            for i in range(len(samples)): 
                f.write(','.join(np.array(samples[i]).astype(np.str)) + f",{gt[i]}")
    
    return (samples, gt, features_description)


def read_dataset(dataset_file_path):
    with open(dataset_file_path, 'r') as f:
        # fields description
        description = f.readline().split(',')
        
        # dataset
        samples = []
        gt = []
        
        while (True):
            line = f.readline()
            
            # end of the file
            if line is None or line == '\n' or line == '':
                break                
            
            fields = line.split(',')
            # sample 
            samples.append(np.array(fields[:-1]).astype(np.float))
            
            # ground truth 
            gt.append(fields[-1])
    
    return (samples, gt, description[:-1])


# region experiences
def analyze_trajectory_demo():
    fishes = read_fishes("resources/detections/v29-fishes.json")
    regions = read_regions("resources/regions-example.json")
    analyze_trajectory("resources/videos/v29.m4v", 
                       regions, 
                       fishes.pop(), 
                       calculation_period=1, 
                       sliding_window=12, 
                       alpha=0.6
                       )


def frequency_impact_demo():
    fishes = read_fishes("resources/detections/v29-fishes.json")
    regions = read_regions("resources/regions-example.json")
    frequency_analysis(fishes.pop(), regions, calculation_periods=[1, 12, 24])
    
    
def build_dataset_v29():
    build_dataset("resources/detections/v29-fishes.json",
                  "resources/classification/species-gt-v29.csv",
                  "resources/regions-example.json",
                  "resources/datasets/v29-dataset1.csv")
# endregion          
                            
                            
if __name__ == "__main__":
    analyze_trajectory_demo()
    # frequency_impact_demo()
    # build_dataset_v29()
