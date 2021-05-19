from os import path
from collections import defaultdict

import pymongo
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import numpy as np
import json


def import_bdm():
    import sys
    sys.path.append(path.abspath("../fish-behavior-detection"))
    
    
import_bdm()
from pre_processing.pre_processing_functions import load_data
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import smooth_positions, identify_descontinuity_points, segment_trajectory
from trajectory_reader.trajectories_reader import Fish
from trajectory_features.trajectory_feature_extraction import extract_features, exponential_weights
from trajectory_features.data_exploration import duration_histogram, positions_histogram
from labeling.regions_selector import read_regions
from rule_based.feeding_baseline import fiffb_time_series
from anomaly_detection.anomaly_detector import most_different_features


URL = "mongodb://localhost:27017/"
BDM_MODULE = "../fish-behavior-detection/"


def process_video(fishes_dict, species_gt, video_info):
    client = pymongo.MongoClient(URL)
    videos_collection = client["fishes"]["videos"]
    all_videos_info = videos_collection.find()
    total_nr_fishes = 0
    for video in all_videos_info:
        total_nr_fishes += len(video["fishes"])
    
    fish_instances, fishes_result = process_fishes(total_nr_fishes, fishes_dict, species_gt)
    global_information = video_general_analysis(fish_instances, species_gt, video_info)
    global_information["fishes"] = fishes_result
    print("inserting to database...")
    videos_collection.insert_one(json.loads(json.dumps(global_information)))  # all keys must be strings
    print("preparing the response...")
    return global_information
    

def process_fishes(total_nr_fishes, fishes, species):
    processing_models = ProcessingModels().get_instance()
    regions = read_regions(path.join(BDM_MODULE, "resources/regions-example.json"))
    
    fishes_instances = set()
    fishes_result = []
    
    id_incrementer = 1
    for fish_id in fishes:
        print(f"processing fish [{id_incrementer}/{len(fishes)}]")
        current_fish = {}
        new_fish = Fish(fish_id)
        new_fish.decode(fishes[fish_id])
        
        fill_gaps_linear(new_fish.trajectory, new_fish)
        fe_obj = extract_features(new_fish, regions, sliding_window=24, 
                                  alpha=0.3, with_xy_split=True)
        feature_labels, feature_vector = fe_obj.get_feature_vector()
        
        current_fish["feature-vector"] = np.array(feature_vector).tolist()
        current_fish["features-description"] = np.array(feature_labels).tolist()
        is_anomaly, anomaly_motifs = processing_models.is_anomaly(feature_vector)
        current_fish["is-anomaly"] = bool(is_anomaly)
        current_fish["anomaly-motifs"] = anomaly_motifs
        current_fish["is-interesting"] = bool(processing_models.is_interesting(feature_vector))
        
        weights = exponential_weights(24, 0.01)
        smooth_positions(new_fish, weights)
        fish_encoding = new_fish.encode()
        
        current_fish["fish-id"] = total_nr_fishes + id_incrementer
        current_fish["trajectory"] = fish_encoding["trajectory"]
        current_fish["bounding-boxes"] = fish_encoding["bounding-boxes"]
        current_fish["species"] = species[fish_id]
        
        smooth_fe_obj = extract_features(new_fish, regions, sliding_window=24, 
                                         alpha=0.01, with_xy_split=True)
        time_series_description = smooth_fe_obj.features_order
        time_series_list = smooth_fe_obj.all_time_series
        current_fish["time-series"] = {time_series_description[i]: time_series_list[i] 
                                       for i in range(len(time_series_description))}
        
        descontinuity_points = identify_descontinuity_points(new_fish, regions, 30, 2, 50)[0]
        segments, motifs = segment_trajectory(new_fish, descontinuity_points)
        current_fish["segments"]: {
            "timestamps": [(segment.trajectory[0][0], segment.trajectory[-1][0]) for segment in segments],
            "motifs": motifs
        }
        
        current_fish["seen"] = bool(False)
        current_fish["analysis-text"] = ""
        current_fish["usefullness"] = -1
        current_fish["true-label"] = None
        
        fishes_result.append(current_fish)
        fishes_instances.add(new_fish)
        id_incrementer += 1

    return fishes_instances, fishes_result


def video_general_analysis(fishes, species, video_info):
    print("processing global information...")
    fishes_by_specie = defaultdict(list)
    for fish in fishes:
        if species[fish.fish_id] in ("shark", "manta-ray"):
            fishes_by_specie[species[fish.fish_id]].append(fish)

    video_dict = {}
    video_dict["path"] = path.abspath(path.join(BDM_MODULE, 
                                                f"resources/videos/{video_info['video-name']}")
                                      )
    video_dict["fps"] = video_info["fps"]
    video_dict["resolution"] = video_info["resolution"]
    video_dict["total-frames"] = video_info["total-frames"]
    video_dict["nsharks"] = len(fishes_by_specie["shark"])
    video_dict["nmantas"] = len(fishes_by_specie["manta-ray"])
    
    for species in ("shark", "manta-ray"):
        aggregation_results = fiffb_time_series(fishes_by_specie[species], 0, video_info["total-frames"]-1)
        duration_hist = duration_histogram(fishes_by_specie[species], video_info["fps"])
        positions_hist = positions_histogram(fishes_by_specie[species], video_info["resolution"])
        
        video_dict[f"{species}-aggregation"] = {
            "fiffbs": np.array(aggregation_results[0]).tolist(),
            "mesh-edges": np.array(aggregation_results[1]).tolist(),
            "outliers": np.array(aggregation_results[2]).tolist()
        }
        video_dict[f"{species}-duration-hist"] = {
            "counts": duration_hist[0].tolist(),
            "bin-values": duration_hist[1].tolist() 
        }
        video_dict[f"{species}-positions-hist"] = {
            "counts": positions_hist[0].tolist(),
            "xbin_values": positions_hist[1].tolist(),
            "ybin_values": positions_hist[2].tolist()
        }

    return video_dict
    
    
class ProcessingModels:

    instance = None

    def __instantiate_models(self):
        x_original, y_original, features = load_data(path.join(BDM_MODULE, "resources/datasets/v29-dataset1.csv"), 
                                   ("shark", "manta-ray"),
                                   path.join(BDM_MODULE, "resources/classification/v29-interesting-moments.csv"))
        self.__x_original = x_original
        self.__y_original = y_original
        self.__feature_labels = features
        
        x, y = SMOTE(random_state=0).fit_resample(x_original, y_original)
        x = StandardScaler().fit_transform(x)
        self.__interesting_moments_clf = SVC(C=0.01, kernel="poly", degree=3, gamma=1, random_state=0).fit(x, y)
        
        return self
        
                
    def get_instance(self):
        if ProcessingModels.instance is None:
            ProcessingModels.instance = self.__instantiate_models()
        return ProcessingModels.instance
    
    def is_anomaly(self, new_x):
        xs = np.vstack((self.__x_original, [new_x]))
        ys = np.hstack((self.__y_original, [int(self.is_interesting(new_x))]))
        xs = StandardScaler().fit_transform(xs)
        xs = SelectKBest(k=20).fit_transform(xs, ys)
        
        anomaly_clf = DBSCAN(29, 5, "manhattan")
        labels = anomaly_clf.fit_predict(xs)
        
        if labels[-1] == -1:
            return True, most_different_features(xs[-1], anomaly_clf.components_, self.__feature_labels)[:7]
        
        return False, None
    
    def is_interesting(self, new_x):
        new_x = StandardScaler().fit_transform(np.vstack((self.__x_original, [new_x])))[-1]
        return self.__interesting_moments_clf.predict([new_x])[0] == 1        
    
    @property
    def anomaly_clf(self):
        return self.__anomaly_clf
    
    @property
    def interesting_moments_clf(self):
        return self.__interesting_moments_clf
