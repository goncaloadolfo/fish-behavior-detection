from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

from interesting_episodes_detection.evaluation import holdout_prediction
from labeling.regions_selector import read_regions
from labeling.trajectory_labeling import read_episodes, BEHAVIORS_ALIAS
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import (identify_descontinuity_points, 
                                                 segment_trajectory)
from pre_processing.pre_processing_functions import load_data
from trajectory_features.trajectory_feature_extraction import extract_features
from trajectory_reader.trajectories_reader import read_fishes
from trajectory_reader.visualization import simple_bar_chart

ACCURACY_KEY = "accuracy"
PRECISION_KEY = "precision"
RECALL_KEY = "recall"


def segments_dataset(fishes, regions_path, episodes_path,
                     distance_thr, speed_thr, angle_thr, plot_class_balance=False):
    # new samples and ground truth
    new_x = []
    new_y = []
    
    # needed settings
    regions = read_regions(regions_path)
    episodes = read_episodes(episodes_path)
    
    for fish in fishes:
        # fill gaps and extract segments
        fill_gaps_linear(fish.trajectory, fish)
        descontinuity_points, _, _ = identify_descontinuity_points(fish, regions, distance_thr,
                                                             speed_thr, angle_thr)
        fish_segments, _ = segment_trajectory(fish, descontinuity_points)

        for segment in fish_segments:
            segment_size = len(segment.trajectory)
            print("segment len: ", segment_size)
            
            # at least segments of more less 1 sec
            if segment_size > 30:
                # extract features
                _, segment_features_vector = extract_features(segment, regions, 1, 24, 0.01).get_feature_vector()
                segment_gt = is_interesting_segment(segment, episodes)

                # update dataset
                new_x.append(segment_features_vector)
                new_y.append(segment_gt)
                
    if plot_class_balance:
        plt.figure()
        labels, counts = np.unique(new_y, return_counts=True)
        simple_bar_chart(plt.gca(), range(len(labels)), counts,
                         "Class Balance", "number of samples", "is interesting")
        plt.gca().set_xticks(range(len(labels)))
        plt.gca().set_xticklabels(labels)
    
    return np.array(new_x), new_y

            
def is_interesting_segment(segment, episodes):
    # segment time edges
    segment_t_initial = segment.trajectory[0][0]
    segment_t_final = segment.trajectory[-1][0]
    
    # check if this segment is contained in any interesting episode
    for episode in episodes:
        if episode.fish_id == segment.fish_id and \
            (segment_t_initial < episode.t_initial < segment_t_final) or \
            (segment_t_initial < episode.t_final < segment_t_final):
            return 1
    
    return 0


def setup_svm_pipeline():
    normalizer = StandardScaler()
    svm = SVC(C=0.01, kernel="poly", gamma=1)
    return Pipeline([("normalizer", normalizer), ("svm", svm)])


# region imperative/tests

def segmentation_performance_impact():
    # general settings
    fishes_path = "resources/detections/v29-fishes.json"
    regions_path = "resources/regions-example.json"
    episodes_path = "resources/classification/v29-interesting-moments.csv"
    distance_thr = 30
    speed_thr = 2
    angle_thr = 50
    
    # segments dataset
    fishes = read_fishes(fishes_path)
    x, y = segments_dataset(fishes, regions_path, episodes_path,
                            distance_thr, speed_thr, angle_thr, True)
    nr_samples = x.shape[0]
    print("samples shape: ", x.shape)
    print("gt shape: ", len(y))
    
    # predictions using segments
    new_x, new_y = SMOTE().fit_resample(x, y)
    print("samples shape (after balance): ", new_x.shape)
    print("gt shape (after balance): ", len(new_y))
    svm_pipeline = setup_svm_pipeline()
    predictions = holdout_prediction(svm_pipeline, new_x, new_y)[:nr_samples]
    
    # predictions using original samples
    original_x, original_y, _ = load_data("resources/datasets/v29-dataset1.csv",
                                       ("shark", "manta-ray"))
    nr_original_samples = original_x.shape[0]
    new_original_x, new_original_y = SMOTE().fit_resample(original_x, original_y)
    original_predictions = holdout_prediction(svm_pipeline, new_original_x, 
                                              new_original_y)[:nr_original_samples]
    
    # evaluation
    segments_acc = accuracy_score(y, predictions)
    segments_precision = precision_score(y, predictions)
    segments_recall = recall_score(y, predictions)
    
    original_acc = accuracy_score(original_y, original_predictions)
    original_precision = precision_score(original_y, original_predictions)
    original_recall = recall_score(original_y, original_predictions)

    # plot metrics
    labels = ["acc", "prec", "rec"]
    original_results = [original_acc, original_precision, original_recall]
    segments_results = [segments_acc, segments_precision, segments_recall]
    
    x = np.arange(len(labels))
    width = 0.35
    plt.figure()
    plt.title("Segmentation Impact")
    plt.ylim([0, 1])
    plt.bar(x - width/2, original_results, width, label='original')
    plt.bar(x + width/2, segments_results, width, label='using segmentation')

    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(labels)
    plt.legend()
    
    plt.show()
    
    
def models_by_species_impact():
    svm_pipeline = setup_svm_pipeline()
    case_scenarios = [("shark", "manta-ray"), ("shark"), ("manta-ray")]
    cases_metric_results = {}
    
    for species_case in case_scenarios:
        # load data
        x, y, _ = load_data("resources/datasets/v29-dataset1.csv", species_case)
        nr_samples = x.shape[0]
        
        # predictions
        balanced_x, balanced_y = SMOTE().fit_resample(x, y)
        predictions = holdout_prediction(svm_pipeline, balanced_x, balanced_y)[:nr_samples]
        
        # extract metrics    
        cases_metric_results[species_case] = {
            ACCURACY_KEY: accuracy_score(y, predictions),
            PRECISION_KEY: precision_score(y, predictions),
            RECALL_KEY: recall_score(y, predictions)   
        }
    
    # plot results
    labels = ["acc", "prec", "rec"]
    x = np.arange(len(labels))
    width = 0.2
    
    plt.figure()
    plt.title("Model by Species Impact")
    plt.ylim([0, 1])
    
    counter = 1
    for species_case, results in cases_metric_results.items():
        if counter == 1:
            center = x - width
        elif counter == 2:
            center = x
        else:
            center = x + width
        
        aux = [results[ACCURACY_KEY], results[PRECISION_KEY], results[RECALL_KEY]]
        plt.bar(center, aux, width, label=species_case)
        counter += 1

    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(labels)
    plt.legend()
    plt.show()


def main():
    segmentation_performance_impact()
    # models_by_species_impact()
    
# endregion


if __name__ == "__main__":
    main()
                