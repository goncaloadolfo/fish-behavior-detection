import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.pre_processing import CorrelatedVariablesRemoval, load_data
from pre_processing.trajectory_filtering import (exponential_weights,
                                                 smooth_positions)
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics.cluster import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from trajectory_reader.trajectories_reader import read_fishes
from trajectory_reader.visualization import (draw_trajectory, histogram,
                                             show_fish_trajectory,
                                             simple_bar_chart,
                                             simple_line_plot)

from anomaly_detection.anomaly_detector import most_different_features
from anomaly_detection.kmeans import (draw_cluster_trajectories,
                                      species_distribution)


def apply_dbscan(video_path, fishes_dataset, dataset, species, min_samples, epsilon,
                 metric, dp_pipeline, show_feature_diffs=False):
    x, y, feature_descriptions = load_data(dataset, species)
    for _, dp_node in dp_pipeline:
        x = dp_node.fit_transform(x, y)

    dbscan = DBSCAN(min_samples=min_samples,
                    eps=epsilon, metric=metric).fit(x)
    clusters = dbscan.labels_

    species_distribution(y, clusters)
    outlier_fishes = analyze_outliers(x, clusters, dbscan, feature_descriptions,
                                      fishes_dataset, show_feature_diffs)
    for fish in outlier_fishes:
        show_fish_trajectory(f"outlier fish {fish.fish_id}",
                             video_path, fish, [])


# def compare_dbscan_pipelines(dataset, species, parameters):
#     x, y, _ = load_data(dataset, species)

#     dbscan = DBSCAN(eps=parameters["epsilon"],
#                     min_samples=parameters["min_samples"],
#                     metric=parameters["metric"])
#     normalizer = StandardScaler()
#     pca = PCA(n_components=10)
#     feature_selector = SelectKBest(k=20)
#     correlation_removal = CorrelatedVariablesRemoval(0.9)

#     pipelines = [
#         Pipeline([("normalizer", normalizer), ("dbscan", dbscan)]),
#         Pipeline([("normalizer", normalizer),
#                  ("pca", pca), ("dbscan", dbscan)]),
#         Pipeline([("normalizer", normalizer),
#                  ("fs", feature_selector), ("dbscan", dbscan)]),
#         Pipeline([("normalizer", normalizer),
#                  ("cr", feature_selector), ("dbscan", dbscan)])
#     ]

#     silhouettes = []
#     x_original = x.copy()
#     for pipeline in pipelines:
#         x = x_original
#         for node_description, node in pipeline.named_steps.items():
#             if node_description != "dbscan":
#                 x = node.fit_transform(x, y)
#             else:
#                 node.fit(x)
#                 silhouettes.append(silhouette_score(x, node.labels_))

#     plt.figure()
#     simple_bar_chart(plt.gca(), range(len(pipelines)), silhouettes,
#                      "DBSCAN Pipelines", "Silhouette", "pipeline")
#     model_descriptions = ['+'.join([step_description for step_description in pipeline.named_steps.keys()])
#                           for pipeline in pipelines]
#     plt.grid()
#     plt.gca().set_xticks(range(len(pipelines)))
#     plt.gca().set_xticklabels(model_descriptions, rotation=30)
#     plt.tight_layout()


def dbscan_tuning(dataset, species, min_samples, epsilons, metric, data_preparation_pipeline):
    x, y, _ = load_data(dataset, species)
    for _, dp_node in data_preparation_pipeline:
        x = dp_node.fit_transform(x, y)

    fig, axs = plt.subplots()
    fig2, axs2 = plt.subplots()
    pd = "+".join([step_node[0]
                   for step_node in data_preparation_pipeline])

    for ms in min_samples:
        silhouettes = []
        nr_outliers = []

        for epsilon in epsilons:
            dbscan = DBSCAN(min_samples=ms, eps=epsilon, metric=metric)
            dbscan.fit(x)

            silhouettes.append(silhouette_score(x,
                                                dbscan.labels_,
                                                metric=metric)
                               )
            nr_outliers.append(np.sum(dbscan.labels_ == -1))

        simple_line_plot(axs, epsilons, silhouettes,
                         f"Silhouette for different epsilons/min_samples ({metric})\n{pd}",
                         "silhouette", "epsilon", marker="--o", label=f"min_samples={ms}")
        simple_line_plot(axs2, epsilons, nr_outliers,
                         f"Number of outliers for different epsilons/min_samples ({metric})\n{pd}",
                         "#outliers", "epsilon", marker="--o", label=f"min_samples={ms}")

    axs.legend()
    axs2.legend()
    axs.grid()
    axs2.grid()
    plt.show()


def analyze_outliers(x, clusters, dbscan, feature_descriptions, fishes_dataset,
                     show_feature_diffs=False):
    fishes = list(read_fishes(fishes_dataset))
    fishes.sort(key=lambda x: x.fish_id)
    fishes = np.array(fishes)

    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish)
        smooth_positions(fish, exponential_weights(24, 0.01))

    core_samples = x[dbscan.core_sample_indices_]
    outliers_frame = None
    outlier_fishes = []

    for i in range(len(clusters)):
        if clusters[i] == -1:
            outlier_fishes.append(fishes[i])
            random_color = (random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255))
            outliers_frame = draw_trajectory(fishes[i].trajectory, (480, 720),
                                             random_color, frame=outliers_frame,
                                             path=False, identifier=i)

            if show_feature_diffs:
                diffs = most_different_features(x[i], core_samples,
                                                feature_descriptions)
                features = [pair[0] for pair in diffs][:7]
                values = [pair[1] for pair in diffs][:7]

                plt.figure()
                xticks = range(len(features))
                simple_bar_chart(plt.gca(), xticks, values,
                                 f"Mean Distance to Core Samples (sample {i})",
                                 "mean distance", "feature")
                plt.xticks(xticks, features, rotation=60)
                plt.tight_layout()

    cv2.imshow("outliers", outliers_frame)
    return outlier_fishes


def analyze_distances_pipelines(dataset, species, n, pipelines):
    for pipeline in pipelines:
        dp_pipeline = pipeline.copy()[:-1]
        analyze_distances(dataset, species, "manhattan", 7, dp_pipeline)
        analyze_distances(dataset, species, "euclidean", 7, dp_pipeline)
        plt.show()


def analyze_distances(dataset, species, metric, n, data_preparation_pipeline):
    x, y, _ = load_data(dataset, species)
    for _, dp_node in data_preparation_pipeline:
        x = dp_node.fit_transform(x, y)

    knn = KNeighborsClassifier(n_neighbors=n, metric=metric,
                               algorithm="brute")
    knn.fit(x, y)

    distances, _ = knn.kneighbors(n_neighbors=n)
    plt.figure()
    pipeline_description = "+".join([step[0]
                                    for step in data_preparation_pipeline])
    _, bins, _ = histogram(plt.gca(), distances.flatten(),
                           f"Distance to {n} Nearest Points ({metric})\n{pipeline_description}",
                           "cumulative density", "distance",
                           density=True, cumulative=True)
    plt.xticks(bins)


if __name__ == "__main__":
    video_path = "resources/videos/v29.m4v"
    fishes = "resources/detections/v29-fishes.json"
    dataset = "resources/datasets/v29-dataset1.csv"
    species = ("shark", "manta-ray", "tuna")

    pipelines = [
        [("dbscan", DBSCAN())],
        [("normalizer", StandardScaler()), ("dbscan", DBSCAN())],
        [("normalizer", StandardScaler()),
         ("pca", PCA(10)), ("dbscan", DBSCAN())],
        [("normalizer", StandardScaler()),
         ("fs", SelectKBest(k=20)), ("dbscan", DBSCAN())],
        [("normalizer", StandardScaler()),
         ("cr", CorrelatedVariablesRemoval(0.9)), ("dbscan", DBSCAN())]
    ]
    # analyze_distances_pipelines(dataset, species, 7, pipelines)

    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(290, 580, 71), "manhattan",  pipelines[0][:-1])
    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(83, 186, 23), "euclidean", pipelines[0][:-1])

    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(58, 108, 11), "manhattan",  pipelines[1][:-1])
    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(8, 16, 1.5), "euclidean", pipelines[1][:-1])

    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(12, 32, 4), "manhattan",  pipelines[2][:-1])
    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(7, 13, 2), "euclidean", pipelines[2][:-1])

    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(19, 42, 5), "manhattan",  pipelines[3][:-1])
    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(4.5, 10.5, 1.2), "euclidean", pipelines[3][:-1])

    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(50, 83, 10), "manhattan",  pipelines[4][:-1])
    # dbscan_tuning(dataset, species, [5, 7, 9],
    #               np.arange(8, 14, 1.4), "euclidean", pipelines[4][:-1])

    # apply_dbscan(video_path, fishes, dataset, species, 5, 29,
    #              "manhattan", pipelines[3][:-1])
    apply_dbscan(video_path, fishes, dataset, species, 7, 11,
                 "euclidean", pipelines[2][:-1])
    plt.show()
    cv2.destroyAllWindows()
