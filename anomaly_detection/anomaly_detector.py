import cv2
import matplotlib.pyplot as plt
import numpy as np

from trajectory_features.trajectory_feature_extraction import \
    trajectory_repeated_reading
from trajectory_reader.visualization import simple_line_plot


def identify_outlier_clusters(cluster_labels, percentage_thr):
    # count number of samples in each cluster
    n_total = len(cluster_labels)
    labels, counts = np.unique(cluster_labels, return_counts=True)

    # find cluster with few samples
    outlier_clusters = set()
    for i, label in enumerate(labels):
        if counts[i] <= (n_total * percentage_thr):
            outlier_clusters.add(label)

    # contemplate algorithms that already detect outliers
    if -1 in labels and -1 not in outlier_clusters:
        outlier_clusters.add(-1)

    return outlier_clusters


def outliers_thr_analysis(labels, thrs):
    # count how many outlier samples are identified using different thresholds
    outliers_found = []
    for thr in thrs:
        outlier_clusters = identify_outlier_clusters(labels, thr)
        nsamples = 0
        for outlier_cluster in outlier_clusters:
            nsamples += np.sum(labels == outlier_cluster)
        outliers_found.append(nsamples)

    # plot results
    plt.figure()
    simple_line_plot(plt.gca(), thrs, outliers_found,
                     "Number of outliers found", "#outliers", "percentage threshold", "-o")


def visualize_outlier_trajectories(video_path, fishes, labels, thr):
    # detect outlier clusters
    outlier_labels = identify_outlier_clusters(labels, thr)

    # visualize fish trajectories from those clusters
    for outlier_label in outlier_labels:
        outlier_fishes = fishes[labels == outlier_label]
        for fish in outlier_fishes:
            trajectory_repeated_reading(video_path, None, fish)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def most_different_features(centroid, remaining_centroids, feature_descriptions):
    # mean difference
    differences = np.array([np.abs(centroid - remaining_centroid)
                            for remaining_centroid in remaining_centroids])
    mean_diff = np.mean(differences, axis=0)

    # order features by difference value - descending order
    diffs_list = list(zip(feature_descriptions, mean_diff))
    diffs_list.sort(key=lambda x: x[1], reverse=True)

    return diffs_list
