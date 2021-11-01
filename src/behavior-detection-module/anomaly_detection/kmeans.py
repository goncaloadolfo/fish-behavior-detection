import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from anomaly_detection.anomaly_detector import most_different_features
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.pre_processing_functions import (z_normalization)
from trajectory_features.trajectory_feature_extraction import read_dataset, read_fishes
from trajectory_reader.visualization import (draw_trajectory, simple_bar_chart,
                                             simple_line_plot)


def train_model(samples, k, max_steps, n_init, seed=None):
    if seed is not None:
        n_init = 1

    # instantiate the model
    kmeans_model = KMeans(k, init='k-means++', n_init=n_init,
                          max_iter=max_steps, random_state=seed)

    # train the model
    kmeans_model.fit(samples)

    # not enough iterations to converge
    if kmeans_model.n_iter_ == max_steps:
        print(f"{max_steps} iterations were not enough to converge")

    return kmeans_model


def model_tunning(samples, ks, max_steps, n_init):
    # calculate cohesion for several models
    cohesions = []
    for k in ks:
        # train a new model
        model = train_model(samples, k, max_steps, n_init)

        # calculate cohesion
        cohesions.append(model.inertia_)

    # plot results
    plt.figure()
    simple_line_plot(plt.gca(), ks, cohesions,
                     f"KMeans Tunning", "cohesion", "k", marker='-o')


def evaluate_model(samples, k, max_steps, n_init, seed=None):
    # train the model
    model = train_model(samples, k, max_steps, n_init, seed)
    print(repr(model))

    # resulting distances and predictions
    distances = model.transform(samples)
    resulting_clusters = model.labels_

    # external evaluation metrics
    cohesions = calculate_cohesions(distances, resulting_clusters)
    separations = calculate_separations(distances, resulting_clusters)
    silhouettes = calculate_silhouettes(cohesions, separations)

    # print results to the console
    print("cohesion: ", np.sum(cohesions))
    print("separation: ", np.sum(separations))
    print("silhouette: ", np.mean(silhouettes))

    return model, resulting_clusters


def best_seed(samples, n, k, max_steps):
    best_seed_value = None

    for _ in range(n):
        # train a new model using a new seed
        seed = random.randrange(2 ** 32)
        model = train_model(samples, k, max_steps, 1, seed)

        # get distances and cluster indexes
        distances = model.transform(samples)
        resulting_clusters = model.labels_

        # calculate silhouette
        cohesions = calculate_cohesions(distances, resulting_clusters)
        separations = calculate_separations(distances, resulting_clusters)
        silhouette = np.mean(calculate_silhouettes(cohesions, separations))

        if best_seed_value is None or silhouette > best_seed_value[1]:
            best_seed_value = (seed, silhouette)

    print(f"Best seed: {best_seed_value[0]}, silhouette: {best_seed_value[1]}")


# endregion


# region external metrics
def calculate_cohesions(distances, cluster_info):
    # the cohesion of a point is considered the distance of a point to its centroid
    cohesions = []
    for i, data_point_distances in enumerate(distances):
        cohesions.append(data_point_distances[cluster_info[i]])
    return cohesions


def calculate_separations(distances, cluster_info):
    # the separation of a point is considered the distance of a point to nearest neighbor centroid
    separations = []
    for i, data_point_distances in enumerate(distances):
        cluster = cluster_info[i]
        # remove distance to its cluster centroid
        aux = np.hstack(
            (data_point_distances[:cluster], data_point_distances[cluster + 1:]))
        separations.append(min(aux))
    return separations


def calculate_silhouettes(cohesions, separations):
    # silhouette represents a ratio between the cohesion and separation of a sample
    silhouettes = []

    for i in range(len(cohesions)):
        # cohesion and separation of a given sample
        cohesion = cohesions[i]
        separation = separations[i]

        # silhouette calculation
        if cohesion < separation:
            silhouettes.append(1 - cohesion / separation)

        elif cohesion == separation:
            silhouettes.append(0)

        else:
            silhouettes.append(separation / cohesion - 1)

    return silhouettes


# endregion


# region clustering utils
def plot_most_different_features(centroids, feature_descriptions, n):
    for label, centroid in centroids.items():
        # get features difference from current centroid
        all_centroids = centroids.copy()
        del all_centroids[label]
        features_difference = most_different_features(centroid,
                                                      np.array(
                                                          list(
                                                              all_centroids.values()
                                                          )
                                                      ),
                                                      feature_descriptions)

        # get first n
        if len(centroid) > n:
            features_difference = features_difference[:n]

        # plot features with lower difference
        plt.figure()
        xticks = range(len(features_difference))
        simple_bar_chart(plt.gca(), xticks,
                         [x[1] for x in features_difference],
                         f"Most characterizing features cluster {label}",
                         "mean difference", "feature")
        plt.xticks(xticks, [x[0] for x in features_difference], rotation=60)
        plt.tight_layout()


def draw_cluster_trajectories(trajectories_file_path, cluster_info):
    # possible cluster indexes
    clusters = np.unique(cluster_info)

    # read fishes and sort by fish_id
    fishes = list(read_fishes(trajectories_file_path))
    fishes.sort(key=lambda x: x.fish_id)
    fishes = np.array(fishes)

    # fill gaps
    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish)

    for cluster in clusters:
        cluster_frame = np.full((480, 720, 3), 255, dtype=np.uint8)
        cluster_fishes = fishes[cluster_info == cluster]

        # draw the trajectory of each fish of this cluster
        for fish in cluster_fishes:
            random_color = np.random.rand(3) * 255
            color_tuple = tuple(int(random_color[i])
                                for i in range(len(random_color)))
            draw_trajectory(fish.trajectory, None,
                            color_tuple, frame=cluster_frame)

        cv2.imshow(f"trajectories cluster {cluster}", cluster_frame)


def species_distribution(species_gt, cluster_info):
    clusters = np.unique(cluster_info)
    species = np.unique(species_gt)
    counting = {
        species_tag: [] for species_tag in species
    }

    # count the number of samples per cluster and per species
    species_gt = np.array(species_gt)
    for cluster in clusters:
        cluster_gts = species_gt[cluster_info == cluster]
        for species_tag in species:
            counting[species_tag].append(np.sum(cluster_gts == species_tag))

    # stacked bar chart
    plt.figure()
    bottom = None
    for species_tag in species:
        plt.bar(clusters, counting[species_tag],
                bottom=bottom, label=species_tag)
        bottom = counting[species_tag]
    plt.title("Clusters distribution")
    plt.xlabel("cluster")
    plt.ylabel("number of samples")
    plt.legend()


# endregion


# region experiences
def v29_all_species():
    """
    KMeans on all trajectories from video 29

    Results using z-normalization:
        - Best k: 7
        - Best seed: 2286868185, silhouette: 0.2919
        - Cohesion: 466.2249, separation: 672.2214, silhouette: 0.2919

    Results using z-normalization and removing high correlated variables (> 0.9):
        - Best k: 7
        - Best seed: 2319272814, silhouette: 0.2787564706508115
        - Cohesion: 407.3607, separation: 572.9619, silhouette: 0.2788

    Results using z-normalization and PCA:
        - Best k: 7
        - Number of principal components: 17
        - Best seed: 4089396988, silhouette: 0.3038
        - Cohesion: 425.6164, separation: 595.9555, silhouette: 0.3038
    """
    # read samples
    samples, gt, descriptions = read_dataset(
        "resources/datasets/v29-dataset1.csv"
    )

    # pre processing
    input_data = z_normalization(np.array(samples))
    # input_data = remove_correlated_variables(input_data, 0.9)
    # analyze_pca_components(input_data)
    # input_data = apply_pca(input_data, 17)

    # tuning
    model_tunning(input_data, ks=range(2, 33), max_steps=300, n_init=10)
    # best_seed(input_data, n=10000, k=7, max_steps=300)

    # evaluation
    model, resulting_clusters = evaluate_model(input_data, k=7,
                                               max_steps=300, n_init=1, seed=2286868185)

    # analysis
    draw_cluster_trajectories("resources/detections/v29-fishes.json",
                              resulting_clusters)
    species_distribution(gt, resulting_clusters)
    plot_most_different_features({label: centroid
                                  for label, centroid in enumerate(model.cluster_centers_)},
                                 descriptions, n=10)
    plt.show()
    cv2.destroyAllWindows()


# endregion


if __name__ == '__main__':
    v29_all_species()
