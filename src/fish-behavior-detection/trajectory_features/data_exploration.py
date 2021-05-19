"""
Module with code that allows the exploration
of a dataset:
    - number of samples and dimensionality
    - features' data type
    - class balance
    - probability density functions
    - correlation
"""
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from labeling.trajectory_labeling import read_species_gt
from pre_processing.pre_processing_functions import CorrelatedVariablesRemoval, load_data
from trajectory_features.trajectory_feature_extraction import read_dataset
from trajectory_reader.trajectories_reader import read_fishes
from trajectory_reader.visualization import (histogram, histogram2d,
                                             simple_bar_chart,
                                             simple_line_plot)


def analyze_trajectories_by_species(path_fishes, path_species, species_groups, video_path, group_by=False):
    fishes = read_fishes(path_fishes)
    species_gt = read_species_gt(path_species)

    fishes_by_species = defaultdict(list)
    for fish in fishes:
        species = species_gt[fish.fish_id]
        if species in species_groups:
            fishes_by_species[species].append(fish)

    if group_by:
        for focus_species in species_groups:
            analyze_trajectories(
                fishes_by_species[focus_species], focus_species, video_path)

    else:
        all_focus_species_trajectories = []
        for focus_species in species_groups:
            all_focus_species_trajectories += fishes_by_species[focus_species]
        analyze_trajectories(all_focus_species_trajectories,
                             species_groups, video_path)


def analyze_trajectories(fishes, tag, video_path):
    trajectory_durations = []
    trajectory_xvalues = []
    trajectory_yvalues = []

    for fish in fishes:
        trajectory = fish.trajectory
        trajectory_durations.append(
            (trajectory[-1][0] - trajectory[0][0]) / 24.0)
        trajectory_xvalues += [data_point[1] for data_point in trajectory]
        trajectory_yvalues += [data_point[2] for data_point in trajectory]

    plt.figure()
    _, bins, _ = histogram(plt.gca(), trajectory_durations,
                           f"Trajectory duration {tag}", "#trajectories", "duration (s)")
    plt.xticks(bins)

    video_capture = cv2.VideoCapture(video_path)
    _, video_frame = video_capture.read()
    frame_size = (video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                  video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.release()

    plt.figure()
    _, binsx, binsy, quad_img, frame = histogram2d(plt.gca(), trajectory_xvalues, trajectory_yvalues,
                                                   f"Positions Distribution {tag}", "y position", "x position",
                                                   bins_range=[[0, frame_size[0]], [0, frame_size[1]]], with_text=True,
                                                   frame=video_frame)
    plt.gca().set_xticks(binsx)
    plt.gca().set_yticks(binsy)
    plt.gca().invert_yaxis()
    plt.colorbar(quad_img)
    cv2.imshow(f"Positions Distribution {tag}", frame)


def duration_histogram(fishes, fps):
    durations = [len(fish.trajectory) / fps for fish in fishes]
    return np.histogram(durations)


def positions_histogram(fishes, frame_size):
    xs = []
    ys = []
    for fish in fishes:
        for t, x, y in fish.trajectory:
            xs.append(x)
            ys.append(y)

    return np.histogram2d(xs, ys,
                          range=[[0, frame_size[0]],
                                 [0, frame_size[1]]]
                          )


def full_analysis(samples, gt, features_description):
    """
    Dataset analysis: general info, class balance, correlation, and distributions
    """
    # general information
    general_info(samples, features_description)
    class_balance(gt, "species")
    correlation_analysis(samples, features_description, 0.8)

    # distribution analysis
    samples = np.array(samples).T
    for i in range(samples.shape[0]):
        distribution_analysis(samples[i], gt, features_description[i])
        plt.show()


def general_info(samples, features_labels):
    """
    Prints the number of samples,
    dimensionality and data type of each of the
    features.
    """
    # validation
    nsamples = len(samples)
    if nsamples == 0:
        print(f"Warning from {general_info.__name__}: \
              received a dataset with 0 samples")
        return
    nfeatures = len(samples[0])

    if nfeatures != len(features_labels):
        labels = [f"var {n}" for n in range(nfeatures)]
        print(f"Warning from {general_info.__name__}: \
              the dimensionality is different from \
              the number of labels received as argument")
    else:
        labels = features_labels

    # general info
    print("Dataset information:")
    print("\t - number of samples: ", nsamples)
    print("\t - dimensionality: ", nfeatures)

    # features info
    if nfeatures > 0:
        dtypes_counting = {}
        print("Features")
        for i in range(nfeatures):
            dtype = type(samples[0][i])
            print(f"\t - {labels[i]}: {dtype}")
            dtypes_counting[dtype] = dtypes_counting[dtype] + 1 \
                if dtype in dtypes_counting else 1

        # data types histogram
        dtypes = list(dtypes_counting.keys())
        counting = list(dtypes_counting.values())
        plt.figure()
        simple_bar_chart(plt.gca(), range(len(dtypes)), counting,
                         "Data Types", "counting", "dtype")
        plt.gca().set_xticks(range(len(dtypes)))
        plt.gca().set_xticklabels(dtypes)


def class_balance(gt, context_label):
    """
    Illustrates a bar chart with the counting
    of the number of samples from each class.
    """
    # validation
    if len(gt) == 0:
        print(f"Warning from {class_balance.__name__}: \
              received an empty ground truth")
        return

    # class balance histogram
    labels, counts = np.unique(gt, return_counts=True)
    plt.figure()
    ax = plt.gca()
    xticks = range(len(labels))
    simple_bar_chart(ax, xticks, counts,
                     f"Class Balance - {context_label}", "#samples", "class")
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)


def distribution_analysis(feature_values, gt, feature_label):
    """
    Illustrates a plot with the probability density function
    of the given feature for each of the classes.
    """
    # validation
    nfeatures = len(feature_values)
    ngt = len(gt)
    if nfeatures == 0:
        print(f"Warning from {distribution_analysis.__name__}: " +
              "received an empty list of values")
        return
    elif ngt == 0:
        print(f"Warning from {distribution_analysis.__name__}: " +
              "received an empty ground truth")
        return
    elif nfeatures != len(gt):
        print(f"Warning from {distribution_analysis.__name__}: " +
              "number of values differ from number of received gt labels")
        return

    # convert lists to a numpy array if necessary
    features_values_array = np.array(feature_values) \
        if type(feature_values) is not np.array else feature_values
    gt_array = np.array(gt) \
        if type(gt) is not np.array else gt

    # plot density prob functions for each class
    labels = np.unique(gt)
    plt.figure()
    for label in labels:
        values_from_label = features_values_array[gt_array == label]
        # kde - gaussian kernel
        sns.distplot(values_from_label, hist=False, kde=True, label=label)
    plt.title(f"Density plot - {feature_label}")
    plt.xlabel(feature_label)
    plt.ylabel("density")
    plt.legend()


def correlation_analysis(samples, features_labels, interest_thr):
    """
    Illustrates a heatmap and a histogram
    with the correlation values between the features.
    """
    # validation
    if len(samples) == 0:
        print(f"Warning from {correlation_analysis.__name__}: " +
              "no samples received, empty list")
        return
    n_variables = len(samples[0])
    if n_variables < 2:
        print(f"Warning from {correlation_analysis.__name__}: " +
              "the number of variables of each sample must be at least 2")
        return

    # calculate correlation
    samples_array = np.array(samples) \
        if type(samples) is not np.array else samples
    correlation_matrix = np.corrcoef(samples_array.T)

    # heatmap
    lower_triangle_mask = np.ones_like(correlation_matrix, dtype=bool)
    lower_triangle_mask[np.tril_indices(n_variables, k=-1)] = False
    plt.figure()
    sns.heatmap(correlation_matrix, vmin=-1, vmax=1,
                cmap="YlGnBu", mask=lower_triangle_mask)
    plt.title("Correlation Heatmap")
    plt.xlabel("variable index")
    plt.ylabel("variable index")

    # histogram
    plt.figure()
    plt.hist(correlation_matrix[lower_triangle_mask])
    plt.title("Correlation Histogram")
    plt.xlabel("correlation")
    plt.ylabel("counts")

    # correlations of interest
    if len(samples[0]) == len(features_labels):
        print("Variables with correlation" +
              f"> {interest_thr} or < {-interest_thr}:")
        for i in range(n_variables):
            for j in range(i):  # only through lower triangle part
                correlation = correlation_matrix[i][j]
                if correlation > interest_thr or correlation < -interest_thr:
                    print(f"\t - {features_labels[i]}, {features_labels[j]}: " +
                          f"{correlation}")
    else:
        print(f"Warning from {correlation_analysis.__name__}: " +
              "number of variables differ from features label length")

    # dimensionality analysis
    dimensionality_analysis(samples_array)


def dimensionality_analysis(samples):
    correlation_thrs = np.arange(1, 0, -0.1)
    dimensionality_values = []

    for thr in correlation_thrs:
        new_matrix = CorrelatedVariablesRemoval(thr).fit_transform(samples, [])
        # check new dimensionality
        dimensionality_values.append(new_matrix.shape[1])

    # plot results
    plt.figure()
    simple_line_plot(plt.gca(), correlation_thrs, dimensionality_values, "High correlated variable removal",
                     "new dimensionality", "correlation thr", marker="-o")


def v29_analysis(path):
    samples, species_gt, features_descriptions = read_dataset(path)
    full_analysis(samples, species_gt, features_descriptions)


def v29_episodes_analysis(species):
    samples, episodes_gt, features_descriptions = load_data(
        "../resources/datasets/v29-dataset1.csv", species
    )
    general_info(samples, features_descriptions)
    class_balance(episodes_gt, "Interesting episodes")
    samples = np.array(samples).T
    for i in range(samples.shape[0]):
        distribution_analysis(
            samples[i], episodes_gt, features_descriptions[i]
        )
        plt.show()


if __name__ == "__main__":
    # v29_analysis("resources/datasets/v29-dataset1.csv")
    # v29_episodes_analysis(("shark", "manta-ray"))
    # v29_episodes_analysis(("shark", ))
    # v29_episodes_analysis(("manta-ray", ))

    analyze_trajectories_by_species("../resources/detections/v29-fishes.json",
                                    "../resources/classification/species-gt-v29.csv", (
                                        "shark", "manta-ray"),
                                    "../resources/videos/v29.m4v")
    analyze_trajectories_by_species("../resources/detections/v29-fishes.json",
                                    "../resources/classification/species-gt-v29.csv", (
                                        "shark", "manta-ray"),
                                    "../resources/videos/v29.m4v", True)
    plt.show()
    cv2.destroyAllWindows()
