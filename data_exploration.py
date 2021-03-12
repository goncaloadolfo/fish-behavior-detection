"""
Module with code that allows the exploration
of a dataset:
    - number of samples and dimensionality
    - features' data type
    - class balance
    - probability density functions
    - correlation
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from visualization import simple_bar_chart
from trajectory_feature_extraction import read_dataset


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


if __name__ == "__main__":
    # analysis of the video 29 dataset
    samples, gt, features_descriptions = read_dataset("resources/datasets/v29-dataset1.csv")
    full_analysis(samples, gt, features_descriptions)
