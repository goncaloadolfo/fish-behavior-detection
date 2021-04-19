import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

from trajectory_reader.visualization import simple_bar_chart


METRICS = ["accuracy", "precision", "recall", "f1-score"]


def evaluation_metrics(true_labels, predicted_labels):
    return (
        accuracy_score(true_labels, predicted_labels),
        precision_score(true_labels, predicted_labels),
        recall_score(true_labels, predicted_labels),
        f1_score(true_labels, predicted_labels)
    )


def plot_metrics(values, x_values, classifier_label, width):
    simple_bar_chart(plt.gca(), x_values, values,
                     "Evaluation metrics", "value", "metric",
                     label=classifier_label, width=width)


def plot_confusion_matrix(true_labels, predicted_labels):
    # bug: heatmap poorly formatted
    labels = np.unique(true_labels)
    matrix = confusion_matrix(true_labels, predicted_labels,
                              labels=labels)
    plt.figure()
    sns.heatmap(matrix, cmap="YlGnBu", xticklabels=labels, yticklabels=np.flip(labels),
                vmin=0, vmax=len(true_labels), annot=True)
    plt.title("Confusion matrix")
    plt.ylabel("true labels")
    plt.xlabel("predicted label")


def holdout_prediction(model, samples, gt):
    predictions = []

    for i, sample in enumerate(samples):
        # first sample
        if i == 0:
            x = samples[1:]
            y = gt[1:]

        # last sample
        elif i == len(samples) - 1:
            x = samples[:-1]
            y = gt[:-1]

        # middle samples
        else:
            x = np.vstack((samples[:i], samples[i+1:]))
            y = gt[:i] + gt[i+1:]

        # fit with the rest of the samples
        model.fit(x, y)

        # predict the label of the current sample
        predictions.append(model.predict([sample])[0])

    return predictions
