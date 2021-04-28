from collections import defaultdict

import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from labeling.trajectory_labeling import read_episodes
from matplotlib import pyplot as plt
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.pre_processing import CorrelatedVariablesRemoval, load_data
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from trajectory_features.trajectory_feature_extraction import read_dataset
from trajectory_reader.trajectories_reader import read_fishes
from trajectory_reader.visualization import (show_fish_trajectory,
                                             simple_bar_chart,
                                             simple_line_plot)

from interesting_episodes_detection.evaluation import holdout_prediction

SEED = 0


def svm_tuning(dataset, param_grid):
    x, y, feature_labels = load_data(
        dataset, ("shark", "manta-ray")
    )

    x, y = SMOTE(random_state=SEED).fit_resample(x, y)
    x = StandardScaler().fit_transform(x)

    models = []
    scores = []

    for c in param_grid["C"]:
        for kernel in param_grid["kernel"]:
            for degree in param_grid["degree"]:
                for gamma in param_grid["gamma"]:
                    model = SVC(C=c, kernel=kernel, degree=degree,
                                gamma=gamma, random_state=SEED)
                    predictions = holdout_prediction(model, x, y)

                    models.append((c, kernel, degree, gamma))
                    scores.append(accuracy_score(y, predictions))

    best_score = np.max(scores)
    best_model = models[scores.index(best_score)]

    print(
        f"Best parameters: SVC(c={best_model[0]}, kernel={best_model[1]}," +
        f"degree={best_model[2]}, gamma={best_model[3]})"
    )
    print("Best score: ", best_score)
    plot_tuning_results(models, scores)


def svm_pipelines(dataset, svm_params):
    x, y, _ = load_data(dataset, ("shark", "manta-ray"))
    svm = SVC(C=svm_params["c"], kernel=svm_params["kernel"],
              degree=svm_params["degree"], gamma=svm_params["gamma"],
              random_state=SEED)

    balancer = SMOTE(random_state=SEED)
    normalizer = StandardScaler()
    select_kbest = SelectKBest(k=20)
    pca = PCA(n_components=10)
    correlation_removal = CorrelatedVariablesRemoval(0.9)

    pipelines = (
        [("svm", svm)],
        [("normalizer", normalizer), ("svm", svm)],
        [("balancer", balancer), ("svm", svm)],
        [("balancer", balancer), ("normalizer", normalizer), ("svm", svm)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("pca", pca), ("svm", svm)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("select kbest", select_kbest), ("svm", svm)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("correlation removal", correlation_removal), ("svm", svm)],
    )

    original_x = x.copy()
    original_y = y.copy()
    scores = []
    pipelines_predictions = []

    for pipeline in pipelines:
        x = original_x
        y = original_y

        for pipeline_node in pipeline:
            if pipeline_node[0] == "svm":
                svm = pipeline_node[1].fit(x, y)
                predictions = holdout_prediction(svm, x, y)
                pipelines_predictions.append(predictions)
                scores.append(accuracy_score(y, predictions))
            elif pipeline_node[0] == "balancer":
                x, y = pipeline_node[1].fit_resample(x, y)
            else:
                x = pipeline_node[1].fit_transform(x, y)

    plt.figure()
    simple_bar_chart(plt.gca(), range(len(pipelines)), scores,
                     "SVM pipelines", "accuracy", "pipeline")
    model_descriptions = ['+'.join([n[0] for n in pipeline])
                          for pipeline in pipelines]
    plt.grid()
    plt.gca().set_xticks(range(len(pipelines)))
    plt.gca().set_xticklabels(model_descriptions, rotation=30)
    plt.tight_layout()
    analyze_errors(pipelines_predictions, scores, original_y, dataset)


def analyze_errors(predictions, scores, true_y, dataset):
    best_predictions = predictions[np.argmax(scores)]
    fishes = list(read_fishes("resources/detections/v29-fishes.json"))
    fishes.sort(key=lambda x: x.fish_id)

    _, species_gt, _ = read_dataset(dataset)
    fishes = [fishes[i] for i in range(len(fishes))
              if species_gt[i] != "tuna"]
    nr_fishes = len(fishes)

    if len(best_predictions) > nr_fishes:
        best_predictions = best_predictions[:nr_fishes]
        true_y = true_y[:nr_fishes]

    nr_errors = np.sum(np.array(best_predictions) != np.array(true_y))
    counter = 1
    for i in range(len(true_y)):
        if true_y[i] != best_predictions[i]:
            print(f"{counter}/{nr_errors}")
            fill_gaps_linear(fishes[i].trajectory, fishes[i])
            show_fish_trajectory(
                f"Fish {fishes[i].fish_id} true class {true_y[i]}, classified as {best_predictions[i]}",
                "resources/videos/v29.m4v", fishes[i],
                read_episodes(
                    "resources/classification/v29-interesting-moments.csv"
                )
            )
            cv2.destroyAllWindows()
            counter += 1


def plot_tuning_results(models, scores):
    models_with_scores = zip(models, scores)
    models_by_kernel = defaultdict(list)
    for model in models_with_scores:
        models_by_kernel[model[0][1]].append(model)

    for kernel in models_by_kernel.keys():
        plt.figure(f"SVM with {kernel} kernel")
        plt.gca().set_xscale('log')
        plt.gca().set_ylim((0, 1))
        plt.grid()

        kernel_models = models_by_kernel[kernel]

        cs = [model[0][0] for model in kernel_models]
        degrees = [model[0][2] for model in kernel_models]
        gammas = [model[0][3] for model in kernel_models]
        scores = [model[1] for model in kernel_models]

        if kernel == "linear":
            simple_line_plot(
                plt.gca(), cs, scores, f"SVM with {kernel} kernel", "accuracy", "c", marker="--o"
            )

        elif kernel == "poly":
            group_by_degree_gamma = _group_by_parameter(kernel_models, (2, 3))
            for degree_gamma, results in group_by_degree_gamma.items():
                simple_line_plot(
                    plt.gca(), results[0], results[1],
                    f"SVM with {kernel} kernel", "accuracy", "c", marker="--o",
                    label=f"degree={degree_gamma[0]} gamma={degree_gamma[1]}"
                )

        else:
            group_by_gamma = _group_by_parameter(kernel_models, (3,))
            for gamma, results in group_by_gamma.items():
                simple_line_plot(
                    plt.gca(), results[0], results[1],
                    f"SVM with {kernel} kernel",
                    "accuracy", "c", marker="--o", label=f"gamma={gamma}"
                )

        if kernel != "linear":
            plt.legend()


def _group_by_parameter(models, parameters_indexes):
    group_by_results = defaultdict(lambda: [[], []])
    for model in models:
        parameter_values = tuple([model[0][i] for i in parameters_indexes])
        group_by_results[parameter_values][0].append(model[0][0])
        group_by_results[parameter_values][1].append(model[1])
    return group_by_results


if __name__ == "__main__":
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4],
        "gamma": [0.01, 0.1, 1]
    }
    svm_tuning("resources/datasets/v29-dataset1.csv", param_grid)
    svm_pipelines("resources/datasets/v29-dataset1.csv",
                  {"c": 0.1, "kernel": "poly", "degree": 3, "gamma": 0.1})
    plt.show()
