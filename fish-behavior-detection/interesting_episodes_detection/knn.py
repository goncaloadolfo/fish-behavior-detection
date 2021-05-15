from collections import defaultdict

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from interesting_episodes_detection.evaluation import holdout_prediction
from pre_processing.pre_processing_functions import CorrelatedVariablesRemoval, load_data
from trajectory_reader.visualization import simple_line_plot, simple_hbar_chart


def knn_tuning(dataset, species, parameters_grid):
    x, y, feature_names = load_data(dataset, species)
    x = StandardScaler().fit_transform(x, y)

    scores = defaultdict(list)
    ns = parameters_grid["n_neighbors"]
    metrics = parameters_grid["metric"]

    for metric in metrics:
        for n in ns:
            knn = KNeighborsClassifier(n_neighbors=n, metric=metric,
                                       algorithm="brute")
            predictions = holdout_prediction(knn, x, y)
            scores[metric].append(accuracy_score(
                y, predictions
            ))

    plt.figure()
    for metric in metrics:
        simple_line_plot(plt.gca(), ns, scores[metric], "KNN Tuning",
                         "accuracy", "k", "--o", label=metric)
    plt.ylim((0, 1))
    plt.grid()
    plt.legend()

    best_score = -1
    best_k = -1
    best_metric = "nd"
    for metric in metrics:
        for i in range(len(scores[metric])):
            if scores[metric][i] > best_score:
                best_score = scores[metric][i]
                best_k = ns[i]
                best_metric = metric
    print(f"Best model: k={best_k} metric:{best_metric} score:{best_score}")


def knn_pipelines(dataset, species, parameters):
    x, y, features_description = load_data(dataset, species)
    knn = KNeighborsClassifier(
        n_neighbors=parameters["n_neighbors"],
        algorithm="brute",
        metric=parameters["metric"]
    )
    balancer = SMOTE()
    normalizer = StandardScaler()
    select_kbest = SelectKBest(k=20)
    pca = PCA(n_components=10)
    correlation_removal = CorrelatedVariablesRemoval(0.9)

    pipelines = (
        [("knn", knn)],
        [("normalizer", normalizer), ("knn", knn)],
        [("balancer", balancer), ("knn", knn)],
        [("balancer", balancer), ("normalizer", normalizer), ("knn", knn)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("pca", pca), ("knn", knn)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("select kbest", select_kbest), ("knn", knn)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("correlation removal", correlation_removal), ("knn", knn)],
    )

    original_x = x.copy()
    original_y = y.copy()
    scores = []

    for pipeline in pipelines:
        x = original_x
        y = original_y

        for pipeline_node in pipeline:
            if pipeline_node[0] == "knn":
                knn = pipeline_node[1].fit(x, y)
                predictions = holdout_prediction(knn, x, y)
                scores.append(accuracy_score(
                    y[:len(original_y)], predictions[:len(original_y)]
                ))
            elif pipeline_node[0] == "balancer":
                x, y = pipeline_node[1].fit_resample(x, y)
            else:
                x = pipeline_node[1].fit_transform(x, y)

    plt.figure()
    model_descriptions = ['+'.join([n[0] for n in pipeline])
                          for pipeline in pipelines]
    simple_hbar_chart(plt.gca(), scores[::-1], model_descriptions[::-1],
                      "Decision tree pipelines", "accuracy", "pipeline")

    plt.grid()
    plt.tight_layout()
    plt.xlim(0, 1)


def main():
    dataset = "../resources/datasets/v29-dataset1.csv"
    species = ("shark", "manta-ray")
    knn_tuning(dataset, species,
               {
                   "n_neighbors": [3, 5, 7, 9],
                   "metric": ["euclidean", "manhattan", "chebyshev"]
               })
    knn_pipelines(dataset, species,
                  {
                      "n_neighbors": 5,
                      "metric": "euclidean"
                  })
    plt.show()


if __name__ == "__main__":
    main()
