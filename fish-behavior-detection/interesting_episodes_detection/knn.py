from collections import defaultdict

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pre_processing.pre_processing import CorrelatedVariablesRemoval, load_data
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from trajectory_reader.visualization import simple_bar_chart, simple_line_plot

from interesting_episodes_detection.evaluation import holdout_prediction

SEED = 2


def knn_tuning(dataset, species, parameters_grid):
    x, y_original, feature_names = load_data(dataset, species)
    x = StandardScaler().fit_transform(x, y_original)
    x, y = SMOTE(random_state=SEED).fit_resample(x, y_original)

    scores = defaultdict(list)
    ns = parameters_grid["n_neighbors"]
    metrics = parameters_grid["metric"]

    for metric in metrics:
        for n in ns:
            knn = KNeighborsClassifier(n_neighbors=n, metric=metric,
                                       algorithm="brute")
            predictions = holdout_prediction(knn, x, y)
            scores[metric].append(accuracy_score(
                y[:len(y_original)], predictions[:len(y_original)]
            ))

    plt.figure()
    for metric in metrics:
        simple_line_plot(plt.gca(), ns, scores[metric], "KNN Tuning",
                         "accuracy", "k", "--o", label=metric)
    plt.ylim((0, 1))
    plt.grid()
    plt.legend()


def knn_pipelines(dataset, species, parameters):
    x, y, features_description = load_data(dataset, species)
    knn = KNeighborsClassifier(
        n_neighbors=parameters["n_neighbors"],
        algorithm="brute",
        metric=parameters["metric"]
    )
    balancer = SMOTE(random_state=SEED)
    normalizer = StandardScaler()
    select_kbest = SelectKBest(k=20)
    pca = PCA(n_components=10)
    correlation_removal = CorrelatedVariablesRemoval(0.9)

    pipelines = (
        [("knn", knn)],
        [("normalizer", normalizer), ("knn", knn)],
        [("balancer", balancer), ("knn", knn)],
        [("normalizer", normalizer), ("balancer", balancer), ("knn", knn)],

        [("normalizer", normalizer), ("pca", pca),
         ("balancer", balancer), ("knn", knn)],

        [("normalizer", normalizer), ("select kbest", select_kbest),
         ("balancer", balancer), ("knn", knn)],

        [("normalizer", normalizer), ("correlation removal", correlation_removal),
         ("balancer", balancer), ("knn", knn)],
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
    simple_bar_chart(plt.gca(), range(len(pipelines)), scores,
                     "KNN Pipelines", "accuracy", "pipeline")
    model_descriptions = ['+'.join([n[0] for n in pipeline])
                          for pipeline in pipelines]
    plt.grid()
    plt.gca().set_xticks(range(len(pipelines)))
    plt.gca().set_xticklabels(model_descriptions, rotation=30)
    plt.tight_layout()


if __name__ == "__main__":
    dataset = "resources/datasets/v29-dataset1.csv"
    species = ("shark", "manta-ray")
    knn_tuning(dataset, species,
               {
                   "n_neighbors": [5, 7, 9],
                   "metric": ["euclidean", "manhattan", "chebyshev"]
               })
    knn_pipelines(dataset, species,
                  {
                      "n_neighbors": 9,
                      "metric": "manhattan"
                  })
    plt.show()
