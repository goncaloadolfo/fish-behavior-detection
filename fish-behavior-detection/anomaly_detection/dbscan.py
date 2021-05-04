import numpy as np
from matplotlib import pyplot as plt
from pre_processing.pre_processing import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from trajectory_reader.visualization import histogram, simple_line_plot
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import silhouette_score


def analyze_distances(dataset, species, metric, n):
    x, y, _ = load_data(dataset, species)
    x = StandardScaler().fit_transform(x)
    knn = KNeighborsClassifier(n_neighbors=n, metric=metric,
                               algorithm="brute")
    knn.fit(x, y)

    distances, _ = knn.kneighbors(n_neighbors=n)
    plt.figure()
    _, bins, _ = histogram(plt.gca(), distances.flatten(),
                           f"Distance to {n} Nearest Points ({metric})", "cumulative density", "distance",
                           density=True, cumulative=True)
    plt.xticks(bins)


def dbscan_tuning(dataset, species, min_samples, epsilons, metric):
    x, _, _ = load_data(dataset, species)
    x = StandardScaler().fit_transform(x)

    fig, axs = plt.subplots()
    fig2, axs2 = plt.subplots()

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
                         f"Silhouette to different epsilons/min_samples ({metric})",
                         "silhouette", "epsilon", marker="--o", label=f"min_samples={ms}")
        simple_line_plot(axs2, epsilons, nr_outliers,
                         f"Number of outliers to different epsilons/min_samples ({metric})",
                         "#outliers", "epsilon", marker="--o", label=f"min_samples={ms}")

    plt.legend()
    plt.grid()


if __name__ == "__main__":
    dataset = "resources/datasets/v29-dataset1.csv"
    species = ("shark", "manta-ray")

    analyze_distances(dataset, species, "manhattan", 7)
    analyze_distances(dataset, species, "euclidean", 7)

    dbscan_tuning(dataset, species, [5, 7, 9],
                  [73.3, 84.9, 96.4, 108.0], "manhattan")
    dbscan_tuning(dataset, species, [5, 7, 9],
                  [10.72, 12.35, 13.97, 15.59], "euclidean")

    plt.show()
