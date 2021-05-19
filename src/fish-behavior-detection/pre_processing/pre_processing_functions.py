import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from labeling.trajectory_labeling import read_episodes
from trajectory_features.trajectory_feature_extraction import read_dataset
from trajectory_reader.visualization import simple_line_plot


class CorrelatedVariablesRemoval:

    def __init__(self, thr):
        self.__thr = thr
        self.__correlation_matrix = None

    def fit(self, x, _):
        x = np.array(x)
        self.__correlation_matrix = np.corrcoef(x.T)
        return self

    def transform(self, x):
        # erase variables with high correlation value
        x = np.array(x)
        variables_to_erase = []

        # go trough correlation values (only half of the matrix)
        for var1_index in range(x.shape[1]):
            for var2_index in range(var1_index + 1, x.shape[1]):
                # too high correlation and none of the variables was already erased
                if abs(self.__correlation_matrix[var1_index, var2_index]) \
                        >= self.__thr \
                        and var1_index not in variables_to_erase \
                        and var2_index not in variables_to_erase:
                    variables_to_erase.append(var1_index)

        # delete the columns of the correlated variables
        return np.delete(x, variables_to_erase, axis=1)

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)


def load_data(dataset_path, species,
              moments_path="resources/classification/v29-interesting-moments.csv"):
    # read dataset
    samples, species_gt, feature_descriptions = read_dataset(
        dataset_path
    )

    # get samples from species received as argument
    x = []
    fish_ids = []
    for i in range(len(samples)):
        if species_gt[i] in species:
            fish_ids.append(i)
            x.append(samples[i])

    return (np.array(x),
            get_episode_gt(fish_ids,
                           moments_path,
                           "interesting"
                           ),
            feature_descriptions
            )


def get_episode_gt(fish_ids, episodes_path, episode_label):
    episodes = read_episodes(episodes_path)
    gt = []

    for fish_id in fish_ids:
        # search for an episode of this fish according to the received label
        found = False
        for episode in episodes:
            if episode.fish_id == fish_id and episode.description == episode_label:
                found = True
                break

        # append gt
        fish_gt = 1 if found else 0
        gt.append(fish_gt)

    return gt


def z_normalization(samples):
    # normalize the received data to z score
    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)

    # delete variables with 0 variance
    variable_indexes = np.where(stds == 0)
    new_samples = np.delete(samples, variable_indexes, axis=1)
    new_means = np.delete(means, variable_indexes)
    new_stds = np.delete(stds, variable_indexes)

    return (new_samples - new_means) / new_stds


def analyze_pca_components(samples, subtitle=""):
    # apply pca using all components
    pca = PCA().fit(samples)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # plot cumulative variance
    plt.figure()
    simple_line_plot(plt.gca(), range(len(cumulative_variance)), cumulative_variance,
                     "Explained variance by pca components" + subtitle,
                     "explained variance", "n components", "-o")


def apply_pca(samples, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(samples)


if __name__ == "__main__":
    x_all, _, _ = load_data(
        "../resources/datasets/v29-dataset1.csv", ("shark",))
    x_sharks, _, _ = load_data(
        "../resources/datasets/v29-dataset1.csv", ("shark",))
    x_mantas, _, _ = load_data(
        "../resources/datasets/v29-dataset1.csv", ("manta-ray",))

    analyze_pca_components(x_all, "- all species")
    analyze_pca_components(x_sharks, "- sharks")
    analyze_pca_components(x_mantas, "- mantas")
    plt.show()
