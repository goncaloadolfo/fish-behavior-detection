import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interesting_episodes_detection.evaluation import (METRICS,
                                                       evaluation_metrics,
                                                       holdout_prediction,
                                                       plot_metrics)
from pre_processing.pre_processing_functions import CorrelatedVariablesRemoval, load_data

SEED = 0


def compare_gnb_pipelines(path_dataset, species, balance=False):
    x, y, _ = load_data(path_dataset, species)

    if balance:
        x, y = SMOTE(random_state=SEED).fit_resample(x, y)

    pipelines = (
        # gaussian NB with no pre processing
        Pipeline([("gnb", GaussianNB())]),

        # removing correlated variables
        Pipeline([("cvr", CorrelatedVariablesRemoval(0.9)),
                  ("gnb", GaussianNB())]),

        # normalization only
        Pipeline([("z-norm", StandardScaler()),
                  ("gnb", GaussianNB())]),

        # Applying PCA
        Pipeline([("z-norm", StandardScaler()),
                  ("pca", PCA(6)),
                  ("gnb", GaussianNB())])
    )

    # plot results of each pipeline
    nr_pipelines = len(pipelines)
    nr_metrics = len(METRICS)
    x_positions = np.arange(nr_metrics)
    bar_width = 0.2

    plt.figure()
    for i, pipeline in enumerate(pipelines):
        predictions = holdout_prediction(pipeline, x, y)
        results = evaluation_metrics(y, predictions)

        # bar positions
        if i < nr_pipelines / 2:
            bar_positions = x_positions - bar_width * \
                            (nr_metrics / 2 - i) + bar_width / 2
        else:
            bar_positions = x_positions + bar_width * \
                            (i + 1 - nr_metrics / 2) - bar_width / 2

        plot_metrics(results, bar_positions,
                     '+'.join(pipeline.named_steps.keys()),
                     bar_width)

    # plot settings
    balance_str = "balanced" if balance else "not balanced"
    plt.title(f"Evaluation of GNB pipelines " +
              f"({','.join(list(species))}, {balance_str})")
    plt.gca().set_xticks(x_positions)
    plt.gca().set_xticklabels(METRICS)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()


if __name__ == "__main__":
    np.random.seed(0)
    dataset = "resources/datasets/v29-dataset1.csv"

    compare_gnb_pipelines(dataset, ("shark", "manta-ray"))
    # compare_gnb_pipelines(dataset, ("shark",))
    # compare_gnb_pipelines(dataset, ("manta-ray",))

    compare_gnb_pipelines(dataset, ("shark", "manta-ray"), balance=True)
    # compare_gnb_pipelines(dataset, ("manta-ray",), balance=True)
    # compare_gnb_pipelines(dataset, ("shark",), balance=True)

    plt.show()
