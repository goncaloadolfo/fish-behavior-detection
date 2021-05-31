import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

from interesting_episodes_detection.evaluation import holdout_prediction
from pre_processing.pre_processing_functions import CorrelatedVariablesRemoval, load_data
from trajectory_reader.visualization import simple_hbar_chart

SEED = 0


def dt_tunning(dataset, species, parameters_grid):
    x, y, feature_names = load_data(dataset, species)
    dt = DecisionTreeClassifier(random_state=SEED)
    gridsearch_obj = GridSearchCV(estimator=dt, param_grid=parameters_grid,
                                  scoring="accuracy", cv=KFold(n_splits=len(x)))
    gridsearch_obj.fit(x, y)
    print("best parameters: ", gridsearch_obj.best_estimator_)
    print("Tree depth: ", gridsearch_obj.best_estimator_.tree_.max_depth)
    print("best score: ", gridsearch_obj.best_score_)


def dt_pipelines(dataset, species, parameters):
    x, y, features_description = load_data(dataset, species)
    dt = DecisionTreeClassifier(
        criterion=parameters["criterion"],
        min_samples_split=parameters["min_samples_split"],
        min_samples_leaf=parameters["min_samples_leaf"],
        min_impurity_decrease=parameters["min_impurity_decrease"],
        random_state=SEED
    )
    balancer = SMOTE(random_state=SEED)
    normalizer = StandardScaler()
    select_kbest = SelectKBest(k=20)
    pca = PCA(n_components=10)
    correlation_removal = CorrelatedVariablesRemoval(0.9)

    pipelines = (
        [("dt", dt)],
        [("normalizer", normalizer), ("dt", dt)],
        [("balancer", balancer), ("dt", dt)],
        [("balancer", balancer), ("normalizer", normalizer), ("dt", dt)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("pca", pca), ("dt", dt)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("select kbest", select_kbest), ("dt", dt)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("correlation removal", correlation_removal), ("dt", dt)],
    )

    original_x = x.copy()
    original_y = y.copy()
    scores = []

    for pipeline in pipelines:
        x = original_x
        y = original_y

        for pipeline_node in pipeline:
            if pipeline_node[0] == "dt":
                dt = pipeline_node[1].fit(x, y)
                predictions = holdout_prediction(dt, x, y)[:len(original_y)]
                scores.append(accuracy_score(original_y, predictions))
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
    plot_best_estimator_tree(pipelines, scores, features_description)


def plot_best_estimator_tree(pipelines, scores, features_description):
    best = np.argmax(scores)
    plt.figure(figsize=(14, 12), dpi=80)
    plt.title("Decision tree")
    plot_tree(pipelines[best][-1][1],
              feature_names=features_description, fontsize=8, ax=plt.gca())


def main():
    dataset = "resources/datasets/v29-dataset1.csv"

    parameters_grid = {
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 3, 5],
        "min_impurity_decrease": [0, 0.1, 0.2]
    }
    dt_tunning(dataset, ("shark", "manta-ray"), parameters_grid)

    dt_pipelines(dataset,
                 ("shark", "manta-ray"),
                 {
                     "criterion": "entropy",
                     "min_samples_split": 2,
                     "min_samples_leaf": 1,
                     "min_impurity_decrease": 0
                 })
    plt.show()


if __name__ == "__main__":
    main()
