import numpy as np
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from interesting_episodes_detection.evaluation import holdout_prediction
from pre_processing.pre_processing import CorrelatedVariablesRemoval, load_data
from trajectory_reader.visualization import simple_bar_chart, simple_hbar_chart


def random_forest_tuning(dataset, species, parameters_grid):
    x, y, _ = load_data(dataset, species)
    random_forest = RandomForestClassifier()
    gridsearch_obj = GridSearchCV(estimator=random_forest, param_grid=parameters_grid,
                                  scoring="accuracy", cv=KFold(n_splits=len(x)), verbose=4)
    gridsearch_obj.fit(x, y)
    print("best parameters: ", gridsearch_obj.best_estimator_)
    print("best score: ", gridsearch_obj.best_score_)


def random_forest_pipelines(dataset, species, parameters):
    x, y, features_description = load_data(dataset, species)
    random_forest = RandomForestClassifier(n_estimators=parameters["n_estimators"],
                                           criterion=parameters["criterion"],
                                           max_depth=parameters["max_depth"],
                                           max_features=parameters["max_features"],
                                           min_samples_leaf=parameters["min_samples_leaf"],
                                           min_samples_split=parameters["min_samples_split"],
                                           bootstrap=parameters["bootstrap"],
                                           random_state=parameters["random_state"]
                                           )
    balancer = SMOTE()
    normalizer = StandardScaler()
    select_kbest = SelectKBest(k=20)
    pca = PCA(n_components=10)
    correlation_removal = CorrelatedVariablesRemoval(0.9)

    pipelines = [
        [("rf", random_forest)],
        [("normalizer", normalizer), ("rf", random_forest)],
        [("balancer", balancer), ("rf", random_forest)],
        [("balancer", balancer), ("normalizer", normalizer), ("rf", random_forest)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("pca", pca), ("rf", random_forest)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("select kbest", select_kbest), ("rf", random_forest)],

        [("balancer", balancer), ("normalizer", normalizer),
         ("correlation removal", correlation_removal), ("rf", random_forest)],
    ]

    original_x = x.copy()
    original_y = y.copy()
    scores = []

    for pipeline in pipelines:
        x = original_x
        y = original_y

        for pipeline_node in pipeline:
            if pipeline_node[0] == "rf":
                rf = pipeline_node[1].fit(x, y)
                predictions = holdout_prediction(rf, x, y)[:len(original_y)]
                scores.append(accuracy_score(original_y, predictions))
            elif pipeline_node[0] == "balancer":
                x, y = pipeline_node[1].fit_resample(x, y)
            else:
                x = pipeline_node[1].fit_transform(x, y)

    plt.figure()
    model_descriptions = ['+'.join([n[0] for n in pipeline])
                          for pipeline in pipelines]
    simple_hbar_chart(plt.gca(), scores[::-1], model_descriptions[::-1],
                      "Random Forest Pipelines", "accuracy", "pipeline")

    plt.grid()
    plt.tight_layout()
    plt.xlim(0, 1)


def plot_features_importance(pipelines, scores, features_description, n):
    while True:
        best_pipeline_index = np.argmax(scores)
        best_pipeline = pipelines[best_pipeline_index]

        steps = [step[0] for step in best_pipeline]
        if "pca" not in steps and "select kbest" not in steps:
            break

        del scores[best_pipeline_index]
        del pipelines[best_pipeline_index]

    rf = best_pipeline[-1][1]
    features_importance = list(
        zip(rf.feature_importances_, features_description)
    )
    features_importance.sort(key=lambda x: x[0], reverse=True)
    most_important_features = features_importance[:n]

    plt.figure()
    simple_bar_chart(plt.gca(), range(n), [feature[0] for feature in most_important_features],
                     "Most important features", "importance", "feature")
    plt.gca().set_xticks(range(n))
    plt.gca().set_xticklabels([feature[1] for feature in most_important_features],
                              rotation=30)
    plt.tight_layout()


def main():
    # parameters_grid = {"n_estimators": [5, 10, 20],
    #                    "criterion": ["gini", "entropy"],
    #                    "max_depth": [5, 10, 20],
    #                    "max_features": ["sqrt", "log2"],
    #                    "min_samples_leaf": [3, 5, 7],
    #                    "min_samples_split": [2, 4, 6],
    #                    "bootstrap": [False, True],
    #                    "random_state": [0, 1, 2, 3, 4]}
    # random_forest_tuning("../resources/datasets/v29-dataset1.csv",
    #                      ("shark", "manta-ray"), parameters_grid)

    # seed 1
    random_forest_pipelines("../resources/datasets/v29-dataset1.csv", ("shark", "manta-ray"),
                            {"n_estimators": 5,
                             "criterion": "entropy",
                             "max_depth": 5,
                             "max_features": "sqrt",
                             "min_samples_leaf": 3,
                             "min_samples_split": 2,
                             "bootstrap": True,
                             "random_state": 4})
    plt.show()


if __name__ == "__main__":
    main()
