from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from pre_processing.pre_processing_functions import load_data

PRECISIONS_KEY = "precisions"
RECALL_KEY = "recalls"


def models_pr_curve(models, data, ground_truth):
    precisions_recalls = {}

    for model in models:
        model_name = get_model_name(model)

        # get precisions and recalls for different thrs
        predict_probas = get_probas_pred(model, data, ground_truth)
        precisions, recalls, _ = precision_recall_curve(
            ground_truth, predict_probas)

        # append model precisions/recalls
        precisions_recalls[model_name] = {
            PRECISIONS_KEY: precisions,
            RECALL_KEY: recalls
        }

    return precisions_recalls


def get_model_name(model):
    if type(model) == Pipeline:
        return list(model.named_steps.values())[-1].__class__.__name__
    else:
        return model.__class__.__name__


def get_probas_pred(model, data, ground_truth):
    predict_probas = []

    for i in range(len(data)):
        # hold out
        evaluation_sample = data[i]
        model.fit(np.delete(data, (i), axis=0),
                  np.delete(ground_truth, (i), axis=0))

        # append prediction probability
        try:
            predict_probas.append(
                model.predict_proba([evaluation_sample])[0][1]
            )
        except:
            predict_probas.append(
                model.decision_function([evaluation_sample])[0]
            )

    return predict_probas


def plot_precisions_recalls(models_precisions_recalls):
    setup_plot("Precision Recall Curves")

    # curve for each model
    for model_name, precisions_recalls in models_precisions_recalls.items():
        # get precisions and recalls
        precisions = precisions_recalls[PRECISIONS_KEY]
        recalls = precisions_recalls[RECALL_KEY]

        # curve and AUC
        plt.plot(recalls, precisions, label=model_name)
        print(f"{model_name} AUC: {auc(recalls, precisions)}")

    plt.legend(fontsize=14)


def setup_plot(title):
    # new figure
    plt.figure()

    # labels
    plt.title(title, fontsize=14)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)

    # axis limits
    plt.xlim([0, 1])
    plt.ylim([0, 1])


# region imperative/tests

def models_curves_test():
    np.random.seed(777)
    # read and balance data
    x, y, feature_names = load_data("resources/datasets/v29-dataset1.csv",
                                    ("shark", "manta-ray"))
    x, y = SMOTE().fit_resample(x, y)

    # needed components
    normalizer = StandardScaler()
    dt = DecisionTreeClassifier(criterion="entropy")
    knn = KNeighborsClassifier(metric="euclidean")
    gnb = GaussianNB()
    rf = RandomForestClassifier(n_estimators=5, criterion="entropy",
                                max_depth=5, max_features="sqrt",
                                min_samples_leaf=3)
    svm = SVC(C=0.01, kernel="poly", gamma=1)

    # pipelines
    models = [
        dt, gnb, rf,
        Pipeline([("normalizer", normalizer), ("knn", knn)]),
        Pipeline([("normalizer", normalizer), ("svm", svm)])
    ]

    # precision-recall curves
    precisions_recalls = models_pr_curve(models, x, y)
    plot_precisions_recalls(precisions_recalls)
    plt.show()


def main():
    models_curves_test()

# endregion


if __name__ == "__main__":
    main()
