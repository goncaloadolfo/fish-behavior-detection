import os
import random

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.activations import relu, sigmoid
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.model_selection._split import train_test_split

from feeding.utils import ErrorTracker

# important keys
NR_CONV_LAYERS = "nr-conv-layers"
NR_FC_LAYERS = "nr-fc-layers"
HL_NR_NEURONS = "hl-nr-neurons"
CONV_NR_FILTERS = "conv-nr-filters"
LEARNING_RATE = "learning-rate"
DROPOUT_RATE = "dropout-rate"
ACTIVATION_FUNCTION = "activation-function"
LOSS_FUNCTION = "loss-function"


def build_dataset(videos, ground_truth, resize_resolution,
                  dataset_base_dir, training_dir, test_dir):
    """
    Save frames of feeding related videos into proper dataset 
    folders, with the specified resolution 
    """
    create_folders(dataset_base_dir, training_dir, test_dir)
    video_captures = set_video_captures(videos)

    # convert gt information into a flat 1D ground truth
    gt_flat = gt_flat_array(video_captures, ground_truth)
    train_split, test_split = dataset_split_class(video_captures,
                                                  gt_flat, 0.8)
    print("train samples: ", len(train_split))
    print("test samples: ", len(test_split))
    frame_counter = 0

    # for each video
    for video_capture in video_captures:
        while True:
            _, frame = video_capture.read()
            if frame is None:
                break
            print(f"processing frame [{frame_counter+1}/{len(gt_flat)}]")

            # define frame name based on feeding gt
            frame_name = f"frame-{frame_counter}"
            frame_name += f"-normal" if not is_feeding_frame(gt_flat, frame_counter) \
                else "-feeding"
            frame_path = training_dir + frame_name if frame_counter in train_split \
                else test_dir + frame_name

            # save frame
            frame_path = training_dir + frame_name if frame_counter in train_split \
                else test_dir + frame_name
            resized_frame = cv2.resize(frame, resize_resolution)
            cv2.imwrite(frame_path + ".jpg", resized_frame)

            frame_counter += 1

    # release resources
    release_video_captures(video_captures)


def read_dataset(training_dir, testing_dir):
    # read both data folders
    data = []
    processed_frames = 0

    for folder in [training_dir, testing_dir]:
        # folder imgs
        folder_imgs = []

        # images gt
        imgs_gt = []

        for img_name in os.listdir(folder):
            # update images and gt
            folder_imgs.append(cv2.imread(folder + img_name))
            filename = img_name.split(".")[0]
            imgs_gt.append(1.0 if filename.endswith("feeding") else 0.0)

            # progress control
            processed_frames += 1
            print("frames processed: ", processed_frames)

        # append folder data
        data.append((np.array(folder_imgs), np.array(imgs_gt)))

    return data


def plot_class_balance(data):
    # bar chart with class balance on each of the sets
    plt.title("Feeding Class Distribution")
    plt.xlabel("set")
    plt.ylabel("number of frames")

    bar_width = 0.2
    sets_center = np.array([1, 2])

    for feeding_label in range(2):
        nr_frames_each_set = [np.sum(gt == feeding_label) for _, gt in data]
        bars_positions = sets_center + bar_width/2 if feeding_label == 1 \
            else sets_center - bar_width/2
        bars_label = "normal frames" if feeding_label == 0 else "feeding frames"
        plt.bar(bars_positions, nr_frames_each_set,
                label=bars_label, width=bar_width)

    plt.xticks(sets_center, ["training set", "testing set"])
    plt.legend()


def define_layers():
    model = Sequential()

    # convolution layers
    model.add(Conv2D(5, kernel_size=5, activation='relu', input_shape=(50, 80, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(10, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # classification layers
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer="adam", loss="mean_squared_error",
                  metrics=["accuracy"])

    return model


def define_custom_layers(nr_conv_layers, nr_fc_layers, nr_perceptrons, nr_filters,
                         learning_rate=0.001, dropout_rate=None,
                         activation_function=relu, loss_function="mean_squared_error"):
    # create a custom model
    model = Sequential()

    # convolutional layers
    for _ in range(nr_conv_layers):
        model.add(Conv2D(nr_filters, kernel_size=5,
                         activation=activation_function,
                         input_shape=(50, 80, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # classification layers
    model.add(Flatten())
    for _ in range(nr_fc_layers):
        if dropout_rate is not None:
            model.add(Dropout(dropout_rate))
        model.add(Dense(nr_perceptrons, activation=activation_function))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss_function, metrics=["accuracy"])

    return model


def training_plots(history):
    # plot loss and training accuracy along epochs
    features = ["loss", "accuracy"]

    for feature in features:
        feature_values = history.history[feature]
        plt.figure()
        plt.title(f"{feature.title()} Along Training Epochs")
        plt.xlabel("epoch")
        plt.ylabel(feature)
        plt.plot(range(len(feature_values)), feature_values)


def check_different_architectures(arch_parameters, train_data, test_data):
    # evaluate different models in terms of net architecture
    nr_conv_layers, nr_fc_layers, nr_perc, nr_filters = arch_parameters
    total_models = len(nr_conv_layers) * len(nr_fc_layers) * \
        len(nr_perc) * len(nr_filters)
    current_model = 1
    results = []

    # grid search
    for ncl in nr_conv_layers:
        for nfl in nr_fc_layers:
            for nrp in nr_perc:
                for nrf in nr_filters:
                    print(f"model [{current_model}/{total_models}]")
                    model = define_custom_layers(ncl, nfl, nrp, nrf)
                    model.fit(train_data[0], train_data[1],
                              epochs=10, batch_size=40)
                    acc = model_evaluation(model, test_data[0],
                                           test_data[1], plot=False)
                    results.append((ncl, nfl, nrp, nrf, acc))
                    current_model += 1

    # print all results
    results.sort(key=lambda x: x[-1])
    print("--- Search Results ---")
    for model_results in results:
        print(f"\t- {model_results}")


def check_different_hyperparameters(architecture_params, hyperparameters,
                                    train_data, test_data):
    # architecture params
    nr_conv_layers = architecture_params[NR_CONV_LAYERS]
    nr_fc_layers = architecture_params[NR_FC_LAYERS]
    hl_nr_neurons = architecture_params[HL_NR_NEURONS]
    conv_nr_filters = architecture_params[CONV_NR_FILTERS]

    # hyparameters grid
    learning_rate_list = hyperparameters[LEARNING_RATE]
    dropout_rate_list = hyperparameters[DROPOUT_RATE]
    activation_function_list = hyperparameters[ACTIVATION_FUNCTION]
    loss_function_list = hyperparameters[LOSS_FUNCTION]

    # status vars
    total_models = len(learning_rate_list) * len(dropout_rate_list) * \
        len(activation_function_list) * len(loss_function_list)
    current_model = 1
    results = []

    # grid search
    for lr in learning_rate_list:
        for dr in dropout_rate_list:
            for af in activation_function_list:
                for lf in loss_function_list:
                    print(f"model [{current_model}/{total_models}]")
                    model = define_custom_layers(nr_conv_layers, nr_fc_layers,
                                                 hl_nr_neurons, conv_nr_filters,
                                                 lr, dr, af, lf)
                    model.fit(train_data[0], train_data[1],
                              epochs=10, batch_size=40)
                    acc = model_evaluation(model, test_data[0],
                                           test_data[1], plot=False)
                    results.append((lr, dr, af, lf, acc))
                    current_model += 1

    # print all results
    results.sort(key=lambda x: x[-1])
    print("--- Search Results ---")
    for model_results in results:
        print(f"\t- {model_results}")


def model_evaluation(model, x_test, y_test, plot=True):
    # evaluate model on a testing set and plot confusion matrix

    # error tracker
    error_tracker = ErrorTracker()

    # calculate confusion matrix and accuracy
    predictions = model.predict(x_test)
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(predictions)):
        classification_label = 1 if predictions[i][0] > 0.5 else 0
        true_label = int(y_test[i])
        confusion_matrix[true_label][classification_label] += 1

        # error
        if true_label != classification_label:
            error_tracker.append_new_timestamp(i+1)

    accuracy = (confusion_matrix[0][0] +
                confusion_matrix[1][1]) / len(predictions)

    if plot:
        # heatmap
        plt.figure()
        plt.title(f"Results test set - accuracy={accuracy}")
        plt.xlabel("true class")
        plt.ylabel("predicted class")
        seaborn.heatmap(confusion_matrix.astype(np.int),
                        annot=True, cmap="YlGnBu", fmt='d')
        plt.tight_layout()

        # errors timeline
        feeding_start = np.where(y_test == 1)
        feeding_start = feeding_start[0][0] if len(
            feeding_start[0]) != 0 else None
        error_tracker.draw_errors_timeline(1, len(predictions),
                                           "Errors Timeline", feeding_start)

    return accuracy


def create_folders(dataset_base_dir, training_dir, test_dir):
    # create needed dataset folders
    needed_folders = [dataset_base_dir, training_dir, test_dir]
    for folder in needed_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


def dataset_split_class(video_captures, gt_flat, train_split_percentage):
    # split samples into train set and test set
    # maintaining class balance and avoiding consecutive frames
    train_set = np.array([], dtype=np.int)
    test_set = np.array([], dtype=np.int)
    video_sizes = [0] + [int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                         for video_capture in video_captures]

    for i in range(len(video_sizes)-1):
        # get flat gt for the frames of this video
        gt_start_index = sum(video_sizes[:i+1])
        gt_end_index = sum(video_sizes[:i+1]) + video_sizes[i+1]
        video_gt = gt_flat[gt_start_index: gt_end_index]
        print("video frames: ", len(video_gt))

        # train-test split each of the classes
        for feeding_class in [0, 1]:
            # class indexes
            class_frames_indexes = np.where(video_gt == feeding_class)[0]
            print(f"class {feeding_class} samples: ",
                  len(class_frames_indexes))
            if len(class_frames_indexes) != 0:
                class_start = class_frames_indexes[0] + gt_start_index
                class_end = class_frames_indexes[-1] + gt_start_index

                # split train-test
                train_split, test_split = train_test_split(
                    list(range(class_start, class_end+1)), shuffle=False,
                    train_size=int(len(class_frames_indexes)
                                   * train_split_percentage)
                )
                print("train samples: ", len(train_split))
                print("test samples: ", len(test_split))
                train_set = np.hstack((train_set, train_split))
                test_set = np.hstack((test_set, test_split))

    return train_set, test_set


def is_feeding_frame(ground_truth, frame_index):
    # return whetever or not a given frame is a feeding frame
    return ground_truth[frame_index] == 1


def set_video_captures(videos):
    # create VideoCapture objs for each video path
    return [cv2.VideoCapture(video_path) for video_path in videos]


def release_video_captures(vc_objs):
    # release all VideoCapture objs
    for vc in vc_objs:
        vc.release()


def gt_flat_array(video_captures, ground_truth):
    """
    Produce a flat feeding ground truth array of all videos
    """
    final_gt = np.array([])

    for i in range(len(video_captures)):
        # form video ground truth
        video_capture = video_captures[i]
        video_total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_gt = np.zeros((video_total_frames,))

        for t_initial, t_final in ground_truth[i]:
            video_gt[t_initial: t_final+1] = 1

        # stack with previous videos ground truth
        final_gt = np.hstack((final_gt, video_gt))

    return final_gt.astype(np.int)


def build_datasets():
    # bottom feeding dataset
    build_dataset(["./resources/videos/feeding-v1-trim.mp4",
                   "./resources/videos/feeding-v1-trim2.mp4",
                   "./resources/videos/feeding-v2.mp4"],
                  [[(5370, 9090)], [], [(0, 16550)]], (80, 50),
                  "./resources/datasets/feeding-dataset/",
                  "./resources/datasets/feeding-dataset/train-samples/",
                  "./resources/datasets/feeding-dataset/test-samples/")

    # surface feeding dataset
    build_dataset(["./resources/videos/feeding-v3.mp4"],
                  [[(0, 40600)]], (80, 50),
                  "./resources/datasets/feeding-surface-dataset/",
                  "./resources/datasets/feeding-surface-dataset/train-samples/",
                  "./resources/datasets/feeding-surface-dataset/test-samples/")

    # go pro - bottom feeding
    build_dataset(["./resources/videos/feeding-v4.mp4"],
                  [[(14940, 21600)]], (80, 50),
                  "./resources/datasets/gopro-feeding-dataset/",
                  "./resources/datasets/gopro-feeding-dataset/train-samples/",
                  "./resources/datasets/gopro-feeding-dataset/test-samples/")


def train_and_evaluate(train_data, test_data):
    # define model
    model = define_layers()
    model.summary()

    # train
    x_train, y_train = train_data
    x_test, y_test = test_data
    training_history = model.fit(x_train, y_train, epochs=10, batch_size=40)

    # evaluate
    training_plots(training_history)
    model_evaluation(model, x_test, y_test)


def baseline_results():
    # bottom dataset
    bottom_data = read_dataset("./resources/datasets/feeding-dataset/train-samples/",
                               "./resources/datasets/feeding-dataset/test-samples/")
    plot_class_balance(bottom_data)
    train_and_evaluate(bottom_data[0], bottom_data[1])

    # # surface dataset
    # random.seed(0)
    # surface_data = read_dataset(
    #     "./resources/datasets/feeding-surface-dataset/train-samples/",
    #     "./resources/datasets/feeding-surface-dataset/test-samples/"
    # )
    # plot_class_balance(surface_data)
    # train_and_evaluate(surface_data[0], surface_data[1])

    # # go pro - bottom dataset
    # random.seed(0)
    # gopro_bottom_data = read_dataset(
    #     "./resources/datasets/gopro-feeding-dataset/train-samples/",
    #     "./resources/datasets/gopro-feeding-dataset/test-samples/"
    # )
    # x_train = np.vstack((bottom_data[0][0], bottom_data[1][0]))
    # y_train = np.hstack((bottom_data[0][1], bottom_data[1][1]))
    # x_test = np.vstack((gopro_bottom_data[0][0], gopro_bottom_data[1][0]))
    # y_test = np.hstack((gopro_bottom_data[0][1], gopro_bottom_data[1][1]))
    # plot_class_balance([(x_train, y_train), (x_test, y_test)])
    # train_and_evaluate((x_train, y_train), (x_test, y_test))

    plt.show()


def tuning_results():
    # grids of parameters
    architecture_parameters = [(1, 2), (1, 2), (80, 120), (5, 15)]
    hyperparameters = {
        LEARNING_RATE: [0.001, 0.01],
        DROPOUT_RATE: [None, 0.2],
        ACTIVATION_FUNCTION: [relu, sigmoid],
        LOSS_FUNCTION: ["mean_squared_error", "binary_crossentropy"]
    }

    # bottom dataset
    # bottom_data = read_dataset("./resources/datasets/feeding-dataset/train-samples/",
    #                            "./resources/datasets/feeding-dataset/test-samples/")
    # check_different_architectures(architecture_parameters,
    #                               bottom_data[0], bottom_data[1])

    # best_archicture = {
    #     NR_CONV_LAYERS: 1,
    #     NR_FC_LAYERS: 2,
    #     HL_NR_NEURONS: 120,
    #     CONV_NR_FILTERS: 5
    # }
    # check_different_hyperparameters(best_archicture, hyperparameters,
    #                                 bottom_data[0], bottom_data[1])

    # surface dataset
    surface_data = read_dataset(
        "./resources/datasets/feeding-surface-dataset/train-samples/",
        "./resources/datasets/feeding-surface-dataset/test-samples/"
    )
    # check_different_architectures(architecture_parameters,
    #                               surface_data[0], surface_data[1])

    best_archicture = {
        NR_CONV_LAYERS: 1,
        NR_FC_LAYERS: 2,
        HL_NR_NEURONS: 120,
        CONV_NR_FILTERS: 5
    }
    check_different_hyperparameters(best_archicture, hyperparameters,
                                    surface_data[0], surface_data[1])


def main():
    # datasets
    # build_datasets()

    # baseline results for the different datasets
    baseline_results()

    # tuning
    # tuning_results()


if __name__ == '__main__':
    main()
