import os

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.model_selection._split import train_test_split


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
    train_split, test_split = dataset_split_class(gt_flat, 0.8)
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


def model_evaluation(model, x_test, y_test):
    # evaluate model on a testing set and plot confusion matrix

    # calculate confusion matrix and accuracy
    predictions = model.predict(x_test)
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(predictions)):
        classification_label = 1 if predictions[i][0] > 0.5 else 0
        confusion_matrix[int(y_test[i])][classification_label] += 1
    accuracy = (confusion_matrix[0][0] +
                confusion_matrix[1][1]) / len(predictions)

    # heatmap
    plt.figure()
    plt.title(f"Results test set - accuracy={accuracy}")
    plt.xlabel("true class")
    plt.ylabel("predicted class")
    seaborn.heatmap(confusion_matrix.astype(np.int),
                    annot=True, cmap="YlGnBu", fmt='d')
    plt.tight_layout()


def create_folders(dataset_base_dir, training_dir, test_dir):
    # create needed dataset folders
    needed_folders = [dataset_base_dir, training_dir, test_dir]
    for folder in needed_folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


def dataset_split_class(ground_truth, train_split_percentage):
    # split samples into train set and test set
    # maintaining class balance
    return train_test_split(list(range(len(ground_truth))), train_size=train_split_percentage)


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


def test_case1():
    # build dataset
    # build_dataset(["./resources/videos/feeding-v1-trim.mp4",
    #                "./resources/videos/feeding-v1-trim2.mp4",
    #                "./resources/videos/feeding-v2.mp4"],
    #               [[(5370, 9090)], [], [(0, 16550)]], (80, 50),
    #               "./resources/datasets/feeding-dataset/",
    #               "./resources/datasets/feeding-dataset/train-samples/",
    #               "./resources/datasets/feeding-dataset/test-samples/")

    # read data
    data = read_dataset("./resources/datasets/feeding-dataset/train-samples/",
                        "./resources/datasets/feeding-dataset/test-samples/")
    for data_set in data:
        print("images shape: ", data_set[0].shape)
        print("gt shape: ", data_set[1].shape)
    plot_class_balance(data)

    # train model
    model = define_layers()
    model.summary()

    x_train, y_train = data[0]
    x_test, y_test = data[1]
    training_history = model.fit(x_train, y_train, epochs=10, batch_size=40)
    training_plots(training_history)
    model_evaluation(model, x_test, y_test)

    plt.show()


def test_case2():
    # build dataset
    # build_dataset(["./resources/videos/feeding-v3.mp4"],
    #               [[(0, 40600)]], (80, 50),
    #               "./resources/datasets/feeding-surface-dataset/",
    #               "./resources/datasets/feeding-surface-dataset/train-samples/",
    #               "./resources/datasets/feeding-surface-dataset/test-samples/")

    # read data
    data = read_dataset("./resources/datasets/feeding-surface-dataset/train-samples/",
                        "./resources/datasets/feeding-surface-dataset/test-samples/")
    for data_set in data:
        print("images shape: ", data_set[0].shape)
        print("gt shape: ", data_set[1].shape)
    plot_class_balance(data)

    # train model
    model = define_layers()
    model.summary()

    x_train, y_train = data[0]
    x_test, y_test = data[1]
    training_history = model.fit(x_train, y_train, epochs=10, batch_size=40)
    training_plots(training_history)
    model_evaluation(model, x_test, y_test)

    plt.show()


def main():
    # script entry point
    test_case2()


if __name__ == '__main__':
    main()
