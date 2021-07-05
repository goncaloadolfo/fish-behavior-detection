import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
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
            imgs_gt.append(1 if filename.endswith("feeding") else 0)

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


def main():
    # script entry point

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
    plt.show()


if __name__ == '__main__':
    main()
