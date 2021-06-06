import cv2
import pre_processing.pre_processing_functions as ppm
from anomaly_detection.switching_model.model_initialization import SquareGrid, initialize_fields, initialize_init_position_priors, load_model_from_file, save_model_to_file, visualize_fields
from anomaly_detection.switching_model.model_maximization import \
    fit_switching_model
from labeling.trajectory_labeling import read_episodes
from matplotlib import pyplot as plt
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import smooth_positions
from trajectory_features.trajectory_feature_extraction import \
    exponential_weights
from trajectory_reader.trajectories_reader import read_fishes_filter
from anomaly_detection.switching_model.model_expectation import joint_probability


def group_by_normal_interesting(fishes, episodes):
    fishes_groups = {
        "normal": [],
        "interesting": []
    }

    for fish in fishes:
        found_episode = False
        for episode in episodes:
            if episode.fish_id == fish.fish_id:
                found_episode = True
                break

        if found_episode:
            fishes_groups["interesting"].append(fish)

        else:
            fishes_groups["normal"].append(fish)

    return fishes_groups


def illustrate_model_results(model_description, complete_likelihoods, fields_images):
    plt.figure()
    plt.title(f"Complete Log Likelihoods Per Iteration - {model_description}")
    plt.xlabel("iteration")
    plt.ylabel("complete log likelihood")
    plt.plot(complete_likelihoods)

    for i in range(len(fields_images)):
        cv2.imshow(f"{model_description} - field {i}", fields_images[i])


def train_save_model(model_description, trajectories, background_image, resolution,
                     nr_nodes, nr_fields, delta, alpha, iterations, show_results=False):
    grid = SquareGrid(resolution[1], resolution[0], nr_nodes, nr_fields)
    initialize_init_position_priors(grid, trajectories)
    initialize_fields(grid, trajectories)

    complete_likelihoods = fit_switching_model(grid, trajectories,
                                               delta, alpha, iterations)
    save_model_to_file(grid,
                       f"resources/models/{model_description}-n{nr_nodes}-f{nr_fields}-d{delta}-a{alpha}.model")

    if show_results:
        fields_images = visualize_fields(background_image, grid, 10)
        illustrate_model_results(model_description, complete_likelihoods,
                                 fields_images)
        plt.show()
        cv2.destroyAllWindows()

    return grid, complete_likelihoods


def classify_trajectory(models, trajectory):
    best_fit_model = None
    best_likelihood = -1

    for model_description in models:
        model = models[model_description]
        likelihood = joint_probability(model, trajectory)

        if likelihood > best_likelihood:
            best_fit_model = model_description
            best_likelihood = likelihood

    return best_fit_model


def tune_model(model_description, trajectories, resolution, nr_nodes_list, nr_fields_list,
               delta_list, alpha_list, iterations_list):
    likelihood_results = []
    nr_total_models = len(nr_nodes_list) * len(nr_fields_list) * \
        len(delta_list) * len(alpha_list) * len(iterations_list)
    model_counter = 0

    for nr_nodes in nr_nodes_list:
        for nr_fields in nr_fields_list:
            for delta in delta_list:
                for alpha in alpha_list:
                    for nr_iterations in iterations_list:
                        model_counter += 1
                        print(
                            f"tuning model [{model_counter}/{nr_total_models}]"
                        )
                        _, likelihoods = train_save_model(model_description, trajectories,
                                                          None, resolution, nr_nodes, nr_fields,
                                                          delta, alpha, nr_iterations, show_results=False)
                        likelihood_results.append(
                            (nr_nodes, nr_fields, delta, alpha, nr_iterations, likelihoods[-1]))

    likelihood_results.sort(key=lambda x: x[-1])
    for i in range(len(likelihood_results)):
        model = likelihood_results[i]
        print("\nModel: ", i)
        print("\tnumber of nodes: ", model[0])
        print("\tnumber of fields: ", model[1])
        print("\tdelta: ", model[2])
        print("\talpha: ", model[3])
        print("\tnumber of iterations: ", model[4])
        print("\tcomplete log likelihood: ", model[5])


def train_models():
    video_path = "resources/videos/v29.m4v"
    fishes_path = "resources/detections/v29-fishes.json"
    species_path = "resources/classification/species-gt-v29.csv"
    episodes_path = "resources/classification/v29-interesting-moments.csv"

    nr_nodes = 16
    nr_fields = 4
    alpha = 1
    delta = 250
    iterations = 10

    resolution = ppm.video_resolution(video_path)
    background_image = ppm.background_image_estimation(video_path, 300)

    fishes = read_fishes_filter(fishes_path, species_path,
                                ("shark", "manta-ray"))
    interesting_episodes_gt = read_episodes(episodes_path)
    fishes_groups = group_by_normal_interesting(fishes,
                                                interesting_episodes_gt)

    for group in fishes_groups:
        # for group in ["interesting"]:
        fishes = fishes_groups[group]
        trajectories = []

        for fish in fishes:
            fill_gaps_linear(fish.trajectory, fish, False)
            smooth_positions(fish, exponential_weights(24, 0.01))
            trajectories.append(fish.trajectory)

        train_save_model(group, trajectories, background_image, resolution,
                         nr_nodes, nr_fields, delta, alpha, iterations, show_results=True)


def tuning_models(trajectories_class):
    video_path = "resources/videos/v29.m4v"
    fishes_path = "resources/detections/v29-fishes.json"
    species_path = "resources/classification/species-gt-v29.csv"
    episodes_path = "resources/classification/v29-interesting-moments.csv"

    nr_nodes_list = [16, 25]
    nr_fields_list = [4, 6]
    alpha_list = [1, 2]
    delta_list = [125, 250]
    iterations_list = [10, 30]

    resolution = ppm.video_resolution(video_path)
    fishes = read_fishes_filter(fishes_path, species_path,
                                ("shark", "manta-ray"))
    interesting_episodes_gt = read_episodes(episodes_path)
    fishes_groups = group_by_normal_interesting(fishes,
                                                interesting_episodes_gt)

    fishes = fishes_groups[trajectories_class]
    trajectories = []

    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish, False)
        smooth_positions(fish, exponential_weights(24, 0.01))
        trajectories.append(fish.trajectory)

    tune_model(trajectories_class, trajectories, resolution, nr_nodes_list, nr_fields_list,
               delta_list, alpha_list, iterations_list)


def evaluate_models():
    fishes_path = "resources/detections/v29-fishes.json"
    species_path = "resources/classification/species-gt-v29.csv"
    episodes_path = "resources/classification/v29-interesting-moments.csv"
    normal_trajectories_model = "resources/models/normal-n16-f4-d250-a1.model"
    interesting_trajectories_model = "resources/models/interesting-n16-f4-d250-a1.model"

    models = {
        "normal": load_model_from_file(normal_trajectories_model),
        "interesting": load_model_from_file(interesting_trajectories_model)
    }

    all_fishes = read_fishes_filter(fishes_path, species_path,
                                    ("shark", "manta-ray"))
    fish_groups = group_by_normal_interesting(all_fishes,
                                              read_episodes(episodes_path))

    for fish in all_fishes:
        fill_gaps_linear(fish.trajectory, fish, False)
        smooth_positions(fish, exponential_weights(24, 0.01))

    total_errors = 0
    for group in fish_groups:
        group_errors = 0
        fishes = fish_groups[group]

        for fish in fishes:
            prediction = classify_trajectory(models, fish.trajectory)
            if prediction != group:
                group_errors += 1
                total_errors += 1

        print(f"{group}: {len(fishes) - group_errors}/{len(fishes)}")

    print(f"total: {len(all_fishes) - total_errors}/{len(all_fishes)}")


if __name__ == "__main__":
    # train_models()
    # evaluate_models()
    tuning_models("normal")
    # tuning_models("interesting")
