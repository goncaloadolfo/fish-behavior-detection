import cv2
from matplotlib import pyplot as plt

import pre_processing.pre_processing_functions as ppm
from anomaly_detection.switching_model.model_initialization import create_grid, visualize_fields
from anomaly_detection.switching_model.model_maximization import fit_switching_model

video_path = "resources/videos/v29.m4v"
fishes_path = "resources/detections/v29-fishes.json"
species_path = "resources/classification/species-gt-v29.csv"

nr_nodes = 16
nr_fields = 4
alpha = 1
delta = 250
iterations = 10

grid, trajectories = create_grid(video_path, fishes_path, species_path,
                                 ("shark", "manta-ray"), nr_nodes, nr_fields, True)
complete_likelihoods = fit_switching_model(grid, trajectories, delta, alpha, iterations)

plt.figure()
plt.title("Complete Log Likelihoods Per Iteration")
plt.xlabel("iteration")
plt.ylabel("complete log likelihood")
plt.plot(complete_likelihoods)

background_image = ppm.background_image_estimation(video_path, 300)
fields_images = visualize_fields(background_image, grid, 10)
for i in range(len(fields_images)):
    cv2.imshow(f"field {i}", fields_images[i])
plt.show()
