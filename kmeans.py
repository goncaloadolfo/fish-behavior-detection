import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from trajectory_feature_extraction import read_dataset
from pre_processing import z_normalization
from visualization import simple_line_plot, simple_bar_chart

DISTANCE_METRICS = [
    tf.compat.v1.estimator.experimental.KMeans.SQUARED_EUCLIDEAN_DISTANCE, 
    tf.compat.v1.estimator.experimental.KMeans.COSINE_DISTANCE
]


def input_fn():
    # input source
    return tf.data.Dataset.from_tensors(
        tf.convert_to_tensor(input_data, dtype=tf.float32)
    ).repeat(1)


def train_model(k, distance_metric, max_steps):
    # instatiate the model
    kmeans_model = tf.compat.v1.estimator.experimental.KMeans(k, distance_metric=distance_metric, use_mini_batch=False)
    
    # train until centers converge or reach max steps
    previous_centroids = None
    for i in range(max_steps):
        kmeans_model.train(input_fn)
        new_centroids = kmeans_model.cluster_centers()
        if previous_centroids is not None and (previous_centroids == new_centroids).all():
            break
        previous_centroids = new_centroids
    
    # not enough iterations to converge 
    if i == max_steps - 1:
        print(f"{max_steps} iterations were not enough to converge")
    
    return kmeans_model


def model_tunning(ks, max_steps):
    # calculate cohesion for several models
    cohesions = {distance_metric: [] for distance_metric in DISTANCE_METRICS}
    for distance_metric in DISTANCE_METRICS:
        for k in ks:
            # train a new model
            model = train_model(k, distance_metric, max_steps)
            distances = np.array(list(model.transform(input_fn)))
            np.nan_to_num(distances, copy=False)
            resulting_clusters = np.array(list(model.predict_cluster_index(input_fn)))
            
            # calculate cohesion
            cohesions_samples = calculate_cohesions(distances, resulting_clusters)
            cohesion = np.mean(cohesions_samples)
            cohesions[distance_metric].append(cohesion)
            
    # plot results
    for distance_metric in DISTANCE_METRICS:
        plt.figure()
        simple_line_plot(plt.gca(), ks, cohesions[distance_metric], 
                         f"KMeans Tunning (distance={distance_metric})", "cohesion", "k", marker='-o')


def evaluate_model(k, distance_metric, max_steps):
    # train the model
    model = train_model(k, distance_metric, max_steps)
    print(f"KMeans model(k={k}, distance_metric={distance_metric})")
    
    # resulting distances and predictions
    distances = np.array(list(model.transform(input_fn)))
    resulting_clusters = np.array(list(model.predict_cluster_index(input_fn)))
    
    # external evaluation metrics
    cohesions = calculate_cohesions(distances, resulting_clusters)
    separations = calculate_separations(distances, resulting_clusters)
    silhouettes = calculate_silhouettes(cohesions, separations)
    
    # print results to the console
    print("cohesion: ", np.mean(cohesions))
    print("separation: ", np.mean(separations))
    print("silhouette: ", np.mean(silhouettes))
    
    # cluster balance
    plt.figure()
    clusters, cluster_counts = np.unique(resulting_clusters, return_counts=True)
    simple_bar_chart(plt.gca(), clusters, cluster_counts, 
                     f"Clusters Balance using {distance_metric}", "number of samples", "cluster index")    
    
    return model


def calculate_cohesions(distances, cluster_info):
    # the cohesion of a point is considered the distance of a point to its centroid
    cohesions = []
    for i, data_point_distances in enumerate(distances):
        cohesions.append(data_point_distances[cluster_info[i]])
    return cohesions


def calculate_separations(distances, cluster_info):
    # the separation of a point is considered the distance of a point to nearest neighbor centroid
    separations = []
    for i, data_point_distances in enumerate(distances):
        cluster = cluster_info[i]
        # remove distance to its cluster centroid
        aux = np.hstack((data_point_distances[:cluster], data_point_distances[cluster+1:]))  
        separations.append(min(aux))
    return separations


def calculate_silhouettes(cohesions, separations):
    # silhouette represents a ratio between the cohesion and separation of a sample
    silhouettes = []
    
    for i in range(len(cohesions)):
        # cohesion and separation of a given sample 
        cohesion = cohesions[i]
        separation = separations[i]
        
        # silhouette calculation       
        if cohesion < separation:
            silhouettes.append(1 - cohesion/separation)
        
        elif cohesion == separation:
            silhouettes.append(0)
            
        else:
            silhouettes.append(separation/cohesion - 1)

    return silhouettes

    
if __name__ == '__main__':
    ## clustering on dataset1 (video 29)
    # load and pre process data
    samples, gt, descriptions = read_dataset("resources/datasets/v29-dataset1.csv")
    input_data = z_normalization(np.array(samples))
    
    # model experiences
    # model_tunning(ks=[2, 4, 8, 16, 32], max_steps=300)
    evaluate_model(k=8, 
                   distance_metric=tf.compat.v1.estimator.experimental.KMeans.SQUARED_EUCLIDEAN_DISTANCE,
                   max_steps=300)
    evaluate_model(k=8, 
                   distance_metric=tf.compat.v1.estimator.experimental.KMeans.COSINE_DISTANCE,
                   max_steps=300)
    
    plt.show()
    