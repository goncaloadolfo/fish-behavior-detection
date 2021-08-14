from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

from labeling.regions_selector import read_regions
from pre_processing.trajectory_filtering import (identify_descontinuity_points, 
                                                 segment_trajectory)
from pre_processing.interpolation import fill_gaps_linear
from trajectory_reader.trajectories_reader import read_fishes


def extract_events_rules(fishes, regions, distance_thr, speed_thr, angle_thr, 
                         min_support, min_confidence):
    # represent each trajectory in a symbolic way
    events_dataset = [symbolic_representation(fish, regions, distance_thr, 
                                              speed_thr, angle_thr) for fish in fishes]
    print("Symbolic Representation:")
    for event_set in events_dataset:
        print("\t - ", event_set)
    
    # extract frequent event sets
    frequent_events_set = extract_frequent_events(events_dataset, min_support)
    print("Frequent itemsets:")
    print(frequent_events_set)
    
    # extract cool rules
    rules = association_rules(frequent_events_set, metric="confidence", min_threshold=min_confidence)
    print("Rules:")
    print(rules)
    
    return frequent_events_set, rules     
    

def extract_frequent_events(dataset, min_support):
    # frequent events
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return apriori(df, min_support=min_support, use_colnames=True)


def symbolic_representation(fish, regions, distance_thr, 
                            speed_thr, angle_thr, ignore_segments_thr=30):
    # trajectory segmentation
    descontinuity_points, _, _ = identify_descontinuity_points(fish, regions, distance_thr, 
                                                               speed_thr, angle_thr)
    segments, segment_motifs = segment_trajectory(fish, descontinuity_points)

    # symbolic tags
    symbols = []
    for i in range(len(segments)):
        segment = segments[i]
        segment_size = len(segment.trajectory)
        
        if segment_size >= ignore_segments_thr:
            symbols.append(segment_motifs[i] + "," + time_level_str(segment_size))
    
    return symbols


def time_level_str(nr_frames):
    if nr_frames < 50:
        return "0-50"
    else:
        return f"{nr_frames - (nr_frames % 50)}-{nr_frames - (nr_frames % 50) + 50}"


# region imperative/tests

def v29_rules_for_fishes(fishes_ids, min_support, min_confidence):
    # extract rules for specified fishes
    fishes = read_fishes("resources/detections/v29-fishes.json")
    fishes = [fish for fish in fishes if fish.fish_id in fishes_ids or len(fishes_ids) == 0]
    regions = read_regions("resources/regions-example.json")
    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish)
    return extract_events_rules(fishes, regions, 30, 2, 50, min_support, min_confidence)


def all_trajectories_rules_test():
    # common rules regarding all fishes
    v29_rules_for_fishes([], 0.1, 0.5)


def surface_trajectories_rules_test():
    # common rules regarding fishes that approached the surface
    v29_rules_for_fishes([3, 17], 1, 0.7)


def exclusive_surface_rules_test():
    # found exclusive rules for surface cases
    _, general_rules = v29_rules_for_fishes([], 0.1, 0.5)
    _, surface_rules = v29_rules_for_fishes([3, 17], 1, 0.7)
    exclusive_rules = surface_rules[~surface_rules.index.isin(general_rules.index)]
    print("initial number of rules: ", len(surface_rules))
    print("exclusive rules:")
    print(exclusive_rules)
    

def main():
    # all_trajectories_rules_test()
    # surface_trajectories_rules_test()
    exclusive_surface_rules_test()
        
# endregion


if __name__ == "__main__":
    main()
    