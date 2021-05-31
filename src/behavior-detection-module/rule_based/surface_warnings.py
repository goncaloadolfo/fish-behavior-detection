from labeling.regions_selector import read_regions
from labeling.trajectory_labeling import read_species_gt
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import smooth_positions
from trajectory_features.trajectory_feature_extraction import exponential_weights
from trajectory_reader.trajectories_reader import read_fishes


def provide_surface_warnings(fishes, species_gt, regions, surface_tag, min_duration):
    swarns = []

    surface_region = None
    for region in regions:
        if region.region_tag == surface_tag:
            surface_region = region
            break
    if surface_region is None:
        return swarns

    for fish in fishes:
        if species_gt[fish.fish_id] != "shark":
            continue

        consecutive_frames = 0
        t_initial = -1
        for t, x, y in fish.trajectory:
            if (x, y) in surface_region:
                if consecutive_frames == 0:
                    t_initial = t
                consecutive_frames += 1

            if (x, y) not in surface_region and consecutive_frames > 0 or t == fish.trajectory[-1][0] \
                    and consecutive_frames > 0:
                if consecutive_frames >= min_duration:
                    swarns.append([fish, t_initial, t - 1])
                t_initial = -1
                consecutive_frames = 0

    return swarns


def surface_warnings_test():
    fishes = read_fishes("resources/detections/v29-fishes.json")
    species_gt = read_species_gt("resources/classification/species-gt-v29.csv")

    exp_weights = exponential_weights(24, 0.01)
    for fish in fishes:
        if species_gt[fish.fish_id] == "shark":
            fill_gaps_linear(fish.trajectory, fish, True)
            smooth_positions(fish, exp_weights)
    regions = read_regions("resources/regions-example.json")

    detected_warnings = provide_surface_warnings(fishes, species_gt, regions, "surface", 1)
    print("Detected warnings:")
    for warn in detected_warnings:
        print(f"\t- fish ID={warn[0].fish_id}, t initial={warn[1]}, t final={warn[2]}")


if __name__ == "__main__":
    surface_warnings_test()
