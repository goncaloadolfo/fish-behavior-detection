import random
import sys
from collections import defaultdict

from labeling.regions_selector import read_regions
from labeling.trajectory_labeling import read_species_gt
from matplotlib import pyplot as plt
from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import (_set_timestamps,
                                                 exponential_weights,
                                                 smooth_positions)
from trajectory_features.trajectory_feature_extraction import (
    TrajectoryFeatureExtraction, extract_features)
from trajectory_reader.trajectories_reader import read_fishes
from trajectory_reader.visualization import simple_line_plot


class HighlightMoment():

    def __init__(self, t_initial, t_final, rule):
        self.t_initial = t_initial
        self.t_final = t_final
        self.rule = rule


class Rule():

    SPEED = "speed"
    ACCELERATION = "acceleration"
    DIRECTION = "direction"
    CURVATURE = "curvature"
    POSTURE = "posture"
    REGION = "region"
    TRANSITION = "transition"

    def __init__(self, feature, values, duration):
        self.feature = feature
        self.values = values
        self.duration = duration

    def pass_the_rule(self, value):
        return self.values[0] <= value <= self.values[1]

    def is_region_related(self):
        return self.feature == Rule.REGION or self.feature == Rule.TRANSITION


RULE_FEATURES_MAPPING = {
    Rule.SPEED: TrajectoryFeatureExtraction.SPEEDS_ATR_NAME,
    Rule.ACCELERATION: TrajectoryFeatureExtraction.ACCELERATIONS_ATR_NAME,
    Rule.DIRECTION: TrajectoryFeatureExtraction.TAS_ATR_NAME,
    Rule.CURVATURE: TrajectoryFeatureExtraction.CURVATURES_ATR_NAME,
    Rule.POSTURE: TrajectoryFeatureExtraction.NBBS_ATR_NAME,
    Rule.REGION: TrajectoryFeatureExtraction.REGION_ATR_NAME
}


def highlight_moments(fish, regions, rules, species):
    if species not in rules:
        return (set(), None)

    fill_gaps_linear(fish.trajectory, fish)
    weights = exponential_weights(24, 0.01)
    smooth_positions(fish, weights)
    fe_obj = extract_features(fish, regions, sliding_window=24, alpha=0.01)

    species_rules = rules[species]
    highlighting_moments = set()
    region_rules = set()

    for rule in species_rules:
        if rule.is_region_related():
            region_rules.add(rule)
        else:
            time_series_name = RULE_FEATURES_MAPPING[rule.feature]
            new_moments = _search_moments_time_series(
                _set_timestamps(fish.trajectory[0][0],
                                len(fish.trajectory),
                                getattr(fe_obj, time_series_name)),
                rule
            )
            highlighting_moments = highlighting_moments.union(new_moments)

    highlighting_moments = highlighting_moments.union(
        _search_moments_regions(fish.trajectory, fe_obj.region_time_series,
                                fe_obj.pass_by_info, region_rules)
    )
    return (highlighting_moments, fe_obj)


def plot_hms(fish, feature, fe_obj, hms):
    original_time_series = getattr(fe_obj, RULE_FEATURES_MAPPING[feature])
    time_diff = len(fish.trajectory) - len(original_time_series)
    ts = list(range(fish.trajectory[time_diff][0], fish.trajectory[-1][0]+1))
    plt.figure()
    simple_line_plot(plt.gca(), ts, original_time_series,
                     feature, "value", "t", label="original")

    for hm in hms:
        hm_ts = range(hm.t_initial, hm.t_final + 1)
        t_initial_index = hm.t_initial - fish.trajectory[time_diff][0]
        t_final_index = hm.t_final - fish.trajectory[time_diff][0]
        simple_line_plot(plt.gca(), hm_ts, original_time_series[t_initial_index: t_final_index + 1],
                         feature, "value", "t", marker="-r", label="highlight")
    handles, labels = plt.gca().get_legend_handles_labels()
    group_by_label = dict(zip(labels, handles))
    plt.gca().legend(group_by_label.values(), group_by_label.keys())


def _search_moments_time_series(time_series, rule):
    highlight_moments = set()
    duration = 0

    for t, value in time_series:
        if (not rule.pass_the_rule(value) or t == time_series[-1][0]) and duration > 0:
            highlight_moments.add(HighlightMoment(t - duration, t-1, rule))
            duration = 0

        elif rule.pass_the_rule(value):
            duration += 1

    return highlight_moments


def _search_moments_regions(trajectory, regions_time_series, pass_by_results, rules):
    highlight_moments = set()

    last_region = -1
    duration = 0
    time_diff = len(trajectory) - len(regions_time_series)

    for i in range(time_diff, len(trajectory)):
        t, x, y = trajectory[i]
        region = regions_time_series[i - time_diff]

        if last_region == region or last_region == -1:
            duration += 1

        if (last_region != region or t == trajectory[-1][0]) and duration > 0:
            for rule in rules:
                if rule.feature == Rule.REGION and rule.values[0] == last_region and rule.duration < duration:
                    highlight_moments.add(
                        HighlightMoment(t - duration, t - 1, rule)
                    )
            duration = 0

        last_region = region

    for rule in rules:
        if rule.feature != Rule.TRANSITION:
            continue

        for transition, nr_transitions in pass_by_results.items():
            if len(transition) == 1:
                continue

            if (rule.values[0] == transition[0] and rule.values[1] == transition[1]):
                total_transitions = pass_by_results[(transition[0], transition[1])] + \
                    pass_by_results[(transition[1], transition[0])]

                if total_transitions >= rule.duration:
                    highlight_moments.add(
                        HighlightMoment(None, None, rule)
                    )

    return highlight_moments


def highlight_moments_test(rules):
    random.seed(10000)
    fishes = list(read_fishes("resources/detections/v29-fishes.json"))
    fishes.sort(key=lambda x: x.fish_id)
    regions = read_regions("resources/regions-example.json")
    species_gt = read_species_gt("resources/classification/species-gt-v29.csv")

    highlight_fishes = {}
    for fish in fishes:
        hm = highlight_moments(fish, regions, rules, species_gt[fish.fish_id])
        if len(hm[0]) > 0:
            highlight_fishes[fish.fish_id] = (fish, hm)

    print(
        f"Fishes that validate some of the rules: {len(highlight_fishes)}/{len(fishes)}"
    )

    if len(highlight_fishes) > 0:
        example_fish_id = random.choice(list(highlight_fishes.keys()))
        print(
            f"Example fish: {example_fish_id} ({species_gt[example_fish_id]})"
        )
        all_hms, fe_obj = highlight_fishes[example_fish_id][1]

        grouped_moments = defaultdict(set)
        for hm in all_hms:
            grouped_moments[hm.rule.feature].add(hm)

        for feature, hms in grouped_moments.items():
            if feature != Rule.TRANSITION:
                plot_hms(highlight_fishes[example_fish_id][0],
                         feature, fe_obj, hms)


if __name__ == "__main__":
    rules = {
        "shark": [
            Rule(Rule.SPEED, (3, sys.maxsize), 24),
            Rule(Rule.ACCELERATION, (0.1, sys.maxsize), 24),
            Rule(Rule.DIRECTION, (60, 120), 24),
            Rule(Rule.DIRECTION, (-120, -60), 24),
            Rule(Rule.CURVATURE, (0.6, sys.maxsize), 24),
            Rule(Rule.REGION, (1,), 24),
            Rule(Rule.TRANSITION, (1, 2), 2),
            Rule(Rule.POSTURE, (2.3, sys.maxsize), 24)
        ],
        "manta-ray": [
            Rule(Rule.SPEED, (2, sys.maxsize), 24),
            Rule(Rule.ACCELERATION, (0.1, sys.maxsize), 24),
            Rule(Rule.DIRECTION, (60, 120), 24),
            Rule(Rule.DIRECTION, (-120, -60), 24),
            Rule(Rule.CURVATURE, (0.7, sys.maxsize), 24),
            Rule(Rule.REGION, (3,), 24),
            Rule(Rule.TRANSITION, (2, 3), 2),
            Rule(Rule.POSTURE, (1.7, sys.maxsize), 24)
        ],
        "tuna": [
            Rule(Rule.SPEED, (2, sys.maxsize), 24),
            Rule(Rule.ACCELERATION, (0.15, sys.maxsize), 24),
            Rule(Rule.DIRECTION, (60, 120), 24),
            Rule(Rule.DIRECTION, (-120, -60), 24),
            Rule(Rule.CURVATURE, (0.5, sys.maxsize), 24),
            Rule(Rule.POSTURE, (2.7, sys.maxsize), 24)
        ]
    }
    highlight_moments_test(rules)
    plt.show()
