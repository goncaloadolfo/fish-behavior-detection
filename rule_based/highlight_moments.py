from pre_processing.interpolation import fill_gaps_linear
from pre_processing.trajectory_filtering import exponential_weights, smooth_positions, _set_timestamps
from trajectory_features.trajectory_feature_extraction import TrajectoryFeatureExtraction, extract_features

RULE_FEATURES_MAPPING = {
    Rule.SPEED: TrajectoryFeatureExtraction.SPEEDS_ATR_NAME,
    Rule.ACCELERATION: TrajectoryFeatureExtraction.ACCELERATIONS_ATR_NAME,
    Rule.DIRECTION: TrajectoryFeatureExtraction.TAS_ATR_NAME,
    Rule.CURVATURE: TrajectoryFeatureExtraction.CURVATURES_ATR_NAME,
    Rule.POSTURE: TrajectoryFeatureExtraction.NBBS_ATR_NAME
}


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
        return values[0] <= value <= values[1]

    def is_region_related(self):
        return self.feature == REGION or self.feature == TRANSITION


def highlight_moments(fish, regions, rules):
    fill_gaps_linear(fish.trajectory, fish)
    weights = exponential_weights(24, 0.01)
    smooth_positions(fish, regions)
    fe_obj = extract_features(fish, regions, sliding_window=24, alpha=0.01)

    highlighting_moments = set()
    region_rules = set()

    for rule in rules:
        if rule.is_region_related():
            region_rules.add(rule)
        else:
            time_series_name = RULE_FEATURES_MAPPING[rule.feature]
            new_moments = _search_moments_time_series(
                _set_timestamps(fish.trajectory[0][0],
                                len(fish.trajectory),
                                getattr(fe_obj, time_series_name))
            )
            highlighting_moments.union(new_moments)

    highlight_moments.union(
        _search_moments_regions(fish.trajectory, regions,
                                fe_obj.pass_by_info, region_rules)
    )
    return highlighting_moments


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


def _search_moments_regions(trajectory, regions, pass_by_results, rules):
    highlight_moments = set()

    last_region = -1
    duration = 0
    for t, x, y in trajectory:
        for region in regions:
            if (x, y) in region:
                break

        if last_region == region.region_id or last_region == -1:
            duration += 1

        if (last_region != region.region_id or t == trajectory[-1][0]) and duration > 0:
            for rule in rules:
                if rule.feature == Rule.REGION and rule.values[0] == region.region_id and rule.duration < duration:
                    highlight_moments.add(
                        HighlightMoment(t - duration, t - 1, rule)
                    )
            duration = 0

        last_region = region.region_id

    for rule in rules:
        if rule.feature != Rule.TRANSITION:
            continue

        for transition, nr_transitions in pass_by_results.items():
            if len(transition) == 1:
                continue

            if (rule.values[0] == transition[0] and rule.values[1] == transition[1]):
                total_transitions = pass_by_results[(transition[0], transition[1])] + \
                    pass_by_results[(transition[1], transition[0])]

                if total_transitions > rule.duration:
                    highlight_moments.add(
                        HighlightMoment(None, None, rule)
                    )

    return highlight_moments
