import matplotlib.pyplot as plt


class ErrorTracker:

    def __init__(self):
        # timestamps initialization
        self.__timestamps = []

    def append_new_timestamp(self, t):
        # new error found
        self.__timestamps.append(t)

    def draw_errors_timeline(self, t_initial, t_final,
                             title, feeding_start=None):
        # draw a line with red dots on the error timestamps
        plt.figure()
        plt.title(title)
        plt.hlines(1, t_initial, t_final)
        plt.plot(self.__timestamps, [1.0] * len(self.__timestamps),
                 "-o", color='r', label="frame classification errors")

        # if feeding edge is set
        if feeding_start:
            plt.eventplot([feeding_start], orientation="horizontal",
                          colors="g", label="feeding")
        plt.gca().yaxis.set_visible(False)
        plt.xlabel("t")
        plt.legend()


def extract_feeding_warnings(motion_time_series, feeding_thr, duration):
    # extract feeding warnings from a feature timeseries
    # threshold and duration based
    feeding_warnings = []
    feeding_flag = False
    feeding_duration = 0

    for t in range(len(motion_time_series)):
        nr_active_pixels = motion_time_series[t]

        if nr_active_pixels >= feeding_thr:
            feeding_duration += 1
            feeding_flag = True

        elif feeding_flag and feeding_duration >= duration:
            feeding_warnings.append((t - 1 - feeding_duration, t - 1))
            feeding_duration = 0
            feeding_flag = False

        else:
            feeding_duration = 0
            feeding_flag = False

        if t == len(motion_time_series) - 1 and feeding_flag:
            feeding_warnings.append((t - feeding_duration, t))

    return feeding_warnings


# true and predicted class
def _get_predicted_class(frame, predicted_warnings):
    for t_initial, t_final in predicted_warnings:
        if t_initial <= frame <= t_final:
            return 1

        if frame < t_initial:
            break

    return 0


def _get_true_class(frame, true_feeding_period):
    for t_initial, t_final in true_feeding_period:
        if t_initial <= frame <= t_final:
            return 1

        if frame < t_initial:
            break

    return 0
