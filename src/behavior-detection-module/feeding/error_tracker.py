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
