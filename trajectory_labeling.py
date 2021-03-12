"""
Script that allow users to visualize a trajectory carefully,
and assign a species label or define behavior episodes.
"""

import cv2
import sys
from threading import Thread

from trajectories_reader import read_fishes

# alias for user input related to behaviors
BEHAVIORS_ALIAS = {
    's': "swallowing-air",
    'f': "feeding",
    'l': "lack-of-interest",
    'a': "abnormal"
}

# alias for user input related species
SPECIES_ALIAS = {
    's': "shark",
    'm': "manta-ray",
    't': "tuna"
}


class Episode():
    """
    Entity for a behavior episode.
    Described by:
        - fish ID
        - description (label)
        - initial and final timestamp
    """

    def __init__(self, fish_id, description, t_initial, t_final):
        self.__fish_id = fish_id
        self.__description = description
        self.__t_initial = t_initial
        self.__t_final = t_final

    def __repr__(self):
        return f"Episode({self.__fish_id}, {self.__description}, {self.__t_initial}, {self.__t_final})"

    @property
    def fish_id(self):
        return self.__fish_id

    @property
    def description(self):
        return self.__description

    @property
    def t_initial(self):
        return self.__t_initial

    @property
    def t_final(self):
        return self.__t_final


class TrajectoryLabeling():
    """
    Visualize fishes and assign
    labels to each one (species and behaviors).
    """

    def __init__(self, fishes, video_path, episodes_out, classification_out):
        # files
        self.__video_capture = cv2.VideoCapture(video_path)
        self.__episodes_out = episodes_out
        self.__classification_out = classification_out
        # entities
        self.__fishes = fishes
        self.__episodes = set()
        self.__classification = {}
        # state
        self.__current_fish = 0
        self.__running = True

    def start(self):
        # validations
        if len(self.__fishes) == 0:
            print(f"Warning from {TrajectoryLabeling.__name__}: " +
                  "received an empty list of fishes")
        elif not self.__video_capture.isOpened():
            print(f"Error from {TrajectoryLabeling.__name__}: " +
                  "could not open video")
        elif len(self.__episodes_out.strip()) == 0 or \
                (self.__classification_out is not None and len(self.__classification_out.strip())) == 0:
            print(f"Error from {TrajectoryLabeling.__name__}: " +
                  "received an empty string as output path")

        # passed all preliminary validations
        else:
            # analyze each of the fishes
            for fish in self.__fishes:
                self.__current_fish = fish
                if self.__classification_out is not None:
                    Thread(target=self.__get_species_label, daemon=True).start()
                self.__analyze_trajectory()
            cv2.destroyAllWindows()

            # save ground truth to file
            self.__save_to_file()

    def __analyze_trajectory(self):
        # current trajectory information
        trajectory = self.__current_fish.trajectory
        t_initial = trajectory[0][0]
        t_final = trajectory[-1][0]
        t = t_initial

        while(True):
            # draw frame showing the fish position
            self.__draw_frame(t)

            # callbacks
            key = cv2.waitKey(1) & 0xFF \
                if self.__running else cv2.waitKey(0) & 0xFF
            if key == ord('a') and not self.__running:  # left key
                t = t - 1 if t > t_initial else t_initial
            elif key == ord('d') and not self.__running:  # right key
                t = t + 1 if t < t_final else t_final
            elif key == 32:  # space
                self.__running = not self.__running
            elif key == ord('n'):  # jump to the next trajectory
                self.__running = True
                cv2.destroyAllWindows()
                break
            elif key == ord('e'):  # insert a new episode
                Thread(target=self.__insert_new_episode,
                       args=(t_initial, t_final)).start()
            elif key == ord('q'):  # save current information to file
                cv2.destroyAllWindows()
                self.__save_to_file()
                exit(0)
            elif self.__running:
                t += 1

            # repeat when the trajectory gets to the end
            if t > t_final:
                t = t_initial

    def __draw_frame(self, t):
        # read frame
        if not self.__running or t == self.__current_fish.trajectory[0][0]:
            self.__video_capture.set(cv2.CAP_PROP_POS_FRAMES, t)
        _, frame = self.__video_capture.read()

        # draw fish position
        position = self.__current_fish.get_position(t)
        if position is not None:
            cv2.circle(frame,
                       center=position,
                       radius=4,
                       color=(0, 255, 0),
                       thickness=-1)

        # timestamp information
        cv2.putText(frame, f"t={t}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        cv2.imshow(f"fish {self.__current_fish.fish_id}", frame)

    def __get_species_label(self):
        # input related to the species of the current trajectory
        print("Species trajectory ", self.__current_fish.fish_id)
        while True:
            species = input("> ").strip()
            if species not in SPECIES_ALIAS:
                print(f"Error from {TrajectoryLabeling.__name__}: " +
                      f"not a valid species - valid: {SPECIES_ALIAS}")
            else:
                break
        print("OK!")
        self.__classification[self.__current_fish.fish_id] = SPECIES_ALIAS[species]

    def __insert_new_episode(self, t_initial, t_final):
        # episode description
        while True:
            description = input("\nepisode description\n> ").strip()
            if description not in BEHAVIORS_ALIAS[description]:
                print(f"Error from {TrajectoryLabeling.__name__}:" +
                      f"invalid alias. Valid ones: {BEHAVIORS_ALIAS}")
            else:
                break

        # timestamps
        while True:
            timestamps = input("initial and final timestamps (two integers " +
                               "separated by a space)\n>").strip()
            timestamp_values = timestamps.split()

            # only two values
            if len(timestamp_values) != 2:
                print(f"Error from {TrajectoryLabeling.__name__}: " +
                      "expecting 2 values")
                continue

            # try to parse values
            try:
                t1 = int(timestamp_values[0])
                t2 = int(timestamp_values[1])
            except ValueError as e:
                print(f"Error from {TrajectoryLabeling.__name__}: " +
                      f"cannot parse values - {e}")
                continue

            # check boundaries
            if t1 < t_initial or t2 > t_final:
                print(f"Error from {TrajectoryLabeling.__name__}: " +
                      f"values out of bounds - must be between [{t_initial},{t_final}]")
                continue
            elif t1 > t2:
                print(f"Error from {TrajectoryLabeling.__name__}: " +
                      "initial timestamp cannot be higher than final timestamp")
                continue

            # passed all verifications
            break
        print("OK!")

        # instantiate a new episode
        self.__episodes.add(Episode(self.__current_fish.fish_id,
                                    BEHAVIORS_ALIAS[description],
                                    t1,
                                    t2)
                            )

        # print episodes to the date
        print("Current episodes:")
        for episode in self.__episodes:
            print("\t - ", repr(episode))

    def __save_to_file(self):
        # episodes
        try:
            if len(self.__episodes) > 0:
                with open(self.__episodes_out, "w") as f:
                    f.write("fish-id,description,t-initial,t-final\n")
                    for episode in self.__episodes:
                        f.write(f"{episode.fish_id}, \
                                {episode.description}, \
                                {episode.t_initial}, \
                                {episode.t_final}\n")
        except Exception as e:
            print(f"Error from {TrajectoryLabeling.__name__}: " +
                  f"problems writing episodes to file - {e}")

        # species
        try:
            if len(self.__classification) > 0:
                with open(self.__classification_out, "w") as f:
                    f.write("fish-id,species\n")
                    for fish_id, label in self.__classification.items():
                        f.write(f"{fish_id},{label}\n")
        except Exception as e:
            print(f"Error from {TrajectoryLabeling.__name__}: " +
                  f"problems writing species information to file - {e}")


def read_episodes(file_path):
    """
    Reads a set of episodes in a file. 
    """
    episodes = set()

    # read all lines in a file
    try:
        with open(file_path, 'r') as f:
            f.readline()  # description line
            while (True):
                line = f.readline().replace(' ', '')
                if line == None or line == '\n' or line == "":
                    break
                # parse every field and instantiate a new episode
                fields = line.split(",")
                episodes.add(Episode(int(fields[0]),
                                     fields[1],
                                     int(fields[2]),
                                     int(fields[3]))
                             )
        return episodes

    # report some problem
    except Exception as e:
        print(f"Error from {read_episodes.__name__}: " +
              f"problems reading episodes from file - {e}")


def read_species_gt(file_path):
    """
    Reads the species ground truth in a file.
    """
    species_gt = {}

    # read all lines in a file
    try:
        with open(file_path, 'r') as f:
            f.readline()  # description line
            while (True):
                line = f.readline().replace(' ', '')
                if line == None or line == '\n' or line == "":
                    break
                # parse fields
                fields = line.split(",")
                species_gt[int(fields[0])] = fields[1]
        return species_gt

    # report some problem
    except Exception as e:
        print(f"Error from {read_episodes.__name__}: " +
              f"problems reading episodes from file - {e}")


def main():
    """
    Starts trajectory labeling using system arguments. 
    """
    if len(sys.argv) < 5:
        print(f"Error from {sys.argv[0]}: " +
              "some arguments are missing - " +
              f"try 'python {sys.argv[0]} " +
              "<fishes-file> <video-path> <episodes-out> <classification_out>'")
        exit(-1)

    # get arguments
    fishes_file_path = sys.argv[1].strip()
    video_path = sys.argv[2].strip()
    episodes_path = sys.argv[3].strip()
    classification_path = sys.argv[4].strip()
    if len(fishes_file_path) == 0 or len(video_path) == 0 \
            or len(episodes_path) == 0 or len(classification_path) == 0:
        print(f"Error from {__name__}: " +
              "received empty string arguments")
        exit(-1)

    # start trajectory labeling
    fishes = read_fishes(fishes_file_path)
    TrajectoryLabeling(fishes, video_path, episodes_path,
                       classification_path).start()


if __name__ == "__main__":
    main()
