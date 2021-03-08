"""
Script that allow users to visualize a trajectory carefully.
A species label and several episodes will be assigned 
to each input trajectory. 
"""

import cv2
import logging
import sys
from itertools import cycle
from threading import Thread

from trajectories_reader import produce_trajectories

BEHAVIORS_ALIAS = {
    's': "swallowing-air",
    'f': "feeding",
    'l': "lack-of-interest",
    'a': "abnormal"
}
trajectory_labeling_logger = logging.getLogger(__name__)
trajectory_labeling_logger.addHandler(logging.StreamHandler())


class Episode():
    
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
    
    def __init__(self, fishes, video_path, output_path):
        # files
        self.__video_capture = cv2.VideoCapture(video_path)
        self.__output_path = output_path
        # entities
        self.__fishes = fishes
        self.__episodes = set()
        # state 
        self.__current_fish = 0
        self.__running = False
    
    def start(self):
        for i in range(len(self.__fishes)): 
            self.__current_fish = self.__fishes[i]
            self.__analyze_trajectory()
        cv2.destroyAllWindows()
        self.__save_to_file()
    
    def __analyze_trajectory(self):
        # trajectory information
        trajectory = self.__current_fish.trajectory
        t_initial = trajectory[0][0]
        t_final = trajectory[-1][0]
        t = t_initial
        
        # indeterminate cycle showing trajectory
        while(True):
            self.__draw_frame(t)
            
            # callbacks 
            key = cv2.waitKey(1) & 0xFF if self.__running else cv2.waitKey(0) & 0xFF
            if key == ord('a'):  # left key
                t = t - 1 if t > t_initial else t_initial
            elif key == ord('d'):  # right key
                t = t + 1 if t < t_final else t_final
            elif key == 32:  # space
                self.__running = not self.__running
            elif key == ord('n'):  # jump to the next trajectory
                self.__running = False
                cv2.destroyAllWindows()
                break
            elif key == ord('e'):  # insert a new episode 
                Thread(target=self.__insert_new_episode).start()                
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
        self.__video_capture.set(cv2.CAP_PROP_POS_FRAMES, t)
        _, frame = self.__video_capture.read()
        # draw fish position 
        position = self.__current_fish.get_position(t)
        if position is not None:
            cv2.circle(frame, 
                       center=(position[1], position[2]), 
                       radius=4, 
                       color=(0, 255, 0), 
                       thickness=-1)
        cv2.putText(frame, f"t={t}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=(0, 255, 0), thickness=2)
        cv2.imshow(f"fish {self.__current_fish.fish_id}", frame)
                    
    def __insert_new_episode(self):
        # user input
        description = input("\nepisode description\n> ")     
        timestamps = input("timestamps\n>").split("-")
        # create a new episode
        self.__episodes.add(Episode(self.__current_fish.fish_id, 
                                    BEHAVIORS_ALIAS[description], 
                                    int(timestamps[0]), 
                                    int(timestamps[1]))
        )        
        # print episodes
        if trajectory_labeling_logger.level == logging.DEBUG:
            for episode in self.__episodes:
                trajectory_labeling_logger.debug(repr(episode))
        
    def __save_to_file(self):
        with open(self.__output_path, "w") as f:
            f.write("fish-id,description,t-initial,t-final\n")
            for episode in self.__episodes:
                f.write(f"{episode.fish_id},{episode.description},{episode.t_initial},{episode.t_final}\n")
                    
                    
def read_episodes(file_path):
    episodes = set()
    with open(file_path, 'r') as f:
        f.readline()  # description line
        while (True):
            line = f.readline()
            if line == None or line == '\n' or line == "":
                break
            # read every field and instantiate a new episode 
            fields = line.split(",")
            episodes.add(Episode(int(fields[0]), fields[1], int(fields[2]), int(fields[3])))
    return episodes


if __name__ == "__main__":
    fishes = list(produce_trajectories(sys.argv[1]).values())
    TrajectoryLabeling(fishes, sys.argv[2], sys.argv[3]).start()
    