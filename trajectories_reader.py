"""
Fish entity; capability to read detections file.
"""

import numpy as np
import cv2

from visualization import draw_trajectory


class Fish:
    """
    Fish entity
    """
    
    def __init__(self, fish_id):
        self.__fish_id = fish_id
        self.__trajectory = []
    
    def add_position(self, data_point):
        self.__trajectory.append(data_point)
        return self
    
    @property
    def fish_id(self):
        return self.__fish_id    
    
    @property
    def trajectory(self):
        return self.__trajectory    
    


def produce_trajectories(detections_file):
    trajectories = {}
    with open(detections_file, 'r') as f:
        for line in f:
            fields = list(map(int, line.split(",")))
            frame = fields[0]
            nr_fish = fields[1]
            # add the new detected positions to the fish
            for i in range(int(nr_fish)):
                detection = fields[2+(i*5):2+(i*5)+5]
                # centroid 
                data_point = (frame, 
                              int((detection[0] + detection[2])/2),
                              int((detection[1] + detection[3])/2))
                fish_id = detection[-1]
                if fish_id in trajectories:
                    trajectories[fish_id].add_position(data_point)
                # new fish detected
                else: 
                    trajectories[fish_id] = Fish(fish_id).add_position(data_point)
    return trajectories    
    

if __name__ == "__main__":
    trajectories = produce_trajectories("../data/Dsc 0029-lowres_gt.txt")
    example_trajectory = list(trajectories.values())[0].trajectory
    print(example_trajectory)
    draw_trajectory(example_trajectory, (480, 720), (0, 0, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    