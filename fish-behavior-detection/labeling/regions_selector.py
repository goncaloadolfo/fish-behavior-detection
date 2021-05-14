"""
Script to help defining geographic regions in the image plan.
"""

import json
import logging
import os
import random
import sys

import cv2

CONF_DIR_NAME = "../resources"
FRAME_NAME = "regions selector"
SELECTION_COLOR = (0, 0, 255)
regions_selector_logger = logging.getLogger(__name__)
regions_selector_logger.setLevel(logging.INFO)
regions_selector_logger.addHandler(logging.StreamHandler())


class Rectangle():

    def __init__(self):
        self.__pt1 = None
        self.__pt2 = None

    @property
    def pt1(self):
        return self.__pt1

    @property
    def pt2(self):
        return self.__pt2

    @pt1.setter
    def pt1(self, v):
        self.__pt1 = v

    @pt2.setter
    def pt2(self, v):
        self.__pt2 = v

    @property
    def centroid(self):
        return (int((self.__pt1[0] + self.__pt2[0]) / 2), int((self.__pt1[1] + self.__pt2[1]) / 2))

    def __contains__(self, v):
        xs = sorted([self.__pt1[0], self.__pt2[0]])
        ys = sorted([self.__pt1[1], self.__pt2[1]])
        return v[0] >= xs[0] and v[0] <= xs[1] and v[1] >= ys[0] and v[1] <= ys[1]


def rectangle_decoder(rectangle_dict):
    rect = Rectangle()
    rect.pt1 = tuple(rectangle_dict['pt1'])
    rect.pt2 = tuple(rectangle_dict['pt2'])
    return rect


class Region():

    def __init__(self, region_id, region_tag):
        self.__region_id = region_id
        self.__region_tag = region_tag \
            if region_tag.strip() != "" \
            else f"r{self.__region_id}"
        self.__region_rectangles = []
        self.__color = (random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255))

    @property
    def region_id(self):
        return self.__region_id

    @property
    def region_tag(self):
        return self.__region_tag

    @property
    def rectangles(self):
        return self.__region_rectangles

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, v):
        self.__color = v

    def __contains__(self, v):
        for rectangle in self.__region_rectangles:
            if v in rectangle:
                return True
        return False

    def __len__(self):
        return len(self.__region_rectangles)

    def __str__(self):
        obj_str = f"- region {self.__region_id}\n- {self.__region_tag}\n"
        for rect in self.__region_rectangles:
            obj_str += f"- rect {rect.pt1}, {rect.pt2}\n"
        return obj_str

    def append(self, rectangle):
        self.__region_rectangles.append(rectangle)

    def remove(self, position):
        for i in range(len(self.__region_rectangles)):
            rect = self.__region_rectangles[i]
            if position in rect:
                del self.__region_rectangles[i]
                return

    def draw(self, frame):
        for rect in self.__region_rectangles:
            cv2.rectangle(frame, rect.pt1, rect.pt2, self.__color, 2)
            cv2.putText(frame,
                        str(self.__region_id),
                        rect.centroid,
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        self.__color)


class RegionEncoder(json.JSONEncoder):

    def default(self, obj):
        # region instance
        if isinstance(obj, Region):
            return {
                'region-id': obj.region_id,
                'region-tag': obj.region_tag,
                'color': obj.color,
                'rectangles': [self.default(rect) for rect in obj.rectangles]
            }
        # rectangle instance
        elif isinstance(obj, Rectangle):
            return {
                'pt1': obj.pt1,
                'pt2': obj.pt2
            }
        # encoder from super class (default)
        else:
            return super().default(obj)


def regions_decoder(dict):
    # region obj
    if "region-id" in dict:
        # instantiate new region
        region_obj = Region(dict['region-id'], dict['region-tag'])
        region_obj.color = tuple(dict['color'])
        # decode region rectangles
        rectangles = dict['rectangles']
        for rect_dict in rectangles:
            region_obj.append(rectangle_decoder(rect_dict))
        return region_obj
    # main dict or rectangle dict
    else:
        return dict


class RegionsSelector():

    def __init__(self, input_file_path, output_filename, video_input_type=False):
        self.__regions = {}
        self.__output_filename = output_filename
        # state attributes
        self.__croping = False
        self.__rectangle = None
        self.__mouse_position = None
        # results frame
        if video_input_type:
            video_capture = cv2.VideoCapture(input_file_path)
            _, self.__frame = video_capture.read()
            video_capture.release()
        else:
            self.__frame = cv2.imread(input_file_path)
        # create window and set mouse callback
        cv2.namedWindow(FRAME_NAME)
        cv2.setMouseCallback(FRAME_NAME, self.__mouse_callback)

    def __mouse_callback(self, event, x, y, flags, param):
        # left mouse click down - region selection
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__rectangle = Rectangle()
            self.__rectangle.pt1 = (x, y)
            self.__croping = True
        # left mouse click up - region selection
        elif event == cv2.EVENT_LBUTTONUP:
            self.__croping = False
            self.__rectangle.pt2 = (x, y)
            region_id = int(input("region number? \n > "))
            # already exist
            if region_id in self.__regions.keys():
                self.__regions[region_id].append(self.__rectangle)
            # doesn't exist
            else:
                region_tag = input("region tag? \n > ")
                self.__regions[region_id] = Region(region_id, region_tag)
                self.__regions[region_id].append(self.__rectangle)
            self.__rectangle = None
            regions_selector_logger.info("")
            self.__print_regions()
        # right click - region removal
        elif event == cv2.EVENT_RBUTTONDOWN:
            for region_id, region in self.__regions.items():
                if (x, y) in region:
                    region.remove((x, y))
                    if len(region) == 0:
                        del self.__regions[region_id]
                        break
            self.__print_regions()
        # mouse position updates
        elif event == cv2.EVENT_MOUSEMOVE:
            self.__mouse_position = (x, y)

    def __print_regions(self):
        if regions_selector_logger.level == logging.DEBUG:
            for region in self.__regions.values():
                print(region)

    def __draw_regions(self):
        frame_copy = self.__frame.copy()
        for region in self.__regions.values():
            region.draw(frame_copy)
        # if cropping
        if self.__croping == True:
            cv2.rectangle(frame_copy,
                          self.__rectangle.pt1,
                          self.__mouse_position,
                          SELECTION_COLOR,
                          2)
        return frame_copy

    def __save_to_file(self):
        if not os.path.isdir(CONF_DIR_NAME):
            os.mkdir(CONF_DIR_NAME, mode=0o666)
        conf_path = os.path.join(CONF_DIR_NAME,
                                 self.__output_filename + ".json")
        with open(conf_path, "w") as f:
            json.dump(self.__regions, f, cls=RegionEncoder, indent=4)

    def start(self):
        regions_selector_logger.info("Select ROIs using left mouse click\n"
                                     + "Remove rectangle using right click\n"
                                     + "Press 's' to write into file\n")
        while True:
            cv2.imshow(FRAME_NAME, self.__draw_regions())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                self.__save_to_file()
                return


def read_regions(config_path):
    with open(config_path, "r") as f:
        return json.load(f, object_hook=regions_decoder).values()


if __name__ == "__main__":
    RegionsSelector(sys.argv[1], sys.argv[2], bool(sys.argv[3])).start()
