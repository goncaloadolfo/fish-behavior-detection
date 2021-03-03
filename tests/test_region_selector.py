"""
Set of tests to regions selector module.
"""

import logging

from regions_selector import RegionsSelector, regions_selector_logger, read_regions


def regions_selector_behavior_test():
    regions_selector_logger.setLevel(logging.DEBUG)
    RegionsSelector("../data/Dsc 0029-lowres.m4v",
                    "regions-example",
                    True).start()


def config_reading_test(config_path):
    regions = read_regions(config_path)
    for region in regions:
        print(region)


if __name__ == "__main__":
    regions_selector_behavior_test()
    # config_reading_test("conf/regions-example.json")
