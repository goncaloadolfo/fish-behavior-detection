from django.test import TestCase
from os import path

import requests
import json

from newvideo.models import import_bdm, BDM_MODULE

import_bdm()
from labeling.trajectory_labeling import read_species_gt


class SurfaceWarningsTestCase(TestCase):
    
    def test_surface_warnings_request(self):
        url = "http://localhost:8000/surfacewarnings/"

        fishes = None
        with open(path.join(BDM_MODULE, "resources/detections/v29-fishes.json"), "r") as file: 
            fishes = json.load(file)
        species = read_species_gt(path.join(BDM_MODULE, "resources/classification/species-gt-v29.csv"))
        fishes = {fish_id: fish for fish_id, fish in fishes.items() 
                  if species[int(fish_id)] == "shark"}
        
        # todo: have to go in body, its too long
        params = {
            "fishes": json.dumps(fishes),
            "min-duration": 1
        }
        print("sending request...")
        response = requests.get(url, params=params)
        
        print("status code:", response.status_code)
        print("response: ", response.json())
