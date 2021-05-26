from os import path
import sys

import requests
import json
from django.test import TestCase

from fish_services.models import import_bdm, BDM_MODULE
import_bdm()
from trajectory_reader.trajectories_reader import read_fishes
from labeling.trajectory_labeling import read_species_gt
from rule_based.highlight_moments import Rule


class HighlightMomentsTestCase(TestCase):
    
    def test_highlight_moments_request(self):
        rules = {
            "shark": [
                {
                    "feature": Rule.ACCELERATION,
                    "interval": [0.01, sys.maxsize],
                    "duration": 24
                },
                {
                    "feature": Rule.CURVATURE,
                    "interval": [5, sys.maxsize],
                    "duration": 24
                },
                {
                    "feature": Rule.DIRECTION,
                    "interval": [60, 120],
                    "duration": 24
                },
                {
                    "feature": Rule.REGION,
                    "interval": [1, -1],
                    "duration": 24
                },
                {
                    "feature": Rule.TRANSITION,
                    "interval": [1, 2],
                    "duration": 2
                }
            ],
            "manta-ray": [
                {
                    "feature": Rule.ACCELERATION,
                    "interval": [0.01, sys.maxsize],
                    "duration": 24
                },
                {
                    "feature": Rule.CURVATURE,
                    "interval": [5, sys.maxsize],
                    "duration": 24
                },
                {
                    "feature": Rule.DIRECTION,
                    "interval": [-120, -60],
                    "duration": 24
                },
                {
                    "feature": Rule.REGION,
                    "interval": [3, -1],
                    "duration": 24
                },
                {
                    "feature": Rule.TRANSITION,
                    "interval": [2, 3],
                    "duration": 2
                }
            ],
            "tuna": []
        }
        fishes = read_fishes(path.join(BDM_MODULE, "resources/detections/v29-fishes.json"))
        species_gt = read_species_gt(path.join(BDM_MODULE, "resources/classification/species-gt-v29.csv"))
        encoded_fishes = {fish.fish_id:fish.encode() for fish in fishes}
        
        url = "http://localhost:8000/highlighting-moments/"
        params = {
            "fishes": encoded_fishes,
            "rules": rules,
            "species": species_gt
        }
        
        print(f"sending example request to '{url}' ...")
        response = requests.get(url, data=json.dumps(params))
        
        print("status code: ", response.status_code)
        if response.status_code == 200:
            print("response: ", response.json())
            

class SurfaceWarningsTestCase(TestCase):
    
    def test_surface_warnings_request(self):
        url = "http://localhost:8000/surface-warnings/"

        fishes = None
        with open(path.join(BDM_MODULE, "resources/detections/v29-fishes.json"), "r") as file: 
            fishes = json.load(file)
        species = read_species_gt(path.join(BDM_MODULE, "resources/classification/species-gt-v29.csv"))
        fishes = {fish_id: fish for fish_id, fish in fishes.items() 
                  if species[int(fish_id)] == "shark"}
        
        params = {
            "fishes": fishes,
            "min-duration": 1
        }
        print(f"sending example request to '{url}' ...")
        response = requests.get(url, data=json.dumps(params))
        
        print("status code:", response.status_code)
        if response.status_code == 200:
            print("response: ", response.json())
            

class NewVideoProcessingTestCase(TestCase):
    
    def test_new_video_request(self):
        url = "http://localhost:8000/new-video/"
        client = requests.session()
        client.get(url)

        post_data = {}
        post_data["video-info"] = {"video-name": "v29.m4v",
                                "fps": 24, "resolution": (480, 720),
                                "total-frames": 7200}
        with open(path.join(BDM_MODULE, "resources/detections/v29-fishes.json"), "r") as file: 
            post_data["fishes"] = json.load(file)
        post_data["species"] = read_species_gt(path.join(BDM_MODULE, "resources/classification/species-gt-v29.csv"))
        headers = {'Content-type': 'application/json',
                "X-CSRFToken": client.cookies['csrftoken']}

        print(f"sending example request to '{url}' ...")
        response = requests.post(url, data=json.dumps(post_data), 
                                headers=headers, cookies=client.cookies)
        print("status code:", response.status_code)
        