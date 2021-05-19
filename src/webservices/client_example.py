from os import path

import requests
import json

from newvideo.models import import_bdm, BDM_MODULE

import_bdm()
from labeling.trajectory_labeling import read_species_gt


url = "http://localhost:8000/newvideo/"
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

print("sending request...")
response = requests.post(url, data=json.dumps(post_data), 
                         headers=headers, cookies=client.cookies)
print("status code:", response.status_code)
