from os import path
from collections import defaultdict

from django.shortcuts import render
from django.views.generic.base import View
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed, JsonResponse
import json
from django.middleware.csrf import get_token

from fish_services.models import import_bdm, discover_highlighting_moments, \
    get_surface_warnings, process_video
import_bdm()
from trajectory_reader.trajectories_reader import Fish
from rule_based.highlight_moments import Rule
from pre_processing.trajectory_filtering import smooth_positions
from pre_processing.interpolation import fill_gaps_linear
from trajectory_features.trajectory_feature_extraction import exponential_weights


def general_request_validation(request_body, mandatory_keys):
    try:
        request_dict = json.loads(request_body)
        for mandatory_key in mandatory_keys:
            if mandatory_key not in list(request_dict.keys()):
                return None
        return request_dict

    except ValueError:
        return None


def validate_fishes(fishes_dict, species_dict):
    fishes_set = set()
    
    for fish_id in fishes_dict:
        if "trajectory" not in fishes_dict[fish_id] or "bounding-boxes" not in fishes_dict[fish_id]:
            return None
    
        trajectory = fishes_dict[fish_id]["trajectory"]
        bbs = fishes_dict[fish_id]["bounding-boxes"] 

        if not isinstance(trajectory, list) or not isinstance(bbs, dict):
            return None
        
        for data_point in trajectory:
            if len(data_point) != 3:
                return None
            
            t, x, y = data_point
            if not isinstance(t, int) or not isinstance(x, int) or not isinstance(y, int):
                return None
            
            if species_dict is not None and fish_id not in species_dict:
                return None
            
            if species_dict is not None and species_dict[fish_id] not in ["shark", "manta-ray", "tuna"]:
                return None
                        
        fish = Fish(fish_id)
        fish.decode(fishes_dict[fish_id])
        fishes_set.add(fish)
            
    return fishes_set


def validate_rules(rules_dict):
    for species in rules_dict:
        if species not in ["shark", "manta-ray", "tuna"]:
            return False

        if not isinstance(rules_dict[species], list):
            return False
        
        for i in range(len(rules_dict[species])):
            rule = rules_dict[species][i]
            if "feature" not in rule or "interval" not in rule or "duration"not in rule:
                return False     
            
            feature = rule["feature"]
            interval = rule["interval"]
            duration = rule["duration"]

            if not isinstance(feature, str) or not isinstance(interval, list) or \
                not isinstance(duration, (int, float)):
                    return False
            
            if feature not in Rule.RECOGNIZED_FEATURES:
                return False 
            
            if interval[0] != "min" and interval[0] != "max" and not isinstance(interval[0], (int, float)) or \
                not isinstance(interval[1], int):
                    return False   
            
            first_interval_value = None
            if interval[0] == "min":
                first_interval_value = min
            elif interval[0] == "max":
                first_interval_value = max
            else:
                first_interval_value = interval[0]
                
            rules_dict[species][i] = Rule(feature, (first_interval_value, interval[1]), duration)  
            
    return True


def build_hms_response(hms):
    response = {}
    
    for fish_id in hms:
        fish_hms = []
        for hm in hms[fish_id]:
            fish_hms.append(
                {
                    "t-initial": hm.t_initial,
                    "t-final": hm.t_final,
                    "motif": hm.rule.feature
                }
            )
        response[fish_id] = fish_hms
        
    return response


class HighlightingMomentsView(View):
    
    def get(self, request, *args, **kwargs):
        request_dict = general_request_validation(request.body, ["fishes", "rules", "species"])
        if request_dict is None:
            return HttpResponseBadRequest() 
        
        fishes = validate_fishes(request_dict["fishes"], request_dict["species"])
        rules = validate_rules(request_dict["rules"])
        
        if fishes is None or not rules:
            return HttpResponseBadRequest()
        
        hms = discover_highlighting_moments(fishes, request_dict["rules"], request_dict["species"])
        return JsonResponse(build_hms_response(hms))


def pre_process_fishes(fishes):
    exp_weights = exponential_weights(24, 0.01)
    for fish in fishes:
        fill_gaps_linear(fish.trajectory, fish)
        smooth_positions(fish, exp_weights)


def build_sw_response(surface_warnings):
    warnings_dict = defaultdict(list)
    for warning in surface_warnings:
        fish = warning[0]
        t_initial = warning[1]
        t_final = warning[2]
        warnings_dict[fish.fish_id].append({
            "t-initial": t_initial,
            "t-final": t_final
        })
    return warnings_dict
    

class SurfaceWarningsView(View):
    
    def get(self, request, *args, **kwargs):
        request_dict = general_request_validation(request.body, ["fishes", "min-duration"])
        if request_dict is None:
            return HttpResponseBadRequest() 
        
        fishes = validate_fishes(request_dict["fishes"], None)
        pre_process_fishes(fishes)
        species = {fish.fish_id: "shark" for fish in fishes}
        
        surface_warnings = get_surface_warnings(fishes, species, request_dict["min-duration"])
        return JsonResponse(build_sw_response(surface_warnings))


def validate_video_info(video_info):
    video_name_validation = "video-name" in video_info and isinstance(video_info["video-name"], str)
    fps_validation = "fps" in video_info and isinstance(video_info["fps"], (int, float))
    resolution_validation = "resolution" in video_info and isinstance(video_info["resolution"], (tuple, list)) \
        and len(video_info["resolution"]) >= 2 and isinstance(video_info["resolution"][0], (int, float)) \
        and isinstance(video_info["resolution"][1], (int, float))
    total_frames_validation = "total-frames" in video_info and isinstance(video_info["total-frames"], (int, float))
    return video_name_validation and fps_validation and resolution_validation and total_frames_validation
    
    
class NewVideoProcessingView(View):
    
    def get(self, request, *args, **kwargs):
        get_token(request)
        return HttpResponseNotAllowed(permitted_methods=["POST"])
    
    
    def post(self, request, *args, **kwargs):
        data = general_request_validation(request.body, ["fishes", "species", "video-info"])
        fishes = validate_fishes(data["fishes"], data["species"])
        video_info_validation = validate_video_info(data["video-info"]) 
        
        if fishes is None or not video_info_validation:
            return HttpResponseBadRequest()

        added_information = process_video(data["fishes"],
                                        data["species"],
                                        data["video-info"])
        return JsonResponse(added_information)
        