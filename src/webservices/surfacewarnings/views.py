from os import path

from django.shortcuts import render
from newvideo.models import import_bdm, BDM_MODULE
from django.http import JsonResponse
from django.views.generic.base import View 
import json

import_bdm()
from trajectory_reader.trajectories_reader import Fish
from pre_processing.interpolation import fill_gaps_linear
from labeling.regions_selector import read_regions
from rule_based.surface_warnings import provide_surface_warnings


class SurfaceWarningsView(View):
    
    def get(self, request, *args, **kwargs):
        fishes = json.loads(request.GET.get("fishes", "{}"))
        fishes_set = set()
        species = {}
        
        for fish_id in fishes:
            fish = Fish(fish_id).decode(fishes[fish_id])
            fill_gaps_linear(fish.trajectory, fish)
            fishes_set.add(fish)
            species[fish_id] = "shark"
            
        regions = read_regions(path.join(BDM_MODULE, "resources/regions-example.json"))
        min_duration = request.GET.get("min-duration", 1)
        warnings = provide_surface_warnings(fishes_set, species, regions, "surface", min_duration)
        
        for warning in warnings:
            warning[0] = warning[0].encode()
        
        return JsonResponse(warnings)
    