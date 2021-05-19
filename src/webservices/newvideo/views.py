from django.shortcuts import render
from newvideo.models import process_video
from django.http import HttpResponseNotAllowed, HttpResponseBadRequest, HttpResponseServerError, JsonResponse
from django.middleware.csrf import get_token
import json


def process_new_video(request, *args, **kwargs):
    get_token(request)
    if request.method != "POST":
        return HttpResponseNotAllowed(permitted_methods=["POST"])

    data = json.loads(request.body)
    if "fishes" not in data or "species" not in data or "video-info" not in data:
        return HttpResponseBadRequest()

    added_information = process_video(data["fishes"],
                                      data["species"],
                                      data["video-info"])
    return JsonResponse(added_information)
