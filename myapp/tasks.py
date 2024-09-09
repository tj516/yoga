# tasks.py
from celery import shared_task
from django.core.files.storage import FileSystemStorage
from .views import process_uploaded_video

@shared_task
def process_video_task(video_path, pose_option):
    return process_uploaded_video(video_path, pose_option)
