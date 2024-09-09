# middleware.py

from django.core.exceptions import SuspiciousOperation
from django.conf import settings

class UploadSizeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == 'POST' and 'video_file' in request.FILES:
            file = request.FILES['video_file']
            if file.size > settings.MAX_UPLOAD_SIZE:
                raise SuspiciousOperation("File size exceeds the maximum limit.")
        response = self.get_response(request)
        return response
