from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import re_path
from . import consumers

# Define the WebSocket URL patterns
websocket_urlpatterns = [
    re_path(r'^ws/video_feed/$', consumers.VideoFeedConsumer.as_asgi()),
]

# Set up the main application routing
application = ProtocolTypeRouter({
    # HTTP protocol is handled by Django views by default
    # WebSocket protocol handling
    "websocket": AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})
