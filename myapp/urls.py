from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # URL pattern for the main home page (replace 'home' with your actual view function name)
    path('', views.home, name='home'),

    # URL pattern for the health page
    path('Live Health Check/', views.yoga_file, name='yoga_file'),

    # URL pattern for the follow page
    path('follow/', views.follow, name='follow'),

    # URL pattern for the yoga landing page (yoga.html)
    path('yoga/', views.yoga_home, name='yoga_home'),

    # URL pattern for the yoga pose classification dashboard (yoga_dashboard.html)
    path('Video Pose Evaluation/', views.yoga_dashboard, name='yoga_dashboard'),

    # URL pattern for the video feed (corrected)
    path('video_feed/', views.video_feed, name='video_feed'),

    # URL pattern for uploading video
    path('upload_video/', views.upload_video, name='upload_video'),

    path('get_similarity_score/', views.get_similarity_score, name='get_similarity_score'),

    path('generate_report/', views.generate_report, name='generate_report'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
