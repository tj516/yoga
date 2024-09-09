# myproject/settings.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = 'ABC'

DEBUG = True

ALLOWED_HOSTS = ['75.119.150.18','yogix.ai','localhost','127.0.0.1', '*']
CSRF_TRUSTED_ORIGINS = ['https://yogix.ai', 'https://yogix.ai/routine/']

# Define the maximum upload size in bytes (e.g., 50 MB)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

# Middleware settings to handle large file uploads
FILE_UPLOAD_HANDLERS = [
    "django.core.files.uploadhandler.MemoryFileUploadHandler",
    "django.core.files.uploadhandler.TemporaryFileUploadHandler",
]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'daphne',
    'django.contrib.staticfiles',
    'corsheaders',
    'channels',
    'myapp',  # Add your app name here
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'myapp.middleware.UploadSizeMiddleware',
]

CORS_ALLOWED_ORIGINS = [
    "https://www.yogix.ai",
    "https://yogix.ai",
]


ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],  # Add your templates directory
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'myproject.wsgi.application'

ASGI_APPLICATION = 'myproject.asgi.application'


# Ensure ALLOWED_HOSTS is properly configured
#ALLOWED_HOSTS = ['yogix.ai', 'www.yogix.ai','75.119.150.18']

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True


# STATIC_URL = '/myapp/static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# import os
# STATIC_ROOT = '/root/projectdi/myapp/static/'

# STATICFILES_DIRS = [
#     os.path.join(BASE_DIR, 'static'),
# ]

# # MEDIA_ROOT: Directory on the filesystem to store uploaded files
# MEDIA_ROOT = '/root/projectdi/myapp/media/'

# # MEDIA_URL: URL that handles the media served from MEDIA_ROOT, used in the templates
# MEDIA_URL = '/myapp/media/'


# Static files (CSS, JavaScript, Images)
STATIC_URL = '/myapp/static/'

# The path where Django will collect static files for deployment (collected via `collectstatic`)
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Additional directories for Django to look for static files
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'myapp/static'),
]

# Media files (uploads)
MEDIA_URL = '/myapp/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# settings.py

# Paths to your models
MODEL_1_PATH = '/root/projectdi/model.h5'
MODEL_2_PATH = '/root/projectdi/model_1.h5'
MODEL_3_PATH = '/root/projectdi/model_2.h5'
MODEL_4_PATH = '/root/projectdi/model_2.h5'
MODEL_5_PATH = '/root/projectdi/pu.h5'
MODEL_6_PATH = '/root/projectdi/test.h5'
MODEL_7_PATH = '/root/projectdi/test1.h5'

# Paths to your labels
LABELS_1_PATH = '/root/projectdi/labels.npy'
LABELS_2_PATH = '/root/projectdi/labels_1.npy'
LABELS_3_PATH = '/root/projectdi/labels_2.npy'
LABELS_4_PATH = '/root/projectdi/labels_2.npy'
LABELS_5_PATH = '/root/projectdi/pu.npy'
LABELS_6_PATH = '/root/projectdi/test.npy'
LABELS_7_PATH = '/root/projectdi/test1.npy'



