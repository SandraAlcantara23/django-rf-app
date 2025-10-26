# config/settings.py
"""
Django settings for config project (Render + local).
Django 5.2
"""
from pathlib import Path
import os

from dotenv import load_dotenv
import dj_database_url

# ------------------------------------------------------
# Base
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar variables del .env (local). En Render usa env vars del panel.
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "inseguro-dev")
DEBUG = os.getenv("DEBUG", "False") == "True"

# ⚠️ Ajusta el host al de tu servicio en Render si no coincide
# p. ej. "django-rf-app-6.onrender.com"
RENDER_HOST = os.getenv("RENDER_HOST", "django-rf-app-6.onrender.com")

ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    RENDER_HOST,
    ".onrender.com",
]

# Necesario para POST/CSRF vía HTTPS en Render
CSRF_TRUSTED_ORIGINS = [
    f"https://{RENDER_HOST}",
    "https://*.onrender.com",
]

# Render pone X-Forwarded-Proto: https
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Cookies/redirect solo en prod
SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# ------------------------------------------------------
# Apps
# ------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "api",
]

# ------------------------------------------------------
# Middleware
# ------------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # WhiteNoise para servir estáticos en Render
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Plantillas del proyecto
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# ------------------------------------------------------
# Base de datos
# ------------------------------------------------------
# Si DATABASE_URL está definida (Render), úsala; si no, SQLite local.
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASES = {
        "default": dj_database_url.parse(
            DATABASE_URL,
            conn_max_age=600,
            ssl_require=True,  # Render usa Postgres con SSL
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# ------------------------------------------------------
# Passwords
# ------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ------------------------------------------------------
# i18n
# ------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ------------------------------------------------------
# Estáticos
# ------------------------------------------------------
# Ruta pública
STATIC_URL = "/static/"
# Carpeta a la que collectstatic exporta (Render/producción)
STATIC_ROOT = BASE_DIR / "staticfiles"

# WhiteNoise (Django ≥4.2 usa STORAGES en lugar de STATICFILES_STORAGE)
STORAGES = {
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    }
}

# Opcional: si tienes una carpeta "static/" con assets fuente
# descomenta la siguiente línea:
# STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ------------------------------------------------------
# Logging básico (útil en Render)
# ------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": "INFO"},
}
