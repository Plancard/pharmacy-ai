"""Gunicorn·flask CLI 진입점:  gunicorn wsgi:app  /  flask --app wsgi <명령>"""
from web import create_app

app = create_app()
