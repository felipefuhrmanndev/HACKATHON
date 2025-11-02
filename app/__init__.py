from flask import Flask
import logging
import sys
from .routes import register_routes


def _configure_logging() -> None:
    # Configura logging básico para console; nível INFO por padrão
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def create_app() -> Flask:
    _configure_logging()
    app = Flask(__name__)
    # Registra rotas diretamente, sem Blueprint
    register_routes(app)
    return app


# Permite `flask --app app run`
app = create_app()