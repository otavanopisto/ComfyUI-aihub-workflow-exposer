from os import environ, path

AIHUB_DIR = environ.get("AIHUB_DIR", None)
AIHUB_DIR = AIHUB_DIR if AIHUB_DIR is not None else path.join(path.dirname(path.abspath(__file__)), "..", "..", "aihub")

AIHUB_MODELS_DIR = path.join(AIHUB_DIR, "models")
AIHUB_LORAS_DIR = path.join(AIHUB_DIR, "loras")
AIHUB_WORKFLOWS_DIR = path.join(AIHUB_DIR, "workflows")

AIHUB_MODELS_LOCALE_DIR = path.join(AIHUB_DIR, "locale", "models")
AIHUB_LORAS_LOCALE_DIR = path.join(AIHUB_DIR, "locale", "loras")
AIHUB_WORKFLOWS_LOCALE_DIR = path.join(AIHUB_DIR, "locale", "workflows")