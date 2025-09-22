# save as python/vantage_server.py inside your custom node package
from .vantage_project import setup_routes  # route registrar from node.py

def preload(app=None):
    # ComfyUI calls preload(app) on server start for Python extensions
    try:
        if app is not None:
            setup_routes(app)
        else:
            # Some builds call preload() with no args; try global
            from server import PromptServer
            setup_routes(PromptServer.instance.app)
    except Exception as e:
        print(f"[Vantage] Failed to register /vantage/preview: {e}", flush=True)

