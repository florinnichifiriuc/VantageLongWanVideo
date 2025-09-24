import uuid
import os, json, time
from typing import Dict, Any

try:
    from aiohttp import web as aiohttp_web
except Exception:
    aiohttp_web = None

# —— Resolve Comfy root and vantage dir ——
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
for _ in range(6):
    if os.path.isdir(os.path.join(_root, "web")):
        break
    _root = os.path.dirname(_root)

COMFY_ROOT = _root
VANTAGE_DIR = os.path.join(COMFY_ROOT, "vantage")
os.makedirs(VANTAGE_DIR, exist_ok=True)

def _log(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass

_log("[Vantage] node module loaded")
_log(f"[Vantage] COMFY_ROOT={COMFY_ROOT}, VANTAGE_DIR={VANTAGE_DIR}")

def _last_index_plus_one(base_dir: str) -> int:
    try:
        if not os.path.isdir(base_dir):
            return 0
        max_idx = -1
        for name in os.listdir(base_dir):
            if name.isdigit():
                try:
                    idx = int(name)
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    pass
        return max_idx + 1 if max_idx >= 0 else 0
    except Exception:
        return 0

def _list_projects():
    try:
        items = []
        if os.path.isdir(VANTAGE_DIR):
            for name in os.listdir(VANTAGE_DIR):
                if name.lower().endswith(".json"):
                    items.append(name)
        items.sort(key=lambda s: s.lower())
        return items
    except Exception as e:
        _log(f"[Vantage] _list_projects error: {e}")
        return []

# —— HTTP API: preview and list ——
async def vantage_preview(request):
    _log("[Vantage] preview hit")
    if aiohttp_web is None:
        # Fallback: return basic JSON-like dict; Comfy will try to serialize it
        return {"error": "aiohttp not available"}  # middleware will wrap

    fname = request.query.get("file", "")
    fname = _safe_name(fname)
    _log(f"[Vantage] preview file param: {fname}")

    if not fname or not fname.lower().endswith(".json"):
        return aiohttp_web.json_response({"error": "bad file"}, status=400)

    path = os.path.join(VANTAGE_DIR, fname)
    if not os.path.isfile(path):
        return aiohttp_web.json_response({"error": "not found"}, status=404)

    try:
        data = _load_json(path)
    except Exception as e:
        return aiohttp_web.json_response({"error": str(e)}, status=500)

    # Prefer prompt_lines -> prompt (joined) for UI
    plines = data.get("prompt_lines")
    if isinstance(plines, list):
        ui_prompt = "\n".join(str(x) for x in plines)
        line_count = len(plines)
    else:
        ui_prompt = data.get("prompt") or ""
        line_count = len(prompt.split("\n"))  # number of lines [web:19]

    proj_id = data.get("id") or uuid.uuid4().hex
    proj_dir = os.path.join(VANTAGE_DIR, proj_id)
    start_from_folders = _last_index_plus_one(proj_dir)
    
    resp = {
        "id": proj_id,
        "name": data.get("name") or os.path.splitext(fname)[0],
        "prompt": ui_prompt,
        "prompt_lines": plines if isinstance(plines, list) else None,
        "existing": True,
        "start_prompt": start_from_folders,
        "file": fname,
    }
    return aiohttp_web.json_response(resp, status=200)

async def vantage_list_projects(request):
    try:
        files = [f for f in _list_projects() if f.lower().endswith(".json")]
        return aiohttp_web.json_response({"files": files}, status=200)
    except Exception as e:
        _log(f"[Vantage] list_projects error: {e}")
        return aiohttp_web.json_response({"files": []}, status=200)

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _safe_name(s: str) -> str:
    return "".join(c for c in (s or "") if c.isalnum() or c in "._-")

def get_vantage_dir() -> str:
    """Return the absolute path to the shared Vantage directory."""
    return VANTAGE_DIR

def setup_routes(app):
    if aiohttp_web is None:
        _log("[Vantage] aiohttp not available; routes not registered")
        return
    try:
        app.router.add_get("/vantage/preview", vantage_preview)
        _log("[Vantage] Registered /vantage/preview")
    except Exception as e:
        _log(f"[Vantage] preview route exists/failed: {e}")
    try:
        app.router.add_get("/vantage/projects", vantage_list_projects)
        _log("[Vantage] Registered /vantage/projects")
    except Exception as e:
        _log(f"[Vantage] projects route exists/failed: {e}")

# Fallback registration at import time
try:
    from server import PromptServer as _PS
    setup_routes(_PS.instance.app)
    _log("[Vantage] Routes registered via PromptServer fallback")
except Exception as _e:
    _log(f"[Vantage] Fallback route registration failed: {_e}")

# —— Node class (only the relevant apply shown) ——
class VantageProject:
    @classmethod
    def INPUT_TYPES(cls):
        # Include "none" always in enum to avoid validation flaps
        files = _list_projects()
        enum = ["none"] + files
        return {
            "required": {
                "project_name": ("STRING", {"default": "", "multiline": False}),
                "positive_text": ("STRING", {"default": "", "multiline": True}),
                "start_prompt": ("INT", {"default": 0}),
            },
            "optional": {
                "project_list": (enum, {"default": "none"}),
                "existing_project": ("BOOLEAN", {"default": False}),
                "project_id": ("STRING", {"default": ""}),
            },
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_data",)
    FUNCTION = "apply"
    CATEGORY = "Vantage"
    
    def IS_NODE_ONLY_RUNNING_ON_EXECUTED(self):
        # This method is required if your node has 'ui' outputs
        return False
        
    def apply(
        self,
        project_list: str,
        project_name: str,
        positive_text: str,
        start_prompt: int,
        existing_project: bool = False,
        project_id: str = "",
    ):
        sel = (project_list or "").strip()
        is_file = bool(sel and sel.lower().endswith(".json") and sel != "none")

        # Prefer id from selected file, then provided id, else new
        file_id = ""
        if is_file:
            try:
                selected_path = os.path.join(VANTAGE_DIR, sel)
                old = {}
                if os.path.isfile(selected_path):
                    with open(selected_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                file_id = (old.get("id") or "").strip()
                _log(f"[Vantage] apply: selected file {sel} id={file_id}")
            except Exception as e:
                _log(f"[Vantage] apply: load selected id failed: {e}")

        if existing_project:
            pid = (project_id or "").strip() or file_id or uuid.uuid4().hex
        else:
            pid = uuid.uuid4().hex
        _log(f"[Vantage] apply: effective pid={pid}")

        proj_dir = os.path.join(VANTAGE_DIR, pid)
        os.makedirs(proj_dir, exist_ok=True)
        _log(f"[Vantage] apply: start prompt: {start_prompt}")
        effective_start_prompt = start_prompt
        _log(f"[Vantage] apply: effective_start_prompt: {effective_start_prompt}")
        safe_name = project_name or f"project_{pid[:8]}"

        # Decide whether this is an update to an existing file or a new file
        target_file = None
        is_update = False

        if is_file and existing_project:
            # Overwrite the selected file
            target_file = sel
            is_update = True
        else:
            # Create a new file name from project_name
            base_name = _safe_name(project_name) or f"project_{pid[:8]}"
            if not base_name.lower().endswith(".json"):
                base_name = f"{base_name}.json"

            # Ensure we don't overwrite an existing file unintentionally
            candidate = base_name
            if os.path.isfile(os.path.join(VANTAGE_DIR, candidate)):
                # If the exact name exists and we are not in update mode, make it unique
                stem, ext = os.path.splitext(base_name)
                suffix = 1
                while os.path.isfile(os.path.join(VANTAGE_DIR, f"{stem}_{suffix}{ext}")):
                    suffix += 1
                candidate = f"{stem}_{suffix}{ext}"
            target_file = candidate
            is_update = False

        path = os.path.join(VANTAGE_DIR, target_file)

        prompt_val = positive_text or ""
        lines = []
        for ln in prompt_val.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            s = ln.strip()
            if s:  # keep only actual prompts, skip blanks
                lines.append(s)

        data = {
            "id": pid,
            "name": project_name or f"project_{pid[:8]}",
            "prompt": "\n".join(lines),
            "prompt_lines": lines,
            "existing": True,
            "file": target_file,
            "start_prompt": effective_start_prompt,
            "ts": time.time(),
        }

        # Only merge from disk if updating an existing file
        if is_update:
            try:
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if old.get("id"):
                        data["id"] = old.get("id") or pid
                    if "prompt_lines" not in old and isinstance(old.get("prompt"), str):
                        data["prompt_lines"] = [
                            s for s in (p.strip() for p in old["prompt"].replace("\r\n","\n").replace("\r","\n").split("\n"))
                            if s
                        ]
            except Exception as e:
                _log(f"[Vantage] apply: merge read failed: {e}")

        # Save: overwrite only in update mode OR when creating a brand-new unique file
        try:
            _save_json(path, data)
            _log(f"[Vantage] apply: saved {target_file} (update={is_update})")
        except Exception as e:
            _log(f"[Vantage] apply: save failed: {e}")


        payload = json.dumps(data, ensure_ascii=False)
        _log(f"[Vantage] apply: returning payload file={data['file']} id={data['id']}")
        
        project_entry = {
            "id": data["id"],
            "name": data["name"],
            "prompt": data["prompt"],
            "prompt_lines": data["prompt_lines"],
            "existing": bool(data.get("existing", True)),
            "file": data["file"],
            "start_prompt": data["start_prompt"],
            "ts": data["ts"],
        }
        # Option A: group under a single ui key
        ui_payload = {"project": [project_entry]}
        return {"ui": ui_payload, "result": (payload,)}
        #return (payload,)

