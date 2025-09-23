import { app } from "../../../scripts/app.js";

function hideInternalWidgets(node) {
  ["existing_project", "project_id"].forEach((name) => {
    const w = node.widgets?.find((w) => w.name === name);
    if (w) {
      w.hidden = true;
      w.computeSize = () => [0, -4];
    }
  });
}

function setNameDisabled(widget, disabled) {
  if (!widget) return;
  const el = widget.inputEl || widget.textareaEl || widget.element;
  widget.readOnly = !!disabled;
  if (el) {
    el.readOnly = !!disabled;
    el.disabled = !!disabled;
    el.tabIndex = disabled ? -1 : 0;
    const st = el.style;
    if (st) {
      st.pointerEvents = disabled ? "none" : "auto";
      st.opacity = disabled ? "0.7" : "1";
      st.cursor = disabled ? "not-allowed" : "";
    }
  }
  widget.disabled = !!disabled;
}

function sanitizeFileName(name) {
  return (name || "").replace(/[^a-zA-Z0-9._-]/g, "");
}

async function fetchProjectFiles() {
  const res = await fetch("/vantage/projects", { credentials: "same-origin" });
  console.log("[VantageJS] GET /vantage/projects ->", res.status);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  console.log("[VantageJS] list files:", data);
  const files = Array.isArray(data.files) ? data.files.filter(Boolean) : [];
  return ["none", ...files];
}

// Unified hook that wires node.onExecuted and socket "executed"
function hookAfterExec(node, handler) {
  const orig = node.onExecuted?.bind(node);
  node.onExecuted = function (output) {
    console.log("[VantageJS] onExecuted payload:", output);
    try {
      handler(output);
    } catch (e) {
      console.warn(e);
    }
    return orig ? orig(output) : undefined;
  };

  const s = app?.socket;
  if (s && !node.__vp_exec_hooked) {
    node.__vp_exec_hooked = true;
    s.on("executed", (msg) => {
      // Docs show executed contains node id and output = ui payload when ui is returned by node. [web:3]
      const nid = msg?.node_id ?? msg?.node;
      if (nid !== node.id) return;
      console.log("[VantageJS] socket executed:", msg);
      try {
        handler(msg?.output ?? msg);
      } catch (e) {
        console.warn(e);
      }
    });
  }
}

app.registerExtension({
  name: "VantageProjectNode",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "VantageProject") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

      const getW = (n) => this.widgets?.find((w) => w.name === n);
      const projectListW = getW("project_list");
      const projectNameW = getW("project_name");
      const promptW = getW("positive_text");
      const existingW = getW("existing_project");
      const idW = getW("project_id");
      const startPromptW = getW("start_prompt");

      hideInternalWidgets(this);

      // Refresh dropdown and optionally select a file
      this.refreshProjectList = async (preferredFile) => {
        try {
          const items = await fetchProjectFiles();
          const ctrl = projectListW;
          if (!ctrl) return;

          // Support legacy and modern combo widgets
          if (ctrl.options) ctrl.options.values = items;
          ctrl.values = items;

          // Selection logic
          let nextVal = ctrl.value;
          if (preferredFile && items.includes(preferredFile)) nextVal = preferredFile;
          else if (!items.includes(nextVal)) nextVal = items[0] || "none";

          const changed = nextVal !== ctrl.value;
          ctrl.value = nextVal;

          if (typeof ctrl.callback === "function") ctrl.callback(nextVal, this, "project_list");
          this.onWidgetChanged?.(ctrl, nextVal, "project_list");

          this.setDirtyCanvas(true, true);
          requestAnimationFrame(() => this.setDirtyCanvas(true, true));
        } catch (e) {
          console.warn("[VantageJS] refresh failed:", e);
        }
      };

      // Optional partial load helper
      const doPartialLoad = async () => {
        const fileName = projectListW?.value;
        if (!fileName || fileName === "none") return;
        const safe = sanitizeFileName(fileName);
        try {
          const res = await fetch(`/vantage/preview?file=${encodeURIComponent(safe)}`, {
            credentials: "same-origin",
          });
          console.log("[VantageJS] preview", safe, "->", res.status);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();

          if (idW) idW.value = data.id || "";
          if (projectNameW) projectNameW.value = data.name || safe.replace(/\.json$/i, "");
          const uiPrompt = Array.isArray(data.prompt_lines)
            ? data.prompt_lines.map((x) => (x == null ? "" : String(x))).join("\n")
            : data.prompt || "";
          if (promptW) promptW.value = uiPrompt;
          if (existingW) existingW.value = !!data.existing;
          if (startPromptW) startPromptW.value = Number.isFinite(data.start_prompt) ? data.start_prompt : 0;

          setNameDisabled(projectNameW, !!data.existing);
          this.graph?.setDirtyCanvas?.(true, true);
        } catch (e) {
          console.warn("[VantageJS] preview failed:", e);
        }
      };

      // Buttons
      const addBtn = (label, fn) => {
        const b = this.addWidget("button", label, null, () => fn());
        b.serialize = false;
        return b;
      };
      addBtn("Load Project", () => doPartialLoad());
      addBtn("New Project", () => {
        if (existingW) existingW.value = false;
        if (idW) idW.value = "";
        if (projectNameW) projectNameW.value = "";
        if (promptW) promptW.value = "";
        if (startPromptW) startPromptW.value = 0;
        setNameDisabled(projectNameW, false);
        this.graph?.setDirtyCanvas?.(true, true);
      });
      addBtn("Refresh List", () => this.refreshProjectList(projectListW?.value));

      // Executed -> bind state (refresh list, set id, set existing true)
      const postBind = (file, pid) => {
        console.log("[VantageJS] postBind file=", file, "pid=", pid);
        if (!file) return;

        if (existingW) existingW.value = true;
        if (idW && pid) idW.value = pid;

        this.refreshProjectList?.(file);
        setNameDisabled(projectNameW, true);

        this.graph?.setDirtyCanvas?.(true, true);
        requestAnimationFrame(() => this.graph?.setDirtyCanvas?.(true, true));
      };

      // Robust payload extraction:
      // - Prefer UI payload from executed (msg.output object with id/file) per docs. [web:3]
      // - Fallback: plain object typed output with id/file
      // - Fallback: string or tuple[string] JSON
      hookAfterExec(this, (output) => {
      let obj = null;

      // Prefer grouped UI: { project: [ {...} ] }
      if (output && output.project && Array.isArray(output.project) && output.project.length) {
        obj = output.project[0];
      } else if (output && typeof output === "object" && (output.id || output.file)) {
        obj = output;
      } else if (output && output.ui && typeof output.ui === "object") {
        const u = output.ui;
        if (u.project && Array.isArray(u.project) && u.project.length) obj = u.project[0];
      } else if (Array.isArray(output) && output.length && typeof output[0] === "string") {
        try { obj = JSON.parse(output[0]); } catch {}
      } else if (typeof output === "string") {
        try { obj = JSON.parse(output); } catch {}
      }

      if (!obj) { console.warn("[VantageJS] no usable executed payload"); return; }
      postBind(obj.file || null, obj.id || null);
    });

      return r;
    };
  },
});
