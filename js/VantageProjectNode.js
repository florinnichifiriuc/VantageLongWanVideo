import { app } from "../../../scripts/app.js";

function hideInternalWidgets(node) {
  ["existing_project", "project_id"].forEach(name => {
    const w = node.widgets?.find(w => w.name === name);
    if (w) { w.hidden = true; w.computeSize = () => [0, -4]; }
  });
}

function setNameDisabled(widget, disabled) {
  if (!widget) return;
  const el = widget.inputEl || widget.textareaEl || widget.element;
  widget.readOnly = !!disabled;
  if (el) {
    el.readOnly = !!disabled; el.disabled = !!disabled; el.tabIndex = disabled ? -1 : 0;
    const st = el.style; if (st) { st.pointerEvents = disabled ? "none" : "auto"; st.opacity = disabled ? "0.7" : "1"; st.cursor = disabled ? "not-allowed" : ""; }
  }
  widget.disabled = !!disabled;
}

function sanitizeFileName(name) { return (name || "").replace(/[^a-zA-Z0-9._-]/g, ""); }

async function fetchProjectFiles() {
  const res = await fetch("/vantage/projects", { credentials: "same-origin" });
  console.log("[VantageJS] GET /vantage/projects ->", res.status);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  console.log("[VantageJS] list files:", data);
  const files = Array.isArray(data.files) ? data.files.filter(Boolean) : [];
  return ["none", ...files];
}

function hookAfterExec(node, handler) {
  const orig = node.onExecuted?.bind(node);
  node.onExecuted = function (output) {
    console.log("[VantageJS] onExecuted payload:", output);
    try { handler(output); } catch (e) { console.warn(e); }
    return orig ? orig(output) : undefined;
  };
  const s = app?.socket;
  if (s && !node.__vp_exec_hooked) {
    node.__vp_exec_hooked = true;
    s.on("executed", (msg) => {
      if (!msg || msg.node_id !== node.id) return;
      console.log("[VantageJS] socket executed:", msg);
      try { handler(msg.output ?? msg); } catch (e) { console.warn(e); }
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

      const getW = (n) => this.widgets?.find(w => w.name === n);
      const projectListW = getW("project_list");
      const projectNameW = getW("project_name");
      const promptW = getW("positive_text");
      const existingW = getW("existing_project");
      const idW = getW("project_id");
      const startPromptW = getW("start_prompt");

      hideInternalWidgets(this);

      // Refresh dropdown and select a file
      this.refreshProjectList = async (preferredFile) => {
          try {
            const items = await fetchProjectFiles();
            const ctrl = projectListW;
            if (!ctrl) return;

            // Replace items for both modern and legacy widgets
            if (ctrl.options) ctrl.options.values = items;
            ctrl.values = items;

            // Decide selection
            let nextVal = ctrl.value;
            if (preferredFile && items.includes(preferredFile)) nextVal = preferredFile;
            else if (!items.includes(nextVal)) nextVal = items[0] || "none";

            // Apply and notify
            const changed = nextVal !== ctrl.value;
            ctrl.value = nextVal;
            if (typeof ctrl.callback === "function") ctrl.callback(nextVal, this, "project_list");
            this.onWidgetChanged?.(ctrl, nextVal, "project_list");

            // Force repaint
            this.setDirtyCanvas(true, true);
            requestAnimationFrame(() => this.setDirtyCanvas(true, true));
          } catch (e) {
            console.warn("[VantageJS] refresh failed:", e);
          }
        };

      // Partial load button (optional)
      const doPartialLoad = async () => {
        const fileName = projectListW?.value;
        if (!fileName || fileName === "none") return;
        const safe = sanitizeFileName(fileName);
        try {
          const res = await fetch(`/vantage/preview?file=${encodeURIComponent(safe)}`, { credentials: "same-origin" });
          console.log("[VantageJS] preview", safe, "->", res.status);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          if (idW) idW.value = data.id || "";
          if (projectNameW) projectNameW.value = data.name || safe.replace(/\.json$/i, "");
          const uiPrompt = Array.isArray(data.prompt_lines) ? data.prompt_lines.map(x => (x == null ? "" : String(x))).join("\n") : (data.prompt || "");
          if (promptW) promptW.value = uiPrompt;
          if (existingW) existingW.value = !!data.existing;
          if (startPromptW) startPromptW.value = Number.isFinite(data.start_prompt) ? data.start_prompt : 0;
          setNameDisabled(projectNameW, !!data.existing);
          this.graph?.setDirtyCanvas?.(true, true);
        } catch (e) {
          console.warn("[VantageJS] preview failed:", e);
        }
      };

      const addBtn = (label, fn) => { const b = this.addWidget("button", label, null, () => fn()); b.serialize = false; return b; };
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
      
      // Execution hook to bind state
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

      hookAfterExec(this, (output) => {
        let payloadStr = null;
        if (Array.isArray(output) && output.length) payloadStr = typeof output[0] === "string" ? output[0] : null;
        else if (typeof output === "string") payloadStr = output;
        else if (output && typeof output === "object") { try { payloadStr = JSON.stringify(output); } catch {} }
        if (!payloadStr) { console.warn("[VantageJS] no payload string from output"); return; }
        let obj = null; try { obj = JSON.parse(payloadStr); } catch {}
        if (!obj) { console.warn("[VantageJS] output not JSON:", payloadStr); return; }
        postBind(obj.file || null, obj.id || null);
      });

      return r;
    };
  },
});

