import { app } from "../../../scripts/app.js";

const TRUNC = (s, n = 48) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const SHORT = (s, n = 22) => (s.length > n ? s.slice(0, n - 3) + "…" : s);
const LOADER_NAME_WIDGETS = ["unet_name", "clip_name", "vae_name", "ckpt_name", "model_name", "filename"];
const SETGET_WIDGETS = ["setnode_name", "uid", "name", "key", "tag", "identifier", "label", "constant"];

const graphOf = (node) => node?.graph ?? app.graph;

function srcModelName(srcNode) {
  if (!srcNode?.widgets) return null;
  const w = srcNode.widgets.find(
    (x) => LOADER_NAME_WIDGETS.includes(x.name) && typeof x.value === "string" && x.value
  );
  return w ? w.value : null;
}

function setGetKey(node) {
  const names = new Set(SETGET_WIDGETS.map((s) => s.toLowerCase()));
  for (const w of node?.widgets ?? []) {
    if (names.has(w.name?.toLowerCase()) && typeof w.value === "string" && w.value.trim()) {
      return { wn: w.name, v: w.value.trim() };
    }
  }
  for (const inp of node?.inputs ?? []) {
    if (names.has(inp.name?.toLowerCase()) && typeof inp.value === "string" && inp.value.trim() && !inp.link) {
      return { wn: inp.name, v: inp.value.trim() };
    }
  }
  const strWs = (node?.widgets ?? []).filter((x) => typeof x.value === "string" && x.value.trim());
  if (strWs.length === 1) {
    return { wn: strWs[0].name, v: strWs[0].value.trim() };
  }
  return null;
}

function linkSource(node, linkId) {
  const g = graphOf(node);
  const link = g?.links?.[linkId];
  return link ? g.getNodeById(link.origin_id) : null;
}

// Set/Get buses are global key-value stores: the setter may live in any graph
// (root or any nested subgraph container's inner graph).
function findSetter(key, startNode) {
  const seen = new Set();
  let found = null;
  const scan = (g) => {
    for (const cand of g?._nodes ?? []) {
      if (found || seen.has(cand)) continue;
      seen.add(cand);
      if (cand === startNode) continue;
      const ck = setGetKey(cand);
      if (!ck || ck.wn !== key.wn || ck.v !== key.v) continue;
      if (cand.inputs?.some((i) => i.link)) { found = cand; return; }
    }
  };
  const walk = (g) => {
    scan(g);
    for (const n of g?._nodes ?? []) if (n.subgraph) walk(n.subgraph);
  };
  walk(graphOf(startNode));
  if (!found && graphOf(startNode) !== app.graph) scan(app.graph);
  return found;
}

// Map a subgraph container's outer socket slot to the inner node feeding it.
// ComfyUI wires inner outputs to a virtual node id -20 with target_slot = the
// container's output slot; fall back to positional by exposed output order.
function subgraphInnerFor(container, slot = 0) {
  const g = container?.subgraph;
  if (!g) return null;
  const link = Object.values(g?.links ?? {}).find(
    (l) => l.target_id === -20 && (l.target_slot ?? 0) === (slot ?? 0)
  );
  if (link) {
    const inner = g.getNodeById(link.origin_id);
    return inner ? { node: inner, slot: link.origin_slot ?? 0 } : null;
  }
  const exposed = (g._nodes ?? []).filter((n) => (n.outputs ?? []).some((o) => o.links?.length));
  const inner = exposed[slot] ?? (g._nodes ?? [])[slot];
  return inner ? { node: inner, slot: 0 } : null;
}

// Resolve a connected model's display name, walking through:
// loader widgets -> set/get indirection (by shared name key, any graph) ->
// subgraph container descent (slot mapped via the inner -20 links) ->
// single model-typed input chains
function resolveModelSource(node, slot = 0, depth = 0, visited = new Set()) {
  if (!node || depth > 5 || visited.has(node)) return null;
  visited.add(node);
  const direct = srcModelName(node);
  if (direct) return direct;
  const key = setGetKey(node);
  if (key) {
    const setter = findSetter(key, node);
    if (setter) {
      const valIn = setter.inputs?.find((i) => i.link);
      if (valIn) {
        const lk = graphOf(setter)?.links?.[valIn.link];
        const r = resolveModelSource(
          linkSource(setter, valIn.link),
          lk?.origin_slot ?? 0,
          depth + 1,
          visited
        );
        if (r) return r;
      }
    }
  }
  if (node.subgraph) {
    const inner = subgraphInnerFor(node, slot);
    if (inner) {
      const r = resolveModelSource(inner.node, inner.slot, depth + 1, visited);
      if (r) return r;
    }
  }
  const vIn = node.inputs?.find((i) => i.link && /MODEL|CLIP|VAE/.test(i.type ?? ""));
  if (vIn) {
    const lk = graphOf(node)?.links?.[vIn.link];
    const r = resolveModelSource(
      linkSource(node, vIn.link),
      lk?.origin_slot ?? 0,
      depth + 1,
      visited
    );
    if (r) return r;
  }
  return null;
}

// Follow set/get indirection to the node that actually PROVIDES the list value.
function resolveListSource(node, depth = 0, visited = new Set()) {
  if (!node || depth > 5 || visited.has(node)) return null;
  visited.add(node);
  const key = setGetKey(node);
  if (key) {
    const setter = findSetter(key, node);
    if (setter) {
      const valIn = setter.inputs?.find((i) => i.link);
      if (valIn) {
        const r = resolveListSource(linkSource(setter, valIn.link), depth + 1, visited);
        if (r) return r;
      }
    }
  }
  return node;
}

function inputSummaries(node) {
  const out = [];
  for (const inp of node?.inputs ?? []) {
    if (!inp.link) continue;
    const g = graphOf(node);
    const link = g?.links?.[inp.link];
    if (!link) continue;
    const src = g?.getNodeById(link.origin_id);
    const n = src ? (resolveModelSource(src, link.origin_slot ?? 0) ?? src.title) : null;
    out.push({ name: inp.name, label: n ? `[${link.type ?? ""}] ${n}` : null });
  }
  return out;
}

function outputsOf(message) {
  const out = message?.output ?? {};
  if (!out) {
    return { status: message?.text?.[0] ?? "", info: "" };
  }
  if (Array.isArray(out)) {
    return {
      status: typeof out[0] === "string" ? out[0] : "",
      info: typeof out[1] === "string" ? out[1] : "",
    };
  }
  const statusVal = out[0] !== undefined ? out[0] : out["0"];
  const infoVal = out[1] !== undefined ? out[1] : out["1"];
  if (typeof statusVal === "string" || typeof infoVal === "string") {
    return {
      status: typeof statusVal === "string" ? statusVal : "",
      info: typeof infoVal === "string" ? infoVal : "",
    };
  }
  // 0.32 delivers string returns as {text:[...]} on text-consumer events.
  if (Array.isArray(out.text)) {
    return {
      status: typeof out.text[0] === "string" ? out.text[0] : "",
      info: typeof out.text[1] === "string" ? out.text[1] : out.text[0] ?? "",
    };
  }
  return { status: "", info: "" };
}

app.registerExtension({
  name: "DynamicVRAMFree.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isFetch = nodeData.name === "DynamicVRAMModelFetch";
    const isUnload = nodeData.name === "DynamicVRAMFree";
    if (!isFetch && !isUnload) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      this._uiDefaultTitle = this.constructor.defaultTitle ?? this.title;
      this._uiAuto = false;
      this._canSetTitle = () => this._uiAuto || this.title === this._uiDefaultTitle;

      this._updateSocketLabels = function () {
        for (const inp of this.inputs ?? []) {
          const src = inp.link ? linkSource(this, inp.link) : null;
          const lk = inp.link ? graphOf(this)?.links?.[inp.link] : null;
          const n = src ? (resolveModelSource(src, lk?.origin_slot ?? 0) ?? src.title) : null;
          inp.label = n ? SHORT(n) : inp.name;
        }
      };

      this._updateTitleFromLinks = function () {
        if (this._uiInfo) return; // executed info is authoritative, never clobber
        if (!this._canSetTitle()) return;
        const names = inputSummaries(this)
          .map((s) => s.label)
          .filter(Boolean);
        if (names.length) {
          this.title = TRUNC(`Fetch: ${names.join(" · ")}`);
          this._uiAuto = true;
        }
      };

      this._applyInfoTitle = function () {
        if (!this._canSetTitle()) return;
        const lines = (this._uiInfo || "").split("\n").filter(Boolean);
        const first = (lines[0] ?? "").replace(/^\d+\.\s*/, "");
        if (first) {
          this.title = TRUNC(`Fetch: ${first}${lines.length ? ` · ${lines.length} models` : ""}`);
          this._uiAuto = true;
        }
      };

      this._applyModelInfo = function (info) {
        if (!info) return;
        const lines = info.split("\n").filter(Boolean);
        if (isUnload) {
          const useWs = this.widgets?.filter((w) => w.name?.startsWith("use_")) ?? [];
          useWs.forEach((w, i) => {
            const line = lines[i];
            if (line) {
              w.label = SHORT(line.replace(/^\d+\.\s*/, ""), 26);
              if (w.computeSize) w.computeSize();
            }
          });
        }
      };

      if (isUnload) {
        const modeW = this.widgets?.find((w) => w.name === "mode");
        const useWs = this.widgets?.filter((w) => w.name?.startsWith("use_")) ?? [];
        const applyMode = () => {
          if (!modeW || !useWs.length) return;
          const disabled = modeW.value !== "selective";
          for (const w of useWs) {
            w.disabled = disabled;
            if (w.computeSize) w.computeSize();
          }
          if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
        };
        if (modeW) modeW.callback = applyMode;
        const onConfigure = this.onConfigure;
        this.onConfigure = function () {
          const r2 = onConfigure?.apply(this, arguments);
          applyMode();
          const modelsLink = this.inputs?.find((i) => i.name === "models")?.link;
          const src = modelsLink ? linkSource(this, modelsLink) : null;
          const listSrc = src ? resolveListSource(src) : null;
          if (listSrc?._uiInfo) {
            this._applyModelInfo?.(listSrc._uiInfo);
          } else if (listSrc) {
            let labels = inputSummaries(listSrc).map((s) => s.label).filter(Boolean);
            if (!labels.length) {
              const single = resolveModelSource(listSrc);
              if (single) labels = [single];
            }
            const useWs = this.widgets?.filter((w) => w.name?.startsWith("use_")) ?? [];
            useWs.forEach((w, i) => {
              const lbl = labels[i];
              if (lbl) {
                w.label = SHORT(lbl.replace(/^\[[^\]]*\]\s*/, ""), 26);
                if (w.computeSize) w.computeSize();
              }
            });
          }
          return r2;
        };
        applyMode();
      }

      if (isFetch) {
        // 0.3x: OUTPUT nodes emit `executed` only when they return {"ui": ...}
        // (see __init__.py) — this global hook is a safety net for consumer text.
        if (!this._uiStatsHooked) {
          this._uiStatsHooked = true;
          const onAnyExecuted = (e) => {
            const d = e?.detail || {};
            const text = d.output?.text;
            if (!Array.isArray(text) || !text.length) return;
            const g = graphOf(this);
            const evNode = d.node != null ? g.getNodeById(String(d.node)) : null;
            if (!evNode) return;
            const infoSrc = (this.outputs || []).findIndex((o) => o.name === "info");
            const link = g?.links;
            const fromMe = (evNode.inputs || []).some((i) => {
              if (!i.link) return false;
              const lk = link?.[i.link];
              return lk && lk.origin_id === this.id && lk.origin_slot === infoSrc;
            });
            if (!fromMe) return;
            const info = text.join("\n");
            this._uiInfo = info;
            this._applyInfoTitle?.();
            for (const cand of g?._nodes ?? []) {
              if (cand === this || typeof cand._applyModelInfo !== "function") continue;
              const modelsIn = (cand.inputs ?? []).find((i) => i.name === "models");
              if (modelsIn?.link && resolveListSource(linkSource(cand, modelsIn.link)) === this) {
                cand._applyModelInfo(info);
              }
            }
            if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
          };
          app.api?.addEventListener?.("executed", onAnyExecuted);
        }
        const onConfigureFetch = this.onConfigure;
        this.onConfigure = function () {
          const r2 = onConfigureFetch?.apply(this, arguments);
          this._updateSocketLabels?.();
          this._updateTitleFromLinks?.();
          return r2;
        };
        this._updateSocketLabels();
        this._updateTitleFromLinks();
      }
      if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = onConnectionsChange?.apply(this, arguments);
      if (type === LiteGraph.INPUT) {
        if (isFetch) {
          this._uiInfo = null; // wiring changed -> previous executed info is stale
          this._updateSocketLabels?.();
          this._updateTitleFromLinks?.();
        } else if (isUnload && this.inputs?.[index]?.name === "models") {
          const link = connected ? link_info : null;
          const src = link ? linkSource(this, link.id) : null;
          const listSrc = src ? resolveListSource(src) : null;
          if (listSrc?._uiInfo) {
            this._applyModelInfo?.(listSrc._uiInfo);
          } else if (listSrc) {
            let labels = inputSummaries(listSrc).map((s) => s.label).filter(Boolean);
            if (!labels.length) {
              const single = resolveModelSource(listSrc);
              if (single) labels = [single];
            }
            const useWs = this.widgets?.filter((w) => w.name?.startsWith("use_")) ?? [];
            useWs.forEach((w, i) => {
              const lbl = labels[i];
              if (lbl) {
                w.label = SHORT(lbl.replace(/^\[[^\]]*\]\s*/, ""), 26);
                if (w.computeSize) w.computeSize();
              }
            });
          }
        }
        if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
      }
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const { status, info } = outputsOf(message);
      if (isFetch) {
        this._uiInfo = info;
        this._applyInfoTitle?.();
        const outLinks = this.outputs?.[0]?.links ?? [];
        for (const l of outLinks) {
          const link = graphOf(this)?.links?.[l];
          const dst = link ? graphOf(this)?.getNodeById(link.target_id) : null;
          dst?._applyModelInfo?.(info);
        }
      } else if (isUnload) {
        this._applyModelInfo(info);
        const summary = (status || "").split("\n").filter(Boolean).pop() ?? "";
        if (summary && this._canSetTitle()) {
          this.title = TRUNC(summary.replace(/^\[DynamicVRAM-Free\]\s*/, ""));
          this._uiAuto = true;
        }
      }
      if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
      return r;
    };
  },
});
