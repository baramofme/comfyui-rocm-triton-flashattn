"""VRAM unload nodes split by role (v4):

1. DynamicVRAMModelFetch — the "model fetcher". Takes up to 10 MODEL/CLIP/VAE
   sockets, normalizes each to its ModelPatcher (CLIP/VAE expose `.patcher`,
   comfy/sd.py:264/:1044), and emits them as a VRAM_MODEL_LIST of
   (socket_index, patcher) tuples. Does NOT touch VRAM.

2. DynamicVRAMFree — the unload node. Consumes the fetcher's list, displays
   per-model info, and unloads:
   - "unload_all": force-unload every model in mm.current_loaded_models
     (dynamic aimdo: partially_unload(offload_device, 1e32) -> vbar.free_memory
     + pinned RAM + detach; legacy: detach), then clears the torch cache.
   - "selective": unloads only the registry entries whose inner model matches a
     CHECKED position in the fetched list (use_1..use_10 map to list order).
   - "show models info": read-only report of fetched + registry models.

ComfyUI 0.32 facts (source-verified against /workspace/comfy):
- mm.current_loaded_models holds LoadedModel wrappers; .model is the ModelPatcher.
  Neither patchers nor CLIP/VAE/BaseModel carry a model name — the displayable
  identifier is the inner class name (same source as the aimdo staging log).
- unload_all_models() -> LoadedModel.model_unload(1e30) -> detach() works for
  legacy models, but dynamic (aimdo) models keep their VBAR (Virtual Block
  Allocator) reservation -> VRAM leak (ComfyUI issue #12335). The ONLY release
  path is patcher.partially_unload(offload_device, 1e32). (An older version
  called partially_unload(1e32), passing 1e32 as device_to with
  memory_to_free=0 -> freed 0 bytes.)
- Legacy partially_unload(offload_device, 1e32) only frees LOWVRAM-partial
  weights; the full unload for a fully-loaded legacy model is detach().
- Graph-passed patchers are clones sharing the inner model instance, so
  registry matching is done by inner-object identity.
"""
import gc
import hashlib

import torch

import comfy.model_management as mm

WEB_DIRECTORY = "./web"

MAX_INPUTS = 10
MODEL_LIST_TYPE = "VRAM_MODEL_LIST"

_last_reset = None  # {"prompt_id": str|None, "fp": str}
_MODEL_NAME_KEYS = {
    "unet_name", "clip_name", "vae_name", "ckpt_name", "lora_name",
    "model_name", "control_net_name", "filename",
}


def _graph_fingerprint(prompt) -> str | None:
    if not isinstance(prompt, dict):
        return None
    items = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        ct = str(node.get("class_type", ""))
        inp = node.get("inputs", {}) or {}
        stable = []
        for k, v in inp.items():
            if isinstance(v, list):
                stable.append((k, "LINK"))
            elif k in _MODEL_NAME_KEYS and isinstance(v, str):
                stable.append((k, v))
        stable.sort()
        items.append((str(nid), ct, tuple(stable)))
    items.sort()
    return hashlib.sha256(repr(items).encode()).hexdigest()



def _current_prompt():
    try:
        import server

        ps = server.PromptServer.instance
        for item in ps.prompt_queue.currently_running.values():
            if len(item) >= 3 and isinstance(item[2], dict):
                pid = None
                for cand in (item[0] if item else None,
                             item[1].get("id") if len(item) > 1 and isinstance(item[1], dict) else None,
                             item[2].get("id"), item[2].get("prompt_id"), item[2].get("__id__")):
                    if isinstance(cand, str) and cand:
                        pid = cand
                        break
                return item[2], pid
    except Exception:
        pass
    return None, None


def _reset_on_switch() -> tuple[bool, str]:
    global _last_reset
    prompt, pid = _current_prompt()
    fp = _graph_fingerprint(prompt)
    if fp is None:
        return False, "no prompt access"
    if _last_reset is None:
        changed = True
    elif pid is not None and _last_reset.get("prompt_id") == pid:
        # same running prompt -> later reset nodes must not be spuriously skipped
        changed = True
    else:
        changed = fp != _last_reset.get("fp")
    _last_reset = {"prompt_id": pid, "fp": fp}
    return changed, ("first run / workflow changed" if changed else "same workflow")


def _type_tag(inner_name: str) -> str:
    low = inner_name.lower()
    if "clip" in low or "text" in low or "encoder" in low or "temodel" in low:
        return "clip"
    if "vae" in low:
        if "audio" in low:
            return "vae-audio"
        if "video" in low:
            return "vae-video"
        return "vae"
    if "audio" in low:
        return "audio"
    if "video" in low:
        return "video"
    return "diffusion"


def _describe(patcher) -> tuple[str, str]:
    inner = type(patcher.model).__name__
    tag = _type_tag(inner)
    try:
        size_mb = patcher.model_size() / (1024 * 1024)
    except Exception:
        size_mb = 0.0
    dev = getattr(patcher, "load_device", None)
    dyn = " [dynamic]" if patcher.is_dynamic() else ""
    return tag, f"[{tag}]{dyn} {inner} {size_mb:.0f}MB on {dev}"


def _to_patcher(obj):
    if obj is None:
        return None
    if hasattr(obj, "patcher"):
        return obj.patcher
    if hasattr(obj, "model") and hasattr(obj, "offload_device"):
        return obj
    return None


def _unload_one(lm) -> tuple[float, str]:
    p = lm.model
    if p is None:
        return 0.0, "(dead)"
    _, desc = _describe(p)
    before = p.loaded_size()
    freed = 0
    try:
        if p.is_dynamic():
            freed += max(p.partially_unload(p.offload_device, 1e32), 0)
            try:
                freed += max(p.partially_unload_ram(1e32), 0)
            except Exception:
                pass
        try:
            if getattr(lm, "model_finalizer", None) is not None:
                lm.model_unload(1e30)
            else:
                p.detach()
        except Exception:
            try:
                p.detach()
            except Exception:
                pass
    except Exception as e:
        print(f"[DynamicVRAM-Free] unload failed for {desc}: {e}")
    after = p.loaded_size()
    freed_mb = (freed + max(before - after, 0)) / (1024 * 1024)
    try:
        if lm in mm.current_loaded_models:
            mm.current_loaded_models.remove(lm)
    except Exception:
        pass
    return freed_mb, desc


def _registry_snapshot() -> dict:
    snap = {}
    for lm in list(mm.current_loaded_models):
        if lm.model is None or lm.model.model is None:
            continue
        _, desc = _describe(lm.model)
        snap[id(lm.model.model)] = (lm, desc)
    return snap


def _cache_clear():
    try:
        mm.unload_all_models()
    except Exception:
        pass
    try:
        mm.soft_empty_cache(True)
    except Exception:
        pass
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


class DynamicVRAMModelFetch:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"model_{i}": ("MODEL,CLIP,VAE", {}) for i in range(1, MAX_INPUTS + 1)}
        return {"required": {}, "optional": optional}

    RETURN_TYPES = (MODEL_LIST_TYPE, "STRING")
    RETURN_NAMES = ("models", "info")
    FUNCTION = "fetch"
    CATEGORY = "DynamicVRAM"
    OUTPUT_NODE = True

    def fetch(self, **kwargs):
        out = []
        for i in range(1, MAX_INPUTS + 1):
            p = _to_patcher(kwargs.get(f"model_{i}"))
            if p is not None:
                out.append((i, p))
        info_lines = []
        for pos, (idx, p) in enumerate(out, 1):
            _, desc = _describe(p)
            info_lines.append(f"{pos}. {desc}")
        info = "\n".join(info_lines) if info_lines else "no models connected"
        return {"ui": {"text": [info]}, "result": (out, info)}


class DynamicVRAMFree:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "anything": ("STRING,IMAGE,LATENT,MODEL,VAE,CLIP,CONDITIONING", {}),
            "models": (MODEL_LIST_TYPE, {}),
            "mode": (["unload_all", "selective", "show models info", "reset on switch"], {"default": "unload_all"}),
            "clear_cache": ("BOOLEAN", {"default": True}),
            "reset_on_switch": ("BOOLEAN", {"default": False}),
        }
        for i in range(1, MAX_INPUTS + 1):
            optional[f"use_{i}"] = ("BOOLEAN", {"default": False})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "info")
    FUNCTION = "free_vram"
    CATEGORY = "DynamicVRAM"
    OUTPUT_NODE = True
    GLOBAL_STATE_CHANGE = True

    def free_vram(self, anything=None, models=None, mode="unload_all", clear_cache=True, reset_on_switch=False, **kwargs):
        if mode == "reset on switch":
            mode = "unload_all"
            reset_on_switch = True
        if reset_on_switch and mode != "show models info":
            changed, _ = _reset_on_switch()
            if not changed:
                msg = "[DynamicVRAM-Free] reset-on-switch: same workflow -> skipped (models kept for speed)"
                print(msg)
                return {"ui": {"text": [msg, ""]}, "result": (msg, "")}
        fetched = models or []
        if isinstance(fetched, (list, tuple)):
            if fetched and not isinstance(fetched[0], (list, tuple)):
                fetched = [(i + 1, p) for i, p in enumerate(fetched)]
        else:
            p = _to_patcher(fetched)
            fetched = [(1, p)] if p is not None else []
        lines = []
        snap = _registry_snapshot()
        info_lines = []
        for pos, (idx, p) in enumerate(fetched, 1):
            _, desc = _describe(p)
            info_lines.append(f"{pos}. {desc}")
        info = "\n".join(info_lines) if info_lines else "no models fetched" 

        if mode == "show models info":
            lines.append(f"[DynamicVRAM-Free] show models info ({len(fetched)} fetched, {len(snap)} in registry):")
            for pos, (idx, p) in enumerate(fetched, 1):
                _, desc = _describe(p)
                loaded = "yes" if id(p.model) in snap else "no"
                lines.append(f"  #{pos} (socket {idx}): {desc} (in registry: {loaded})")
            if snap:
                lines.append("  registry:")
                for _, desc in snap.values():
                    lines.append(f"    - {desc}")
            else:
                lines.append("  registry: empty")
            lines.append("[DynamicVRAM-Free] show models info: nothing unloaded")
            report = "\n".join(lines)
            print(report)
            return {"ui": {"text": [report, info]}, "result": (report, info)}

        if mode == "unload_all":
            total_freed = 0.0
            n = 0
            for lm in list(mm.current_loaded_models):
                if lm.model is None:
                    continue
                freed_mb, desc = _unload_one(lm)
                total_freed += freed_mb
                n += 1
                lines.append(f"  unloaded: {desc} (weights {freed_mb:.0f}MB)")
            if n == 0:
                lines.append("  no models in current_loaded_models")
            if clear_cache:
                _cache_clear()
            lines.append(
                f"[DynamicVRAM-Free] unload_all: unloaded {n} model(s), "
                f"model weights {total_freed:.0f}MB; "
                f"VRAM {mm.get_free_memory(mm.get_torch_device()) / (1024 * 1024):.0f}MB free"
            )
            report = "\n".join(lines)
            print(report)
            return {"ui": {"text": [report, info]}, "result": (report, info)}

        # selective: unload registry entries matching CHECKED positions of the fetched list
        n = 0
        for pos, (idx, p) in enumerate(fetched, 1):
            _, desc = _describe(p)
            checked = bool(kwargs.get(f"use_{pos}", False))
            if not checked:
                lines.append(f"  #{pos}: {desc} (unchecked, kept)")
                continue
            lm, rdesc = snap.get(id(p.model), (None, None))
            if lm is None:
                lines.append(f"  #{pos}: {desc} (not in registry, nothing to unload)")
                continue
            freed_mb, _ = _unload_one(lm)
            lines.append(f"  #{pos}: unloaded {rdesc} (weights {freed_mb:.0f}MB)")
            n += 1
        if n == 0:
            lines.append("[DynamicVRAM-Free] selective: no checked model was loaded; nothing unloaded")
        if clear_cache:
            _cache_clear()
        lines.append(
            f"[DynamicVRAM-Free] selective: unloaded {n} model(s); "
            f"VRAM {mm.get_free_memory(mm.get_torch_device()) / (1024 * 1024):.0f}MB free"
        )
        report = "\n".join(lines)
        print(report)
        return {"ui": {"text": [report, info]}, "result": (report, info)}


NODE_CLASS_MAPPINGS = {
    "DynamicVRAMFree": DynamicVRAMFree,
    "DynamicVRAMModelFetch": DynamicVRAMModelFetch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicVRAMFree": "DynamicVRAM Free (real VBAR release)",
    "DynamicVRAMModelFetch": "DynamicVRAM Model Fetch",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
