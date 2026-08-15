"""Release real VRAM from dynamic-VRAM (aimdo VBAR) models.

Core unload_all_models()/free_memory() do not free dynamic models: free_memory
sets memory_to_free=0 for dynamic models, and model_unload(1e30) skips
partially_unload (1e30 >= loaded_size) so VBAR stays resident -> OOM when
switching workflows (ComfyUI issue #12335). Calling partially_unload(1e32)
is the exact internal path that frees VBAR (vbar.free_memory); weights reload
on-demand from file. Standard patchers get a full unload.
"""

import gc

import torch

import comfy.model_management as mm


def _release_vram():
    freed_mb = 0.0
    n_models = 0
    for loaded in list(mm.current_loaded_models):
        model = loaded.model
        if model is None:
            continue
        n_models += 1
        try:
            if model.is_dynamic():
                freed_mb += model.partially_unload(1e32) / (1024 * 1024)
            elif model.loaded_size() > 0:
                freed_mb += model.partially_unload(model.offload_device, 1e32) / (1024 * 1024)
        except Exception as e:
            print(f"[DynamicVRAM-Free] unload failed for {type(model).__name__}: {e}")

    mm.unload_all_models()
    mm.soft_empty_cache(True)
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        from comfy import memory_management as cmm
        if getattr(cmm, "aimdo_enabled", False):
            import comfy_aimdo.model_vbar
            dev = mm.get_torch_device()
            comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
            freed = comfy_aimdo.model_vbar.vbars_analyze(dev)
            if freed:
                freed_mb += freed / (1024 * 1024)
    except Exception as e:
        print(f"[DynamicVRAM-Free] aimdo vbar cleanup skipped: {e}")

    return n_models, freed_mb


class DynamicVRAMFree:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": ("STRING,IMAGE,LATENT,MODEL,VAE,CLIP,CONDITIONING", {}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "free_vram"
    CATEGORY = "DynamicVRAM"
    OUTPUT_NODE = True
    GLOBAL_STATE_CHANGE = True

    def free_vram(self, anything):
        n_models, freed_mb = _release_vram()
        total = mm.get_total_memory(mm.get_torch_device()) / (1024 * 1024)
        free = mm.get_free_memory(mm.get_torch_device()) / (1024 * 1024)
        msg = (
            f"[DynamicVRAM-Free] released {freed_mb:.0f}MB from {n_models} model(s); "
            f"VRAM now {free:.0f}MB free / {total:.0f}MB"
        )
        print(msg)
        return (msg,)


NODE_CLASS_MAPPINGS = {"DynamicVRAMFree": DynamicVRAMFree}
NODE_DISPLAY_NAME_MAPPINGS = {"DynamicVRAMFree": "DynamicVRAM Free (real VBAR release)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
