"""LTX-2.5 Smart Controls — full drop-in replacement for ltx25_smart_controls.

Ports the complete LTX25AllModesControlsV2 ("MODE / FINAL RESOLUTION / TIMING /
VAE DECODE") behavior — same 16 outputs, same summary format — PLUS a resolution
selector (aspect_ratio + megapixels + multiple). Independent of the third-party
pack so pack updates never overwrite it.

Resolution is COMPUTED by the node and emitted on the final_width / final_height
outputs; there are no width/height inputs. Use "Custom (manual)" + the
custom_width / custom_height widgets when you want to enter dimensions by hand.

Resolution formula (when aspect_ratio != "Custom (manual)"):
w = snap(ceil(sqrt(MP*1e6*AR)/multiple)*multiple), h = snap(ceil(sqrt(MP*1e6/AR)/multiple)*multiple)
where snap() rounds UP to the selected `multiple` (16..256) and clamps to 256..2048.
The result is a multiple of `multiple`, so multiple=32 yields 32-grid dims (MiniMax H3),
multiple=64 yields the LTX-2.5 Conv VAE 64-grid.
"""
import math

WEB_DIRECTORY = "./web"
MAX_SEED = (1 << 64) - 1

MODES = (
    "T2V (text only)",
    "I2V (first frame)",
    "FLF2V (first + last frame)",
)
TIMING_MODES = (
    "Duration drives frames",
    "Manual exact frames",
)
VAE_DECODE_PRESETS = (
    "Auto (VRAM-based)",
    "Low VRAM (12GB target, experimental)",
    "Medium Tiles (16GB recommended)",
    "Large Tiles (24GB recommended)",
)
ASPECT_RATIOS = {
    "1:1": 1.0,
    "4:3": 4.0 / 3.0,
    "3:2": 3.0 / 2.0,
    "16:9": 16.0 / 9.0,
    "21:9": 21.0 / 9.0,
    "9:16": 9.0 / 16.0,
    "3:4": 3.0 / 4.0,
    "2:3": 2.0 / 3.0,
}
MEGAPIXEL_OPTIONS = (
    0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.0, 1.2, 1.5, 1.8, 2.0,
)
MULTIPLE_OPTIONS = (16, 32, 64, 128, 256)
MIN_DIM = 256
MAX_DIM = 2048


def _snap(v: float, multiple: int) -> int:
    """Round v UP to the nearest multiple, clamped to [MIN_DIM, MAX_DIM]."""
    v = max(float(MIN_DIM), min(float(MAX_DIM), float(v)))
    return int(math.ceil(v / multiple) * multiple)


def _compute_size(ar: float, megapixels: float | str, multiple: int) -> tuple[int, int]:
    area = float(megapixels) * 1e6
    w = _snap(math.sqrt(area * ar), multiple)
    h = _snap(math.sqrt(area / ar), multiple)
    return w, h


def _detect_vram_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device_index = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    except Exception:
        return None


def _decode_profile(host: str) -> tuple[int, int, int, int]:
    if host == "24GB+ Quality":
        return 1024, 256, 128, 48
    if host == "16GB Fast":
        return 768, 192, 128, 32
    return 512, 64, 64, 8


def _resolve_vae_decode_preset(requested: str) -> tuple[str, float | None, str]:
    detected = _detect_vram_gib()
    aliases = {
        "12GB Safe (experimental)": "12GB Safe",
        "12GB Decode Tiles (experimental)": "12GB Safe",
        VAE_DECODE_PRESETS[1]: "12GB Safe",
        "16GB Fast": "16GB Fast",
        VAE_DECODE_PRESETS[2]: "16GB Fast",
        "24GB+ Quality": "24GB+ Quality",
        VAE_DECODE_PRESETS[3]: "24GB+ Quality",
    }
    if requested in aliases:
        host = aliases[requested]
    elif requested in {"Auto (detect VRAM)", VAE_DECODE_PRESETS[0]}:
        if detected is not None and detected >= 22:
            host = "24GB+ Quality"
        elif detected is not None and detected >= 14:
            host = "16GB Fast"
        else:
            host = "12GB Safe"
    else:
        raise ValueError(f"Unsupported VAE decode preset: {requested}")
    label = {
        "12GB Safe": VAE_DECODE_PRESETS[1],
        "16GB Fast": VAE_DECODE_PRESETS[2],
        "24GB+ Quality": VAE_DECODE_PRESETS[3],
    }[host]
    return host, detected, label


class LTX25Controls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(MODES), {"default": MODES[1]}),
                "vae_decode_preset": (list(VAE_DECODE_PRESETS), {"default": VAE_DECODE_PRESETS[0]}),
                "aspect_ratio": (["Custom (manual)"] + list(ASPECT_RATIOS), {"default": "16:9"}),
                "stage1_megapixels": (list(MEGAPIXEL_OPTIONS), {"default": 0.2}),
                "stage2_megapixels": (list(MEGAPIXEL_OPTIONS), {"default": 0.5}),
                "final_megapixels": (list(MEGAPIXEL_OPTIONS), {"default": 0.98}),
                "multiple": (list(MULTIPLE_OPTIONS), {"default": 64}),
                "custom_width": ("INT", {"default": 1344, "min": MIN_DIM, "max": MAX_DIM, "step": 64}),
                "custom_height": ("INT", {"default": 768, "min": MIN_DIM, "max": MAX_DIM, "step": 64}),
                "timing_mode": (list(TIMING_MODES), {"default": TIMING_MODES[0]}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 30.0, "step": 1.0}),
                "duration_seconds": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "manual_frames": ("INT", {"default": 361, "min": 9, "max": 993, "step": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": MAX_SEED, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "BOOLEAN", "BOOLEAN",
        "INT", "INT", "INT", "INT", "INT", "INT", "STRING", "FLOAT",
        "INT", "INT", "FLOAT", "FLOAT", "FLOAT",
    )
    RETURN_NAMES = (
        "final_width", "final_height", "stage1_width", "stage1_height", "frames",
        "fps", "effective_seconds", "need_first_frame", "need_last_frame",
        "tile_size", "overlap", "temporal_size", "temporal_overlap",
        "stage1_seed", "stage2_seed", "resolved_summary", "duration",
        "stage2_width", "stage2_height",
        "stage1_megapixels", "stage2_megapixels", "final_megapixels",
    )
    FUNCTION = "resolve"
    CATEGORY = "LTX-2.5"

    @staticmethod
    def _manual_frames(value: int) -> int:
        value = int(value)
        if 9 <= value <= 993 and (value - 1) % 8 == 0:
            return value
        bounded = min(max(value, 9), 993)
        lower = max(9, ((bounded - 1) // 8) * 8 + 1)
        upper = min(993, lower if lower == value else lower + 8)
        raise ValueError(
            "manual_frames must be exactly 8*n+1 and at most 993 for LTX-2.5 audio. "
            f"Use {lower} or {upper}, not {value}."
        )

    def resolve(
        self,
        mode,
        vae_decode_preset,
        aspect_ratio="16:9",
        stage1_megapixels=0.2,
        stage2_megapixels=0.5,
        final_megapixels=0.98,
        multiple=64,
        custom_width=1344,
        custom_height=768,
        timing_mode=TIMING_MODES[0],
        fps=24.0,
        duration_seconds=15.0,
        manual_frames=361,
        seed=42,
    ):
        if mode not in MODES:
            raise ValueError(f"Unsupported LTX mode: {mode}")
        if timing_mode not in TIMING_MODES:
            raise ValueError(f"Unsupported timing mode: {timing_mode}")
        multiple = int(multiple)

        if aspect_ratio == "Custom (manual)":
            final_width = _snap(custom_width, multiple)
            final_height = _snap(custom_height, multiple)
            # custom stages halve down from final, each snapped to multiple (legacy behavior)
            stage2_width = _snap(final_width // 2, multiple)
            stage2_height = _snap(final_height // 2, multiple)
            stage1_width = _snap(stage2_width // 2, multiple)
            stage1_height = _snap(stage2_height // 2, multiple)
        else:
            ar = ASPECT_RATIOS[aspect_ratio]
            stage1_width, stage1_height = _compute_size(ar, stage1_megapixels, multiple)
            stage2_width, stage2_height = _compute_size(ar, stage2_megapixels, multiple)
            final_width, final_height = _compute_size(ar, final_megapixels, multiple)

        fps = float(fps)
        if not 1.0 <= fps <= 30.0:
            raise ValueError("fps must be between 1 and 30.")

        if timing_mode == TIMING_MODES[0]:
            duration_seconds = float(duration_seconds)
            if not 1.0 <= duration_seconds <= 30.0:
                raise ValueError("duration_seconds must be between 1 and 30.")
            frames = 1 + math.floor(fps * duration_seconds / 8.0) * 8
            if frames < 9:
                raise ValueError(
                    "The requested duration is too short for LTX-2.5. "
                    f"At {fps:g} fps, request at least {8.0 / fps:.3f} seconds."
                )
            if frames > 993:
                raise ValueError(
                    "The requested duration and FPS exceed the 993-frame LTX audio limit. "
                    "Reduce duration or FPS."
                )
            timing_summary = (
                f"Duration: requested {duration_seconds:g}s -> {frames}f "
                f"@{fps:g} = {(frames - 1) / fps:.3f}s"
            )
        else:
            frames = self._manual_frames(manual_frames)
            timing_summary = (
                f"Manual: {frames}f @{fps:g} = {(frames - 1) / fps:.3f}s "
                "(duration field ignored)"
            )

        host, detected, resolved_label = _resolve_vae_decode_preset(vae_decode_preset)
        tile_size, overlap, temporal_size, temporal_overlap = _decode_profile(host)
        stage1_seed = int(seed) & MAX_SEED
        stage2_seed = (stage1_seed + 1) & MAX_SEED
        need_first = mode != MODES[0]
        need_last = mode == MODES[2]
        detected_text = "unknown" if detected is None else f"{detected:.1f}GiB"
        host_label = "12GB Decode Tiles" if host == "12GB Safe" else host
        support = "EXPERIMENTAL" if host == "12GB Safe" else "host profile"
        base_summary = (
            f"{mode} | FINAL {final_width}x{final_height} "
            f"(stage 1 {stage1_width}x{stage1_height} -> stage 2 {stage2_width}x{stage2_height} -> FINAL {final_width}x{final_height}) | "
            f"{timing_summary} | {host_label} {support}, detected={detected_text} | "
            f"Conv VAE tiles {tile_size}/{overlap}/{temporal_size}/{temporal_overlap}. "
            "Canvas and timing are user-controlled and are not auto-clamped for VRAM."
        )
        summary_prefix = " | ".join(base_summary.split(" | ")[:3])
        selection = (
            f"Auto -> {resolved_label}"
            if vae_decode_preset in {"Auto (detect VRAM)", VAE_DECODE_PRESETS[0]}
            else resolved_label
        )
        summary = (
            f"{summary_prefix} | VAE decode: {selection}, detected={detected_text} | "
            f"Conv VAE tiles {tile_size}/{overlap}/{temporal_size}/{temporal_overlap}. "
            "This preset changes decode tiling only, not model precision, sampling, or "
            "generation quality. Larger tiles can reduce tile count but use more VRAM. "
            "Canvas and timing remain user-controlled."
        )
        return {
            "ui": {"text": [summary]},
            "result": (
                final_width, final_height, stage1_width, stage1_height,
                frames, fps, (frames - 1) / fps, need_first, need_last,
                tile_size, overlap, temporal_size, temporal_overlap,
                stage1_seed, stage2_seed, summary, duration_seconds,
                stage2_width, stage2_height,
                float(stage1_megapixels), float(stage2_megapixels), float(final_megapixels),
            ),
        }


class LTX25Resolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (["Custom (manual)"] + list(ASPECT_RATIOS), {"default": "16:9"}),
                "megapixels": (list(MEGAPIXEL_OPTIONS), {"default": 0.98}),
                "multiple": (list(MULTIPLE_OPTIONS), {"default": 64}),
                "custom_width": ("INT", {"default": 1344, "min": MIN_DIM, "max": MAX_DIM, "step": 64}),
                "custom_height": ("INT", {"default": 768, "min": MIN_DIM, "max": MAX_DIM, "step": 64}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("final_width", "final_height", "stage1_width", "stage1_height", "summary")
    FUNCTION = "resolve"
    CATEGORY = "LTX-2.5"

    def resolve(self, aspect_ratio="16:9", megapixels=0.98, multiple=64,
                custom_width=1344, custom_height=768):
        multiple = int(multiple)
        if aspect_ratio == "Custom (manual)":
            final_width = _snap(custom_width, multiple)
            final_height = _snap(custom_height, multiple)
        else:
            ar = ASPECT_RATIOS[aspect_ratio]
            final_width, final_height = _compute_size(ar, megapixels, multiple)
        summary = (
            f"{aspect_ratio} {megapixels}MP x{multiple} -> "
            f"{final_width}x{final_height} (stage1 {final_width // 2}x{final_height // 2})"
        )
        return (final_width, final_height, final_width // 2, final_height // 2, summary)


NODE_CLASS_MAPPINGS = {
    "LTX25Controls": LTX25Controls,
    "LTX25Resolution": LTX25Resolution,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX25Controls": "LTX25 Smart Controls (full)",
    "LTX25Resolution": "LTX25 Resolution Selector",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]