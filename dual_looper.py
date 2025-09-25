import torch
from typing import Any, List, Optional, Tuple
import json
import os, glob, shutil
from pathlib import Path
from .vantage_project import get_vantage_dir

import numpy as np

from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator
from comfy_api.internal import register_versions, ComfyAPIWithVersion
from comfy_api.version_list import supported_versions
from comfy_api.latest import io, ComfyExtension

import comfy.samplers
import node_helpers
import comfy.clip_vision

# ——— logging ———
def _log(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass

# ——— tensor helpers ———
LATENT_DOWNSCALE = 8
LATENT_CHANNELS = 16

def _align_px(x, multiple=16):
    return max(multiple, (int(x) // multiple) * multiple)

def _model_device_dtype(model) -> Tuple[torch.device, torch.dtype]:
    try:
        dev = next(model.model.parameters()).device
        dt = next(model.model.parameters()).dtype
    except Exception:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dt = torch.float32
    return dev, dt

def _to(x: torch.Tensor, device, dtype):
    if x.device != device or x.dtype != dtype:
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return x

# ——— disk I/O helpers ———
def _get_root_vantage_dir() -> Path:
    return Path(get_vantage_dir())
    
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _norm_uint8_bhwc(img_bhwc: torch.Tensor) -> np.ndarray:
    x = img_bhwc.detach().cpu().clamp(0, 1).numpy()  # B,H,W,C in [0,1]
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def _save_loop_frames(loop_dir: Path, img_bhwc: torch.Tensor):
    # img_bhwc: (B,H,W,C) float [0,1]
    _ensure_dir(loop_dir)
    arr = _norm_uint8_bhwc(img_bhwc)
    # Prefer PIL via torchvision to avoid extra deps; fallback to imageio if available
    try:
        from PIL import Image
        for i in range(arr.shape[0]):
            fn = loop_dir / f"{i:06d}.png"
            Image.fromarray(arr[i]).save(str(fn))
    except Exception:
        import imageio.v3 as iio
        for i in range(arr.shape[0]):
            fn = loop_dir / f"{i:06d}.png"
            iio.imwrite(str(fn), arr[i])

def _load_all_frames(project_dir: Path) -> torch.Tensor:
    # Collect subdirs named as integers 0,1,2...
    dirs = [p for p in project_dir.iterdir() if p.is_dir()]
    dirs_sorted = sorted(dirs, key=lambda p: int(p.name) if p.name.isdigit() else 10**9)
    frame_paths: List[str] = []
    for d in dirs_sorted:
        frame_paths.extend(sorted(glob.glob(str(d / "*.png"))))
    if not frame_paths:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)
    # Read with PIL for RGB/RGBA consistency; fallback to imageio
    frames = []
    use_pil = True
    try:
        from PIL import Image
    except Exception:
        use_pil = False
    if use_pil:
        from PIL import Image
        for fp in frame_paths:
            im = Image.open(fp).convert("RGBA" if fp.lower().endswith(".png") else "RGB")
            frames.append(np.array(im))
    else:
        import imageio.v3 as iio
        for fp in frame_paths:
            im = iio.imread(fp)
            if im.ndim == 2:
                im = np.stack([im]*3, axis=-1)
            frames.append(im)
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # F,H,W,C
    t = torch.from_numpy(arr)  # (F,H,W,C) float in [0,1]
    return t

class VantageDualLooperI2V:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "project_data": (IO.ANY, {"tooltip": "Dict with 'prompt_lines' and 'project_id'."}),
                "negative_text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "Negative text."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps_init": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "steps_high": ("INT", {"default": 9, "min": 1, "max": 10000}),
                "steps_low": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "cfg_init": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "width": (IO.INT, {"default": 832, "min": 64, "max": 8192, "step": 16}),
                "height": (IO.INT, {"default": 480, "min": 64, "max": 8192, "step": 16}),
                "batch_size": (IO.INT, {"default": 1, "min": 1, "max": 64}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fps": (["16", "25"], {"default": "16"}),
                "overlap": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1, "tooltip": "How many frames overlap into the next loop. 0 or 1 = last frame; N>=2 = Nth from bottom."}),
                "crop": (["center", "none"],),
                "vae": (IO.VAE, {"tooltip": "The VAE Model used for decoding latent."}),
                "start_image": (IO.IMAGE,),
            },
            "optional": {
                "model_init": ("MODEL",),
                "clip_vision": (IO.CLIP_VISION,),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)
    FUNCTION = "dual_loop_i2v"
    CATEGORY = "video/latent"

    # ——— encoding helpers unchanged except simplified signature ———
    def encode_clip_vision(self, clip_vision, image, crop: str):
        if clip_vision is None:
            return None
        crop_mode = (str(crop or "center").lower())
        if crop_mode not in ("center", "none"):
            crop_mode = "center"
        crop_image = (crop_mode == "center")

        img = image
        if not isinstance(img, torch.Tensor):
            raise ValueError("encode_clip_vision expected IMAGE tensor.")
        img = img.to(torch.float32).clamp(0, 1)
        if img.dim() == 3:
            if img.shape[-1] in (3, 4):
                img = img.unsqueeze(0)
            else:
                img = img.unsqueeze(-1).unsqueeze(0)
        elif img.dim() == 4:
            B, D1, D2, D3 = img.shape
            last_channel_like = (D3 in (1, 3, 4))
            mid_channel_like = (D1 in (1, 3, 4))
            if (not last_channel_like) and mid_channel_like:
                img = img.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError(f"Unsupported IMAGE rank {img.dim()}, expected 3 or 4 dims.")
        if img.dim() != 4 or img.shape[-1] not in (1, 3, 4):
            raise ValueError(f"encode_clip_vision needs BHWC with C in (1,3,4), got {tuple(img.shape)}")
        output = clip_vision.encode_image(img, crop=crop_image)
        return output

    def _encode_prompts_clip(self, clip, pos_text: str, neg_text: str):
        tokens_pos = clip.tokenize(pos_text or "")
        tokens_neg = clip.tokenize(neg_text or "")
        pos_dict = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=True)
        neg_dict = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
        positive = [(pos_dict["cond"], {"pooled_output": pos_dict.get("pooled_output")})]
        negative = []
        if neg_text and neg_dict is not None:
            negative = [(neg_dict["cond"], {"pooled_output": neg_dict.get("pooled_output")})]
        return positive, negative

    def _ksampler_call(self, model, positive, negative, latent, steps, cfg, sampler_name, scheduler, denoise, seed, disable_noise=False, start_step=None, last_step=None,  force_full_denoise=False):
        import nodes
        result_force = not force_full_denoise
        result_disable = not disable_noise
        _log(f"[Vantage Dual Looper] Add Noise: {result_disable} Return with noise {result_force}")
        out, = nodes.common_ksampler(model=model, seed=int(seed), steps=int(steps), cfg=float(cfg), sampler_name=str(sampler_name), scheduler=str(scheduler), positive=positive, negative=negative, latent={"samples": latent["samples"]}, denoise=float(denoise), disable_noise=disable_noise, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise)
        return {"samples": out["samples"]}

    def _get_clip_vision_latent(
        self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None,
    ):
        # sampler input (window-sized latent “noise”/zeros)
        latent = torch.zeros(
            [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
            device=comfy.model_management.intermediate_device()
        )
        tdim = 2  # [B, C, T, H, W]
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)
            image = torch.ones(
                (length, height, width, start_image.shape[-1]),
                device=start_image.device, dtype=start_image.dtype
            ) * 0.5
            image[:start_image.shape[0]] = start_image
            concat_latent_image = vae.encode(image[:, :, :, :3])
            # Keep mask sized to the concat latent
            mask_T = int(concat_latent_image.size(tdim))
            mask = torch.ones(
                (1, 1, mask_T, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
                device=start_image.device, dtype=start_image.dtype
            )
            # Zero region for existing frames
            zeros_T = min(mask_T, ((start_image.shape[0] - 1) // 4) + 1)
            mask[:, :, :zeros_T] = 0.0
            positive = node_helpers.conditioning_set_values(
                positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
            )
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})
        return positive, negative, {"samples": latent}

    def dual_loop_i2v(
        self,
        model_high, model_low, clip, project_data, negative_text,
        width, height, batch_size,
        steps_init, steps_high, steps_low, cfg_init, cfg, sampler_name, scheduler, denoise,
        fps, overlap,
        seed, crop, vae,
        start_image, clip_vision = None,
        model_init = None,
    ):
        # Parse project_data
        pd = project_data or {}
        if isinstance(pd, str):
            try:
                pd = json.loads(pd)
            except Exception:
                pd = {"prompt": str(pd)}
        project_id = str(pd.get("id", "default")).strip() or "default"
        
        prompt_lines: List[str] = []
        if isinstance(pd, dict):
            if isinstance(pd.get("prompt_lines"), list):
                prompt_lines = [str(x).strip() for x in pd["prompt_lines"] if str(x).strip()]
            elif isinstance(pd.get("prompt"), str):
                prompt_lines = [ln.strip() for ln in pd["prompt"].splitlines() if ln.strip()]
        if not prompt_lines:
            prompt_lines = [""]

        # Negative once
        _, negative = self._encode_prompts_clip(clip, "", str(negative_text or ""))

        # fps
        try:
            fps_val = int(fps)
        except Exception:
            fps_val = 16

        window_size = 81 if fps_val == 16 else 129
        
        # overlap policy
        try:
            overlap_n = int(overlap)
        except Exception:
            overlap_n = 1
        pick_n = 1 if overlap_n <= 1 else overlap_n  # seed uses last Nth frame
        discard_n = pick_n                             # frames to drop at next loop boundary

        # Each prompt line equals 5 seconds (unchanged behavior)
        total_seconds = max(1, len(prompt_lines) * 5)
        target_frames = (total_seconds * fps_val) + 1 if fps_val == 16 else (total_seconds * fps_val) + 4

        model_device, model_dtype = _model_device_dtype(model_high)
        cpu = torch.device("cpu")

        L = max(2, int(window_size))
        width = _align_px(width, 16)
        height = _align_px(height, 16)
        B = int(batch_size)

        seed64 = int(seed) & 0xFFFFFFFFFFFFFFFF

        total_frames = int(target_frames) if target_frames > 0 else L
        produced_frames = 0
        tdim = 2
        loop_idx = 0

        # Prepare disk paths
        root_vantage = _get_root_vantage_dir()
        project_dir = root_vantage / project_id
        _ensure_dir(project_dir)
        
        # Parse start_prompt from project_data
        start_prompt_idx = 0
        try:
            start_prompt_idx = int((pd or {}).get("start_prompt", 0))
        except Exception:
            start_prompt_idx = 0

        # Determine prompt_lines now so we can validate start_prompt
        prompt_lines: List[str] = []
        if isinstance(pd, dict):
            if isinstance(pd.get("prompt_lines"), list):
                prompt_lines = [str(x).strip() for x in pd["prompt_lines"] if str(x).strip()]
            elif isinstance(pd.get("prompt"), str):
                prompt_lines = [ln.strip() for ln in pd["prompt"].splitlines() if ln.strip()]
        if not prompt_lines:
            prompt_lines = [""]

        # Validate start_prompt against number of prompt lines
        max_prompt_idx = max(0, len(prompt_lines) - 1)
        if start_prompt_idx < 0:
            start_prompt_idx = 0
        if start_prompt_idx > max_prompt_idx:
            _log(f"[Vantage Dual Looper] start_prompt {start_prompt_idx} out of range (prompt_lines={len(prompt_lines)}); resetting to 0.")
            start_prompt_idx = 0
        
        # After clamping, walk backward until a valid prev restart folder exists (start_prompt_idx - 1)
        def _folder_has_pngs(p: Path) -> bool:
            if not p.exists() or not p.is_dir():
                return False
            pngs = sorted(glob.glob(str(p / "*.png")))
            return len(pngs) > 0

        original_spi = start_prompt_idx 
        while start_prompt_idx > 0:
            prev_dir = project_dir / f"{start_prompt_idx - 1}"
            if _folder_has_pngs(prev_dir):
                break
            # decrement and continue searching backward
            start_prompt_idx -= 1 

        if original_spi != start_prompt_idx:
            _log(f"Vantage Dual Looper: adjusted start_prompt_idx from {original_spi} to {start_prompt_idx} based on restart folder availability.")

        # With final start_prompt_idx, compute produced frames and try to load resume seed
        frames_per_prompt = int(fps_val) * 5  # 5 seconds per prompt line 
        produced_frames = int(start_prompt_idx) * frames_per_prompt 
        _log(f"Vantage Dual Looper: resume produced_frames set to {produced_frames} (start_prompt={start_prompt_idx}, fps={fps_val}, frames_per_prompt={frames_per_prompt}).")

        resume_seed_image = None
        if start_prompt_idx > 0:
            prev_dir = project_dir / f"{start_prompt_idx - 1}"
            if prev_dir.exists() and prev_dir.is_dir():
                # load last image file by name sort
                pngs = sorted(glob.glob(str(prev_dir / "*.png")))
                if pngs:
                    try:
                        from PIL import Image
                        im = Image.open(pngs[-1]).convert("RGBA" if pngs[-1].lower().endswith(".png") else "RGB")
                        arr = np.array(im).astype(np.float32) / 255.0  # H,W,C
                        resume_seed_image = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W,C)
                        _log(f"[Vantage Dual Looper] resume image seed from {pngs[-1]}")
                    except Exception:
                        try:
                            import imageio.v3 as iio
                            im = iio.imread(pngs[-1])
                            if im.ndim == 2:
                                im = np.stack([im]*3, axis=-1)
                            arr = im.astype(np.float32) / 255.0
                            resume_seed_image = torch.from_numpy(arr).unsqueeze(0)
                            _log(f"[Vantage Dual Looper] resume seed via imageio from {pngs[-1]}")
                        except Exception as e:
                            _log(f"[Vantage Dual Looper] failed to read last image for resume: {e}")
        # Clean current and future loop folders
        to_delete = [p for p in project_dir.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name) >= start_prompt_idx]
        for p in sorted(to_delete, key=lambda x: int(x.name)):
            try:
                shutil.rmtree(p, ignore_errors=True)
                _log(f"[Vantage Dual Looper] removed folder {p}")
            except Exception as e:
                _log(f"[Vantage Dual Looper] could not remove {p}: {e}")

        prev_seed_image = None  # BHWC float [0,1], single frame selected by overlap
        
        if start_prompt_idx > 0 and resume_seed_image is not None:
            prev_seed_image = resume_seed_image  # will be used for the first generated loop (loop_idx = start_prompt_idx)

        # Start from start_prompt index
        loop_idx = int(start_prompt_idx)

        def build_window_latent(win_T: int, positive, negative, loopidx):
            parts: List[torch.Tensor] = []

            # Choose seed image: first loop uses start_image; subsequent loops use last decoded image
            if loopidx == 0:
                seed_image = start_image
            else:
                seed_image = prev_seed_image

            clip_vision_output = self.encode_clip_vision(clip_vision, seed_image, crop) if seed_image is not None else None
            positive1, negative1, clip_lats = self._get_clip_vision_latent(
                positive, negative, vae, width, height, win_T, batch_size, seed_image, clip_vision_output
            )

            if clip_lats["samples"].device.type == "cuda":
                _log(f"[Vantage Dual Looper] Clip Vision Latent device cuda")
                parts.append(_to(clip_lats["samples"], cpu, torch.float32))
            else:
                _log(f"[Vantage Dual Looper] Clip Vision Latent device cpu")
                parts.append(clip_lats["samples"])

            full = parts[0]
            return positive1, negative1, {"samples": full}

        while produced_frames < total_frames - 4:
            remaining = (total_frames - produced_frames)
            win_T = min(L, remaining + 1)
            contrib = win_T  # decoded frames this window before any discard
            _log(f"[Vantage Dual Looper] remaining={remaining}, win_T={win_T}, contrib={contrib}, produced={produced_frames}")


            # Positive prompt: select line by loop index (clamped)
            if loop_idx < len(prompt_lines):
                pos_str = prompt_lines[loop_idx]
            else:
                pos_str = prompt_lines[-1] if prompt_lines else ""
            positive, _ = self._encode_prompts_clip(clip, pos_str, "")

            _log(f"[Vantage Dual Looper] {pos_str}")

            # Build latent window on CPU, then move to model device/dtype
            positive, negative, win_lat_cpu = build_window_latent(win_T, positive, negative, loop_idx)
            win_lat_gpu = {"samples": _to(win_lat_cpu["samples"], model_device, model_dtype)}
            
            # Sample
            init_steps = 0
            if model_init is not None:
                init_steps = int(steps_init)
            steps_eff = init_steps + int(steps_high) + int(steps_low) 
            steps_i = init_steps
            steps_h = int(steps_high) + steps_i
            steps_l = int(steps_low) + steps_h
            if model_init is not None:
                out_win = self._ksampler_call(
                    model_init, positive, negative, win_lat_gpu, steps_eff, cfg_init, sampler_name, scheduler, denoise, seed64, False, 0, steps_i, False 
                )
                out_win = self._ksampler_call(
                    model_high, positive, negative, out_win, steps_eff, cfg, sampler_name, scheduler, denoise, seed64, True, steps_i, steps_h, False
                )
            else:
                out_win = self._ksampler_call(
                    model_high, positive, negative, win_lat_gpu, steps_eff, cfg, sampler_name, scheduler, denoise, seed64, False, 0, steps_h, False
                )

            out_win = self._ksampler_call(
                model_low, positive, negative, out_win, steps_eff, cfg, sampler_name, scheduler, denoise, seed64, True, steps_h, steps_l, True
            )
            # Bring to CPU for VAE decode
            out_win_cpu = _to(out_win["samples"], cpu, torch.float32)

            # Decode with VAE to images
            decoded = vae.decode(out_win_cpu)  # expected dict with "image"
            img = decoded["image"] if isinstance(decoded, dict) else decoded

            # Normalize to BHWC with batch across time
            # Accept shapes: [B,C,T,H,W] or [B,T,H,W,C] or [B,H,W,C]
            if img.dim() == 5:
                # Try [B,T,H,W,C] first
                if img.shape[-1] in (3, 4):
                    # Collapse B and T -> batch of frames
                    Bv, Tv, H, W, Cv = img.shape
                    img = img.reshape(Bv * Tv, H, W, Cv)
                else:
                    # Assume [B,C,T,H,W] -> move to BHWC per frame then merge
                    img = img.permute(0, 2, 3, 4, 1).contiguous()  # B,T,H,W,C
                    Bv, Tv, H, W, Cv = img.shape
                    img = img.reshape(Bv * Tv, H, W, Cv)
            elif img.dim() == 4:
                # Either NCHW or BHWC
                if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (3, 4):
                    img = img.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
                # else already BHWC
            elif img.dim() == 3:
                # single image HWC -> add batch
                img = img.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected VAE decode shape {tuple(img.shape)}")

            img = img.to(torch.float32).clamp(0, 1)  # (B,H,W,C)
            # Drop the first discard_n frames for all loops after the first, then save remaining starting from index 0
            if loop_idx > start_prompt_idx and discard_n > 0:
                if img.shape[0] > discard_n:
                    img_to_save = img[discard_n:]   # keep remaining frames
                else:
                    img_to_save = img.new_zeros((0, *img.shape[1:]))  # no frames to save
            else:
                img_to_save = img
            
            # Save frames for this loop
            loop_dir = project_dir / f"{loop_idx}"
            _save_loop_frames(loop_dir, img_to_save)
            
            # Cache next loop seed frame according to overlap:
            # pick the Nth frame from the end of the current decoded batch BEFORE discarding for save
            try:
                if img.shape[0] >= pick_n:
                    # index from end: -pick_n; keep as a single-frame BHWC tensor
                    prev_seed_image = img[-pick_n:-pick_n+1] if pick_n > 1 else img[-1:]
                elif img.shape[0] > 0:
                    prev_seed_image = img[-1:]  # fallback to last frame if not enough frames
                else:
                    prev_seed_image = None
            except Exception as e:
                _log(f"[Vantage Dual Looper] warning: could not cache seed frame: {e}")
                prev_seed_image = None

            # Update counters to reflect frames that advance the timeline after discarding
            if loop_idx > start_prompt_idx and discard_n > 0:
                saved_count = max(int(img.shape[0]) - int(discard_n), 0)
            else:
                saved_count = int(img.shape[0])
            produced_frames += saved_count
            loop_idx += 1

            # GPU sync and cleanup per-iteration
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            del out_win
            del win_lat_gpu
            del decoded
            del img
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.memory.empty_cache()

        _log(f"[Vantage Dual Looper] done, produced {produced_frames} frames (target {total_frames}).")

        # Load all frames from disk and return as IMAGE
        all_frames = _load_all_frames(project_dir)  # (F,H,W,C) float [0,1]
        if all_frames.ndim == 3:
            all_frames = all_frames.unsqueeze(0)  # (1,H,W,C)
        return (all_frames,)

