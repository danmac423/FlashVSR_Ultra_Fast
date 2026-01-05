#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
from tqdm import tqdm

from src import FlashVSRTinyPipeline, ModelManager
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import Causal_LQ4x_Proj, clean_vram, get_device_list

device_choices = get_device_list()


def log(message: str, message_type: str = "normal"):
    if message_type == "error":
        message = "\033[1;41m" + message + "\033[m"
    elif message_type == "warning":
        message = "\033[1;31m" + message + "\033[m"
    elif message_type == "finish":
        message = "\033[1;32m" + message + "\033[m"
    elif message_type == "info":
        message = "\033[1;33m" + message + "\033[m"
    else:
        message = message
    print(f"{message}")


def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final


def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5


def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    # Round UP to next multiple to avoid cropping
    tW = ((sW + multiple - 1) // multiple) * multiple
    tH = ((sH + multiple - 1) // multiple) * multiple
    return sW, sH, tW, tH


def tensor_upscale_then_pad(
    frame_tensor: torch.Tensor, scale: int, tW: int, tH: int
) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> BCHW

    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode="bicubic", align_corners=False)

    # Pad instead of crop
    pad_w = tW - sW
    pad_h = tH - sH

    # Symmetric padding (split difference on both sides)
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # F.pad uses (left, right, top, bottom) for last 2 dimensions
    padded_tensor = F.pad(
        upscaled_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
    )

    return padded_tensor.squeeze(0)


def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape

    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)

    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")

    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = (
            tensor_upscale_then_pad(frame_slice, scale=scale, tW=tW, tH=tH).to("cpu").to(dtype)
            * 2.0
            - 1.0
        )
        frames.append(tensor_chw)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)

    del vid_stacked
    clean_vram()

    return vid_final, tH, tW, F


def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []

    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)

    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride

            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)

            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)

            coords.append((x1, y1, x2, y2))

    return coords


def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)

    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(
        mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1)
    )

    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(
        mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1)
    )

    return mask


def init_pipeline(model: str, device: torch.device, dtype: torch.dtype) -> FlashVSRTinyPipeline:
    """Initialize FlashVSR pipeline with given model and device.

    Args:
        model (str): Model name.
        device (torch.device): Device to load the model on.
        dtype (torch.dtype): Data type for model weights.

    Returns: FlashVSRTinyPipeline: Initialized pipeline instance.

    Raises:
        RuntimeError: If model directory or required files do not exist.
    """
    model_path = os.path.join("models", model)
    if not os.path.exists(model_path):
        raise RuntimeError(
            f'Model directory does not exist!\nPlease save all weights to "{model_path}"'
        )

    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"'
        )

    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')

    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    if not os.path.exists(prompt_path):
        raise RuntimeError(f'"posi_prompt.pth" does not exist!\nPlease save it to "{model_path}"')

    mm = ModelManager(torch_dtype=dtype, device="cpu")

    mm.load_models([ckpt_path])

    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)

    multi_scale_channels = [512, 256, 128, 128]
    pipe.TCDecoder = build_tcdecoder(
        new_channels=multi_scale_channels,
        device=device,
        dtype=dtype,
        new_latent_channels=16 + 768,
    )
    pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
    pipe.TCDecoder.clean_mem()

    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(
        device, dtype=dtype
    )
    pipe.denoising_model().LQ_proj_in.load_state_dict(
        torch.load(lq_path, map_location="cpu"), strict=True
    )
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit"])
    pipe.offload_model()

    return pipe


def flashvsr(
    pipe,
    frames,
    scale,
    color_fix,
    tiled_vae,
    tiled_dit,
    tile_size,
    tile_overlap,
    unload_dit,
    sparse_ratio,
    kv_ratio,
    local_range,
    seed,
    force_offload,
):
    _frames = frames
    _device = pipe.device
    dtype = pipe.torch_dtype

    add = next_8n5(frames.shape[0]) - frames.shape[0]
    padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
    _frames = torch.cat([frames, padding_frames], dim=0)

    if tiled_dit:
        N, H, W, C = _frames.shape

        final_output_canvas = torch.zeros(
            (N, H * scale, W * scale, C), dtype=torch.float16, device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        latent_tiles_cpu = []

        for i, (x1, y1, x2, y2) in enumerate(tqdm(tile_coords, desc="Processing Tiles")):
            log(
                f"[FlashVSR] Processing tile {i + 1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})",
                message_type="info",
            )

            input_tile = _frames[:, y1:y2, x1:x2, :]
            input_tile_h = y2 - y1
            input_tile_w = x2 - x1

            LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
            LQ_tile = LQ_tile.to(_device)

            output_tile_gpu = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=seed,
                tiled=tiled_vae,
                LQ_video=LQ_tile,
                num_frames=F,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
                kv_ratio=kv_ratio,
                local_range=local_range,
                color_fix=color_fix,
                unload_dit=unload_dit,
                force_offload=force_offload,
            )

            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")

            exact_tile_h = input_tile_h * scale
            exact_tile_w = input_tile_w * scale

            pad_h = th - exact_tile_h
            pad_w = tw - exact_tile_w

            crop_top = pad_h // 2
            crop_bottom = crop_top + exact_tile_h
            crop_left = pad_w // 2
            crop_right = crop_left + exact_tile_w
            processed_tile_cpu = processed_tile_cpu[
                :, crop_top:crop_bottom, crop_left:crop_right, :
            ]
            mask_nchw = create_feather_mask(
                (exact_tile_h, exact_tile_w), tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)

            out_x1, out_y1 = x1 * scale, y1 * scale
            out_x2, out_y2 = out_x1 + exact_tile_w, out_y1 + exact_tile_h
            
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += (
                processed_tile_cpu * mask_nhwc
            )
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc

            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log("[FlashVSR] Preparing frames...")
        LQ, th, tw, F = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        LQ = LQ.to(_device)
        log(f"[FlashVSR] Processing {frames.shape[0]} frames...", message_type="info")

        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=seed,
            tiled=tiled_vae,
            progress_bar_cmd=tqdm,
            LQ_video=LQ,
            num_frames=F,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=kv_ratio,
            local_range=local_range,
            color_fix=color_fix,
            unload_dit=unload_dit,
            force_offload=force_offload,
        )

        final_output = tensor2video(video).to("cpu")

        del video, LQ
        clean_vram()

    log("[FlashVSR] Done.", message_type="info")
    if frames.shape[0] == 1:
        final_output = final_output.to(_device)
        stacked_image_tensor = (
            torch.median(final_output, dim=0).values.unsqueeze(0).float().to("cpu")
        )
        del final_output
        clean_vram()
        return stacked_image_tensor

    return final_output[: frames.shape[0], :, :, :]


def main():
    model = "FlashVSR-v1.1"
    scale = 4
    tiled_vae = True
    tiled_dit = True
    unload_dit = True
    seed = 0

    video_path = "../videolq_videos/000.mp4"

    decoder = VideoDecoder(video_path, dimension_order="NHWC")
    frames = decoder[:].float() / 255.0

    _device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "auto"
    )
    if _device == "auto" or _device not in device_choices:
        raise RuntimeError("No devices found to run FlashVSR!")

    pipe = init_pipeline(model, _device, torch.float16)
    output = flashvsr(
        pipe=pipe,
        frames=frames,
        scale=scale,
        color_fix=True,
        tiled_vae=tiled_vae,
        tiled_dit=tiled_dit,
        tile_size=128 + 32 + 32,
        tile_overlap=24,
        unload_dit=unload_dit,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=11,
        seed=seed,
        force_offload=True,
    )

    encoder = VideoEncoder(output.permute(0, 3, 1, 2).mul(255).byte(), frame_rate=25)
    encoder.to_file(
        f"{os.path.splitext(os.path.basename(video_path))[0]}_out.mp4",
        codec="libx264",
        crf=0,
        pixel_format="yuv420p",
    )


if __name__ == "__main__":
    main()
