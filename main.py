#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
import os
from enum import Enum

import torch
import torch.nn.functional as F
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
from torchvision.io import write_png
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


def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.float16):
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


def calculate_temporal_chunks(total_frames, chunk_size, overlap):
    """Calculate temporal chunk ranges with overlap."""
    chunks = []
    stride = chunk_size - overlap

    num_chunks = math.ceil((total_frames - overlap) / stride)

    for i in range(num_chunks):
        start = i * stride
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))

    return chunks


def blend_overlap_region(prev_chunk_tail, current_chunk_head, overlap):
    """
    Blend only the overlapping region of two chunks.

    Args:
        prev_chunk_tail: Last 'overlap' frames from previous chunk
        current_chunk_head: First 'overlap' frames from current chunk
        overlap: Number of overlapping frames

    Returns:
        Blended frames (size: overlap)
    """
    if overlap <= 0 or prev_chunk_tail is None:
        return current_chunk_head

    actual_overlap = min(overlap, prev_chunk_tail.shape[0], current_chunk_head.shape[0])

    if actual_overlap <= 0:
        return current_chunk_head

    # Create blend weights: 1 -> 0 (favor previous chunk at start, current chunk at end)
    blend_weight = torch.linspace(1, 0, actual_overlap).view(-1, 1, 1, 1)

    # Blend
    blended = prev_chunk_tail[-actual_overlap:] * blend_weight + current_chunk_head[
        :actual_overlap
    ] * (1 - blend_weight)

    return blended


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


def process_single_temporal_chunk(
    pipe,
    frames_chunk,
    scale,
    color_fix,
    spatial_tiling,
    spatial_tile_size,
    spatial_tile_overlap,
    unload_dit,
    sparse_ratio,
    kv_ratio,
    local_range,
    seed,
    force_offload,
):
    """Process a single temporal chunk of frames."""
    _device = pipe.device
    dtype = pipe.torch_dtype

    # Add padding frames for model requirements
    add = next_8n5(frames_chunk.shape[0]) - frames_chunk.shape[0]
    if add > 0:
        padding_frames = frames_chunk[-1:, :, :, :].repeat(add, 1, 1, 1)
        _frames = torch.cat([frames_chunk, padding_frames], dim=0)
    else:
        _frames = frames_chunk

    if spatial_tiling:
        N, H, W, C = _frames.shape

        final_output_canvas = torch.zeros(
            (N, H * scale, W * scale, C), dtype=torch.float16, device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, spatial_tile_size, spatial_tile_overlap)

        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
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
                (exact_tile_h, exact_tile_w), spatial_tile_overlap * scale
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
        LQ, th, tw, F = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
        LQ = LQ.to(_device)

        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=seed,
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

    # Remove padding frames
    return final_output[: frames_chunk.shape[0], :, :, :]


class OutputMode(Enum):
    VIDEO = 1
    FRAMES = 2


def flashvsr(
    pipe,
    video_decoder: VideoDecoder,
    total_frames,
    output_dir,
    frame_rate,
    scale,
    output_mode: OutputMode,
    color_fix,
    spatial_tiling,
    spatial_tile_size,
    spatial_tile_overlap,
    temporal_tiling,
    temporal_tile_size,
    temporal_tile_overlap,
    unload_dit,
    sparse_ratio,
    kv_ratio,
    local_range,
    seed,
    force_offload,
):
    """
    Process video using FlashVSR with temporal tiling support.
    Writes output incrementally to avoid RAM overflow.

    Args:
        pipe: FlashVSR pipeline
        video_decoder: VideoDecoder instance for loading frames
        total_frames: Total number of frames in the video
        output_path: Path to save output video
        frame_rate: Output video frame rate
        ... (other parameters)
    """

    # Temporal tiling mode - process chunks and write incrementally
    if temporal_tiling and total_frames > temporal_tile_size:
        log(
            f"[FlashVSR] Using temporal tiling: {total_frames} frames, chunk size: {temporal_tile_size}, overlap: {temporal_tile_overlap}",
            message_type="info",
        )

        chunks = calculate_temporal_chunks(total_frames, temporal_tile_size, temporal_tile_overlap)
        log(f"[FlashVSR] Created {len(chunks)} temporal chunks", message_type="info")

        prev_chunk_tail = None  # Store only the overlapping tail from previous chunk

        frame_counter = 0

        for chunk_idx, (start, end) in enumerate(tqdm(chunks, desc="Processing Temporal Chunks")):
            log(
                f"[FlashVSR] Loading and processing chunk {chunk_idx + 1}/{len(chunks)}: frames {start}-{end}",
                message_type="info",
            )

            # Load and process current chunk
            chunk_frames = (
                video_decoder.get_frames_in_range(start, end).data.to(torch.float16) / 255.0
            )
            current_chunk_output = process_single_temporal_chunk(
                pipe,
                chunk_frames,
                scale,
                color_fix,
                spatial_tiling,
                spatial_tile_size,
                spatial_tile_overlap,
                unload_dit,
                sparse_ratio,
                kv_ratio,
                local_range,
                seed,
                force_offload,
            )
            del chunk_frames
            clean_vram()

            # Determine what to write
            if chunk_idx == 0:
                # First chunk: write everything except the tail (overlap region)
                frames_to_write = (
                    current_chunk_output[:-temporal_tile_overlap]
                    if temporal_tile_overlap > 0
                    else current_chunk_output
                )
                prev_chunk_tail = (
                    current_chunk_output[-temporal_tile_overlap:]
                    if temporal_tile_overlap > 0
                    else None
                )
            else:
                # Subsequent chunks: blend overlap, then write
                overlap_head = current_chunk_output[:temporal_tile_overlap]
                blended_overlap = blend_overlap_region(
                    prev_chunk_tail, overlap_head, temporal_tile_overlap
                )

                # Write blended overlap + rest of current chunk (except new tail)
                rest_of_chunk = current_chunk_output[temporal_tile_overlap:]

                if chunk_idx == len(chunks) - 1:
                    # Last chunk: write everything including tail
                    frames_to_write = torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    prev_chunk_tail = None
                else:
                    # Not last chunk: save tail for next iteration
                    frames_to_write = (
                        torch.cat([blended_overlap, rest_of_chunk[:-temporal_tile_overlap]], dim=0)
                        if temporal_tile_overlap > 0
                        else torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    )
                    prev_chunk_tail = (
                        rest_of_chunk[-temporal_tile_overlap:]
                        if temporal_tile_overlap > 0
                        and rest_of_chunk.shape[0] >= temporal_tile_overlap
                        else rest_of_chunk
                    )

            if output_mode == OutputMode.VIDEO:
                encoder = VideoEncoder(
                    frames=frames_to_write.permute(0, 3, 1, 2)
                    .clamp(0, 1)
                    .mul_(255)
                    .to(torch.uint8),
                    frame_rate=frame_rate,
                )
                encoder.to_file(
                    os.path.join(output_dir, f"chunk_{chunk_idx:03d}.mp4"),
                    codec="libx264",
                    crf=0,
                    pixel_format="yuv420p",
                )
            elif output_mode == OutputMode.FRAMES:
                final_output = (
                    frames_to_write.permute(0, 3, 1, 2).clamp(0, 1).mul_(255).to(torch.uint8)
                )
                N = final_output.shape[0]
                for i in range(N):
                    frame = final_output[i]
                    write_png(
                        frame,
                        os.path.join(output_dir, f"{frame_counter:08d}.png"),
                        compression_level=0,
                    )
                    frame_counter += 1

            log(
                f"[FlashVSR] Saved chunk {chunk_idx + 1} with {frames_to_write.shape[0]} frames",
                message_type="info",
            )

            del current_chunk_output, frames_to_write
            clean_vram()

    else:
        # Process all frames at once (original behavior)
        log(
            f"[FlashVSR] Loading and processing all {total_frames} frames at once...",
            message_type="info",
        )

        frames = video_decoder.get_frames_in_range(0, total_frames).data.to(torch.float16) / 255.0

        add = next_8n5(total_frames) - total_frames
        if add > 0:
            padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
            _frames = torch.cat([frames, padding_frames], dim=0)
        else:
            _frames = frames

        final_output = process_single_temporal_chunk(
            pipe,
            _frames,
            scale,
            color_fix,
            spatial_tiling,
            spatial_tile_size,
            spatial_tile_overlap,
            unload_dit,
            sparse_ratio,
            kv_ratio,
            local_range,
            seed,
            force_offload,
        )

        if output_mode == OutputMode.VIDEO:
            encoder = VideoEncoder(
                frames=final_output.permute(0, 3, 1, 2).clamp(0, 1).mul_(255).to(torch.uint8),
                frame_rate=frame_rate,
            )
            encoder.to_file(
                os.path.join(output_dir, "chunk_000.mp4"),
                codec="libx264",
                crf=0,
                pixel_format="yuv420p",
            )
        elif output_mode == OutputMode.FRAMES:
            final_output = final_output.permute(0, 3, 1, 2).clamp(0, 1).mul_(255).to(torch.uint8)
            N = final_output.shape[0]
            for i in range(N):
                frame = final_output[i]
                write_png(frame, os.path.join(output_dir, f"{i:08d}.png"), compression_level=0)

        log("[FlashVSR] Saved final output video.", message_type="info")

        del frames, _frames, final_output
        clean_vram()

    log("[FlashVSR] Done.", message_type="info")


def main():
    model = "FlashVSR-v1.1"
    scale = 4
    spatial_tiling = True
    temporal_tiling = True
    unload_dit = True
    seed = 0

    spatial_tile_size = 128 + 32 + 32
    spatial_tile_overlap = 24

    temporal_tile_size = 100
    temporal_tile_overlap = 4

    # video_path = "inputs/example0.mp4"
    video_path = "../train_videos/000.mp4"

    decoder = VideoDecoder(video_path, dimension_order="NHWC")
    total_frames = len(decoder)

    log(f"[FlashVSR] Video loaded: {total_frames} frames", message_type="info")

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

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{video_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    flashvsr(
        pipe=pipe,
        video_decoder=decoder,
        total_frames=total_frames,
        output_dir=output_dir,
        frame_rate=30,
        scale=scale,
        output_mode=OutputMode.FRAMES,
        color_fix=True,
        spatial_tiling=spatial_tiling,
        spatial_tile_size=spatial_tile_size,
        spatial_tile_overlap=spatial_tile_overlap,
        temporal_tiling=temporal_tiling,
        temporal_tile_size=temporal_tile_size,
        temporal_tile_overlap=temporal_tile_overlap,
        unload_dit=unload_dit,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=11,
        seed=seed,
        force_offload=True,
    )


if __name__ == "__main__":
    main()
