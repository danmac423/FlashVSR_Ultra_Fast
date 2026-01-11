import math
import os
from enum import Enum
from typing import Tuple, List

import torch
import torch.nn.functional as F
from einops import rearrange
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
from torchvision.io import write_png
from tqdm import tqdm

from src import BasePipeline, FlashVSRTinyPipeline, ModelManager
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import Causal_LQ4x_Proj, clean_vram, get_device_list
from dataclasses import dataclass


class OutputMode(Enum):
    """Output mode for processed video."""

    VIDEO = "video"
    FRAMES = "frames"


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""

    scale: int = 4
    color_fix: bool = True
    seed: int = 0
    sparse_ratio: float = 2.0
    kv_ratio: float = 3.0
    local_range: int = 11
    unload_dit: bool = True
    force_offload: bool = True


@dataclass
class SpatialTilingConfig:
    """Configuration for spatial tiling."""

    enabled: bool = True
    tile_size: Tuple[int, int] = (192, 192)
    tile_overlap: int = 24


@dataclass
class TemporalTilingConfig:
    """Configuration for temporal tiling."""

    enabled: bool = True
    tile_size: int = 100
    tile_overlap: int = 4


@dataclass
class IOConfig:
    """Configuration for input/output."""

    input_path: str = "input/video.mp4"
    output_mode: OutputMode = OutputMode.VIDEO
    output_dir: str = "output"


def convert_tensor_to_video(frames: torch.Tensor) -> torch.Tensor:
    """Convert tensor from model output format to video format.

    Args:
        frames (torch.Tensor): Input tensor of shape (C, N, H, W)

    Returns:
        torch.Tensor: Output tensor of shape (N, H, W, C) with values in [0, 1]
    """
    video_permuted = rearrange(frames, "C N H W -> N H W C")
    video_final = (video_permuted.to(torch.float16) + 1.0) / 2.0
    return video_final


def upscale_and_normalize_tensor(
    frame_tensor: torch.Tensor,
    sh: int,
    sw: int,
    th: int,
    tw: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Upscale and normalize a single frame tensor.

    Args:
        frame_tensor (torch.Tensor): Input frame tensor of shape (H, W, C)
        scaled_height (int): Scaled height before padding
        scaled_width (int): Scaled width before padding
        target_height (int): Target height after padding
        target_width (int): Target width after padding

    Returns:
        Tuple: Upscaled and normalized tensor of shape (H, W, C)
        and padding applied (left, right, top, bottom)
    """
    tensor_bchw = rearrange(frame_tensor, "H W C -> 1 C H W")

    upscaled_tensor = F.interpolate(tensor_bchw, size=(sh, sw), mode="bicubic", align_corners=False)

    pad_h = th - sh
    pad_w = tw - sw
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_tensor = F.pad(
        upscaled_tensor, [pad_left, pad_right, pad_top, pad_bottom], mode="reflect"
    )

    normalized_tensor = padded_tensor.clamp(0.0, 1.0).mul(255).round().div(255).mul(2).sub(1.0)

    return rearrange(normalized_tensor, "1 C H W -> H W C"), (
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
    )


def prepare_input_tensor(
    image_tensor: torch.Tensor,
    device: torch.device,
    scale: int = 4,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, int, int, int, int, int, Tuple[int, int, int, int]]:
    """Prepare input tensor for model processing. Upscales, normalizes, and pads frames
    temporally and spatially to meet model requirements.

    Args:
        image_tensor (torch.Tensor): Input tensor of shape (N, H, W, C)
        device (torch.device): Device to load tensors onto
        scale (int): Upscaling factor
        dtype (torch.dtype): Data type for tensors

    Returns:
        Tuple: Prepared tensor and related dimensions
            - torch.Tensor: Prepared tensor of shape (1, C, N, H, W)
            - int: Scaled height before padding
            - int: Scaled width before padding
            - int: Target height after padding
            - int: Target width after padding
            - int: Number of frames after temporal padding
            - Tuple: Padding applied (left, right, top, bottom)
    """
    N0, h0, w0, _ = image_tensor.shape

    multiple = 128
    sh, sw, th, tw = compute_scaled_and_target_dims(h0, w0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    N = calculate_padded_frame_count(num_frames_with_padding)

    if N == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")

    frames = []
    for i in range(N):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor, padding = upscale_and_normalize_tensor(frame_slice, sh, sw, th, tw)
        tensor = tensor.to("cpu").to(dtype)

        frames.append(tensor)
        del frame_slice

    vid_stacked = torch.stack(frames, 0)
    vid_final = rearrange(vid_stacked, "N H W C -> 1 C N H W")

    del vid_stacked
    clean_vram()

    return vid_final, sh, sw, th, tw, N, padding


def remove_padding(
    frames: torch.Tensor, sh: int, sw: int, th: int, tw: int, padding: Tuple[int, int, int, int]
) -> torch.Tensor:
    """Remove padding from processed frames.

    Args:
        frames (torch.Tensor): Processed frames tensor of shape (N, H, W, C)
        sh (int): Scaled height before padding
        sw (int): Scaled width before padding
        th (int): Target height after padding
        tw (int): Target width after padding
        padding (Tuple[int, int, int, int]): Padding applied (left, right, top, bottom)

    Returns:
        torch.Tensor: Frames tensor with padding removed of shape (N, sh, sw, C)
    """
    pad_left, pad_right, pad_top, pad_bottom = padding

    if th > sh:
        frames = frames[:, pad_top : th - pad_bottom, :, :]
    if tw > sw:
        frames = frames[:, :, pad_left : tw - pad_right, :]

    return frames


def calculate_padded_frame_count(n: int) -> int:
    """Calculate largest frame count in format 8n+1 that is <= n.

    Args:
        n (int): Original frame count

    Returns:
        int: Padded frame count
    """
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def calculate_next_frame_requirement(n: int) -> int:
    """Calculate next frame count in format 8n+5.

    Args:
        n (int): Original frame count

    Returns:
        int: Next frame count
    """
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5


def compute_scaled_and_target_dims(
    h0: int, w0: int, scale: int = 4, multiple: int = 128
) -> Tuple[int, int, int, int]:
    """Compute scaled dimensions and target dimensions aligned to multiple.

    Args:
        h0 (int): Original height
        w0 (int): Original width
        scale (int): Upscaling factor
        multiple (int): Multiple to align target dimensions

    Returns:
        Tuple: scaled_height, scaled_width, target_height, target_width
    """
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sw, sh = w0 * scale, h0 * scale
    th = ((sh + multiple - 1) // multiple) * multiple
    tw = ((sw + multiple - 1) // multiple) * multiple
    return sh, sw, th, tw


def calculate_spatial_tile_coords(
    height: int, width: int, tile_size: Tuple[int, int], overlap: int
) -> List[Tuple[int, int, int, int]]:
    """Calculate spatial tile coordinates with overlap for tiled processing.

    Args:
        height (int): Height of the frame
        width (int): Width of the frame
        tile_size (Tuple[int, int]): (tile_width, tile_height)
        overlap (int): Overlap size

    Returns:
        List: List of tile coordinates (x1, y1, x2, y2)
    """
    coords = []
    tile_w, tile_h = tile_size

    stride_w = tile_w - overlap
    stride_h = tile_h - overlap

    num_rows = math.ceil((height - overlap) / stride_h)
    num_cols = math.ceil((width - overlap) / stride_w)

    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride_h
            x1 = c * stride_w

            y2 = min(y1 + tile_h, height)
            x2 = min(x1 + tile_w, width)

            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)
            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)

            coords.append((x1, y1, x2, y2))

    return coords


def calculate_temporal_tile_ranges(
    total_frames: int, chunk_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """Calculate temporal chunk ranges with overlap for tiled processing.

    Args:
        total_frames (int): Total number of frames in the video
        chunk_size (int): Size of each temporal chunk
        overlap (int): Overlap size between chunks

    Returns:
        List: List of temporal chunk ranges (start_frame, end_frame)
    """
    chunks = []
    stride = chunk_size - overlap

    num_chunks = math.ceil((total_frames - overlap) / stride)

    for i in range(num_chunks):
        start = i * stride
        end = min(start + chunk_size, total_frames)
        chunks.append((start, end))

    return chunks


def create_spatial_blend_mask(size: Tuple[int, int], overlap: int) -> torch.Tensor:
    """Create feather mask for blending spatial tiles.

    Args:
        size (Tuple[int, int]): Size of the mask (height, width)
        overlap (int): Overlap size for blending

    Returns:
        torch.Tensor: Blend mask of shape (1, H, W, 1)
    """
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

    return rearrange(mask, "N C H W -> N H W C")


def blend_temporal_overlap(
    prev_chunk_tail: torch.Tensor, current_chunk_head: torch.Tensor, overlap: int
) -> torch.Tensor:
    """Blend overlapping region of two temporal chunks.

    Args:
        prev_chunk_tail (torch.Tensor): Last 'overlap' frames from previous chunk
        current_chunk_head (torch.Tensor): First 'overlap' frames from current chunk
        overlap (int): Number of overlapping frames

    Returns:
        torch.Tensor: Blended frames
    """
    if overlap <= 0 or prev_chunk_tail is None:
        return current_chunk_head

    actual_overlap = min(overlap, prev_chunk_tail.shape[0], current_chunk_head.shape[0])

    if actual_overlap <= 0:
        return current_chunk_head

    blend_weight = torch.linspace(1, 0, actual_overlap).view(-1, 1, 1, 1)

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
    pipe: BasePipeline,
    frames_chunk: torch.Tensor,
    proc_config: ProcessingConfig,
    spatial_config: SpatialTilingConfig,
) -> torch.Tensor:
    """Process a single temporal chunk of frames.

    Args:
        pipe (BasePipeline): FlashVSR pipeline instance
        frames_chunk (torch.Tensor): Input frames tensor of shape (N, H, W, C)
        proc_config (ProcessingConfig): Processing configuration
        spatial_config (SpatialTilingConfig): Spatial tiling configuration

    Returns:
        torch.Tensor: Processed frames tensor of shape (N, H, W, C)
    """
    _device = pipe.device
    dtype = pipe.torch_dtype

    add = calculate_next_frame_requirement(frames_chunk.shape[0]) - frames_chunk.shape[0]
    if add > 0:
        padding_frames = frames_chunk[-1:, :, :, :].repeat(add, 1, 1, 1)
        _frames = torch.cat([frames_chunk, padding_frames], dim=0)
    else:
        _frames = frames_chunk

    if spatial_config.enabled:
        N, H, W, C = _frames.shape

        final_output_canvas = torch.zeros(
            (N, H * proc_config.scale, W * proc_config.scale, C), dtype=torch.float16, device="cpu"
        )
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_spatial_tile_coords(
            H, W, spatial_config.tile_size, spatial_config.tile_overlap
        )

        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            input_tile = _frames[:, y1:y2, x1:x2, :]

            LQ_tile, sh, sw, th, tw, N, padding = prepare_input_tensor(
                input_tile, _device, scale=proc_config.scale, dtype=dtype
            )
            LQ_tile = LQ_tile.to(_device)

            output_tile_gpu = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=proc_config.seed,
                LQ_video=LQ_tile,
                num_frames=N,
                height=th,
                width=tw,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=proc_config.sparse_ratio * 768 * 1280 / (th * tw),
                kv_ratio=proc_config.kv_ratio,
                local_range=proc_config.local_range,
                color_fix=proc_config.color_fix,
                unload_dit=proc_config.unload_dit,
                force_offload=proc_config.force_offload,
            )

            processed_tile_cpu = convert_tensor_to_video(output_tile_gpu).to("cpu")

            processed_tile_cpu = remove_padding(processed_tile_cpu, sh, sw, th, tw, padding)

            mask = create_spatial_blend_mask(
                (sh, sw), spatial_config.tile_overlap * proc_config.scale
            ).to("cpu")

            out_x1, out_y1 = x1 * proc_config.scale, y1 * proc_config.scale
            out_x2, out_y2 = out_x1 + sw, out_y1 + sh

            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask

            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()

        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        LQ, sh, sw, th, tw, N, padding = prepare_input_tensor(
            _frames, _device, scale=proc_config.scale, dtype=dtype
        )
        LQ = LQ.to(_device)

        video = pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=proc_config.seed,
            LQ_video=LQ,
            num_frames=N,
            height=th,
            width=tw,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=proc_config.sparse_ratio * 768 * 1280 / (th * tw),
            kv_ratio=proc_config.kv_ratio,
            local_range=proc_config.local_range,
            color_fix=proc_config.color_fix,
            unload_dit=proc_config.unload_dit,
            force_offload=proc_config.force_offload,
        )

        final_output = convert_tensor_to_video(video).to("cpu")

        final_output = remove_padding(final_output, sh, sw, th, tw, padding)

        del video, LQ
        clean_vram()

    return final_output[: frames_chunk.shape[0], :, :, :]


def flashvsr(
    pipe: BasePipeline,
    io_config: IOConfig,
    proc_config: ProcessingConfig,
    spatial_tiling_config: SpatialTilingConfig,
    temporal_tiling_config: TemporalTilingConfig,
):
    """Process video using FlashVSR with temporal tiling support.
    Writes output incrementally to avoid RAM overflow.

    Args:
        pipe (BasePipeline): FlashVSR pipeline instance
        io_config (IOConfig): Input/output configuration
        proc_config (ProcessingConfig): Processing configuration
        spatial_tiling_config (SpatialTilingConfig): Spatial tiling configuration
        temporal_tiling_config (TemporalTilingConfig): Temporal tiling configuration
    """
    video_decoder = VideoDecoder(io_config.input_path, dimension_order="NHWC")

    if (
        temporal_tiling_config.enabled
        and video_decoder.metadata.num_frames > temporal_tiling_config.tile_size
    ):
        chunks = calculate_temporal_tile_ranges(
            video_decoder.metadata.num_frames,
            temporal_tiling_config.tile_size,
            temporal_tiling_config.tile_overlap,
        )

        prev_chunk_tail = None

        frame_counter = 0

        for chunk_idx, (start, end) in enumerate(tqdm(chunks, desc="Processing Temporal Chunks")):
            frames_chunk = (
                video_decoder.get_frames_in_range(start, end).data.to(torch.float16) / 255.0
            )
            current_chunk_output = process_single_temporal_chunk(
                pipe=pipe,
                frames_chunk=frames_chunk,
                proc_config=proc_config,
                spatial_config=spatial_tiling_config,
            )
            del frames_chunk
            clean_vram()

            if chunk_idx == 0:
                frames_to_write = (
                    current_chunk_output[: -temporal_tiling_config.tile_overlap]
                    if temporal_tiling_config.tile_overlap > 0
                    else current_chunk_output
                )
                prev_chunk_tail = (
                    current_chunk_output[-temporal_tiling_config.tile_overlap :]
                    if temporal_tiling_config.tile_overlap > 0
                    else None
                )
            else:
                overlap_head = current_chunk_output[: temporal_tiling_config.tile_overlap]
                blended_overlap = blend_temporal_overlap(
                    prev_chunk_tail, overlap_head, temporal_tiling_config.tile_overlap
                )

                rest_of_chunk = current_chunk_output[temporal_tiling_config.tile_overlap :]
                if chunk_idx == len(chunks) - 1:
                    frames_to_write = torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    prev_chunk_tail = None
                else:
                    frames_to_write = (
                        torch.cat(
                            [
                                blended_overlap,
                                rest_of_chunk[: -temporal_tiling_config.tile_overlap],
                            ],
                            dim=0,
                        )
                        if temporal_tiling_config.tile_overlap > 0
                        else torch.cat([blended_overlap, rest_of_chunk], dim=0)
                    )
                    prev_chunk_tail = (
                        rest_of_chunk[-temporal_tiling_config.tile_overlap :]
                        if temporal_tiling_config.tile_overlap > 0
                        and rest_of_chunk.shape[0] >= temporal_tiling_config.tile_overlap
                        else rest_of_chunk
                    )

            if io_config.output_mode == OutputMode.VIDEO:
                encoder = VideoEncoder(
                    frames=rearrange(frames_to_write, "N H W C -> N C H W")
                    .clamp(0, 1)
                    .mul_(255)
                    .to(torch.uint8),
                    frame_rate=video_decoder.metadata.average_fps,
                )
                encoder.to_file(
                    os.path.join(io_config.output_dir, f"chunk_{chunk_idx:03d}.mp4"),
                    codec="libx264",
                    crf=0,
                    pixel_format="yuv420p",
                )
            elif io_config.output_mode == OutputMode.FRAMES:
                final_output = (
                    rearrange(frames_to_write, "N H W C -> N C H W")
                    .clamp(0, 1)
                    .mul_(255)
                    .to(torch.uint8)
                )
                N = final_output.shape[0]
                for i in range(N):
                    frame = final_output[i]
                    write_png(
                        frame,
                        os.path.join(io_config.output_dir, f"{frame_counter:08d}.png"),
                        compression_level=0,
                    )
                    frame_counter += 1

            del current_chunk_output, frames_to_write
            clean_vram()

    else:
        frames = (
            video_decoder.get_frames_in_range(0, video_decoder.metadata.num_frames).data.to(
                torch.float16
            )
            / 255.0
        )

        final_output = process_single_temporal_chunk(
            pipe=pipe,
            frames_chunk=frames,
            proc_config=proc_config,
            spatial_config=spatial_tiling_config,
        )

        if io_config.output_mode == OutputMode.VIDEO:
            encoder = VideoEncoder(
                frames=rearrange(final_output, "N H W C -> N C H W")
                .clamp(0, 1)
                .mul_(255)
                .to(torch.uint8),
                frame_rate=video_decoder.metadata.average_fps,
            )
            encoder.to_file(
                os.path.join(io_config.output_dir, "chunk_000.mp4"),
                codec="libx264",
                crf=0,
                pixel_format="yuv420p",
            )
        elif io_config.output_mode == OutputMode.FRAMES:
            final_output = (
                rearrange(final_output, "N H W C -> N C H W").clamp(0, 1).mul_(255).to(torch.uint8)
            )
            N = final_output.shape[0]
            for i in range(N):
                frame = final_output[i]
                write_png(
                    frame,
                    os.path.join(io_config.output_dir, f"{i:08d}.png"),
                    compression_level=0,
                )

        del frames, final_output
        clean_vram()


def main():
    model = "FlashVSR-v1.1"

    _device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "auto"
    )
    if _device == "auto" or _device not in get_device_list():
        raise RuntimeError("No devices found to run FlashVSR!")

    input_path = "inputs/example3.mp4"

    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = f"{video_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    proc_config = ProcessingConfig(
        scale=4,
        seed=0,
        sparse_ratio=2.0,
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
        unload_dit=True,
        force_offload=True,
    )
    spatial_tiling_config = SpatialTilingConfig(
        enabled=True,
        tile_size=(192, 192),
        tile_overlap=24,
    )
    temporal_tiling_config = TemporalTilingConfig(
        enabled=True,
        tile_size=100,
        tile_overlap=6,
    )
    io_config = IOConfig(
        input_path=input_path,
        output_mode=OutputMode.VIDEO,
        output_dir=output_dir,
    )

    pipe = init_pipeline(model, _device, torch.float16)

    flashvsr(
        pipe=pipe,
        io_config=io_config,
        proc_config=proc_config,
        spatial_tiling_config=spatial_tiling_config,
        temporal_tiling_config=temporal_tiling_config,
    )


if __name__ == "__main__":
    main()
