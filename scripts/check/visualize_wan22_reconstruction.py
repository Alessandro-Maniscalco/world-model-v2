r"""Render a side-by-side reconstruction video with the official Wan 2.2 VAE.

Example:
    . .\.venv\Scripts\Activate.ps1
    python scripts/check/visualize_wan22_reconstruction.py `
      --data-root data/so101_base_sim_pickplace_cache `
      --dataset-format lerobot_so101_base_sim_pickplace `
      --split train `
      --episode 0 `
      --height 208 `
      --width 272 `
      --output-dir outputs/wan22_vae_reconstruction_ep0_208x272
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
from huggingface_hub import hf_hub_download
from einops import rearrange
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.dataset import ValidationClipDataset, load_clip
from world_model_v2.lerobot_video_dataset import (
    LeRobotEpisodeVideoRepository,
    LeRobotVideoValidationClipDataset,
    SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    crop_bgr_frame,
    resolve_default_lerobot_crop_bounds,
)
from world_model_v2.maniskill_dataset import (
    MANISKILL_DEFAULT_CAMERA,
    MANISKILL_DEFAULT_TRAJ_H5,
    MANISKILL_DEFAULT_TRAJ_JSON,
    ManiSkillValidationClipDataset,
)
from world_model_v2.metaworld_dataset import (
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
    ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_IMAGE_COLUMN,
    METAWORLD_DATASET_ID,
    MetaWorldValidationClipDataset,
    AlohaValidationClipDataset,
)
from world_model_v2.utils.checkpointing import save_json
from world_model_v2.utils.visualization import build_side_by_side_grid, write_side_by_side_mp4
from world_model_v2.wan_vae import (
    WanPosteriorEncoder,
    WanVAEConfig,
    WanVideoDecoder,
    dreamdojo_wan_state_dict,
    remap_dreamdojo_wan_state_dict,
)


WAN22_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_FILENAME = "Wan2.2_VAE.pth"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Wan 2.2 reconstruction export."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vae-path",
        default="",
        help="Optional local path to Wan2.2_VAE.pth. Downloads from Hugging Face when omitted.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for inference.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Model and activation dtype.",
    )
    parser.add_argument(
        "--dataset-format",
        default="lerobot_so101_base_sim_pickplace",
        choices=(
            "interactive_world_sim",
            "lerobot_so101_base_sim_pickplace",
            "lerobot_metaworld",
            "lerobot_aloha_sim_transfer_cube_scripted",
            "maniskill_replay",
        ),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task", default="single_grasp")
    parser.add_argument("--split", default="train")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default="camera_1_color")
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=208)
    parser.add_argument("--height", type=int, default=208)
    parser.add_argument("--width", type=int, default=272)
    parser.add_argument("--chunk-latent-frames", type=int, default=4)
    parser.add_argument("--max-grid-frames", type=int, default=24)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--no-squish-crop",
        action="store_true",
        help=(
            "For SO101 LeRobot clips, crop away unused sides/bottom, trim that crop to a multiple of 32, "
            "then exact-2x area-downsample into the requested size."
        ),
    )
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    """Fail early when CUDA is requested but unavailable."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available to PyTorch. "
            "Rerun with --device cpu or install a compatible torch wheel."
        )


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    """Resolve the requested inference dtype for the active device."""

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[dtype_name]
    if device.type == "cpu" and dtype is not torch.float32:
        raise ValueError("CPU inference only supports --dtype float32.")
    return dtype


def download_or_resolve_vae_path(vae_path: str | Path) -> Path:
    """Return a local filesystem path to the official Wan 2.2 VAE weights."""

    if str(vae_path):
        resolved = Path(vae_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Wan 2.2 VAE path not found: {resolved}")
        return resolved
    downloaded = hf_hub_download(repo_id=WAN22_REPO_ID, filename=WAN22_FILENAME, repo_type="model")
    return Path(downloaded)


def load_validation_clip_from_args(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    """Load one validation clip using the requested dataset family."""

    cache_dir = args.cache_dir or None
    if args.dataset_format == "lerobot_so101_base_sim_pickplace":
        if args.no_squish_crop:
            clip = maybe_load_lerobot_clip_with_aligned_half_downsample(args, cache_dir=cache_dir)
            if clip is not None:
                return clip
        dataset = LeRobotVideoValidationClipDataset(
            data_root=args.data_root,
            split=args.split,
            episode=int(args.episode),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            resolution=int(args.resolution),
            height=int(args.height),
            width=int(args.width),
            repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
            image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
            cache_dir=cache_dir,
        )
        return dataset[0]
    if args.dataset_format == "lerobot_metaworld":
        dataset = MetaWorldValidationClipDataset(
            data_root=args.data_root,
            split=args.split,
            episode=int(args.episode),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            resolution=int(args.resolution),
            height=int(args.height),
            width=int(args.width),
            repo_id=METAWORLD_DATASET_ID,
            cache_dir=cache_dir,
        )
        return dataset[0]
    if args.dataset_format == "lerobot_aloha_sim_transfer_cube_scripted":
        dataset = AlohaValidationClipDataset(
            data_root=args.data_root,
            split=args.split,
            episode=int(args.episode),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            resolution=int(args.resolution),
            height=int(args.height),
            width=int(args.width),
            repo_id=ALOHA_SIM_TRANSFER_CUBE_SCRIPTED_DATASET_ID,
            cache_dir=cache_dir,
        )
        return dataset[0]
    if args.dataset_format == "maniskill_replay":
        dataset = ManiSkillValidationClipDataset(
            data_root=args.data_root,
            split=args.split,
            episode=int(args.episode),
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            resolution=int(args.resolution),
            height=int(args.height),
            width=int(args.width),
            traj_h5=MANISKILL_DEFAULT_TRAJ_H5,
            traj_json=MANISKILL_DEFAULT_TRAJ_JSON,
            camera=MANISKILL_DEFAULT_CAMERA,
        )
        return dataset[0]
    clip = load_clip(
        data_root=args.data_root,
        task=args.task,
        split=args.split,
        episode=int(args.episode),
        camera=args.camera,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        resolution=int(args.resolution),
        height=int(args.height),
        width=int(args.width),
        load_actions=True,
    )
    return {
        "frames": clip["frames"],
        "actions": clip["actions"],
        "frame_idx": clip["frame_idx"],
        "episode_idx": clip["episode_idx"],
    }


def resolve_aligned_so101_crop_bounds(
    crop_bounds: tuple[int, int, int, int],
    *,
    multiple: int,
) -> tuple[int, int, int, int]:
    """Trim one SO101 crop to the requested alignment using only the sides and bottom."""

    top, bottom, left, right = crop_bounds
    height = int(bottom - top)
    width = int(right - left)
    aligned_height = height - (height % multiple)
    aligned_width = width - (width % multiple)
    if aligned_height < multiple or aligned_width < multiple:
        raise ValueError(
            f"Aligned crop would be too small: original={(height, width)} aligned={(aligned_height, aligned_width)}."
        )
    extra_height = height - aligned_height
    extra_width = width - aligned_width
    # The bottom of the SO101 frame is unused table area; keep the top fixed.
    bottom -= extra_height
    # Trim width symmetrically because both side margins are unused.
    left += extra_width // 2
    right -= extra_width - (extra_width // 2)
    return top, bottom, left, right


def bgr_frame_to_area_downsampled_tensor(
    frame_bgr: np.ndarray,
    *,
    crop_bounds: tuple[int, int, int, int],
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Crop one BGR frame and exact-downsample it into a normalized RGB tensor."""

    cropped = crop_bgr_frame(frame_bgr, crop_bounds)
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(np.transpose(np.ascontiguousarray(pixels), (2, 0, 1))).contiguous()


def maybe_load_lerobot_clip_with_aligned_half_downsample(
    args: argparse.Namespace,
    *,
    cache_dir: str | None,
) -> dict[str, torch.Tensor] | None:
    """Load one SO101 clip by crop-aligning to /32 and then exact-2x area downsampling."""

    repository = LeRobotEpisodeVideoRepository(
        data_root=args.data_root,
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
        cache_dir=cache_dir,
    )
    record = repository.episode_record(episode=int(args.episode))
    resolved_frame_start = 0 if args.frame_start is None else int(args.frame_start)
    resolved_frame_end = record.length - 1 if args.frame_end is None else int(args.frame_end)
    preview_bgr = repository._read_video_frame(record, resolved_frame_start)
    crop_bounds = resolve_default_lerobot_crop_bounds(
        SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        int(preview_bgr.shape[0]),
        int(preview_bgr.shape[1]),
    )
    if crop_bounds is None:
        return None
    aligned_crop_bounds = resolve_aligned_so101_crop_bounds(crop_bounds, multiple=32)
    top, bottom, left, right = aligned_crop_bounds
    crop_height = int(bottom - top)
    crop_width = int(right - left)
    requested_height = int(args.height)
    requested_width = int(args.width)
    if crop_height != requested_height * 2 or crop_width != requested_width * 2:
        return None
    capture = repository._video_capture(record)
    capture.set(cv2.CAP_PROP_POS_FRAMES, resolved_frame_start)
    frames: list[torch.Tensor] = []
    next_frame_index = resolved_frame_start
    while next_frame_index <= resolved_frame_end:
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            capture.release()
            repository._video_captures.pop(record.episode_index, None)
            capture = repository._video_capture(record)
            capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise ValueError(
                    f"Failed to decode frame {next_frame_index} from episode {record.episode_index}."
                )
        frames.append(
            bgr_frame_to_area_downsampled_tensor(
                frame_bgr,
                crop_bounds=aligned_crop_bounds,
                target_height=requested_height,
                target_width=requested_width,
            )
        )
        next_frame_index += 1
    clip = {
        "frames": torch.stack(frames, dim=0),
        "frame_idx": torch.arange(resolved_frame_start, resolved_frame_end + 1, dtype=torch.long),
        "episode_idx": torch.tensor(record.episode_index, dtype=torch.long),
        "actions": repository.load_action_tensors(record, resolved_frame_start, resolved_frame_end),
        "preprocess_crop_bounds": torch.tensor(aligned_crop_bounds, dtype=torch.long),
    }
    return clip


def load_wan22_modules(
    vae_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[WanVAEConfig, WanPosteriorEncoder, WanVideoDecoder]:
    """Load the official Wan 2.2 encoder and decoder into the local port."""

    cfg = WanVAEConfig()
    payload = torch.load(vae_path, map_location="cpu", weights_only=True)
    state_dict = dreamdojo_wan_state_dict(payload)
    remapped = remap_dreamdojo_wan_state_dict(state_dict)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in remapped.items()
        if key.startswith("encoder.")
    }
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in remapped.items()
        if key.startswith("decoder.")
    }
    encoder = WanPosteriorEncoder(cfg)
    decoder = WanVideoDecoder(cfg)
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    encoder.to(device=device, dtype=dtype).eval()
    decoder.to(device=device, dtype=dtype).eval()
    return cfg, encoder, decoder


def reconstruct_chunk(
    frames: torch.Tensor,
    *,
    encoder: WanPosteriorEncoder,
    decoder: WanVideoDecoder,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode and decode one exact Wan 2.2 video chunk."""

    video = rearrange(frames.unsqueeze(0), "b t c h w -> b c t h w").to(device=device, dtype=dtype)
    with torch.no_grad():
        mu, _ = encoder(video)
        reconstructed = decoder(mu)
    return rearrange(reconstructed[0], "c t h w -> t c h w").float().cpu(), mu.float().cpu()


def reconstruct_video_in_chunks(
    frames: torch.Tensor,
    *,
    cfg: WanVAEConfig,
    encoder: WanPosteriorEncoder,
    decoder: WanVideoDecoder,
    device: torch.device,
    dtype: torch.dtype,
    chunk_latent_frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reconstruct an arbitrary-length clip with exact Wan-sized overlapping chunks."""

    if chunk_latent_frames < 1:
        raise ValueError("chunk_latent_frames must be at least 1.")
    chunk_pixel_frames = cfg.latent_frames_to_pixel_frames(int(chunk_latent_frames))
    stride_frames = cfg.temporal_downsample_factor() * max(int(chunk_latent_frames) - 1, 0)
    if stride_frames < 1:
        stride_frames = 1
    reconstructed_chunks: list[torch.Tensor] = []
    latent_shapes: list[list[int]] = []
    total_frames = int(frames.shape[0])
    for start in range(0, total_frames, stride_frames):
        actual_stop = min(start + chunk_pixel_frames, total_frames)
        frame_chunk = frames[start:actual_stop]
        if frame_chunk.shape[0] < chunk_pixel_frames:
            pad_frame = frame_chunk[-1:].expand(chunk_pixel_frames - frame_chunk.shape[0], -1, -1, -1)
            frame_chunk = torch.cat([frame_chunk, pad_frame], dim=0)
        reconstructed_chunk, latent_chunk = reconstruct_chunk(
            frame_chunk,
            encoder=encoder,
            decoder=decoder,
            device=device,
            dtype=dtype,
        )
        latent_shapes.append(list(latent_chunk.shape))
        reconstructed_chunk = reconstructed_chunk[: actual_stop - start]
        reconstructed_chunks.append(reconstructed_chunk if start == 0 else reconstructed_chunk[1:])
        if actual_stop == total_frames:
            break
    reconstructed = torch.cat(reconstructed_chunks, dim=0)
    stats = {
        "chunk_latent_frames": int(chunk_latent_frames),
        "chunk_pixel_frames": int(chunk_pixel_frames),
        "stride_frames": int(stride_frames),
        "latent_shapes": latent_shapes,
    }
    return reconstructed, stats


def build_stats(
    *,
    args: argparse.Namespace,
    vae_path: Path,
    cfg: WanVAEConfig,
    clip: dict[str, torch.Tensor],
    reconstructed: torch.Tensor,
    chunk_stats: dict[str, Any],
    grid_path: Path,
    video_path: Path,
    exported_frame_count: int,
) -> dict[str, Any]:
    """Assemble the reconstruction metadata written alongside the outputs."""

    frames = clip["frames"]
    return {
        "dataset_format": str(args.dataset_format),
        "data_root": str(args.data_root),
        "task": str(args.task),
        "split": str(args.split),
        "episode": int(clip["episode_idx"].reshape(-1)[0].item()),
        "frame_start": int(clip["frame_idx"][0].item()),
        "frame_end": int(clip["frame_idx"][-1].item()),
        "input_frame_count": int(frames.shape[0]),
        "decoded_frame_count": int(reconstructed.shape[0]),
        "exported_video_frame_count": int(exported_frame_count),
        "input_frame_shape": list(frames.shape),
        "output_frame_shape": list(reconstructed.shape),
        "wan_vae_path": str(vae_path),
        "wan_vae_repo": WAN22_REPO_ID,
        "wan_config": cfg.to_dict(),
        "recon_mse": float(torch.mean((reconstructed - frames) ** 2).item()),
        "preprocess_crop_bounds": (
            clip["preprocess_crop_bounds"].tolist() if "preprocess_crop_bounds" in clip else None
        ),
        "output_grid": str(grid_path),
        "output_video": str(video_path),
        **chunk_stats,
    }


def render_reconstruction(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Wan 2.2 reconstruction export and return the written artifact paths."""

    device = torch.device(args.device)
    validate_device(device)
    dtype = resolve_dtype(args.dtype, device)
    vae_path = download_or_resolve_vae_path(args.vae_path)
    clip = load_validation_clip_from_args(args)
    frames = clip["frames"].detach().cpu()
    if int(frames.shape[-2]) % 16 != 0 or int(frames.shape[-1]) % 16 != 0:
        raise ValueError(
            f"Wan 2.2 requires height and width divisible by 16, received {tuple(frames.shape[-2:])}."
        )
    cfg, encoder, decoder = load_wan22_modules(vae_path, device=device, dtype=dtype)
    reconstructed, chunk_stats = reconstruct_video_in_chunks(
        frames,
        cfg=cfg,
        encoder=encoder,
        decoder=decoder,
        device=device,
        dtype=dtype,
        chunk_latent_frames=int(args.chunk_latent_frames),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = output_dir / "episode_0_grid.png"
    video_path = output_dir / "episode_0.mp4"
    stats_path = output_dir / "episode_0_stats.json"
    build_side_by_side_grid(
        original=frames,
        reconstructed=reconstructed,
        max_frames=min(int(args.max_grid_frames), int(frames.shape[0])),
    ).save(grid_path)
    exported_frame_count = write_side_by_side_mp4(
        original=frames,
        reconstructed=reconstructed,
        output_path=video_path,
        duration_ms=120,
    )
    stats = build_stats(
        args=args,
        vae_path=vae_path,
        cfg=cfg,
        clip=clip,
        reconstructed=reconstructed,
        chunk_stats=chunk_stats,
        grid_path=grid_path,
        video_path=video_path,
        exported_frame_count=int(exported_frame_count),
    )
    if stats["input_frame_count"] != stats["decoded_frame_count"]:
        raise RuntimeError(f"Decoded frame count mismatch: {json.dumps(stats, sort_keys=True)}")
    if stats["decoded_frame_count"] != stats["exported_video_frame_count"]:
        raise RuntimeError(f"Exported frame count mismatch: {json.dumps(stats, sort_keys=True)}")
    save_json(stats_path, stats)
    return {
        "grid_path": str(grid_path),
        "video_path": str(video_path),
        "stats_path": str(stats_path),
        "stats": stats,
    }


def main() -> None:
    """Run the Wan 2.2 reconstruction CLI."""

    args = parse_args()
    result = render_reconstruction(args)
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print(f"Wrote grid to {result['grid_path']}")
    print(f"Wrote mp4 to {result['video_path']}")


if __name__ == "__main__":
    main()
