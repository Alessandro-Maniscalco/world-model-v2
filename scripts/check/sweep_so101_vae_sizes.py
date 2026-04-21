r"""Sweep Wan 2.2 reconstructions across exact SO101 default-crop downsample sizes.

Example:
    . .\.venv\Scripts\Activate.ps1
    python scripts/check/sweep_so101_vae_sizes.py `
      --data-root data/so101_base_sim_pickplace_cache `
      --episode 0 `
      --frame-index 0 `
      --output-dir outputs/checks/so101_vae_size_sweep_ep0_frame0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image, ImageDraw
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.lerobot_video_dataset import (
    LeRobotEpisodeVideoRepository,
    SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    bgr_frame_to_exact_area_downsampled_tensor,
    crop_bgr_frame,
    resolve_default_lerobot_crop_bounds,
    resolve_exact_downsample_crop_bounds,
)
from world_model_v2.wan_vae import (
    WanPosteriorEncoder,
    WanVAEConfig,
    WanVideoDecoder,
    dreamdojo_wan_state_dict,
    remap_dreamdojo_wan_state_dict,
)


WAN22_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_FILENAME = "Wan2.2_VAE.pth"
DISPLAY_HEIGHT = 208
DISPLAY_WIDTH = 272
ROW_GAP = 14
CELL_GAP = 16
TEXT_GAP = 10
OUTER_PAD = 16
LABEL_HEIGHT = 36
TITLE_HEIGHT = 64


@dataclass(frozen=True)
class SweepSpec:
    """Describe one exact SO101 crop and downsample candidate."""

    target_height: int
    target_width: int
    crop_bounds: tuple[int, int, int, int]
    crop_height: int
    crop_width: int
    downsample_factor: int

    @property
    def latent_height(self) -> int:
        """Return the Wan latent height for this target size."""

        return self.target_height // 16

    @property
    def latent_width(self) -> int:
        """Return the Wan latent width for this target size."""

        return self.target_width // 16

    @property
    def target_area(self) -> int:
        """Return the pixel area of the downsampled frame."""

        return self.target_height * self.target_width


@dataclass(frozen=True)
class SweepResult:
    """Bundle one GT/reconstruction pair and its summary metrics."""

    spec: SweepSpec
    mse: float
    psnr_db: float
    gt_image_path: str
    reconstructed_image_path: str

    def to_dict(self) -> dict[str, object]:
        """Return one JSON-serializable result payload."""

        payload = asdict(self)
        payload["spec"] = asdict(self.spec)
        return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the SO101 size sweep."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vae-path", default="")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for the Wan 2.2 VAE.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Inference dtype for the Wan 2.2 VAE.",
    )
    return parser.parse_args()


def validate_device(device: torch.device) -> None:
    """Fail fast when CUDA is requested but unavailable."""

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch.")


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


def configure_torch_attention(device: torch.device) -> None:
    """Prefer the stable math SDPA backend for tiny-shape Wan VAE sweeps."""

    if device.type != "cuda":
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def download_or_resolve_vae_path(vae_path: str | Path) -> Path:
    """Return a local filesystem path to the official Wan 2.2 VAE weights."""

    if str(vae_path):
        resolved = Path(vae_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Wan 2.2 VAE path not found: {resolved}")
        return resolved
    downloaded = hf_hub_download(repo_id=WAN22_REPO_ID, filename=WAN22_FILENAME, repo_type="model")
    return Path(downloaded)


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


def representative_sweep_specs(
    default_crop_bounds: tuple[int, int, int, int],
) -> list[SweepSpec]:
    """Return one largest VAE-safe target size for each exact downsample factor."""

    top, bottom, left, right = default_crop_bounds
    crop_height = int(bottom - top)
    crop_width = int(right - left)
    by_factor: dict[int, SweepSpec] = {}
    target_aspect = crop_height / crop_width
    for target_height in range(16, crop_height + 1, 16):
        for target_width in range(16, crop_width + 1, 16):
            exact = resolve_exact_downsample_crop_bounds(
                default_crop_bounds,
                target_height=target_height,
                target_width=target_width,
            )
            if exact is None:
                continue
            crop_bounds, downsample_factor = exact
            candidate = SweepSpec(
                target_height=target_height,
                target_width=target_width,
                crop_bounds=crop_bounds,
                crop_height=crop_bounds[1] - crop_bounds[0],
                crop_width=crop_bounds[3] - crop_bounds[2],
                downsample_factor=downsample_factor,
            )
            current = by_factor.get(downsample_factor)
            if current is None:
                by_factor[downsample_factor] = candidate
                continue
            current_area = current.target_area
            candidate_area = candidate.target_area
            if candidate_area > current_area:
                by_factor[downsample_factor] = candidate
                continue
            if candidate_area != current_area:
                continue
            current_ratio_error = abs((current.target_height / current.target_width) - target_aspect)
            candidate_ratio_error = abs((candidate.target_height / candidate.target_width) - target_aspect)
            if candidate_ratio_error < current_ratio_error:
                by_factor[downsample_factor] = candidate
    return sorted(by_factor.values(), key=lambda spec: spec.target_area, reverse=True)


def reconstruct_single_frame(
    frame_chw: torch.Tensor,
    *,
    cfg: WanVAEConfig,
    encoder: WanPosteriorEncoder,
    decoder: WanVideoDecoder,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode and decode one resized RGB frame through one exact Wan chunk."""

    chunk_pixel_frames = cfg.latent_frames_to_pixel_frames(cfg.temporal_window)
    repeated_frames = frame_chw.unsqueeze(0).repeat(chunk_pixel_frames, 1, 1, 1)
    video = repeated_frames.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(device=device, dtype=dtype)
    with torch.no_grad():
        mu, _ = encoder(video)
        reconstructed = decoder(mu)
    return reconstructed[0, :, 0].float().cpu().clamp(0.0, 1.0)


def tensor_to_pil(frame_chw: torch.Tensor) -> Image.Image:
    """Convert one normalized CHW tensor into a PIL RGB image."""

    array = frame_chw.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(array), mode="RGB")


def fit_for_display(image: Image.Image, *, width: int, height: int) -> Image.Image:
    """Upscale one image into a fixed display cell while preserving aspect ratio."""

    scale = min(width / image.width, height / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), resample=Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def build_contact_sheet(
    *,
    results: list[SweepResult],
    raw_frame_path: str,
    default_crop_bounds: tuple[int, int, int, int],
    raw_shape: tuple[int, int, int],
) -> Image.Image:
    """Render one comparison sheet with GT and reconstruction pairs per target size."""

    sheet_width = OUTER_PAD * 2 + DISPLAY_WIDTH * 2 + CELL_GAP
    sheet_height = (
        OUTER_PAD * 2
        + TITLE_HEIGHT
        + len(results) * (LABEL_HEIGHT + DISPLAY_HEIGHT)
        + max(len(results) - 1, 0) * ROW_GAP
    )
    canvas = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    title = "SO101 default-crop exact downsample sweep"
    subtitle = (
        f"raw={raw_shape[1]}x{raw_shape[0]}  default_crop={default_crop_bounds}  "
        f"raw_preview={raw_frame_path}"
    )
    draw.text((OUTER_PAD, OUTER_PAD), title, fill=(0, 0, 0))
    draw.text((OUTER_PAD, OUTER_PAD + 22), subtitle, fill=(40, 40, 40))
    draw.text((OUTER_PAD, OUTER_PAD + 42), "Left: GT after crop+downsample   Right: Wan 2.2 reconstruction", fill=(40, 40, 40))
    current_y = OUTER_PAD + TITLE_HEIGHT
    for result in results:
        gt_image = fit_for_display(Image.open(result.gt_image_path), width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT)
        reconstructed_image = fit_for_display(
            Image.open(result.reconstructed_image_path),
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
        )
        label = (
            f"{result.spec.target_height}x{result.spec.target_width} | crop {result.spec.crop_height}x{result.spec.crop_width} "
            f"| x{result.spec.downsample_factor} | latents {result.spec.latent_height}x{result.spec.latent_width} "
            f"| mse {result.mse:.6f} | psnr {result.psnr_db:.2f} dB"
        )
        draw.text((OUTER_PAD, current_y + 8), label, fill=(0, 0, 0))
        image_y = current_y + LABEL_HEIGHT
        canvas.paste(gt_image, (OUTER_PAD, image_y))
        canvas.paste(reconstructed_image, (OUTER_PAD + DISPLAY_WIDTH + CELL_GAP, image_y))
        current_y = image_y + DISPLAY_HEIGHT + ROW_GAP
    return canvas


def raw_frame_with_default_crop_overlay(
    frame_bgr: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
) -> Image.Image:
    """Return one raw frame preview with the default crop overlaid."""

    rgb = frame_bgr[:, :, ::-1]
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    draw = ImageDraw.Draw(image)
    top, bottom, left, right = crop_bounds
    draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 64, 64), width=4)
    draw.rectangle((0, 0, image.width, 30), fill=(0, 0, 0))
    draw.text((8, 8), f"Default crop {crop_bounds}", fill=(255, 255, 255))
    return image


def main() -> None:
    """Run the SO101 VAE size sweep and save comparison artifacts."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    validate_device(device)
    configure_torch_attention(device)
    dtype = resolve_dtype(args.dtype, device)
    vae_path = download_or_resolve_vae_path(args.vae_path)
    repository = LeRobotEpisodeVideoRepository(
        data_root=args.data_root,
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    )
    record = repository.episode_record(episode=int(args.episode))
    frame_bgr = repository._read_video_frame(record, int(args.frame_index))
    default_crop_bounds = resolve_default_lerobot_crop_bounds(
        SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        int(frame_bgr.shape[0]),
        int(frame_bgr.shape[1]),
    )
    if default_crop_bounds is None:
        raise RuntimeError("SO101 default crop bounds were not available.")
    specs = representative_sweep_specs(default_crop_bounds)
    raw_frame_preview = raw_frame_with_default_crop_overlay(frame_bgr, default_crop_bounds)
    raw_frame_path = output_dir / "raw_frame_with_default_crop.png"
    raw_frame_preview.save(raw_frame_path)
    cfg, encoder, decoder = load_wan22_modules(vae_path, device=device, dtype=dtype)
    results: list[SweepResult] = []
    for spec in specs:
        gt_frame = bgr_frame_to_exact_area_downsampled_tensor(
            frame_bgr,
            crop_bounds=spec.crop_bounds,
            target_height=spec.target_height,
            target_width=spec.target_width,
        )
        reconstructed = reconstruct_single_frame(
            gt_frame,
            cfg=cfg,
            encoder=encoder,
            decoder=decoder,
            device=device,
            dtype=dtype,
        )
        mse = float(torch.mean((reconstructed - gt_frame) ** 2).item())
        psnr_db = float("inf") if mse <= 0.0 else float(10.0 * math.log10(1.0 / mse))
        gt_image_path = output_dir / f"gt_{spec.target_height}x{spec.target_width}_x{spec.downsample_factor}.png"
        reconstructed_image_path = (
            output_dir
            / f"reconstructed_{spec.target_height}x{spec.target_width}_x{spec.downsample_factor}.png"
        )
        tensor_to_pil(gt_frame).save(gt_image_path)
        tensor_to_pil(reconstructed).save(reconstructed_image_path)
        results.append(
            SweepResult(
                spec=spec,
                mse=mse,
                psnr_db=psnr_db,
                gt_image_path=str(gt_image_path),
                reconstructed_image_path=str(reconstructed_image_path),
            )
        )
    contact_sheet = build_contact_sheet(
        results=results,
        raw_frame_path=str(raw_frame_path),
        default_crop_bounds=default_crop_bounds,
        raw_shape=tuple(int(value) for value in frame_bgr.shape),
    )
    contact_sheet_path = output_dir / "so101_vae_size_sweep_contact_sheet.png"
    summary_path = output_dir / "so101_vae_size_sweep_summary.json"
    contact_sheet.save(contact_sheet_path)
    summary = {
        "data_root": str(args.data_root),
        "episode": int(args.episode),
        "frame_index": int(args.frame_index),
        "raw_shape": list(int(value) for value in frame_bgr.shape),
        "default_crop_bounds": list(int(value) for value in default_crop_bounds),
        "raw_frame_preview": str(raw_frame_path),
        "contact_sheet": str(contact_sheet_path),
        "wan_vae_path": str(vae_path),
        "results": [result.to_dict() for result in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
