"""Download and render short preview MP4s for selected LeRobot sim datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq


OUTPUT_ROOT = Path("outputs/dataset_previews")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TEXT_BAND_HEIGHT = 132
TITLE_CARD_FRAMES = 18


@dataclass(frozen=True)
class PreviewSpec:
    """Describe one dataset preview clip to download and render."""

    repo_id: str
    short_name: str
    video_path: str
    note: str
    task: str
    episode_metadata_path: str | None = None
    max_preview_seconds: float = 4.0


PREVIEW_SPECS: tuple[PreviewSpec, ...] = (
    PreviewSpec(
        repo_id="lerobot/xarm_lift_medium",
        short_name="xarm_lift_medium",
        video_path="videos/observation.image/chunk-000/file-000.mp4",
        note="Official LeRobot arm sim.",
        task="Lift a cube.",
        episode_metadata_path="meta/episodes/chunk-000/file-000.parquet",
    ),
    PreviewSpec(
        repo_id="lerobot/xarm_push_medium",
        short_name="xarm_push_medium",
        video_path="videos/observation.image/chunk-000/file-000.mp4",
        note="Official LeRobot arm sim.",
        task="Push a cube onto the target.",
        episode_metadata_path="meta/episodes/chunk-000/file-000.parquet",
    ),
    PreviewSpec(
        repo_id="davidlinjiahao/lerobot_so101_base_sim_pickplace",
        short_name="so101_base_sim_pickplace",
        video_path="videos/chunk-000/observation.images.front/episode_000000.mp4",
        note="Community SO-101 sim with high-res front camera.",
        task="Pick and place.",
        episode_metadata_path="data/chunk-000/episode_000000.parquet",
    ),
    PreviewSpec(
        repo_id="LeRobot-worldwide-hackathon/97-LeRobotec-o3de_sim_so101_pick_up_red_ball",
        short_name="o3de_so101_red_ball",
        video_path="videos/chunk-000/observation.images.up/episode_000000.mp4",
        note="Hackathon O3DE single-arm sim.",
        task="Pick up the red ball.",
        episode_metadata_path="data/chunk-000/episode_000000.parquet",
    ),
    PreviewSpec(
        repo_id="lerobot/pusht",
        short_name="pusht_bonus",
        video_path="videos/observation.image/chunk-000/file-000.mp4",
        note="Bonus sanity-check baseline. Not a real robot arm.",
        task="Push a T block onto the T target.",
        episode_metadata_path="meta/episodes/chunk-000/file-000.parquet",
    ),
)


def load_json(path: Path) -> dict[str, Any]:
    """Return one JSON file as a Python dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))


def download_dataset_file(repo_id: str, filename: str) -> Path:
    """Download one dataset asset through the Hugging Face cache."""

    return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename))


def first_video_feature_key(info: dict[str, Any]) -> str:
    """Return the first video feature key declared in dataset metadata."""

    for name, feature in info["features"].items():
        if feature.get("dtype") == "video":
            return str(name)
    raise ValueError("Dataset metadata does not declare any video features.")


def video_shape(info: dict[str, Any], video_key: str) -> tuple[int, int]:
    """Return the source video height and width from info metadata."""

    shape = info["features"][video_key]["shape"]
    return int(shape[0]), int(shape[1])


def action_dim(info: dict[str, Any]) -> int:
    """Return the dataset action dimensionality."""

    shape = info["features"]["action"]["shape"]
    return int(shape[0])


def dataset_fps(info: dict[str, Any], video_key: str) -> float:
    """Return the video frame rate for one dataset preview."""

    video_info = info["features"][video_key]
    if "video_info" in video_info:
        return float(video_info["video_info"]["video.fps"])
    if "info" in video_info:
        return float(video_info["info"]["video.fps"])
    return float(info["fps"])


def first_episode_frame_limit(spec: PreviewSpec, info: dict[str, Any], fps: float) -> int:
    """Return the preview frame budget for the first episode."""

    default_limit = max(1, int(round(spec.max_preview_seconds * fps)))
    if spec.episode_metadata_path is None:
        return default_limit
    metadata_path = download_dataset_file(spec.repo_id, spec.episode_metadata_path)
    table = pq.read_table(metadata_path)
    if "length" in table.column_names:
        return min(default_limit, int(table["length"][0].as_py()))
    return min(default_limit, int(table.num_rows))


def read_video_frames(video_path: Path, frame_limit: int) -> list[np.ndarray]:
    """Read up to the requested number of RGB frames from one MP4 file."""

    frames: list[np.ndarray] = []
    reader = imageio.get_reader(video_path, format="FFMPEG")
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index >= frame_limit:
                break
            frames.append(np.asarray(frame, dtype=np.uint8))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames could be read from {video_path}.")
    return frames


def fit_frame(frame: np.ndarray, target_width: int, target_height: int) -> Image.Image:
    """Resize one frame to fit inside a preview canvas while preserving aspect ratio."""

    source = Image.fromarray(frame)
    scale = min(target_width / source.width, target_height / source.height)
    scaled_width = max(1, int(round(source.width * scale)))
    scaled_height = max(1, int(round(source.height * scale)))
    resample = Image.Resampling.NEAREST if scale >= 1.0 else Image.Resampling.LANCZOS
    return source.resize((scaled_width, scaled_height), resample=resample)


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a readable sans font with a safe fallback."""

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def title_card(spec: PreviewSpec, summary: str) -> np.ndarray:
    """Render one still title card for a dataset preview."""

    canvas = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), color=(18, 21, 28))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44)
    body_font = load_font(28)
    draw.text((80, 120), spec.short_name, fill=(245, 245, 245), font=title_font)
    draw.text((80, 188), spec.repo_id, fill=(172, 184, 196), font=body_font)
    draw.text((80, 278), summary, fill=(236, 236, 236), font=body_font)
    draw.text((80, 332), f"Task: {spec.task}", fill=(236, 236, 236), font=body_font)
    draw.text((80, 386), spec.note, fill=(220, 220, 220), font=body_font)
    draw.rectangle((80, 465, FRAME_WIDTH - 80, 470), fill=(73, 127, 214))
    return np.asarray(canvas, dtype=np.uint8)


def annotate_frame(frame: np.ndarray, spec: PreviewSpec, summary: str, frame_index: int) -> np.ndarray:
    """Overlay one dataset label block above the resized source frame."""

    canvas = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), color=(10, 12, 18))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    body_font = load_font(22)
    draw.text((30, 20), spec.short_name, fill=(250, 250, 250), font=title_font)
    draw.text((30, 56), summary, fill=(210, 218, 228), font=body_font)
    draw.text(
        (30, 86),
        f"Task: {spec.task} | frame {frame_index}",
        fill=(210, 218, 228),
        font=body_font,
    )
    fitted = fit_frame(frame, FRAME_WIDTH - 40, FRAME_HEIGHT - TEXT_BAND_HEIGHT - 28)
    x_offset = (FRAME_WIDTH - fitted.width) // 2
    y_offset = TEXT_BAND_HEIGHT + ((FRAME_HEIGHT - TEXT_BAND_HEIGHT - fitted.height) // 2)
    canvas.paste(fitted, (x_offset, y_offset))
    return np.asarray(canvas, dtype=np.uint8)


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write one RGB frame sequence to an H.264 MP4 file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv444p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        ffmpeg_params=["-crf", "12", "-preset", "medium", "-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def build_summary(info: dict[str, Any], video_key: str) -> str:
    """Format the key dataset stats for one overlay line."""

    height, width = video_shape(info, video_key)
    fps = dataset_fps(info, video_key)
    return (
        f"{height}x{width} | action_dim={action_dim(info)} | "
        f"episodes={int(info['total_episodes'])} | fps={fps:g}"
    )


def render_dataset_preview(spec: PreviewSpec) -> dict[str, Any]:
    """Download one dataset and export its short labeled preview clip."""

    info_path = download_dataset_file(spec.repo_id, "meta/info.json")
    info = load_json(info_path)
    video_key = first_video_feature_key(info)
    fps = dataset_fps(info, video_key)
    frame_limit = first_episode_frame_limit(spec, info, fps)
    video_path = download_dataset_file(spec.repo_id, spec.video_path)
    raw_frames = read_video_frames(video_path, frame_limit=frame_limit)
    summary = build_summary(info, video_key)
    rendered_frames = [title_card(spec, summary) for _ in range(TITLE_CARD_FRAMES)]
    rendered_frames.extend(
        annotate_frame(frame, spec, summary, frame_index=index)
        for index, frame in enumerate(raw_frames)
    )
    output_path = OUTPUT_ROOT / f"{spec.short_name}.mp4"
    write_mp4(output_path, rendered_frames, fps=fps)
    return {
        "repo_id": spec.repo_id,
        "short_name": spec.short_name,
        "task": spec.task,
        "note": spec.note,
        "video_path": str(output_path),
        "summary": summary,
        "source_video_path": spec.video_path,
        "frame_count": len(rendered_frames),
    }


def main() -> None:
    """Create per-dataset preview MP4s and one combined overview video."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_frames: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    for spec in PREVIEW_SPECS:
        preview = render_dataset_preview(spec)
        manifest.append(preview)
        reader = imageio.get_reader(preview["video_path"], format="FFMPEG")
        try:
            for frame in reader:
                combined_frames.append(np.asarray(frame, dtype=np.uint8))
        finally:
            reader.close()
    write_mp4(OUTPUT_ROOT / "lerobot_single_arm_sim_overview.mp4", combined_frames, fps=15.0)
    (OUTPUT_ROOT / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
