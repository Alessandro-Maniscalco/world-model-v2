"""Visualize downloaded Interactive World Sim HDF5 episodes as grids, MP4s, or task sheets.

source .venv/bin/activate
python scripts/check/visualize_interactive_world_sim_data.py episode-grid \
  --data-root data/full \
  --task single_grasp \
  --split val \
  --episode 0 \
  --camera camera_0_color \
  --frames 9 \
  --output /tmp/single_grasp_grid.png

source .venv/bin/activate
python scripts/check/visualize_interactive_world_sim_data.py first-video \
  --data-root data/full \
  --output /tmp/interactive_world_sim_first_video.mp4


"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_DATA_ROOT = Path("data/full")
DEFAULT_TRAIN_TASK = "single_grasp"
DEFAULT_TRAIN_SPLIT = "val"
DEFAULT_TRAIN_EPISODE = 0
DEFAULT_TRAIN_CAMERA = "camera_1_color"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List downloaded tasks, splits, and episodes.")
    list_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))

    first_parser = subparsers.add_parser(
        "first-video", help="Create an MP4 for the default starter training task and camera."
    )
    first_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    first_parser.add_argument("--task", default=DEFAULT_TRAIN_TASK)
    first_parser.add_argument("--split", default=DEFAULT_TRAIN_SPLIT, choices=["train", "val"])
    first_parser.add_argument("--episode", type=int, default=DEFAULT_TRAIN_EPISODE)
    first_parser.add_argument("--camera", default=DEFAULT_TRAIN_CAMERA)
    first_parser.add_argument("--frames", type=int, default=24)
    first_parser.add_argument("--duration-ms", type=int, default=120)
    first_parser.add_argument(
        "--output", default="/tmp/interactive_world_sim_first_video.mp4"
    )

    grid_parser = subparsers.add_parser("episode-grid", help="Create a grid of frames from one episode.")
    add_common_episode_args(grid_parser)
    grid_parser.add_argument("--camera", required=True)
    grid_parser.add_argument("--frames", type=int, default=12)
    grid_parser.add_argument("--output", required=True)

    video_parser = subparsers.add_parser("episode-video", help="Create an MP4 from one episode.")
    add_common_episode_args(video_parser)
    video_parser.add_argument("--camera", required=True)
    video_parser.add_argument("--frames", type=int, default=24)
    video_parser.add_argument("--duration-ms", type=int, default=120)
    video_parser.add_argument("--output", required=True)

    sheet_parser = subparsers.add_parser(
        "task-sheet", help="Create a contact sheet for the same frame index across many episodes."
    )
    sheet_parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    sheet_parser.add_argument("--task", required=True)
    sheet_parser.add_argument("--split", default="val", choices=["train", "val"])
    sheet_parser.add_argument("--camera", required=True)
    sheet_parser.add_argument("--frame-index", type=int, default=0)
    sheet_parser.add_argument("--limit", type=int, default=8)
    sheet_parser.add_argument("--output", required=True)
    return parser.parse_args()


def add_common_episode_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by episode-based commands."""

    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--episode", type=int, default=0)


def episode_path(data_root: Path, task: str, split: str, episode: int) -> Path:
    """Build the path to a raw HDF5 episode file."""

    return data_root / task / split / f"episode_{episode}.hdf5"


def list_available_episodes(split_dir: Path) -> list[Path]:
    """Return all raw HDF5 episodes available in one split directory."""

    return sorted(split_dir.glob("episode_*.hdf5"))


def list_cameras(h5_file: h5py.File) -> list[str]:
    """List image camera keys available in one episode file."""

    return sorted(
        key for key in h5_file["obs"]["images"].keys() if key.endswith("_color")
    )


def list_available_tasks(data_root: Path) -> list[Path]:
    """Return all downloaded task directories."""

    return sorted(path for path in data_root.iterdir() if path.is_dir())


def load_camera_frames(episode_file: Path, camera: str) -> np.ndarray:
    """Load RGB frames for one camera from an episode file."""

    with h5py.File(episode_file, "r") as handle:
        return handle["obs"]["images"][camera][()]


def load_actions(episode_file: Path) -> np.ndarray:
    """Load the action array for one episode."""

    with h5py.File(episode_file, "r") as handle:
        return handle["action"][()]


def frame_indices(num_frames: int, wanted: int) -> np.ndarray:
    """Choose evenly spaced frame indices across an episode."""

    if wanted >= num_frames:
        return np.arange(num_frames, dtype=int)
    return np.linspace(0, num_frames - 1, num=wanted, dtype=int)


def to_pil(image_array: np.ndarray) -> Image.Image:
    """Convert one image array into a PIL image."""

    if image_array.dtype != np.uint8:
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(image_array)


def format_action(action: np.ndarray) -> str:
    """Format one action vector for display."""

    return "[" + ", ".join(f"{value:.2f}" for value in action) + "]"


def annotate(image: Image.Image, text: str) -> Image.Image:
    """Add a visible label and border to an image."""

    bordered = ImageOps.expand(image, border=4, fill="white")
    draw = ImageDraw.Draw(bordered)
    draw.rectangle((6, 6, min(bordered.width - 6, 360), 42), fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255))
    return bordered


def duration_ms_to_fps(duration_ms: int) -> float:
    """Convert a fixed frame duration in milliseconds into frames per second."""

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive.")
    return 1000.0 / float(duration_ms)


def pad_even_frame(image_array: np.ndarray) -> np.ndarray:
    """Pad one RGB frame to even width and height for MP4 export."""

    pad_height = image_array.shape[0] % 2
    pad_width = image_array.shape[1] % 2
    if pad_height == 0 and pad_width == 0:
        return image_array
    return np.pad(image_array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def write_mp4(images: list[Image.Image], output: Path, duration_ms: int) -> None:
    """Write a list of PIL frames to an MP4 file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output,
        format="FFMPEG",
        mode="I",
        fps=duration_ms_to_fps(duration_ms),
        codec="libx264",
        pixelformat="yuv444p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        ffmpeg_params=["-crf", "12", "-preset", "medium", "-movflags", "+faststart"],
    ) as writer:
        for image in images:
            writer.append_data(pad_even_frame(np.asarray(image)))


def make_grid(images: list[Image.Image], columns: int | None = None) -> Image.Image:
    """Arrange images into a rectangular grid."""

    if not images:
        raise ValueError("No images were provided for the grid.")
    width, height = images[0].size
    if columns is None:
        columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), "black")
    for idx, image in enumerate(images):
        x = (idx % columns) * width
        y = (idx // columns) * height
        canvas.paste(image, (x, y))
    return canvas


def command_list(data_root: Path) -> None:
    """Print the downloaded task/split/episode inventory."""

    for task_dir in list_available_tasks(data_root):
        print(task_dir.name)
        for split in ("train", "val"):
            split_dir = task_dir / split
            if not split_dir.exists():
                continue
            hdf5_files = list_available_episodes(split_dir)
            cache_exists = (split_dir / "cache.zarr.zip").exists()
            print(
                f"  {split}: cache={'yes' if cache_exists else 'no'},"
                f" raw_episodes={len(hdf5_files)}"
            )
            if hdf5_files:
                with h5py.File(hdf5_files[0], "r") as handle:
                    cameras = list_cameras(handle)
                    action_shape = handle["action"].shape
                    print(f"    sample_file={hdf5_files[0].name}")
                    print(f"    cameras={cameras}")
                    print(f"    action_shape={action_shape}")


def command_first_video(
    data_root: Path,
    task: str,
    split: str,
    episode: int,
    camera: str,
    frames: int,
    duration_ms: int,
    output: Path,
) -> None:
    """Write an MP4 for the default starter training episode."""

    episode_file = episode_path(data_root, task, split, episode)
    if not episode_file.exists():
        raise ValueError(f"Episode file not found: {episode_file}")
    with h5py.File(episode_file, "r") as handle:
        cameras = list_cameras(handle)
    if camera not in cameras:
        raise ValueError(f"Camera {camera} not found in {episode_file}. Available: {cameras}")
    command_episode_video(
        data_root=data_root,
        task=task,
        split=split,
        episode=episode,
        camera=camera,
        frames=frames,
        duration_ms=duration_ms,
        output=output,
    )
    print(f"Used task={task} split={split} episode={episode} camera={camera}")


def command_episode_grid(
    data_root: Path, task: str, split: str, episode: int, camera: str, frames: int, output: Path
) -> None:
    """Write a frame grid from one episode."""

    ep_path = episode_path(data_root, task, split, episode)
    frame_array = load_camera_frames(ep_path, camera)
    actions = load_actions(ep_path)
    indices = frame_indices(len(frame_array), frames)
    images = [
        annotate(
            to_pil(frame_array[idx]),
            f"f {idx}  a {format_action(actions[min(idx, len(actions) - 1)])}",
        )
        for idx in indices
    ]
    make_grid(images).save(output)
    print(f"Wrote grid to {output}")


def command_episode_video(
    data_root: Path,
    task: str,
    split: str,
    episode: int,
    camera: str,
    frames: int,
    duration_ms: int,
    output: Path,
) -> None:
    """Write an MP4 from one episode."""

    ep_path = episode_path(data_root, task, split, episode)
    frame_array = load_camera_frames(ep_path, camera)
    actions = load_actions(ep_path)
    indices = frame_indices(len(frame_array), frames)
    images = [
        annotate(
            to_pil(frame_array[idx]),
            f"f {idx}  a {format_action(actions[min(idx, len(actions) - 1)])}",
        )
        for idx in indices
    ]
    write_mp4(images, output, duration_ms)
    print(f"Wrote MP4 to {output}")


def command_task_sheet(
    data_root: Path,
    task: str,
    split: str,
    camera: str,
    frame_index: int,
    limit: int,
    output: Path,
) -> None:
    """Write a contact sheet using the same frame index across many episodes."""

    split_dir = data_root / task / split
    episode_files = list_available_episodes(split_dir)[:limit]
    images: list[Image.Image] = []
    for ep_path in episode_files:
        frames = load_camera_frames(ep_path, camera)
        actions = load_actions(ep_path)
        chosen = min(frame_index, len(frames) - 1)
        label = ep_path.stem.replace("episode_", "ep ")
        images.append(
            annotate(
                to_pil(frames[chosen]),
                f"{label} f {chosen}  a {format_action(actions[min(chosen, len(actions) - 1)])}",
            )
        )
    make_grid(images).save(output)
    print(f"Wrote task sheet to {output}")


def main() -> None:
    """Run the requested visualization command."""

    args = parse_args()
    data_root = Path(args.data_root)
    if args.command == "list":
        command_list(data_root)
    elif args.command == "first-video":
        command_first_video(
            data_root=data_root,
            task=args.task,
            split=args.split,
            episode=args.episode,
            camera=args.camera,
            frames=args.frames,
            duration_ms=args.duration_ms,
            output=Path(args.output),
        )
    elif args.command == "episode-grid":
        command_episode_grid(
            data_root=data_root,
            task=args.task,
            split=args.split,
            episode=args.episode,
            camera=args.camera,
            frames=args.frames,
            output=Path(args.output),
        )
    elif args.command == "episode-video":
        command_episode_video(
            data_root=data_root,
            task=args.task,
            split=args.split,
            episode=args.episode,
            camera=args.camera,
            frames=args.frames,
            duration_ms=args.duration_ms,
            output=Path(args.output),
        )
    elif args.command == "task-sheet":
        command_task_sheet(
            data_root=data_root,
            task=args.task,
            split=args.split,
            camera=args.camera,
            frame_index=args.frame_index,
            limit=args.limit,
            output=Path(args.output),
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
