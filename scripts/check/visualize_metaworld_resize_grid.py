"""Create a side-by-side MT50 resize comparison grid.

source .venv/bin/activate
python scripts/check/visualize_metaworld_resize_grid.py \
  --task-index 0 \
  --episode 0 \
  --output outputs/checks/metaworld_task0_episode0_resize_grid.png
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.metaworld_dataset import MetaWorldFrameRecord, MetaWorldRepository


DEFAULT_SIZES = (480, 384, 320, 256, 240, 192, 160, 128)
DEFAULT_OUTPUT = Path("outputs/checks/metaworld_task0_episode0_resize_grid.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the resize-comparison grid."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--cache-dir", default="data/metaworld_cache")
    parser.add_argument("--repo-id", default="lerobot/metaworld_mt50")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="*", default=[])
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def evenly_spaced_frame_indices(length: int, count: int) -> list[int]:
    """Return evenly spaced frame indices across one episode."""

    if count <= 0:
        raise ValueError("frame_count must be positive.")
    if count == 1:
        return [0]
    return [int(index) for index in np.linspace(0, length - 1, num=count, dtype=int)]


def decode_frame_image(
    repository: MetaWorldRepository,
    task_index: int,
    episode: int,
    frame_index: int,
) -> Image.Image:
    """Load one MT50 frame as a full-resolution RGB PIL image."""

    record = repository.episode_record(episode=episode, task_index=task_index)
    image_bytes = repository.load_frame_bytes(
        MetaWorldFrameRecord(
            episode_index=record.episode_index,
            task_index=record.task_index,
            data_chunk_index=record.data_chunk_index,
            data_file_index=record.data_file_index,
            row_index_in_file=repository.frame_row_index_in_file(record, frame_index),
            frame_index=frame_index,
        )
    )
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def render_tile(image: Image.Image, frame_index: int, size: int) -> Image.Image:
    """Render one labeled comparison tile for a target resize."""

    if size == image.width:
        compared = image.copy()
    else:
        compared = image.resize((size, size), resample=Image.Resampling.BILINEAR)
        compared = compared.resize((image.width, image.height), resample=Image.Resampling.BILINEAR)
    labeled = ImageOps.expand(compared, border=4, fill="white")
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((6, 6, min(260, labeled.width - 6), 34), fill=(0, 0, 0))
    draw.text((10, 10), f"frame {frame_index} | {size}", fill=(255, 255, 255))
    return labeled


def build_grid(images: list[Image.Image], columns: int) -> Image.Image:
    """Arrange annotated images into a rectangular grid."""

    if not images:
        raise ValueError("No images were provided for the resize grid.")
    width, height = images[0].size
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * width, rows * height), "black")
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(image, (x, y))
    return canvas


def main() -> None:
    """Generate and save the requested MT50 resize comparison grid."""

    args = parse_args()
    repository = MetaWorldRepository(
        data_root=args.data_root or None,
        repo_id=args.repo_id,
        cache_dir=args.cache_dir or None,
    )
    record = repository.episode_record(episode=args.episode, task_index=args.task_index)
    frame_indices = args.frames or evenly_spaced_frame_indices(record.length, args.frame_count)
    tiles: list[Image.Image] = []
    for frame_index in frame_indices:
        original = decode_frame_image(
            repository=repository,
            task_index=args.task_index,
            episode=args.episode,
            frame_index=frame_index,
        )
        for size in args.sizes:
            tiles.append(render_tile(original, frame_index=frame_index, size=size))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_grid(tiles, columns=len(args.sizes)).save(output_path)
    print(f"Wrote resize grid to {output_path}")


if __name__ == "__main__":
    main()
