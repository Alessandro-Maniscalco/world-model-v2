"""Compare SO101 resize filters at a target training resolution.

Example command:
  ./.venv/Scripts/python.exe scripts/check/compare_resize_filters.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.image_resize import supported_resize_filters
from world_model_v2.lerobot_video_dataset import (
    LeRobotEpisodeVideoRepository,
    SO101_BASE_SIM_PICKPLACE_DATASET_ID,
    SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    crop_bgr_frame,
    resolve_default_lerobot_crop_bounds,
)
from world_model_v2.metaworld_dataset import bgr_frame_to_tensor
from world_model_v2.utils.visualization import annotate_frame, tensor_to_uint8_images


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the resize-filter comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument(
        "--resize-filter",
        action="append",
        default=[],
        help="Resize filters to compare. Defaults to bilinear, bicubic, lanczos.",
    )
    return parser.parse_args()


def resolved_resize_filters(requested_filters: list[str]) -> tuple[str, ...]:
    """Resolve the requested filter list or fall back to all supported filters."""

    if not requested_filters:
        return supported_resize_filters()
    supported = set(supported_resize_filters())
    normalized = tuple(filter_name.strip().lower() for filter_name in requested_filters)
    invalid = [filter_name for filter_name in normalized if filter_name not in supported]
    if invalid:
        supported_list = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported resize filters {invalid!r}. Expected only: {supported_list}."
        )
    return normalized


def select_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    """Select evenly spaced frame indices across one episode."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if frame_count <= max_frames:
        return list(range(frame_count))
    return sorted({int(index) for index in np.linspace(0, frame_count - 1, num=max_frames)})


def fit_rgb_frame_to_canvas(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Fit one RGB frame into a black canvas while preserving aspect ratio."""

    canvas = Image.new("RGB", (width, height), "black")
    contained = ImageOps.contain(
        Image.fromarray(rgb, mode="RGB"),
        (width, height),
        method=Image.Resampling.LANCZOS,
    )
    offset_x = (width - contained.width) // 2
    offset_y = (height - contained.height) // 2
    canvas.paste(contained, (offset_x, offset_y))
    return np.asarray(canvas)


def render_frame_row(
    cropped_bgr: np.ndarray,
    *,
    frame_index: int,
    height: int,
    width: int,
    resize_filters: tuple[str, ...],
) -> Image.Image:
    """Render one comparison row for a single cropped frame."""

    source_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    panels = [
        annotate_frame(
            fit_rgb_frame_to_canvas(source_rgb, height=height, width=width),
            f"source {frame_index}",
        )
    ]
    for resize_filter in resize_filters:
        tensor = bgr_frame_to_tensor(
            cropped_bgr,
            height=height,
            width=width,
            resize_filter=resize_filter,
        )
        panel = annotate_frame(
            tensor_to_uint8_images(tensor.unsqueeze(0))[0],
            f"{resize_filter} {frame_index}",
        )
        panels.append(panel)
    row_width = sum(panel.width for panel in panels)
    row_height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (row_width, row_height), "black")
    offset_x = 0
    for panel in panels:
        canvas.paste(panel, (offset_x, 0))
        offset_x += panel.width
    return canvas


def build_contact_sheet(rows: list[Image.Image]) -> Image.Image:
    """Stack rendered comparison rows into one contact sheet."""

    if not rows:
        raise ValueError("Expected at least one rendered row.")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), "black")
    offset_y = 0
    for row in rows:
        canvas.paste(row, (0, offset_y))
        offset_y += row.height
    return canvas


def main() -> None:
    """Render and save a resize-filter comparison contact sheet."""

    args = parse_args()
    resize_filters = resolved_resize_filters(args.resize_filter)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repository = LeRobotEpisodeVideoRepository(
        data_root=args.data_root,
        repo_id=SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        image_column=SO101_BASE_SIM_PICKPLACE_IMAGE_COLUMN,
    )
    record = repository.episode_record(args.episode)
    frame_indices = select_frame_indices(record.length, args.max_frames)
    first_frame = repository._read_video_frame(record, frame_indices[0])
    crop_bounds = resolve_default_lerobot_crop_bounds(
        SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        first_frame.shape[0],
        first_frame.shape[1],
    )

    rows: list[Image.Image] = []
    for frame_index in frame_indices:
        raw_frame = repository._read_video_frame(record, frame_index)
        cropped_bgr = crop_bgr_frame(raw_frame, crop_bounds)
        rows.append(
            render_frame_row(
                cropped_bgr,
                frame_index=frame_index,
                height=args.height,
                width=args.width,
                resize_filters=resize_filters,
            )
        )

    sheet = build_contact_sheet(rows)
    image_path = output_dir / f"resize_filters_ep{args.episode:03d}_{args.height}x{args.width}.png"
    sheet.save(image_path)
    summary = {
        "dataset": SO101_BASE_SIM_PICKPLACE_DATASET_ID,
        "episode": args.episode,
        "frame_count": record.length,
        "selected_frame_indices": frame_indices,
        "crop_bounds": crop_bounds,
        "height": args.height,
        "width": args.width,
        "resize_filters": list(resize_filters),
        "image_path": str(image_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
