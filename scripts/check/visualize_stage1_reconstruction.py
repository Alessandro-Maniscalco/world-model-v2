"""Run Stage-1 checkpoint reconstruction on an Interactive World Sim episode.

source .venv/bin/activate
python scripts/check/visualize_stage1_reconstruction.py \
  --checkpoint outputs/stage1/<run_name>/checkpoints/last.pt \
  --output-dir /tmp/stage1_recon
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_model_v2.infer.reconstruct_episode import main


if __name__ == "__main__":
    main()
