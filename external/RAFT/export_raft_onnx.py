#!/usr/bin/env python3
"""
Export RAFT model to ONNX for C++ runtime migration.

Usage:
  python3 export_raft_onnx.py --ckpt /path/to/raft-kitti.pth --output raft_kitti.onnx
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "core"))

from core.raft import RAFT  # type: ignore


class _Args:
    small = False
    mixed_precision = False
    alternate_corr = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="RAFT checkpoint .pth")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--width", type=int, default=1241)
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()

    model = RAFT(_Args())
    state = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    img1 = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)
    img2 = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (img1, img2, args.iters, True),
            args.output,
            opset_version=17,
            input_names=["image1", "image2", "iters", "test_mode"],
            output_names=["flow_low", "flow_up"],
            dynamic_axes={
                "image1": {0: "batch", 2: "height", 3: "width"},
                "image2": {0: "batch", 2: "height", 3: "width"},
                "flow_up": {0: "batch", 2: "height", 3: "width"},
            },
        )

    print(f"[OK] Exported ONNX: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
