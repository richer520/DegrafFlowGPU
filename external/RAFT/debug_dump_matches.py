#!/usr/bin/env python3
"""
Dump sparse RAFT matches for one frame pair (Python baseline).
Output format: src_x src_y dst_x dst_y
"""

import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from degraf_raft_matcher import load_model, load_image, load_points, sample_flow_at_points  # noqa: E402
from utils.utils import InputPadder  # noqa: E402


def save_matches(matches: np.ndarray, output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# src_x src_y dst_x dst_y\n")
        for row in matches:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Python RAFT sparse matches for one frame pair")
    parser.add_argument("--model", required=True, help="RAFT checkpoint path (.pth)")
    parser.add_argument("--image1", required=True, help="First frame path")
    parser.add_argument("--image2", required=True, help="Second frame path")
    parser.add_argument("--points", required=True, help="Sparse points txt (x y per line)")
    parser.add_argument("--output", required=True, help="Output matches file")
    parser.add_argument("--iters", type=int, default=12, help="RAFT iterations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, small=False, mixed_precision=False, alternate_corr=False)
    image1 = load_image(args.image1)
    image2 = load_image(args.image2)
    points = load_points(args.points)
    if points.size == 0:
        save_matches(np.zeros((0, 4), dtype=np.float32), args.output)
        print(f"Saved empty matches: {args.output}")
        return

    img_h, img_w = image1.shape[2], image1.shape[3]
    with torch.no_grad():
        padder = InputPadder(image1.shape, mode="kitti")
        image1_pad, image2_pad = padder.pad(image1, image2)
        _, flow_up = model(image1_pad, image2_pad, iters=args.iters, test_mode=True)
        matches = sample_flow_at_points(flow_up, points, img_w, img_h)
    save_matches(matches, args.output)
    print(f"[OK] device={device} matches={matches.shape[0]} output={args.output}")


if __name__ == "__main__":
    main()
