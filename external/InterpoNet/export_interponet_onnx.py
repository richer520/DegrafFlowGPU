#!/usr/bin/env python3
"""
Placeholder exporter for InterpoNet TF1 checkpoint -> ONNX.

This script validates checkpoint presence and prints recommended conversion
steps using tf2onnx. InterpoNet is TF1-based and often needs manual graph
freezing before ONNX conversion.
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_prefix", required=True, help="Path prefix of TF1 ckpt")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--width", type=int, default=1241)
    args = parser.parse_args()

    meta_path = args.ckpt_prefix + ".meta"
    index_path = args.ckpt_prefix + ".index"

    if not os.path.exists(meta_path) or not os.path.exists(index_path):
        print("[ERROR] Missing TF1 checkpoint files:")
        print(f"  - {meta_path}")
        print(f"  - {index_path}")
        return 1

    print("[INFO] InterpoNet checkpoint found.")
    print("[INFO] Recommended conversion pipeline:")
    print("  1) Freeze TF1 graph to .pb")
    print("  2) Convert frozen graph with tf2onnx")
    print("")
    print("Example:")
    print("  python -m tf2onnx.convert --graphdef frozen_graph.pb "
          "--output", args.output, "--opset 17")
    print("")
    print("[NOTE] Input/output tensor names must be filled according to InterpoNet.py graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
