#!/usr/bin/env python3
"""
Export InterpoNet TF1 checkpoint to ONNX.

Pipeline:
1) Build TF1 graph from model.getNetwork(...)
2) Restore variables from ckpt
3) Freeze graph (variables -> const)
4) Convert frozen graph to ONNX with tf2onnx
"""

import argparse
import os
import sys

import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import model  # type: ignore  # noqa: E402


def _align_downscaled_size(height: int, width: int, downscale: int) -> tuple:
    if downscale <= 0:
        raise ValueError("downscale must be positive")
    h_trim = height - (height % downscale)
    w_trim = width - (width % downscale)
    if h_trim <= 0 or w_trim <= 0:
        raise ValueError("invalid downscaled shape; check height/width/downscale")
    return h_trim // downscale, w_trim // downscale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_prefix", required=True, help="Path prefix of TF1 ckpt")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=375, help="Original image height")
    parser.add_argument("--width", type=int, default=1242, help="Original image width")
    parser.add_argument("--downscale", type=int, default=8, help="InterpoNet downscale factor")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    meta_path = args.ckpt_prefix + ".meta"
    index_path = args.ckpt_prefix + ".index"
    data_path = args.ckpt_prefix + ".data-00000-of-00001"

    missing = [p for p in (meta_path, index_path, data_path) if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing TF1 checkpoint files:")
        for p in missing:
            print(f"  - {p}")
        return 1

    try:
        import tf2onnx  # type: ignore
    except Exception as e:
        print("[ERROR] tf2onnx is required but not available.")
        print("[HINT] Install with: python3 -m pip install tf2onnx")
        print(f"[DETAIL] {e}")
        return 2

    tf.compat.v1.disable_eager_execution()
    down_h, down_w = _align_downscaled_size(args.height, args.width, args.downscale)

    graph = tf.Graph()
    with graph.as_default():
        image_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, down_h, down_w, 2), name="image_ph"
        )
        mask_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, down_h, down_w, 1), name="mask_ph"
        )
        edges_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, down_h, down_w, 1), name="edges_ph"
        )
        prediction = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=0)

    with tf.compat.v1.Session(graph=graph) as sess:
        saver.restore(sess, args.ckpt_prefix)
        output_op_name = prediction.op.name
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_op_name]
        )

    input_names = ["image_ph:0", "mask_ph:0", "edges_ph:0"]
    output_names = [f"{output_op_name}:0"]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tf2onnx.convert.from_graph_def(
        frozen_graph,
        input_names=input_names,
        output_names=output_names,
        opset=args.opset,
        output_path=args.output,
    )

    print("[OK] Exported InterpoNet ONNX:", args.output)
    print(f"[OK] Input shape: batchx{down_h}x{down_w}x(2/1/1), downscale={args.downscale}")
    print("[OK] Input tensors:", ", ".join(input_names))
    print("[OK] Output tensor:", output_names[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
