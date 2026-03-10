#!/usr/bin/env python3
import argparse
import csv
import glob
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def frame_id(i: int) -> str:
    return f"{i:06d}"


def to_hwc2(out: np.ndarray) -> np.ndarray:
    if out.ndim != 4 or out.shape[0] != 1:
        raise ValueError(f"unexpected output shape: {out.shape}")
    x = out[0]
    if x.ndim != 3:
        raise ValueError(f"unexpected output rank after squeeze: {x.shape}")
    if x.shape[-1] == 2:
        return x.astype(np.float32, copy=False)
    if x.shape[0] == 2:
        return np.transpose(x, (1, 2, 0)).astype(np.float32, copy=False)
    raise ValueError(f"cannot infer output layout: {out.shape}")


def compare(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    epe = np.sqrt(np.sum(d * d, axis=2))
    return {
        "mean_epe": float(np.mean(epe)),
        "p90_epe": float(np.percentile(epe, 90)),
        "max_epe": float(np.max(epe)),
        "mean_abs_du": float(np.mean(np.abs(d[:, :, 0]))),
        "mean_abs_dv": float(np.mean(np.abs(d[:, :, 1]))),
    }


def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_module_from_path(module_name: str, module_path: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_components_exist(prefix: str) -> bool:
    return (
        os.path.exists(prefix + ".index")
        and os.path.exists(prefix + ".meta")
        and bool(glob.glob(prefix + ".data-*"))
    )


def resolve_checkpoint_prefix(raw_path: str) -> str:
    raw_path = os.path.expanduser(raw_path.strip())
    raw_path = raw_path.rstrip("/\\")
    candidates: List[str] = []

    def add_candidate(p: str) -> None:
        if p and p not in candidates:
            candidates.append(p)

    add_candidate(raw_path)
    for suffix in (".meta", ".index", ".data-00000-of-00001"):
        if raw_path.endswith(suffix):
            add_candidate(raw_path[: -len(suffix)])

    if os.path.isdir(raw_path):
        for index_path in sorted(glob.glob(os.path.join(raw_path, "*.ckpt.index"))):
            add_candidate(index_path[: -len(".index")])

    parent_dir = raw_path if os.path.isdir(raw_path) else os.path.dirname(raw_path) or "."
    base_name = os.path.basename(raw_path)
    if os.path.isdir(parent_dir):
        for index_path in sorted(glob.glob(os.path.join(parent_dir, "*.ckpt.index"))):
            prefix = index_path[: -len(".index")]
            if base_name and base_name in os.path.basename(prefix):
                add_candidate(prefix)

    for prefix in candidates:
        if checkpoint_components_exist(prefix):
            print(f"[INFO] Resolved checkpoint prefix: {prefix}")
            return prefix

    discovered = sorted(glob.glob(os.path.join(parent_dir, "*"))) if os.path.isdir(parent_dir) else []
    discovered_text = "\n".join(f"  - {p}" for p in discovered[:50]) or "  - <none>"
    raise FileNotFoundError(
        "could not resolve a valid TF checkpoint prefix.\n"
        f"input: {raw_path}\n"
        f"searched directory: {parent_dir}\n"
        f"directory contents:\n{discovered_text}"
    )


def import_interponet_modules(project_root: str):
    interpo_py = os.path.join(project_root, "external", "InterpoNet")
    io_utils_path = os.path.join(interpo_py, "io_utils.py")
    utils_path = os.path.join(interpo_py, "utils.py")
    model_path = os.path.join(interpo_py, "model.py")
    missing = [p for p in (io_utils_path, utils_path, model_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "missing InterpoNet helper files:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    import tensorflow.compat.v1 as tf  # type: ignore

    tf.disable_v2_behavior()

    if not hasattr(tf, "contrib"):
        tf.contrib = types.SimpleNamespace()  # type: ignore[attr-defined]
    if not hasattr(tf.contrib, "layers"):
        tf.contrib.layers = types.SimpleNamespace()  # type: ignore[attr-defined]
    if not hasattr(tf.contrib.layers, "xavier_initializer_conv2d"):
        tf.contrib.layers.xavier_initializer_conv2d = tf.glorot_uniform_initializer  # type: ignore[attr-defined]

    sys.modules["tensorflow"] = tf
    io_utils = load_module_from_path("io_utils", io_utils_path)
    utils = load_module_from_path("utils", utils_path)
    model = load_module_from_path("model", model_path)
    return io_utils, utils, tf, model


class TFRunnerCache:
    def __init__(self, tf_mod, model_mod, ckpt_prefix: str):
        self.tf = tf_mod
        self.model = model_mod
        self.ckpt_prefix = ckpt_prefix
        self.runners: Dict[Tuple[int, int], Tuple[object, Dict[str, object], object]] = {}

    def _build_runner(self, low_h: int, low_w: int):
        graph = self.tf.Graph()
        with graph.as_default():
            image_ph = self.tf.placeholder(
                self.tf.float32, shape=(None, low_h, low_w, 2), name="image_ph"
            )
            mask_ph = self.tf.placeholder(
                self.tf.float32, shape=(None, low_h, low_w, 1), name="mask_ph"
            )
            edges_ph = self.tf.placeholder(
                self.tf.float32, shape=(None, low_h, low_w, 1), name="edges_ph"
            )
            prediction = self.model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False)
            saver = self.tf.train.Saver(self.tf.global_variables(), max_to_keep=0)

        config = self.tf.ConfigProto()
        config.allow_soft_placement = True
        if hasattr(config, "gpu_options"):
            config.gpu_options.allow_growth = True

        sess = self.tf.Session(graph=graph, config=config)
        with graph.as_default():
            saver.restore(sess, self.ckpt_prefix)

        placeholders = {
            "image_ph": image_ph,
            "mask_ph": mask_ph,
            "edges_ph": edges_ph,
        }
        return sess, placeholders, prediction

    def run(self, img_ds: np.ndarray, mask_ds: np.ndarray, edges_ds: np.ndarray) -> np.ndarray:
        low_h, low_w = img_ds.shape[:2]
        key = (low_h, low_w)
        if key not in self.runners:
            self.runners[key] = self._build_runner(low_h, low_w)

        sess, placeholders, prediction = self.runners[key]
        out = sess.run(
            prediction,
            feed_dict={
                placeholders["image_ph"]: np.expand_dims(img_ds.astype(np.float32), axis=0),
                placeholders["mask_ph"]: np.reshape(
                    mask_ds.astype(np.float32), (1, low_h, low_w, 1)
                ),
                placeholders["edges_ph"]: np.expand_dims(
                    np.expand_dims(edges_ds.astype(np.float32), axis=0), axis=3
                ),
            },
        )
        return out[0].astype(np.float32, copy=False)

    def close(self) -> None:
        for sess, _, _ in self.runners.values():
            try:
                sess.close()
            except Exception:
                pass
        self.runners.clear()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare InterpoNet TF checkpoint output with ONNXRuntime output on identical low-res inputs."
    )
    ap.add_argument("--ckpt-prefix", required=True, help="Path prefix of TF1 checkpoint")
    ap.add_argument("--onnx", required=True, help="Path to InterpoNet ONNX")
    ap.add_argument("--project-root", default="", help="Project root for importing external/InterpoNet modules")
    ap.add_argument("--image-dir", required=True, help="KITTI image_2 directory")
    ap.add_argument("--edges-dir", required=True, help="Directory with *_edges.dat")
    ap.add_argument("--matches-dir", required=True, help="Directory with sparse matches txt files")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--csv-out", default="")
    args = ap.parse_args()

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        raise RuntimeError("onnxruntime is required. Please install it first.") from e

    root = args.project_root or str(Path(__file__).resolve().parents[1])
    ckpt_prefix = resolve_checkpoint_prefix(args.ckpt_prefix)
    io_utils, utils, tf_mod, model_mod = import_interponet_modules(root)
    tf_runner = TFRunnerCache(tf_mod, model_mod, ckpt_prefix)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name

    rows: List[Dict[str, float]] = []
    try:
        for i in range(args.start, args.start + args.count):
            fid = frame_id(i)
            img1 = os.path.join(args.image_dir, f"{fid}_10.png")
            edges = first_existing(
                [
                    os.path.join(args.edges_dir, f"{fid}_edges.dat"),
                    os.path.join(args.edges_dir, f"{fid}_10_edges.dat"),
                ]
            )
            matches = first_existing(
                [
                    os.path.join(args.matches_dir, f"{fid}_matches.txt"),
                    os.path.join(args.matches_dir, f"cpp_trt_matches_frame_{fid}.txt"),
                    os.path.join(args.matches_dir, f"{fid}.txt"),
                ]
            )
            if not img1 or not edges or not matches or not os.path.exists(img1):
                continue

            im = cv2.imread(img1, cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]

            edges_arr = io_utils.load_edges_file(edges, width=w, height=h)
            img_arr, mask_arr = io_utils.load_matching_file(matches, width=w, height=h)
            img_ds, mask_ds, edges_ds = utils.downscale_all(img_arr, mask_arr, edges_arr, args.downscale)

            tf_flow = tf_runner.run(img_ds, mask_ds, edges_ds)
            inp = {
                input_names[0]: np.expand_dims(img_ds.astype(np.float32), axis=0),
                input_names[1]: np.reshape(
                    mask_ds.astype(np.float32), (1, mask_ds.shape[0], mask_ds.shape[1], 1)
                ),
                input_names[2]: np.expand_dims(np.expand_dims(edges_ds.astype(np.float32), axis=0), axis=3),
            }
            onnx_flow = to_hwc2(sess.run([output_name], inp)[0])
            if tf_flow.shape != onnx_flow.shape:
                continue

            m = compare(tf_flow, onnx_flow)
            m["frame"] = float(i)
            m["pixels"] = float(tf_flow.shape[0] * tf_flow.shape[1])
            rows.append(m)
    finally:
        tf_runner.close()

    print(f"frames_compared={len(rows)}")
    if not rows:
        print("No comparable frames.")
        return

    tot_px = sum(r["pixels"] for r in rows)
    w_epe = sum(r["mean_epe"] * r["pixels"] for r in rows) / tot_px
    w_du = sum(r["mean_abs_du"] * r["pixels"] for r in rows) / tot_px
    w_dv = sum(r["mean_abs_dv"] * r["pixels"] for r in rows) / tot_px
    print(f"weighted_mean_epe={w_epe:.6f}")
    print(f"weighted_mean_abs_du={w_du:.6f}")
    print(f"weighted_mean_abs_dv={w_dv:.6f}")

    rows = sorted(rows, key=lambda x: x["mean_epe"], reverse=True)
    print("worst_frames:")
    for r in rows[:10]:
        print(
            f"frame={int(r['frame']):06d} mean_epe={r['mean_epe']:.6f} "
            f"p90={r['p90_epe']:.6f} max={r['max_epe']:.6f}"
        )

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            wri = csv.writer(f)
            wri.writerow(["frame", "mean_epe", "p90_epe", "max_epe", "mean_abs_du", "mean_abs_dv", "pixels"])
            for r in rows:
                wri.writerow([
                    int(r["frame"]),
                    f"{r['mean_epe']:.8f}",
                    f"{r['p90_epe']:.8f}",
                    f"{r['max_epe']:.8f}",
                    f"{r['mean_abs_du']:.8f}",
                    f"{r['mean_abs_dv']:.8f}",
                    int(r["pixels"]),
                ])
        print(f"csv_saved={args.csv_out}")


if __name__ == "__main__":
    main()
