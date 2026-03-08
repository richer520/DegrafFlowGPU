#!/usr/bin/env python3
import argparse
import csv
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


MAGIC = 202021.25


def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - MAGIC) > 1e-4:
            raise ValueError(f"invalid .flo magic: {path}")
        w = struct.unpack("<i", f.read(4))[0]
        h = struct.unpack("<i", f.read(4))[0]
        data = np.fromfile(f, np.float32, count=w * h * 2)
    if data.size != w * h * 2:
        raise ValueError(f"truncated .flo: {path}")
    return data.reshape(h, w, 2)


def frame_id(i: int) -> str:
    return f"{i:06d}"


def to_hwc2(out: np.ndarray) -> np.ndarray:
    if out.ndim != 4 or out.shape[0] != 1:
        raise ValueError(f"unexpected output shape: {out.shape}")
    x = out[0]
    if x.ndim != 3:
        raise ValueError(f"unexpected output rank after squeeze: {x.shape}")
    # NHWC: [1,H,W,2]
    if x.shape[-1] == 2:
        return x.astype(np.float32, copy=False)
    # NCHW: [1,2,H,W]
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare InterpoNet ONNX output with TRT low-res .flo dumps.")
    ap.add_argument("--onnx", required=True, help="Path to InterpoNet ONNX")
    ap.add_argument("--project-root", default="", help="Project root for importing external/InterpoNet utils")
    ap.add_argument("--image-dir", required=True, help="KITTI image_2 directory")
    ap.add_argument("--edges-dir", required=True, help="Directory with *_edges.dat")
    ap.add_argument("--matches-dir", required=True, help="Directory with cpp_trt_matches_frame_XXXXXX.txt")
    ap.add_argument("--trt-low-dir", required=True, help="Directory with XXXXX_trt_low.flo")
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
    interpo_py = os.path.join(root, "external", "InterpoNet")
    if interpo_py not in sys.path:
        sys.path.insert(0, interpo_py)
    import io_utils  # type: ignore
    import utils  # type: ignore

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_name = sess.get_outputs()[0].name

    rows: List[Dict[str, float]] = []
    for i in range(args.start, args.start + args.count):
        fid = frame_id(i)
        img1 = os.path.join(args.image_dir, f"{fid}_10.png")
        edges = os.path.join(args.edges_dir, f"{fid}_edges.dat")
        matches = os.path.join(args.matches_dir, f"cpp_trt_matches_frame_{fid}.txt")
        trt_low = os.path.join(args.trt_low_dir, f"{fid}_trt_low.flo")
        if not (os.path.exists(img1) and os.path.exists(edges) and os.path.exists(matches) and os.path.exists(trt_low)):
            continue

        im = cv2.imread(img1, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]

        edges_arr = io_utils.load_edges_file(edges, width=w, height=h)
        img_arr, mask_arr = io_utils.load_matching_file(matches, width=w, height=h)
        img_ds, mask_ds, edges_ds = utils.downscale_all(img_arr, mask_arr, edges_arr, args.downscale)

        inp = {
            input_names[0]: np.expand_dims(img_ds.astype(np.float32), axis=0),
            input_names[1]: np.reshape(mask_ds.astype(np.float32), (1, mask_ds.shape[0], mask_ds.shape[1], 1)),
            input_names[2]: np.expand_dims(np.expand_dims(edges_ds.astype(np.float32), axis=0), axis=3),
        }
        out = sess.run([output_name], inp)[0]
        onnx_flow = to_hwc2(out)
        trt_flow = read_flo(trt_low)
        if onnx_flow.shape != trt_flow.shape:
            continue

        m = compare(onnx_flow, trt_flow)
        m["frame"] = float(i)
        m["pixels"] = float(onnx_flow.shape[0] * onnx_flow.shape[1])
        rows.append(m)

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

