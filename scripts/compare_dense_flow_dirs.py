#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


MAGIC = 202021.25
FRAME_RE = re.compile(r"(\d{6})")


def read_flo(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - MAGIC) > 1e-4:
            raise ValueError(f"invalid .flo magic in {path}")
        width = struct.unpack("<i", f.read(4))[0]
        height = struct.unpack("<i", f.read(4))[0]
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid shape in {path}: {width}x{height}")
        data = np.fromfile(f, np.float32, count=width * height * 2)
        if data.size != width * height * 2:
            raise ValueError(f"truncated .flo file: {path}")
    return data.reshape(height, width, 2)


def frame_id_from_path(path: str) -> Optional[str]:
    name = os.path.basename(path)
    m = FRAME_RE.search(name)
    if not m:
        return None
    return m.group(1)


def build_index(file_paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(file_paths):
        fid = frame_id_from_path(p)
        if fid is None:
            continue
        out[fid] = p
    return out


def compute_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = a - b
    epe = np.sqrt(np.sum(diff * diff, axis=2))
    return {
        "mean_epe": float(np.mean(epe)),
        "p90_epe": float(np.percentile(epe, 90)),
        "max_epe": float(np.max(epe)),
        "mean_abs_du": float(np.mean(np.abs(diff[:, :, 0]))),
        "mean_abs_dv": float(np.mean(np.abs(diff[:, :, 1]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dense .flo outputs between TRT and Python InterpoNet directories."
    )
    parser.add_argument("--trt-dir", required=True, help="Directory containing TRT .flo files")
    parser.add_argument("--py-dir", required=True, help="Directory containing Python .flo files")
    parser.add_argument("--trt-glob", default="*_trt_dense.flo", help="Glob under --trt-dir")
    parser.add_argument("--py-glob", default="*.flo", help="Glob under --py-dir")
    parser.add_argument("--top-k", type=int, default=20, help="Print worst K frames")
    parser.add_argument("--csv-out", default="", help="Optional output CSV path")
    args = parser.parse_args()

    trt_files = glob.glob(os.path.join(args.trt_dir, args.trt_glob))
    py_files = glob.glob(os.path.join(args.py_dir, args.py_glob))
    trt_map = build_index(trt_files)
    py_map = build_index(py_files)
    common_ids = sorted(set(trt_map.keys()) & set(py_map.keys()))

    print(f"trt_files={len(trt_files)} indexed={len(trt_map)}")
    print(f"py_files={len(py_files)} indexed={len(py_map)}")
    print(f"common_frames={len(common_ids)}")

    if not common_ids:
        print("No common frame ids found.")
        return

    rows: List[Dict[str, float]] = []
    skipped = 0
    for fid in common_ids:
        trt_path = trt_map[fid]
        py_path = py_map[fid]
        try:
            trt_flow = read_flo(trt_path)
            py_flow = read_flo(py_path)
        except Exception as e:
            skipped += 1
            print(f"[WARN] skip frame {fid}: {e}")
            continue

        if trt_flow.shape != py_flow.shape:
            skipped += 1
            print(f"[WARN] skip frame {fid}: shape mismatch {trt_flow.shape} vs {py_flow.shape}")
            continue

        m = compute_metrics(trt_flow, py_flow)
        m["frame"] = float(fid)
        m["pixels"] = float(trt_flow.shape[0] * trt_flow.shape[1])
        rows.append(m)

    if not rows:
        print("No comparable frames after filtering.")
        return

    # Global weighted mean by pixels (more stable).
    total_px = sum(r["pixels"] for r in rows)
    w_mean_epe = sum(r["mean_epe"] * r["pixels"] for r in rows) / total_px
    w_mean_du = sum(r["mean_abs_du"] * r["pixels"] for r in rows) / total_px
    w_mean_dv = sum(r["mean_abs_dv"] * r["pixels"] for r in rows) / total_px

    print("=== Dense Flow Compare Summary ===")
    print(f"frames_compared={len(rows)} skipped={skipped}")
    print(f"weighted_mean_epe={w_mean_epe:.6f}")
    print(f"weighted_mean_abs_du={w_mean_du:.6f}")
    print(f"weighted_mean_abs_dv={w_mean_dv:.6f}")

    rows_sorted = sorted(rows, key=lambda x: x["mean_epe"], reverse=True)
    k = min(max(args.top_k, 1), len(rows_sorted))
    print(f"=== Worst {k} Frames by mean_epe ===")
    for r in rows_sorted[:k]:
        print(
            f"frame={int(r['frame']):06d} "
            f"mean_epe={r['mean_epe']:.6f} p90={r['p90_epe']:.6f} max={r['max_epe']:.6f} "
            f"mean_abs_du={r['mean_abs_du']:.6f} mean_abs_dv={r['mean_abs_dv']:.6f}"
        )

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "mean_epe", "p90_epe", "max_epe", "mean_abs_du", "mean_abs_dv", "pixels"])
            for r in rows_sorted:
                writer.writerow(
                    [
                        int(r["frame"]),
                        f"{r['mean_epe']:.8f}",
                        f"{r['p90_epe']:.8f}",
                        f"{r['max_epe']:.8f}",
                        f"{r['mean_abs_du']:.8f}",
                        f"{r['mean_abs_dv']:.8f}",
                        int(r["pixels"]),
                    ]
                )
        print(f"csv_saved={args.csv_out}")


if __name__ == "__main__":
    main()

