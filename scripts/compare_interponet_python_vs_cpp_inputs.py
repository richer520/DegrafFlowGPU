#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


MAGIC = 202021.25


def frame_id(i: int) -> str:
    return f"{i:06d}"


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


def import_interponet_helpers(project_root: str):
    interpo_py = os.path.join(project_root, "external", "InterpoNet")
    io_utils = load_module_from_path("interpo_io_utils", os.path.join(interpo_py, "io_utils.py"))
    utils = load_module_from_path("interpo_utils", os.path.join(interpo_py, "utils.py"))
    return io_utils, utils


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


def read_dat(path: str, h: int, w: int) -> np.ndarray:
    data = np.fromfile(path, np.float32)
    if data.size != h * w:
        raise ValueError(f"unexpected .dat element count: {path}, got={data.size}, expected={h*w}")
    return data.reshape(h, w)


def compare_flow(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    epe = np.sqrt(np.sum(d * d, axis=2))
    return {
        "mean_epe": float(np.mean(epe)),
        "p90_epe": float(np.percentile(epe, 90)),
        "max_epe": float(np.max(epe)),
        "mean_abs_du": float(np.mean(np.abs(d[:, :, 0]))),
        "mean_abs_dv": float(np.mean(np.abs(d[:, :, 1]))),
    }


def compare_scalar(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> Dict[str, float]:
    d = np.abs(a - b)
    return {
        "mean_abs": float(np.mean(d)),
        "max_abs": float(np.max(d)),
        "count_gt_tol": float(np.sum(d > tol)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare Python-rebuilt InterpoNet inputs against C++ dumped TRT inputs."
    )
    ap.add_argument("--cpp-dump-dir", required=True, help="Directory with *_cpp_input_{image,mask,edges} dumps")
    ap.add_argument("--project-root", default="", help="Project root for importing external/InterpoNet helpers")
    ap.add_argument("--image-dir", required=True, help="Directory with 000000_10.png / 000000_11.png")
    ap.add_argument("--edges-dir", required=True, help="Directory with *_edges.dat")
    ap.add_argument("--matches-dir", required=True, help="Directory with sparse matches txt files")
    ap.add_argument("--matches-ba-dir", default="", help="Optional directory with B->A sparse matches txt files")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--downscale", type=int, default=8)
    ap.add_argument("--csv-out", default="")
    args = ap.parse_args()

    root = args.project_root or str(Path(__file__).resolve().parents[1])
    io_utils, utils = import_interponet_helpers(root)

    rows: List[Dict[str, float]] = []
    for i in range(args.start, args.start + args.count):
        fid = frame_id(i)
        cpp_image_path = os.path.join(args.cpp_dump_dir, f"{fid}_cpp_input_image.flo")
        cpp_mask_path = os.path.join(args.cpp_dump_dir, f"{fid}_cpp_input_mask.dat")
        cpp_edges_path = os.path.join(args.cpp_dump_dir, f"{fid}_cpp_input_edges.dat")
        img1_path = os.path.join(args.image_dir, f"{fid}_10.png")
        edges_path = first_existing(
            [
                os.path.join(args.edges_dir, f"{fid}_edges.dat"),
                os.path.join(args.edges_dir, f"{fid}_10_edges.dat"),
            ]
        )
        matches_path = first_existing(
            [
                os.path.join(args.matches_dir, f"{fid}_matches.txt"),
                os.path.join(args.matches_dir, f"cpp_trt_matches_frame_{fid}.txt"),
                os.path.join(args.matches_dir, f"{fid}.txt"),
            ]
        )
        matches_ba_path = None
        if args.matches_ba_dir:
            matches_ba_path = first_existing(
                [
                    os.path.join(args.matches_ba_dir, f"{fid}_matches.txt"),
                    os.path.join(args.matches_ba_dir, f"cpp_trt_matches_frame_{fid}.txt"),
                    os.path.join(args.matches_ba_dir, f"{fid}.txt"),
                ]
            )

        if not all(os.path.exists(p) for p in (cpp_image_path, cpp_mask_path, cpp_edges_path, img1_path)):
            continue
        if not edges_path or not matches_path:
            continue

        cpp_image = read_flo(cpp_image_path)
        low_h, low_w = cpp_image.shape[:2]
        cpp_mask = read_dat(cpp_mask_path, low_h, low_w)
        cpp_edges = read_dat(cpp_edges_path, low_h, low_w)

        img = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        py_edges = io_utils.load_edges_file(edges_path, width=w, height=h)
        py_image, py_mask = io_utils.load_matching_file(matches_path, width=w, height=h)
        py_image, py_mask, py_edges = utils.downscale_all(py_image, py_mask, py_edges, args.downscale)
        if matches_ba_path:
            py_image_ba, py_mask_ba = io_utils.load_matching_file(matches_ba_path, width=w, height=h)
            py_image_ba, py_mask_ba, _ = utils.downscale_all(py_image_ba, py_mask_ba, None, args.downscale)
            py_image, py_mask = utils.create_mean_map_ab_ba(
                py_image, py_mask, py_image_ba, py_mask_ba, args.downscale
            )

        if py_image.shape != cpp_image.shape or py_mask.shape != cpp_mask.shape or py_edges.shape != cpp_edges.shape:
            print(
                f"skip frame={fid} shape_mismatch "
                f"py_image={py_image.shape} cpp_image={cpp_image.shape} "
                f"py_mask={py_mask.shape} cpp_mask={cpp_mask.shape} "
                f"py_edges={py_edges.shape} cpp_edges={cpp_edges.shape}"
            )
            continue

        image_stats = compare_flow(py_image.astype(np.float32), cpp_image.astype(np.float32))
        mask_stats = compare_scalar(py_mask.astype(np.float32), cpp_mask.astype(np.float32))
        edges_stats = compare_scalar(py_edges.astype(np.float32), cpp_edges.astype(np.float32))
        rows.append(
            {
                "frame": float(i),
                "pixels": float(low_h * low_w),
                "image_mean_epe": image_stats["mean_epe"],
                "image_p90_epe": image_stats["p90_epe"],
                "image_max_epe": image_stats["max_epe"],
                "image_mean_abs_du": image_stats["mean_abs_du"],
                "image_mean_abs_dv": image_stats["mean_abs_dv"],
                "mask_mean_abs": mask_stats["mean_abs"],
                "mask_max_abs": mask_stats["max_abs"],
                "mask_count_gt_tol": mask_stats["count_gt_tol"],
                "edges_mean_abs": edges_stats["mean_abs"],
                "edges_max_abs": edges_stats["max_abs"],
                "edges_count_gt_tol": edges_stats["count_gt_tol"],
            }
        )

    print(f"frames_compared={len(rows)}")
    if not rows:
        print("No comparable frames.")
        return

    tot_px = sum(r["pixels"] for r in rows)
    def weighted(name: str) -> float:
        return sum(r[name] * r["pixels"] for r in rows) / tot_px

    print(f"weighted_image_mean_epe={weighted('image_mean_epe'):.6f}")
    print(f"weighted_image_mean_abs_du={weighted('image_mean_abs_du'):.6f}")
    print(f"weighted_image_mean_abs_dv={weighted('image_mean_abs_dv'):.6f}")
    print(f"weighted_mask_mean_abs={weighted('mask_mean_abs'):.6f}")
    print(f"weighted_edges_mean_abs={weighted('edges_mean_abs'):.6f}")
    print("worst_frames_by_image_epe:")
    rows = sorted(rows, key=lambda x: x["image_mean_epe"], reverse=True)
    for r in rows[:10]:
        print(
            f"frame={int(r['frame']):06d} "
            f"img_mean_epe={r['image_mean_epe']:.6f} img_p90={r['image_p90_epe']:.6f} img_max={r['image_max_epe']:.6f} "
            f"mask_mean_abs={r['mask_mean_abs']:.6f} mask_gt_tol={int(r['mask_count_gt_tol'])} "
            f"edges_mean_abs={r['edges_mean_abs']:.6f} edges_gt_tol={int(r['edges_count_gt_tol'])}"
        )

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "frame",
                    "image_mean_epe",
                    "image_p90_epe",
                    "image_max_epe",
                    "image_mean_abs_du",
                    "image_mean_abs_dv",
                    "mask_mean_abs",
                    "mask_max_abs",
                    "mask_count_gt_tol",
                    "edges_mean_abs",
                    "edges_max_abs",
                    "edges_count_gt_tol",
                    "pixels",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        int(r["frame"]),
                        f"{r['image_mean_epe']:.8f}",
                        f"{r['image_p90_epe']:.8f}",
                        f"{r['image_max_epe']:.8f}",
                        f"{r['image_mean_abs_du']:.8f}",
                        f"{r['image_mean_abs_dv']:.8f}",
                        f"{r['mask_mean_abs']:.8f}",
                        f"{r['mask_max_abs']:.8f}",
                        int(r["mask_count_gt_tol"]),
                        f"{r['edges_mean_abs']:.8f}",
                        f"{r['edges_max_abs']:.8f}",
                        int(r["edges_count_gt_tol"]),
                        int(r["pixels"]),
                    ]
                )
        print(f"csv_saved={args.csv_out}")


if __name__ == "__main__":
    main()
