#!/usr/bin/env python3
import argparse
import math
from typing import List, Tuple


Point = Tuple[float, float, float, float]  # src_x, src_y, dst_x, dst_y


def load_matches(path: str) -> List[Point]:
    rows: List[Point] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            sx, sy, dx, dy = map(float, parts[:4])
            rows.append((sx, sy, dx, dy))
    return rows


def mean_abs(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(abs(v) for v in values) / len(values)


def compare(cpp_rows: List[Point], py_rows: List[Point]) -> None:
    n = min(len(cpp_rows), len(py_rows))
    if n == 0:
        print("No comparable rows (empty file).")
        return

    if len(cpp_rows) != len(py_rows):
        print(f"[WARN] count mismatch: cpp={len(cpp_rows)}, python={len(py_rows)}, compare_first={n}")

    dx_err: List[float] = []
    dy_err: List[float] = []
    dst_x_err: List[float] = []
    dst_y_err: List[float] = []
    epe: List[float] = []

    flow_cpp_u: List[float] = []
    flow_cpp_v: List[float] = []
    flow_py_u: List[float] = []
    flow_py_v: List[float] = []

    for i in range(n):
        csx, csy, cdx, cdy = cpp_rows[i]
        psx, psy, pdx, pdy = py_rows[i]
        if abs(csx - psx) > 1e-4 or abs(csy - psy) > 1e-4:
            # src points differ means upstream point ordering changed; still compare by row index.
            pass

        dxe = cdx - pdx
        dye = cdy - pdy
        dst_x_err.append(dxe)
        dst_y_err.append(dye)
        epe.append(math.sqrt(dxe * dxe + dye * dye))

        cdu = cdx - csx
        cdv = cdy - csy
        pdu = pdx - psx
        pdv = pdy - psy

        flow_cpp_u.append(cdu)
        flow_cpp_v.append(cdv)
        flow_py_u.append(pdu)
        flow_py_v.append(pdv)

        dx_err.append(cdu - pdu)
        dy_err.append(cdv - pdv)

    # Heuristic checks for common bugs
    normal = (mean_abs(dx_err), mean_abs(dy_err))
    swap = (
        mean_abs([flow_cpp_u[i] - flow_py_v[i] for i in range(n)]),
        mean_abs([flow_cpp_v[i] - flow_py_u[i] for i in range(n)]),
    )
    neg = (
        mean_abs([flow_cpp_u[i] + flow_py_u[i] for i in range(n)]),
        mean_abs([flow_cpp_v[i] + flow_py_v[i] for i in range(n)]),
    )
    swap_neg = (
        mean_abs([flow_cpp_u[i] + flow_py_v[i] for i in range(n)]),
        mean_abs([flow_cpp_v[i] + flow_py_u[i] for i in range(n)]),
    )

    print("=== Sparse Match Compare ===")
    print(f"rows_compared: {n}")
    print(f"mean_abs_dst_error_x: {mean_abs(dst_x_err):.6f}")
    print(f"mean_abs_dst_error_y: {mean_abs(dst_y_err):.6f}")
    print(f"mean_epe_dst: {sum(epe) / n:.6f}")
    print("--- flow delta checks ---")
    print(f"normal (u,v): ({normal[0]:.6f}, {normal[1]:.6f})")
    print(f"uv_swap      : ({swap[0]:.6f}, {swap[1]:.6f})")
    print(f"sign_flip    : ({neg[0]:.6f}, {neg[1]:.6f})")
    print(f"swap+flip    : ({swap_neg[0]:.6f}, {swap_neg[1]:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sparse match dumps from C++ TRT and Python RAFT")
    parser.add_argument("--cpp", required=True, help="Path to cpp_trt_matches_frame_XXXXXX.txt")
    parser.add_argument("--py", required=True, help="Path to python_raft_matches_frame_XXXXXX.txt")
    args = parser.parse_args()

    cpp_rows = load_matches(args.cpp)
    py_rows = load_matches(args.py)
    compare(cpp_rows, py_rows)


if __name__ == "__main__":
    main()
