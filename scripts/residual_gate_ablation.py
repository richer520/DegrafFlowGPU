#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


SCENEFLOW_RE = re.compile(
    r"SceneFlow averages for .*?\| EPE3d:\s*([0-9.]+)\s*\| AccS:\s*([0-9.]+)\s*\| "
    r"AccR:\s*([0-9.]+)\s*\| Outlier:\s*([0-9.]+)\s*\| Time:\s*([0-9.]+)\s*ms"
)


def run_case(
    project_root: Path, name: str, start: int, count: int, methods: str, env_overrides: Dict[str, str]
) -> Tuple[str, Dict[str, float]]:
    cmd = [
        "bash",
        "run_gpu_pipeline.sh",
        "--start",
        str(start),
        "--count",
        str(count),
        "--methods",
        methods,
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    print(f"[RUN] {name} -> {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout
    metrics = None
    for line in output.splitlines():
        m = SCENEFLOW_RE.search(line)
        if m:
            metrics = {
                "EPE3d": float(m.group(1)),
                "AccS": float(m.group(2)),
                "AccR": float(m.group(3)),
                "Outlier": float(m.group(4)),
                "TimeMs": float(m.group(5)),
            }
    if metrics is None:
        print(output)
        raise RuntimeError("Could not find SceneFlow averages in output.")

    if proc.returncode != 0:
        print(
            f"[WARN] {name} exited with code {proc.returncode}, "
            "but metrics were parsed successfully."
        )
        metrics["ExitCode"] = float(proc.returncode)
    else:
        metrics["ExitCode"] = 0.0
    return output, metrics


def append_markdown_row(
    metrics_file: Path,
    date_str: str,
    commit: str,
    machine: str,
    start: int,
    count: int,
    methods: str,
    thresh: float,
    m_off: Dict[str, float],
    m_full: Dict[str, float],
    m_gate: Dict[str, float],
) -> None:
    section_header = "## 8) variational residual gate 对照实验"
    table_header = (
        "| 日期 | commit | 机器 | 配置 | EPE3d(off/full/gate) | Time(ms,off/full/gate) | "
        "AccS(off/full/gate) | AccR(off/full/gate) | Outlier(off/full/gate) | 结论 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    config = f"start={start},count={count},methods={methods},gate_thresh={thresh}"

    gain_full = m_off["EPE3d"] - m_full["EPE3d"]
    gain_gate = m_off["EPE3d"] - m_gate["EPE3d"]
    cost_full = m_full["TimeMs"] - m_off["TimeMs"]
    cost_gate = m_gate["TimeMs"] - m_off["TimeMs"]
    conclusion = (
        "gate 保留主要精度收益且明显降时延"
        if (gain_gate > 0 and cost_gate < cost_full)
        else "gate 收益有限，需调阈值/参数"
    )
    conclusion += (
        f" (exit off/full/gate={int(m_off['ExitCode'])}/{int(m_full['ExitCode'])}/{int(m_gate['ExitCode'])})"
    )

    row = (
        f"| {date_str} | {commit} | {machine} | `{config}` | "
        f"{m_off['EPE3d']:.4f}/{m_full['EPE3d']:.4f}/{m_gate['EPE3d']:.4f} | "
        f"{m_off['TimeMs']:.2f}/{m_full['TimeMs']:.2f}/{m_gate['TimeMs']:.2f} | "
        f"{m_off['AccS']:.2f}/{m_full['AccS']:.2f}/{m_gate['AccS']:.2f} | "
        f"{m_off['AccR']:.2f}/{m_full['AccR']:.2f}/{m_gate['AccR']:.2f} | "
        f"{m_off['Outlier']:.2f}/{m_full['Outlier']:.2f}/{m_gate['Outlier']:.2f} | "
        f"{conclusion} |\n"
    )

    text = metrics_file.read_text(encoding="utf-8") if metrics_file.exists() else ""
    if section_header not in text:
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n{section_header}\n\n{table_header}"
    elif table_header not in text:
        idx = text.find(section_header) + len(section_header)
        text = text[:idx] + "\n\n" + table_header + text[idx:]

    text += row
    metrics_file.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run residual-gate ablation: off vs full variational vs gated variational."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metrics-file", default="teaching_learning_center/METRICS_BASELINE.md")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--methods", default="degraf_flow_interponet")
    parser.add_argument("--machine", default="cloud")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--gate-thresh", type=float, default=3.0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    metrics_file = (project_root / args.metrics_file).resolve()
    logs_dir = project_root / "logs" / "ablation"
    logs_dir.mkdir(parents=True, exist_ok=True)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(project_root), text=True
    ).strip()

    out_off, m_off = run_case(
        project_root,
        "variational_off",
        args.start,
        args.count,
        args.methods,
        {
            "DEGRAF_ENABLE_VARIATIONAL": "0",
            "DEGRAF_VARIATIONAL_RESIDUAL_GATE": "0",
        },
    )
    out_full, m_full = run_case(
        project_root,
        "variational_full",
        args.start,
        args.count,
        args.methods,
        {
            "DEGRAF_ENABLE_VARIATIONAL": "1",
            "DEGRAF_VARIATIONAL_RESIDUAL_GATE": "0",
        },
    )
    out_gate, m_gate = run_case(
        project_root,
        "variational_gate",
        args.start,
        args.count,
        args.methods,
        {
            "DEGRAF_ENABLE_VARIATIONAL": "1",
            "DEGRAF_VARIATIONAL_RESIDUAL_GATE": "1",
            "DEGRAF_VARIATIONAL_RESIDUAL_THRESH": str(args.gate_thresh),
        },
    )

    (logs_dir / f"{args.date}_residual_gate_off.log").write_text(out_off, encoding="utf-8")
    (logs_dir / f"{args.date}_residual_gate_full.log").write_text(out_full, encoding="utf-8")
    (logs_dir / f"{args.date}_residual_gate_gate.log").write_text(out_gate, encoding="utf-8")

    append_markdown_row(
        metrics_file=metrics_file,
        date_str=args.date,
        commit=git_commit,
        machine=args.machine,
        start=args.start,
        count=args.count,
        methods=args.methods,
        thresh=args.gate_thresh,
        m_off=m_off,
        m_full=m_full,
        m_gate=m_gate,
    )

    print("[OK] Residual-gate ablation done.")
    print(f"[OK] off  metrics: {m_off}")
    print(f"[OK] full metrics: {m_full}")
    print(f"[OK] gate metrics: {m_gate}")
    print(f"[OK] Appended to: {metrics_file}")
    print(f"[OK] Logs saved in: {logs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
