#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path


SCENEFLOW_RE = re.compile(
    r"SceneFlow averages for .*?\| EPE3d:\s*([0-9.]+)\s*\| AccS:\s*([0-9.]+)\s*\| "
    r"AccR:\s*([0-9.]+)\s*\| Outlier:\s*([0-9.]+)\s*\| Time:\s*([0-9.]+)\s*ms"
)


def run_case(project_root: Path, variational: int, start: int, count: int, methods: str) -> tuple[str, dict]:
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
    env["DEGRAF_ENABLE_VARIATIONAL"] = str(variational)

    print(f"[RUN] variational={variational} -> {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout
    if proc.returncode != 0:
        print(output)
        raise RuntimeError(f"Command failed with exit code {proc.returncode} for variational={variational}")

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
        raise RuntimeError("Could not find SceneFlow averages in output.")
    return output, metrics


def append_markdown_row(
    metrics_file: Path,
    date_str: str,
    commit: str,
    machine: str,
    start: int,
    count: int,
    methods: str,
    m0: dict,
    m1: dict,
) -> None:
    section_header = "## 7) variational=0/1 对照实验记录"
    table_header = (
        "| 日期 | commit | 机器 | 配置 | EPE3d(v=0) | EPE3d(v=1) | "
        "AccS(v=0) | AccS(v=1) | AccR(v=0) | AccR(v=1) | Outlier(v=0) | Outlier(v=1) | "
        "Time(ms,v=0) | Time(ms,v=1) | 结论 |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    config = f"start={start},count={count},methods={methods}"
    conclusion = (
        "variational 有收益" if m1["EPE3d"] < m0["EPE3d"] else "variational 收益有限/需复核"
    )
    row = (
        f"| {date_str} | {commit} | {machine} | `{config}` | "
        f"{m0['EPE3d']:.4f} | {m1['EPE3d']:.4f} | "
        f"{m0['AccS']:.2f} | {m1['AccS']:.2f} | "
        f"{m0['AccR']:.2f} | {m1['AccR']:.2f} | "
        f"{m0['Outlier']:.2f} | {m1['Outlier']:.2f} | "
        f"{m0['TimeMs']:.3f} | {m1['TimeMs']:.3f} | {conclusion} |\n"
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
    parser = argparse.ArgumentParser(description="Run variational on/off ablation and append results.")
    parser.add_argument("--project-root", default=".", help="Project root (where run_gpu_pipeline.sh exists)")
    parser.add_argument("--metrics-file", default="teaching_learning_center/METRICS_BASELINE.md")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--methods", default="degraf_flow_interponet")
    parser.add_argument("--machine", default="cloud")
    parser.add_argument("--date", default=dt.date.today().isoformat())
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    metrics_file = (project_root / args.metrics_file).resolve()

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(project_root), text=True
    ).strip()

    out0, m0 = run_case(project_root, 0, args.start, args.count, args.methods)
    out1, m1 = run_case(project_root, 1, args.start, args.count, args.methods)

    logs_dir = project_root / "logs" / "ablation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"{args.date}_variational0.log").write_text(out0, encoding="utf-8")
    (logs_dir / f"{args.date}_variational1.log").write_text(out1, encoding="utf-8")

    append_markdown_row(
        metrics_file=metrics_file,
        date_str=args.date,
        commit=git_commit,
        machine=args.machine,
        start=args.start,
        count=args.count,
        methods=args.methods,
        m0=m0,
        m1=m1,
    )

    print("[OK] Ablation done.")
    print(f"[OK] v=0 metrics: {m0}")
    print(f"[OK] v=1 metrics: {m1}")
    print(f"[OK] Appended to: {metrics_file}")
    print(f"[OK] Logs saved in: {logs_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
