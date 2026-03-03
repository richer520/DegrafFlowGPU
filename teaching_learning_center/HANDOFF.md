# Handoff（新窗口交接卡）

用途：在新窗口快速恢复上下文，避免重复沟通。

## 使用规则（最小版）

每次会话结束前，只更新这三行：

- 当前分支：
- 最新commit：
- 下一步：
- 最新学习记录：

每次新窗口开始时：

1. 先让助手阅读本文件  
2. 再补充本次新增报错/截图/目标
3. 打开最近一份 `teaching_learning_center/DAILY_LOG /DAILY_LOG_YYYY-MM-DD.md`，确认上次做到哪一步
4. 查看 `teaching_learning_center/KNOWLEDGE_GAP_LIST.md`，先选 1 个基础短板作为本次补洞目标
5. 打开当日学习任务单 `teaching_learning_center/Daily_Study/DAILY_STUDY_TASK_YYYY-MM-DD.md`

---

## 当前状态（请持续维护）

- 当前分支：`feature-optimize`
- 最新commit：`646e005`（默认 `DEGRAF_VARIATIONAL_RESIDUAL_THRESH=3.0`）
- 下一步：继续做“精度接近论文 + 速度超过论文”的单目标优化；优先 RAFT 真 ONNX/TensorRT 并入
- 最新学习记录：`teaching_learning_center/DAILY_LOG /DAILY_LOG_2026-03-03.md`

## 项目目标（固定）

- 目标不是双模式对比，而是单目标：
  - 精度接近或达到论文
  - 速度快于论文（GPU 加速应有速度）

## 学习执行目标（固定）

- 方法：边优化边学习（教学模式 + 单点优化循环）。
- 每日要求：2小时保底，不设硬上限；每天必须有“改动 + 验证 + 四行复盘”。
- 记录要求：每日新增一份按日期日志，作为学习证据与简历素材。
- 汇总材料：`METRICS_BASELINE.md`（指标基线）与 `INTERVIEW_STORY_BANK.md`（面试故事库）。
- 基础补洞：持续维护 `KNOWLEDGE_GAP_LIST.md`，每周至少清理 2 条 `OPEN` 项。
- 任务执行：每日维护并勾选 `Daily_Study/DAILY_STUDY_TASK_YYYY-MM-DD.md`。

## 已完成关键点（简版）

- 已建立 `feature-optimize` 分支进行优化实验，`feature` 保持基线不动
- variational 已参数化并支持残差门控
- 增加了 `PROFILE` 日志，确认主要瓶颈在 variational
- 当前默认：
  - `DEGRAF_VARIATIONAL_MODE=external`（稳定优先）
  - `DEGRAF_VARIATIONAL_RESIDUAL_THRESH=3.0`

## 环境信息（云端）

- TensorRT：`10.8.0.5`
- CUDA：`12.4`
- GPU：`RTX 2080 Ti`

## 常用运行命令

```bash
bash run_gpu_pipeline.sh --start 0 --count 50 --batch-size 10 --methods degraf_flow_interponet
```

```bash
bash run_gpu_pipeline.sh --start 0 --count 200 --batch-size 20 --methods degraf_flow_interponet
```

```bash
bash run_gpu_pipeline.sh --start 0 --count 200 --batch-size 20 --methods degraf_flow_interponet,degraf_flow_rlof,DISflow_fast,deepflow
```

## 交接给新窗口时可直接粘贴的话术

```text
请先阅读 `teaching_learning_center/HANDOFF.md`、最新 `teaching_learning_center/DAILY_LOG /DAILY_LOG_YYYY-MM-DD.md`、`teaching_learning_center/KNOWLEDGE_GAP_LIST.md`、当日 `teaching_learning_center/Daily_Study/DAILY_STUDY_TASK_YYYY-MM-DD.md` 再继续。目标是单目标优化（精度接近论文+速度超过论文），不是双模式。请沿着“RAFT 真 ONNX/TensorRT 并入 -> InterpoNet 真模型并入 -> variational 进程内稳定版”的路线推进，并按教学模式执行（先解释后修改、单点改动、改后复盘）。
```

