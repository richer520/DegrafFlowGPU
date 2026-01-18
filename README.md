## 🚀 Solution Overview (Pre-Sales View)

### TL;DR
A **GPU-accelerated optical / scene flow pipeline** for **robotics and autonomous driving perception**, demonstrating **strong accuracy gains with a clear path toward real-time deployment**.

- **Optical Flow (KITTI 2015):** EPE **8.61 → 2.64** (↓ **69%**), Fl-all **32.39% → 28.98%**
- **Scene Flow (KITTI 2015):** EPE3D **0.3832 m → 0.1602 m** (↓ **58%**), AccS **75.38% → 79.08%**, AccR **82.14% → 86.68%**
- **Deployment insight:** current end-to-end runtime bottleneck is mainly **system integration overhead (I/O / cross-container communication)** rather than algorithmic complexity (thesis evaluation).

> Notes: Metrics are reported from my MSc dissertation evaluation on **KITTI 2015** with **end-to-end measurement** (not kernel-only).

---

### Why it matters
Real-time perception systems are **latency-sensitive**.
Many optical/scene flow pipelines:
- Fail to meet **30–60 FPS** requirements
- Are hard to deploy on **edge GPUs**
- Trade accuracy for speed without controllable knobs

This project focuses on **deployability**, not just algorithmic accuracy:
- modular pipeline design (feature → sparse flow → interpolation)
- measurable KPIs (accuracy, latency/FPS, stability)
- reproducible setup for PoC delivery (Docker / scripts)

---

### Target Use Cases
- Autonomous driving perception pipelines
- Robotics motion estimation & tracking
- Temporal consistency for downstream tasks
- Edge GPU deployment with real-time constraints

---

### Demo & PoC Scope

**Demo (30–90s)**
1. Show baseline pipeline runtime + flow visualization  
2. Switch to GPU-accelerated / optimized pipeline components  
3. Compare: **runtime + flow quality (visual + error metrics)**  
4. Show reproducibility: build/run scripts or Docker commands  

**PoC (1–2 weeks)**
- Integrated GPU flow module (or accelerated components)
- Benchmark report (**latency / FPS / quality**) on customer-like data
- Dockerized deployment / reproducible scripts
- Performance tuning knobs (speed–accuracy modes)

**What we need from customer**
- Target hardware constraints (GPU model / FPS requirement / power budget)
- Sample data (3–5 representative sequences)
- Success criteria (latency threshold + acceptable quality)

---

# 🚗 DeGraF Flow GPU

**GPU-Based Scene Flow Recovery using Dense Gradient-based Features (DeGraF)**  
This project implements a sparse-to-dense optical flow pipeline based on DeGraF feature points, now leveraging OpenCV’s built-in SparseRLOFOpticalFlow, with optional CUDA acceleration for future GPU support.

> Based on:  
> 📝 *Stephenson et al., DeGraF-Flow: Extending DeGraF Features for Accurate and Efficient Sparse-to-Dense Optical Flow Estimation (ICIP 2019)*  
> 🔗 [paper link](https://breckon.org/toby/publications/papers/stephenson19degraf-flow.pdf)

---

## 📦 Project Structure

```
DEGRAF_FLOW_GPU/
├── include/                # Header files
├── src/                    # CPU implementation (.cpp)
├── gpu/                    # CUDA modules and main
├── data/                   # Input images + GT
├── CMakeLists.txt          # CMake build config
├── native_build_setup.sh   # One-click build
```

---

## 🚀 Features

- ✅ DeGraF feature point detection (CPU / planned GPU)
- ✅ Optical flow via OpenCV’s SparseRLOFOpticalFlow
- ✅ Sparse-to-dense interpolation with EPIC
- ✅ KITTI-compatible evaluation
- ✅ Easy native build with OpenCV 4.9

---

## 🛠️ Build & Run

### Prerequisites

- OpenCV 4.9 (with `optflow` and `ximgproc` modules)
- CMake ≥ 3.12
- GCC / Clang with C++14 support

### 1. Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. Run

```bash
./degraf_flow
```

Modify `main.cpp` if you wish to change image paths or test sequences.

---

## 📂 Data Format

```
data/
├── images/
│   ├── 000006_10.png
│   └── 000006_11.png
├── flow_gt/
│   └── gt_000006_10.png
├── outputs/
```

---

## 📊 Output

- Per-frame evaluation (EPE, R2.0, runtime, etc.)
- Visualized flow + ground truth overlays
- Output written to `data/outputs/`

---

## 📚 References

- I. Katramados & T. Breckon, *DeGraF: Dense Gradient-based Features*, ICIP 2016  
- F. Stephenson et al., *DeGraF-Flow: Sparse-to-Dense Optical Flow Estimation*, ICIP 2019

---

## 📃 License

Academic and research use only.

---

## 🙋 Author

- 💻 Adapted and extended by: *Gang Wang*
- 🏫 Durham University, 2025

