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

