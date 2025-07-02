# 🚗 DeGraF Flow GPU

**GPU-Based Scene Flow Recovery using Dense Gradient-based Features (DeGraF)**  
This project implements a sparse-to-dense optical flow pipeline based on DeGraF feature points, with optional CUDA acceleration, intended for autonomous driving applications.

> Based on:  
> 📝 *Stephenson et al., DeGraF-Flow: Extending DeGraF Features for Accurate and Efficient Sparse-to-Dense Optical Flow Estimation (ICIP 2019)*  
> 🔗 [paper link](https://breckon.org/toby/publications/papers/stephenson19degraf-flow.pdf)

---

## 📦 Project Structure

```
DEGRAF_FLOW_GPU/
├── include/                # Header files
├── src/                    # CPU implementation (.cpp)
├── gpu/                    # GPU modules
│   ├── cuda/               # CUDA kernel and headers
│   └── gpu_main.cpp        # GPU test entry
├── data/                   # (You supply) KITTI-style images + ground truth
├── external/               # External libraries (see below)
├── CMakeLists.txt          # CMake build config
├── Dockerfile              # CPU-only dev container
├── Dockerfile.cuda         # CUDA-enabled dev container
├── run_dev.sh              # Start CPU container
├── run_dev_gpu.sh          # Start GPU container
├── native_build_setup.sh   # One-click build & run on native Linux
```

---

## 🚀 Features

- ✅ DeGraF feature point detection (CPU / GPU)
- ✅ Sparse-to-dense flow: DeGraF + RLOF + EPIC
- ✅ Flow evaluation on KITTI ground truth
- ✅ CUDA module pluggable via `ENABLE_CUDA` CMake option
- ✅ Docker + Native build supported

---

## 🖥️ Build & Run (Native Linux)

### 1. Clone the repo and setup external dependency

```bash
git clone https://github.com/yourname/degraf_flow_gpu.git
cd degraf_flow_gpu
git clone https://github.com/tsenst/RLOFLib.git external/RLOFLib
cd external/RLOFLib
mkdir build && cd build
cmake ..
make
```

> Make sure `libRLOF_64.so` is placed in: `external/RLOFLib/lib/`

---

### 2. Build and run (CPU or GPU)

```bash
chmod +x native_build_setup.sh
./native_build_setup.sh
```

> Modify `native_build_setup.sh` to switch between `gpu_main` and `degraf_flow`

---

## 🐳 Build with Docker (Optional)

### CPU-only container

```bash
docker build -f Dockerfile -t degraf_flow_cpu .
./run_dev.sh
```

### GPU-enabled container

```bash
docker build -f Dockerfile.cuda -t degraf_flow_cuda .
./run_dev_gpu.sh
```

> Make sure your host system supports `nvidia-smi` and has NVIDIA Container Toolkit installed.

---

## 📂 Data Format

Place images and ground truth in:

```
data/
├── images/
│   ├── 000006_10.png
│   └── 000006_11.png
├── flow_gt/
│   └── gt_000006_10.png
├── outputs/
```

You may test with a few KITTI image pairs.

---

## 📊 Evaluation Output

The program outputs average EPE, R2.0, R3.0, runtime per frame etc., and visualizes flow/GT comparison.

---

## 📚 References

- I. Katramados & T. Breckon, *DeGraF: Dense Gradient-based Features*, ICIP 2016
- F. Stephenson et al., *DeGraF-Flow: Sparse-to-Dense Optical Flow Estimation*, ICIP 2019

---

## 📃 License

This project is for academic research and educational use only.

---

## 🙋 Author

- 💻 Modified and extended by: *Gang Wang*
- 🏫 Durham University, 2025
