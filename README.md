# GPU-Based Scene Flow with DeGraF, RAFT, and InterpoNet

This repository implements a **GPU-accelerated sparse-to-dense scene flow pipeline** built on **Dense Gradient-based Features (DeGraF)**.  
The pipeline combines:

- **CUDA DeGraF detector** for uniform feature extraction  
- **RAFT** for dense optical flow estimation  
- **InterpoNet** for learned edge-preserving interpolation  
- KITTI-compatible evaluation for both optical and scene flow  

This project extends the original **DeGraF-Flow** framework (Stephenson et al., ICIP 2019) with modern GPU and deep-learning modules.

---

## Project Structure

```
DEGRAF_FLOW_GPU/
├── include/                   # C++ headers
├── src/                       # Core CPU/C++ implementation
├── gpu/                       # CUDA modules (DeGraF detector, kernels)
├── external/                  # Third-party models (RAFT / InterpoNet)
│   ├── RAFT/                  # Cloned from RAFT GitHub
│   └── InterpoNet/            # Cloned from InterpoNet GitHub
├── data/
│   ├── data_scene_flow/       # KITTI 2015 dataset
│   ├── devkit_scene_flow/     # KITTI devkit evaluation tools
│   └── outputs/               # Flow predictions, visualizations, metrics
├── CMakeLists.txt             # C++/CUDA build config
└── README.md                  # Project documentation
```

---

## External Dependencies

### RAFT (ECCV 2020)
- Repository: [https://github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)    
- **Added files** in this project:
  - `degraf_raft_matcher.py` – sampling RAFT dense flow at DeGraF feature locations  
  - `raft_batch_tcp_server.py` – TCP server for batch inference  
  - `Dockerfile` – defines RAFT environment (PyTorch + CUDA)  
  - `run_raft_tcp_server.sh` – helper script to launch RAFT container  

### InterpoNet (CVPR 2017)
- Repository: [https://github.com/shayzweig/InterpoNet](https://github.com/shayzweig/InterpoNet)    
- **Modified files** (replace originals with the provided versions in this repo):
  - `InterpoNet.py`  
  - `io_utils.py`  
  - `utils.py`  
  - `model.py`  
- **Added files**:
  - `interponet_batch_tcp_server.py` – TCP server for batch interpolation  
  - `Dockerfile` – defines InterpoNet environment (TensorFlow 1.15 + CUDA 10.x)  
  - `enter_interponet.sh` – helper script to enter the container  

---

##  Requirements

- **C++/CUDA**
  - CUDA ≥ 12.0  
  - OpenCV 4.9 (built with `optflow`, `ximgproc`)  
  - CMake ≥ 3.12, GCC ≥ 9
- **Python/Docker**
  - Docker with GPU support (`nvidia-docker2` or `--gpus all`)  
  - Separate containers for **RAFT** (PyTorch) and **InterpoNet** (TF1.15)  

---

## Build (C++/CUDA Core)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This builds the CUDA-accelerated DeGraF detector and the C++ evaluation tools.

---

## Dataset Setup

Download the **KITTI 2015 Scene Flow dataset** and **devkit**:

- KITTI 2015: [http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)

Place data as:

```
data/
├── data_scene_flow/
│   ├── training/
│   └── testing/
├── devkit_scene_flow/
└── outputs/
```

---

## Running the Pipeline

1. **Start RAFT server**

```bash
cd external/RAFT
./run_raft_tcp_server.sh
# Inside container:
python raft_batch_tcp_server.py
```

2. **Start InterpoNet server**

```bash
cd external/InterpoNet
./enter_interponet.sh
# Inside container:
python interponet_batch_tcp_server.py
```

3. **Run C++ main program**

```bash
./build/degraf_flow
```

This will:

- Extract DeGraF features (CUDA)
- Request RAFT dense flow, sample at features
- Send sparse matches to InterpoNet server
- Receive interpolated dense flow
- Write KITTI-format outputs to `/data/outputs/`

---

## Evaluation

We use the official **KITTI devkit_scene_flow** for evaluation.

```bash
cd data/devkit_scene_flow
make
./evaluate_scene_flow ../outputs/ results.txt
```

Outputs include:

- Optical flow metrics: **Fl-bg, Fl-fg, Fl-all**
- Scene flow metrics: **EPE3D, AccS, AccR, Outlier %**
- Visualization: flow/error maps (PNG)

---

## References

- F. Stephenson, T. Breckon, I. Katramados,
  *DeGraF-Flow: Extending DeGraF Features for Accurate and Efficient Sparse-to-Dense Optical Flow Estimation*, ICIP 2019.
- Z. Teed, J. Deng,
  *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020.
- S. Zweig, L. Wolf,
  *InterpoNet: A Brain Inspired Neural Network for Optical Flow Dense Interpolation*, CVPR 2017.
- M. Menze, A. Geiger,
  *Object Scene Flow for Autonomous Vehicles*, CVPR 2015.

---

## License

Academic and research use only.

---

## Author

- Gang Wang
- Durham University, 2025

```mermaid
flowchart LR
    K["KITTI images + GT"]

    %% 推理 pipeline
    subgraph pipeline["Scene-flow pipeline (C++ & CUDA)"]
        D["CUDA DeGraF<br/>degraf_detector.cu"]
        RC["RAFT client (C++)"]
        RCont["RAFT container (TCP)"]
        IC["InterpoNet client (C++)"]
        ICont["InterpoNet container (TCP)"]
        F["FeatureMatcher.cpp<br/>predicted optical flow"]
        S["SceneFlowReconstructor.cpp<br/>predicted scene flow"]
    end

    %% 评估模块
    subgraph eval["Evaluation"]
        EvalOF["EvaluateOptFlow.cpp"]
        EvalSF["EvaluateSceneFlow.cpp"]
    end

    main["main.cpp<br/>orchestrator"]

    %% data flow 只保留核心数据流
    K --> D
    D --> RC
    RC <--> RCont
    RC --> IC
    IC <--> ICont
    ICont --> F
    F --> S

    %% evaluation 使用 KITTI GT + 预测结果
    F --> EvalOF
    S --> EvalSF
    K --> EvalOF
    K --> EvalSF

    %% main 只连到两个子系统，减少电线
    main --> pipeline
    main --> eval
 ```
