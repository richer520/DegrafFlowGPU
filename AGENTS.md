# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
DeGraF Flow GPU is a C++14 optical flow pipeline using OpenCV contrib modules (`optflow`, `ximgproc`). See `README.md` for full description.

### Build (CPU-only, no CUDA)
```bash
mkdir -p build && cd build
cmake .. -DENABLE_CUDA=OFF -DCMAKE_CXX_COMPILER=g++
make -j$(nproc)
```

### Run
The executable expects to be run from the `build/` directory (data paths are relative `../data/...`):
```bash
cd build && xvfb-run -a ./degraf_flow
```
The program blocks at `waitKey(0)` after displaying results (GUI window). In headless environments use `xvfb-run`. The flow evaluation completes and writes output files to `data/outputs/` before blocking.

### Key gotchas
- **Clang linker issue**: The default system compiler (Clang 18) fails to link due to `-lstdc++` not being found. Use `-DCMAKE_CXX_COMPILER=g++` explicitly.
- **CUDA is optional**: The original `CMakeLists.txt` declared `LANGUAGES CXX CUDA` unconditionally. The fix makes CUDA conditional via `-DENABLE_CUDA=ON/OFF` (defaults OFF).
- **Odometry module**: `Odometry::run()` in `main.cpp` has hardcoded Windows paths and will print an error and return -1 gracefully. This does not crash the program.
- **No formal test suite or linter**: This is an academic C++ project. Validation is done by running the executable against sample KITTI data in `data/`.
- **xvfb required**: OpenCV `imshow`/`waitKey` calls require a display. Install `xvfb` and prefix runs with `xvfb-run -a`.
- **System deps**: `libopencv-dev`, `libopencv-contrib-dev`, `xvfb` (Ubuntu 24.04 provides OpenCV 4.6.0 with needed contrib modules).
