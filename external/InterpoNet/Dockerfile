# ============================
# Ubuntu 18.04 + CUDA 10.0
# ============================
FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y tzdata

RUN apt-get update && apt-get install -y \
    python3.6 python3-pip python3.6-dev python3-setuptools \
    curl git wget unzip make g++ \
    libpng-dev liblapack-dev libjpeg-dev \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


# Set python and pip to point to python3.6 by default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip

# TF 1.15 GPU version, and InterpoNet supporting dependencies
RUN pip install tensorflow-gpu==1.15.0 \
    numpy==1.19.5 \
    scikit-image==0.14.3 \
    opencv-python==4.1.2.30 \
    cython \
    matplotlib

# ============================
# Install CUDA 10 and cuDNN manually decompress
# ============================
COPY files/cuda_10.0.130_410.48_linux.run /tmp/cuda_installer.run
COPY files/cudnn-10.0-linux-x64-v7.6.5.32.tgz /tmp/cudnn.tgz

RUN mkdir -p /usr/local/cuda-10.0 && \
    tar -xzf /tmp/cudnn.tgz -C /usr/local/cuda-10.0 && \
    rm /tmp/cudnn.tgz

RUN mkdir -p /tmp/cuda_extract && \
    sh /tmp/cuda_installer.run --silent --toolkit --toolkitpath=/usr/local/cuda-10.0 --override && \
    rm /tmp/cuda_installer.run

# Set environment variables (compatible with TF)
ENV CUDA_HOME=/usr/local/cuda-10.0
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV PATH=$CUDA_HOME/bin:$PATH

WORKDIR /app
COPY . /app

# Compile C++ part: SrcVariational/
RUN cd /app/SrcVariational && make

CMD ["/bin/bash"]
