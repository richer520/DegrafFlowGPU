
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /app

# Add NVIDIA GPG public key (required)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    wget \
    unzip \
    git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch/torchvision does not require cu101)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    matplotlib \
    opencv-python \
    scipy \
    tensorboard \
    scikit-image \
    pandas \
    moviepy

#Copy the current directory files to the container
COPY . /app

# Set the default entry
CMD ["bash"]
