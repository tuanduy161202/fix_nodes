FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL com.nvidia.cudnn.version="9.7.0.66-1"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV TORCH_CUDA_ARCH_LIST="9.0+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    gnupg2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install --no-cache-dir \
    opencv-python \
    librosa \
    moviepy \
    onnx \
    onnxruntime-gpu \
    sageattention \
    rotary-embedding-torch \
    "fastapi[standard]" \
    uvicorn

EXPOSE 8188 9001

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--lowvram", "--fast"]