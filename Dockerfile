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

# 1. Copy file requirements chính của dự án
COPY requirements.txt .

# 2. Copy tất cả requirements.txt từ các custom_nodes vào các file tạm
# (Chỉ copy file, không copy toàn bộ code để tối ưu cache)
COPY custom_nodes/ComfyUI-Crystools/requirements.txt ./crystools_req.txt
COPY custom_nodes/ComfyUI-Easy-Use/requirements.txt ./easyuse_req.txt
COPY custom_nodes/ComfyUI-Manager/requirements.txt ./manager_req.txt
COPY custom_nodes/ComfyUI-KJNodes/requirements.txt ./kjnodes_req.txt
COPY custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ./vhs_req.txt
COPY custom_nodes/was-node-suite-comfyui/requirements.txt ./was_req.txt
COPY custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt ./wan_wrapper_req.txt
COPY custom_nodes/ComfyUI-wanBlockswap/requirements.txt ./wan_swap_req.txt
COPY custom_nodes/ComfyUI-MelBandRoFormer/requirements.txt ./roformer_req.txt
COPY custom_nodes/ComfyUI-MieNodes/requirements.txt ./mie_req.txt
COPY custom_nodes/ComfyUI-GGUF/requirements.txt ./gguf_req.txt

# 3. Cài đặt tất cả trong một lệnh RUN duy nhất
# Điều này giúp giảm số lượng layer và dung lượng image
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir \
    -r ./crystools_req.txt \
    -r ./easyuse_req.txt \
    -r ./manager_req.txt \
    -r ./kjnodes_req.txt \
    -r ./vhs_req.txt \
    -r ./was_req.txt \
    -r ./wan_wrapper_req.txt \
    -r ./wan_swap_req.txt \
    -r ./roformer_req.txt \
    -r ./mie_req.txt \
    -r ./gguf_req.txt && \
    rm *_req.txt

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