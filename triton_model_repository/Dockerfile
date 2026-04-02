FROM nvcr.io/nvidia/tritonserver:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip && \
    pip install numpy && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip install torch-geometric==2.7.0 pyg-lib==0.5.0+pt26cu124 \
        torch-scatter==2.1.2+pt26cu124 torch-sparse==0.6.18+pt26cu124 \
        torch-cluster==1.6.3+pt26cu124 torch-spline-conv==1.2.2+pt26cu124 \
        -f https://data.pyg.org/whl/torch-2.6.0+cu124.html