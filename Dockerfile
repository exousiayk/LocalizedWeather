FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 시스템 환경 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 2. 시스템 의존성 설치 (가나다순 정렬하여 가독성 확보)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    git \
    libgdal-dev \
    libgeos-dev \
    libglib2.0-0 \
    libproj-dev \
    libsm6 \
    libspatialindex-dev \
    libxext6 \
    libxrender1 \
    proj-bin \
    tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/Source

# 3. [최적화] 파이썬 의존성 설치
RUN pip install --upgrade pip && \
        pip install \
            nvitop \
            numpy \
            pandas \
            xarray \
            netCDF4 \
            h5netcdf \
            geopandas \
            shapely \
            fiona \
            pyproj \
            scipy \
            scikit-learn \
            matplotlib \
            Pillow \
            tqdm \
            wandb \
            requests \
            pyyaml \
            cdsapi \
            ecmwf-datastores-client \
            dask \
            pyogrio && \
        pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# 4. 나머지 소스 코드 복사 (코드 수정 시 여기서부터만 다시 빌드됨)
COPY . /workspace

# 환경 변수 설정
ENV PYTHONPATH=/workspace/Source:/workspace

CMD ["bash"]

# docker run --gpus all -it \
#   --name weather_dev \
#   -v /projects3/home/flag0220/LocalizedWeather:/workspace \
#   -w /workspace/Source \
#   weather:v1 bash