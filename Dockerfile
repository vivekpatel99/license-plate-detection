# This assumes the container is running on a system with a CUDA GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter

# tensorflow/tensorflow:2.18.0-gpu-jupyter
# FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3
FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3 

# Set DEBIAN_FRONTEND temporarily for build-time only
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variable to enable dynamic GPU memory allocation
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_VISIBLE_DEVICES=0

RUN apt-get update -y && \
    apt-get upgrade -y  \
    && apt-get install wget curl unzip git ffmpeg libsm6 libxext6 \
    gpg-agent \
    python3-pip \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone the TensorFlow Object Detection API repository
RUN git clone https://github.com/tensorflow/models.git

# Install the Object Detection API dependencies
WORKDIR /opt/models/research

# Compile protobuf configs
RUN python3 -m pip install -U pip && pip install "protobuf<4"
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py ./

RUN python3 -m pip install .

# Add object_detection to PYTHONPATH
ENV PYTHONPATH $PYTHONPATH:/opt/models/research:/opt/models/research/object_detection

# (Optional) Test the installation - run a simple import test
RUN python3 -c "from object_detection.utils import label_map_util; print('Object Detection API installed successfully')"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV VIRTUAL_ENV="/opt/.venv" 
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml /opt/
RUN uv sync --active 


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME
ENV PATH="/root/.local/bin:${PATH}"


