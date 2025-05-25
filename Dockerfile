# base image with cuda 12.4
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# install python 3.11 and git
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# set python3.11 as the default python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# install uv from pip
RUN pip install uv

# create venv
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.11 /.venv

# install torch with automatic backend detection
ENV UV_TORCH_BACKEND=auto
RUN uv pip install torch

# install remaining dependencies from PyPI
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt

# copy files
COPY download_weights.py schemas.py handler.py test_input.json /

# download the weights from hugging face
RUN python /download_weights.py

# run the handler
CMD python -u /handler.py
