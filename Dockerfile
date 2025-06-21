FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /workspace

ENV HF_ENDPOINT=https://hf-mirror.com
ENV PIP_ROOT_USER_ACTION=ignore
ENV MAX_JOBS=32
#ENV HF_HUB_ENABLE_HF_TRANSFER="1"

ARG APT_SOURCE=https://mirrors.tuna.tsinghua.edu.cn/ubuntu/
ARG PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    { \
    echo "deb ${APT_SOURCE} jammy main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-updates main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-backports main restricted universe multiverse"; \
    echo "deb ${APT_SOURCE} jammy-security main restricted universe multiverse"; \
    } > /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd && \
    apt-get clean

RUN apt-get update && \
    apt-get install -y tini aria2 && \
    apt-get clean

RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    python -m pip install --upgrade pip

RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    xgboost transformer_engine flash_attn apex megatron-core grpcio

#RUN aria2c  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
#    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
COPY cuda-ubuntu2204.pin /workspace/cuda-ubuntu2204.pin
RUN mv /workspace/cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

#RUN aria2c --always-resume=true --max-tries=99999 https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
COPY cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb /workspace/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
RUN dpkg -i /workspace/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-4 && \
    rm /workspace/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb && \
    update-alternatives --set cuda /usr/local/cuda-12.4 && \
    rm -rf /usr/local/cuda-12.6

#RUN aria2c --max-tries=99999 https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb && \
COPY cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb /workspace/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
RUN dpkg -i /workspace/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb && \
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cudnn-cuda-12 && \
    rm /workspace/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb

RUN pip install --resume-retries 999 --no-cache-dir uv ninja
