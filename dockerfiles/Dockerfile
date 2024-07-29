#  Copyright (c) 2024. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Imaging-based Computational Biomedicine Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

FROM --platform=$BUILDPLATFORM continuumio/miniconda3:latest AS miniconda3_image
LABEL authors="gu"

FROM --platform=$BUILDPLATFORM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as build
LABEL authors="gu"
CMD ["bin/bash"]

# build tools
RUN apt-get update && \
    apt-get install -y openssl wget build-essential unzip curl git cmake ccache ninja-build && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH
COPY --from=miniconda3_image /opt/conda /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda install -y python=3.12

# Basic package
RUN python3 -m pip install numpy cupy-cuda12x scikit-learn scikit-learn-intelex mypy scikit-image
# Faster build
RUN python3 -m pip install ninja psutil

RUN python3 -m pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python3 -c "import torchvision"
# https://stackoverflow.com/questions/76216778/userwarning-applied-workaround-for-cudnn-issue-install-nvrtc-s
RUN cd /opt/conda/lib/python3.12/site-packages/torch/lib && \
    ln -s libnvrtc-*.so.11.2 libnvrtc.so

RUN python3 -c "import torchvision"
# Regular package
RUN python3 -m pip install pandas polars matplotlib seaborn && \
    python3 -m pip install eigen einops timm && \
    python3 -m pip install opencv-python opencv-contrib-python imageio[pyav] && \
    python3 -m pip install SciencePlots plotly && \
    python3 -m pip install tensorboard tensorboardx tqdm torchmetrics kornia && \
    python3 -m pip install omegaconf openpyxl xlsx2csv python-calamine fastexcel yacs python-pptx && \
    python3 -m pip install pyarrow && \
    python3 -m pip install toml && \
    python3 -m pip install jupyter && \
    python3 -m pip install pygifsicle && \
    python3 -m pip install importlib_metadata && \
    python3 -m pip install regex
RUN python3 -c "import torchvision"

# Flash attention
RUN python3 -m pip install flash-attn --no-build-isolation

# Open-MMlab
RUN python3 -m pip install openmim
#  0.10.4
RUN python3 -m pip install --upgrade setuptools && \
    python3 -m mim install mmengine

RUN python3 -c 'import torch;print(torch.__version__)'
# 2.1.0
RUN python3 -m mim install "mmcv==2.1.0"

# 1.2.0
RUN cd /root && git clone https://github.com/open-mmlab/mmpretrain.git && \
    cd mmpretrain && \
    python3 -m mim install . && \
    cd /root && rm -r mmpretrain

# 1.2.2
# 1.2.2 equires mmcv<=2.1.0
RUN python3 -m pip install ftfy
RUN cd /root && \
    git clone -b main https://github.com/open-mmlab/mmsegmentation.git && \
    cd mmsegmentation && \
    python3 -m pip install -v . &&\
    cd /root && rm -r mmsegmentation
RUN python3 -c "import torchvision"

FROM --platform=$BUILDPLATFORM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
LABEL authors="gu"
CMD ["bin/bash"]

RUN apt-get update && apt-get install -y openssl

ENV PATH /opt/conda/bin:$PATH
COPY --from=build /opt/conda /opt/conda
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN python3 -c "import torchvision"

# OpenCV runtime
RUN apt-get update && apt-get install -y libglib2.0-0 ffmpeg libsm6 libxext6
RUN python3 -c "import cv2"

# For pygifsicle runtime
RUN apt-get update &&  apt-get install -y gifsicle

# Latex
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get update && apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended cm-super  && rm -rf /var/lib/apt/lists/*
#COPY --from=akihikokayaba/tex2023-ubuntu22 /usr/local/texlive /usr/local/texlive
#ENV PATH /usr/local/texlive/2023/bin/x86_64-linux:$PATH