FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y gnupg2 curl lsb-core vim wget python3-pip libpng16-16 libjpeg-turbo8 libtiff5

RUN apt-get install -y \
        # Base tools
        cmake \
        build-essential \
        git \
        unzip \
        pkg-config \
        python3-dev \
        # OpenCV dependencies
        python3-numpy \
        # Pangolin dependencies
        libgl1-mesa-dev \
        libglew-dev \
        libpython3-dev \
        libeigen3-dev \
        apt-transport-https \
        ca-certificates\
        software-properties-common

# Build OpenCV
RUN apt-get install -y python3-dev python3-numpy python2-dev
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
RUN apt-get install -y libgtk-3-dev

RUN cd /tmp && git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout 4.4.0 && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_EXAMPLES=OFF  -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$nproc && make install && \
    cd / && rm -rf /tmp/opencv

# # Build Pangolin
RUN cd /tmp && git clone https://github.com/stevenlovegrove/Pangolin && \
    cd Pangolin && git checkout v0.6 && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-std=c++11 .. && \
    make -j$nproc && make install && \
    cd / && rm -rf /tmp/Pangolin

# pybind11 setup
RUN pip install pybind11
RUN apt-get update && apt-get install -y pybind11-dev libboost-all-dev libssl-dev

# ml-depth-pro setup
RUN python3 -m pip install --upgrade pip
RUN pip install --default-timeout=100 torch
RUN pip install torchvision
RUN pip install matplotlib
RUN pip install --default-timeout=100 opencv-python
RUN pip install --default-timeout=1000 scipy
RUN pip install tqdm
RUN pip install pillow
RUN pip install pyvista
COPY ml-depth-pro/ /root/deep_orb/ml-depth-pro/
WORKDIR /root/deep_orb/ml-depth-pro/
RUN apt-get remove -y python3-sympy || true
RUN pip install --no-cache-dir -e .

# RUN chmod +x get_pretrained_models.sh
# RUN ./get_pretrained_models.sh
