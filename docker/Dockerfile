# Dockerfile. Tegra R32.3.1. Tensorflow 1.13 OpenCV 4.2.0 + K4a
FROM nvcr.io/nvidia/l4t-base:r32.3.1

RUN apt-get update && \
    apt-get install -y curl

RUN apt install -y --reinstall software-properties-common

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add && \
    apt-add-repository https://packages.microsoft.com/ubuntu/18.04/multiarch/prod && \
    apt-get update

# TF1.13
RUN apt-get install -y \
    libhdf5-serial-dev \
    hdf5-tools \
    libhdf5-dev \
    zlib1g-dev \
    zip \
    libjpeg8-dev \
    pkg-config

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu/hdf5/serial/"

RUN apt-get install -y \
    python3-pip

RUN pip3 install -U pip \
                    testresources \
                    setuptools

RUN pip3 install -U numpy==1.16.1 \
                    future==0.17.1 \
                    mock==3.0.5 \
                    keras_preprocessing==1.0.5 \
                    keras_applications==1.0.8 \
                    gast==0.2.2 \
                    enum34 \
                    futures \
                    protobuf \
                    h5py==2.9.0

RUN apt install -y git

# OPENCV 
RUN apt install -y \
    libglew-dev \
    libtiff5-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    # libjasper-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswscale-dev \
    libeigen3-dev \
    libtbb-dev \
    libgtk2.0-dev \
    cmake \
    pkg-config \
    && apt-get -y clean all

# GStreamer
RUN apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

WORKDIR /tmp
ENV CV_VERSION 4.2.0
RUN wget https://github.com/opencv/opencv/archive/$CV_VERSION.zip -O opencv.zip && \
    unzip -q opencv.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$CV_VERSION.zip -O opencv_contrib.zip && \
    unzip -q opencv_contrib.zip && \
    mkdir opencv-$CV_VERSION/build && \
    cd opencv-$CV_VERSION/build && \
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr \
        -DBUILD_PNG=OFF \
        -DBUILD_TIFF=OFF \
        -DBUILD_TBB=OFF \
        -DBUILD_JPEG=OFF \
        -DBUILD_JASPER=OFF \
        -DBUILD_ZLIB=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_opencv_java=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_opencv_python3=ON \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DWITH_OPENCL=OFF \
        -DWITH_OPENMP=OFF \
        -DWITH_FFMPEG=ON \
        -DWITH_GSTREAMER=ON \
        -DWITH_GSTREAMER_0_10=OFF \
        -DWITH_CUDA=ON \
        -DWITH_GTK=ON \
        -DWITH_VTK=OFF \
        -DWITH_TBB=ON \
        -DWITH_1394=OFF \
        -DWITH_OPENEXR=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCUDA_ARCH_BIN=6.2 \
        -DCUDA_ARCH_PTX="" \
        -DINSTALL_C_EXAMPLES=OFF \
        -DINSTALL_TESTS=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-$CV_VERSION/modules .. && \
    make -j4 && \
    make install && \
    rm /tmp/opencv.zip && \
    rm /tmp/opencv_contrib.zip && \
    rm -r /tmp/opencv-$CV_VERSION && \
    rm -r /tmp/opencv_contrib-$CV_VERSION

RUN export ACCEPT_EULA=y
# Trick because accept_eula=y doesn't work... found info using debconf-get-selections | grep libk4a
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    echo libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76 | debconf-set-selections && \
    echo libk4a1.4 libk4a1.4/accept-eula boolean true | debconf-set-selections && \
    apt-get install -y libk4a1.4 libk4a1.4-dev k4a-tools

ENV DEBIAN_FRONTEND="noninteractive"
RUN apt install -y tzdata
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN pip3 install jupyter matplotlib image

RUN pip3 uninstall -y enum34 
RUN pip3 install jupyterlab flake8 git+https://github.com/microsoft/debugpy.git paho-mqtt aiohttp 
RUN apt install -y libcanberra-gtk-module libcanberra-gtk3-module
RUN pip3 install git+https://github.com/etiennedub/pyk4a.git@develop
RUN pip3 install azure-storage-blob==12.3.0 azure-identity

WORKDIR /app
# COPY src/utils /app/utils/
COPY config.json /app
# COPY model/label_map.pbtxt /app/model/label_map.pbtxt
COPY src/*.py /app/
CMD [ "/bin/sh", "-c", "python3 startup.py --config_file config.json"]