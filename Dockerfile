# Stage 0: OpenCV Builder
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 AS opencv-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config wget curl ca-certificates \
    python3 python3-dev python3-pip python3-venv \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran ffmpeg unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir numpy

ENV OPENCV_VERSION=4.13.0
RUN git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git /opt/opencv \
    && git clone -b ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib \
    && mkdir -p /opt/opencv/build \
    && cd /opt/opencv/build \
    && cmake -D CMAKE_BUILD_TYPE=Release \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
             -D BUILD_opencv_python3=ON \
             -D PYTHON3_EXECUTABLE=/opt/venv/bin/python \
             -D BUILD_EXAMPLES=OFF \
             -D WITH_CUDA=ON \
             -D WITH_CUBLAS=ON \
             -D WITH_IPP=ON \
             -D OPENCV_ENABLE_NONFREE=ON \
             -D ENABLE_FAST_MATH=1 \
             -D CUDA_FAST_MATH=1 \
             -D WITH_CUDNN=ON \
             -D OPENCV_DNN_CUDA=ON \
             -D PYTHON3_PACKAGES_PATH=/opt/venv/lib/python3.12/site-packages \
             -D OPENCV_PYTHON3_INSTALL_PATH=/opt/venv/lib/python3.12/site-packages .. 2>&1 | tee /var/log/opencv_cmake.log \
    && make -j"$(nproc)" 2>&1 | tee /var/log/opencv_build.log \
    && make install 2>&1 | tee /var/log/opencv_install.log \
    && ldconfig \
    && mkdir -p /opt/opencv_artifacts \
    && cp -r /opt/venv/lib/python3.12/site-packages/cv2 /opt/opencv_artifacts/ \
    && cp -P /usr/local/lib/libopencv_*.so* /opt/opencv_artifacts/ \
    && cp /var/log/opencv_*.log /opt/opencv_artifacts/ \
    && rm -rf /opt/opencv /opt/opencv_contrib

# Stage 1: Dependency Builder
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies needed for OpenCV validation during build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    python3 python3-dev python3-pip python3-venv \
    libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 libfontconfig1 libgl1 \
    libgtk-3-0 libice6 libxfixes3 libatk1.0-0 libfreetype6-dev libpng-dev libjpeg-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

# Use our specialized script to populate a local wheelhouse
RUN --mount=type=cache,target=/root/.cache/pip \
    bash scripts/make_wheelhouse.sh /workspace/requirements.cuda.txt /opt/wheelhouse

# Create the final VIRTUAL_ENV and install everything from wheelhouse
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# First install build helpers to ensure sdist builds work, and a stub for opencv-python
# so that ultralytics is satisfied without trying to download it.
# Note: wheel_stub is required by some TensorRT sdists as a build backend.
RUN /opt/venv/bin/python -m pip install --no-index --find-links /opt/wheelhouse \
    pip setuptools wheel wheel_stub && \
    mkdir -p /tmp/opencv_stub && \
    echo "from setuptools import setup; setup(name='opencv-python', version='4.13.0')" > /tmp/opencv_stub/setup.py && \
    /opt/venv/bin/python -m pip install /tmp/opencv_stub

# Install everything with special handling for TensorRT sdists (no-build-isolation)
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/python -m pip install --no-index --find-links /opt/wheelhouse --no-build-isolation \
    -r /workspace/requirements.cuda.txt

# Copy OpenCV python bindings and shared libraries from the other stage
COPY --from=opencv-builder /opt/opencv_artifacts/ /opt/venv/lib/python3.12/site-packages/
RUN cp /opt/venv/lib/python3.12/site-packages/libopencv_*.so* /usr/local/lib/ && ldconfig

# Comprehensive check for missing shared libraries
RUN find /opt/venv/lib/python3.12/site-packages/cv2 -name "*.so" -exec ldd {} \; | grep "not found" && \
    (echo "ERROR: Missing shared libraries detected by ldd!" && exit 1) || echo "All shared libraries resolved."

# Move logs to a dedicated folder
RUN mkdir -p /workspace/logs && \
    mv /opt/venv/lib/python3.12/site-packages/opencv_*.log /workspace/logs/ || true

# Check TensorRT and OpenCV
RUN /opt/venv/bin/python -c "import tensorrt; print('TensorRT Version:', tensorrt.__version__)" && \
    /opt/venv/bin/python -c "import cv2; print('OpenCV Version:', cv2.__version__)"

# Stage 2: Final Runtime Image
FROM nvidia/cuda:13.1.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime system dependencies (X11, GL, etc. for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    libglib2.0-0 libsm6 libxrender1 libxext6 libx11-6 libfontconfig1 libgl1 \
    libgtk-3-0 libice6 libxfixes3 libatk1.0-0 libfreetype6 libpng16-16t64 libjpeg-turbo8 libtiff6 \
    libwebp7 libwebpdemux2 libwebpmux3 \
    ffmpeg libv4l-0 libxvidcore4 libx264-164 \
    && rm -rf /var/lib/apt/lists/*

# Copy the prepared virtualenv and libraries
COPY --from=builder /opt/venv /opt/venv
COPY --from=opencv-builder /opt/opencv_artifacts/libopencv_*.so* /usr/local/lib/
RUN ldconfig

# Setup environment
WORKDIR /workspace
COPY . /workspace

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
