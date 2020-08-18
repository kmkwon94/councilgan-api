FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    nano \
    vim \
    wget \
    locales \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
RUN update-locale en_US.UTF-8
# Create a working directory
RUN mkdir /app
WORKDIR /app

# All users can use /home/user as their home directory
ENV HOME=/home/user
#RUN chmod 777 /home/user

# Install Miniconda
WORKDIR /home/user
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh \
 && chmod +x ~/Miniconda3-4.6.14-Linux-x86_64.sh\
 && ~/Miniconda3-4.6.14-Linux-x86_64.sh -b -p ~/miniconda \
 && rm ~/Miniconda3-4.6.14-Linux-x86_64.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
    "torchvision=0.5.0=py36_cu101" \
 && conda clean -ya
# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# Install Council_GAN requirements
RUN conda install -c conda-forge tensorboardx 
RUN conda install -c omnia termcolor 
RUN conda install -c conda-forge python-telegram-bot \
 && conda clean -ya

# Install PyQt5
RUN conda install -y pyqt=5
# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

#Set environment of Cuda 10.1
ENV PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV CUDA_HOME=/usr/local/cuda

#Set ascii environment
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN pip3 install flask
RUN pip3 install flask_cors
COPY . .
RUN mkdir upload 
RUN mkdir upload/person2anime
RUN mkdir upload/male2female
RUN mkdir upload/no_glasses
#change permission
RUN chmod 777 /home/user/pretrain/m2f/128 
RUN echo "uwsgi_read_timeout 300s;"
EXPOSE 80
# Set the default command to python3
CMD python3 ./main.py

