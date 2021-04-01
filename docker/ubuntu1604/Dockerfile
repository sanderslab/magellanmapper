# Docker build file for MagellanMapper
# Author: David Young, 2020

FROM ubuntu:16.04

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]
ENV BASE_DIR /app

# install wget to download Miniconda and vim for any basic text editing
RUN  apt-get update && apt-get install -y \
    wget \
    sudo \
    vim \
    && rm -rf /var/lib/apt/lists/*

# set up non-root user with sudo access
ARG username=magellan
RUN mkdir /home/$username \
    && groupadd -r $username \
    && useradd -r -s /bin/false -g $username $username \
    && echo "$username:$username" | chpasswd \
    && usermod -aG sudo $username \
    && chown -R $username:$username /home/$username \
    && mkdir $BASE_DIR \
    && chown -R $username:$username $BASE_DIR

# set up appliction base directory and change to non-root user
WORKDIR $BASE_DIR
USER $username

# set up Conda environment for MagellanMapper
COPY --chown=$username:$username bin/setup_conda bin/libmag.sh ./bin/
COPY --chown=$username:$username environment.yml \
    "SimpleITK-2.0.0rc2.dev908+g8244e-cp36-cp36m-linux_x86_64.whl" ./
RUN echo "y" | bin/setup_conda \
    && echo "conda activate mag" >> ~/.bashrc \
    && eval "$(/home/"$username"/miniconda3/bin/conda shell.bash hook)" \
    && conda clean --all \
    && rm -rf /home/$username/.cache/pip \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && conda activate mag \
    && pip uninstall -y SimpleITK \
    && pip install "SimpleITK-2.0.0rc2.dev908+g8244e-cp36-cp36m-linux_x86_64.whl"

# extract application contents from a git archive to use only files in
# the repository; copy after Conda setup to avoid triggering rebuilding
# prior layers for code updates
COPY --chown=$username:$username magellanmapper_gitarc.tar.gz ./
RUN tar xzvf magellanmapper_gitarc.tar.gz \
    && rm magellanmapper_gitarc.tar.gz \
    "SimpleITK-2.0.0rc2.dev908+g8244e-cp36-cp36m-linux_x86_64.whl"
