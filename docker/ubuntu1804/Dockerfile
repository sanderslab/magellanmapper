# Docker build file for MagellanMapper
# Author: David Young, 2020

FROM ubuntu:18.04

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]

# set up non-root user and allow access to Conda installation folder
ENV BASE_DIR /app
ARG username=magellan
RUN mkdir /home/$username && groupadd -r $username \
    && useradd -r -s /bin/false -g $username $username \
    && chown -R $username:$username /home/$username && mkdir $BASE_DIR \
    && chown -R $username:$username $BASE_DIR

# install wget to download Miniconda
RUN  apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

# set up appliction base diretory and change to non-root user
WORKDIR $BASE_DIR
USER $username

# set up Conda environment for MagellanMapper
COPY --chown=$username:$username bin/ ./bin/
COPY --chown=$username:$username environment.yml ./
RUN echo "y" | bin/setup_conda && echo "conda activate mag" >> ~/.bashrc \
    && /home/$username/miniconda3/bin/conda clean --all \
    && rm -rf /home/$username/.cache/pip \
    && rm Miniconda3-latest-Linux-x86_64.sh

# copy in rest of MagellanMapper files
COPY --chown=$username:$username run.py setup.py LICENSE.txt ./
COPY --chown=$username:$username magmap/ ./magmap/
COPY --chown=$username:$username stitch/ ./stitch/
