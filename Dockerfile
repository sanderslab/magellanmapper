# Docker build file for MagellanMapper
# Author: David Young, 2020

FROM continuumio/miniconda3

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]
ENV BASE_DIR /app
RUN mkdir $BASE_DIR
WORKDIR $BASE_DIR

# set up Conda environment for MagellanMapper
COPY environment.yml ./
RUN conda env create -n mag environment.yml && conda init bash \
    && echo "conda activate mag" >> ~/.bashrc

# copy in rest of MagellanMapper files
COPY run.py setup.py LICENSE.txt ./
COPY magmap/ ./magmap/
COPY bin/ ./bin/
COPY stitch/ ./stitch/
