# Docker build file for MagellanMapper
# Author: David Young, 2020

FROM continuumio/miniconda3

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]

# set up non-root user and allow access to Conda installation folder
ARG username=mag
RUN mkdir /home/$username && groupadd -r $username \
    && useradd -r -s /bin/false -g $username $username \
    && chown -R $username:$username /home/$username \
    && chown -R $username:$username /opt/conda/

# set up appliction base diretory and change to non-root user
ENV BASE_DIR /app
RUN mkdir $BASE_DIR && chown -R $username:$username $BASE_DIR
WORKDIR $BASE_DIR
USER $username

# set up Conda environment for MagellanMapper
COPY --chown=$username:$username environment.yml ./
RUN conda env create -n mag environment.yml && conda init bash \
    && echo "conda activate mag" >> ~/.bashrc

# copy in rest of MagellanMapper files
COPY --chown=$username:$username run.py setup.py LICENSE.txt ./
COPY --chown=$username:$username magmap/ ./magmap/
COPY --chown=$username:$username bin/ ./bin/
COPY --chown=$username:$username stitch/ ./stitch/
