# FROM nvidia/cuda:10.1-runtime-ubuntu18.04
# FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM nvidia/cuda:11.0-runtime-ubuntu20.04
FROM nvidia/cuda:10.1-devel-ubuntu18.04
LABEL maintainer="Hebi Li <hebi@lihebi.com>"
RUN chmod 1777 /tmp && chmod 1777 /var/tmp
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
USER root


# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen
# Install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    vim-tiny \
    git \
    inkscape \
    jed \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    python-dev \
    # ---- nbconvert dependencies ----
    # texlive-xetex \
    # texlive-fonts-recommended \
    # texlive-plain-generic \
    # ----
    tzdata \
    unzip \
    nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ffmpeg for matplotlib anim & dvipng for latex labels
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg dvipng && \
    rm -rf /var/lib/apt/lists/*

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Create NB_USER wtih name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

# R pre-requisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc && \
    rm -rf /var/lib/apt/lists/*


ARG PYTHON_VERSION=default

# Install conda as jovyan and check the md5 sum provided on the download site
ENV MINICONDA_VERSION=4.8.3 \
    MINICONDA_MD5=d63adf39f2c220950a063e0529d4ff74 \
    CONDA_VERSION=4.8.3


WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_MD5} *Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority strict && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes conda && \
    conda install --quiet --yes pip && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER


# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
    conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install Jupyter Notebook, Lab, and Hub
# Generate a notebook server config
# Cleanup temporary files
# Correct permissions
# Do all this in a single RUN command to avoid duplicating all of the
# files across image layers when the permissions change
RUN conda install --quiet --yes \
    'notebook=6.1.3' \
    'jupyterhub=1.1.0' \
    'jupyterlab=2.2.5' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    jupyter notebook --generate-config && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Julia
ENV JULIA_VERSION=1.5.0
WORKDIR /tmp

RUN mkdir "/opt/julia-${JULIA_VERSION}" && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/$(echo "${JULIA_VERSION}" | cut -d. -f 1,2)"/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" && \
    echo "be7af676f8474afce098861275d28a0eb8a4ece3f83a11027e3554dcdecddb91 *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
    tar xzf "julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -C "/opt/julia-${JULIA_VERSION}" --strip-components=1 && \
    rm "/tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz"
RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

# My additional packages
RUN apt-get update && \
    apt-get install -yq --no-install-recommends\
    iputils-ping iproute2 htop locate libzmq5

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

##############################
USER $NB_UID

# UPDATE I don't need these. I would just install per-user, and those persist
# with the mounted volume.  Even if these are set, they will be removed.
#
# CAUTION this is slow
RUN julia -e 'import Pkg; Pkg.update()' && \
    (test $TEST_ONLY_BUILD || julia -e 'import Pkg; Pkg.add("HDF5")') && \
    julia -e "using Pkg; pkg\"add IJulia\"; pkg\"precompile\""

USER $NB_UID
# jupyterlab extensions
RUN pip install --upgrade jupyterlab-git
# FIXME NOT WORKING seems that I have to install tensorflow and tensorboard 2.2 for nbtensorboard extension to work
# tensorflow==2.2 tensorboard==2.2
RUN pip install tensorflow==2.2
RUN pip install jupyter-tensorboard
RUN jupyter labextension install --no-build jupyterlab_tensorboard
RUN jupyter labextension install --no-build @jupyterlab/toc
RUN pip install pynvml bokeh jupyterlab-nvdashboard
RUN jupyter labextension install --no-build jupyterlab-nvdashboard
RUN jupyter labextension install --no-build @ijmbarr/jupyterlab_spellchecker
RUN jupyter labextension install --no-build @aquirdturtle/collapsible_headings
RUN jupyter labextension install --no-build https://github.com/lihebi/jupyterlab-racket
# TODO ipywidgets and its lab companion @jupyter-widgets/jupyterlab-manager@2.0
#
# # FIXME this really feels too slow
# # FIXME but if I have minimize set to False, the download (page open) will be slow
# --dev-build=False --minimize=False
RUN jupyter lab build -y --dev-build=False --minimize=False && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# DEBUG Additional packages
USER root
RUN apt-get update && apt-get install -yq --no-install-recommends\
    tmux silversearcher-ag emacs



# TODO DEBUG install BHDL deps, including:
USER $NB_UID
# ENV JULIA_DEPOT_PATH=$HOME/.julia:/opt/julia

# FIXME this is private repo
# RUN git clone --recursive https://github.com/lihebi/bhdl $HOME/bhdl
# RUN cd $HOME/bhdl &&\
#         raco pkg install --batch --deps search-auto --link; exit 0
# RUN cd $HOME/bhdl/placement &&\
#         julia --project -e 'import Pkg; Pkg.update(); Pkg.instantiate(verbose=true); Pkg.precompile(verbose=true)'

# RUN pip uninstall -y tensorflow
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.2
# RUN pip install tensorflow==2.2

# TODO install R and setup kernels
# RUN conda install r-base=3.6.3

# TODO all the R packages for DAG-EQ
# RUN R -e "install.packages('devtools')"


USER $NB_UID

# Files
# COPY start.sh start-notebook.sh start-singleuser.sh /usr/local/bin/
COPY start.sh /usr/local/bin/
COPY jupyter_notebook_config.py /etc/jupyter/

USER root
RUN chmod a+rx /usr/local/bin/start.sh
USER $NB_UID


EXPOSE 8888
WORKDIR $HOME


ENTRYPOINT ["/tini", "-g", "--"]
# CMD [ "/usr/local/bin/start-notebook.sh" ]
# CMD [ "jupyter-lab" ]
CMD [ "/usr/local/bin/start.sh", "jupyter", "lab" ]
