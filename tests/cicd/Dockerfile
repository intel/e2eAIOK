# Dockerfile for AIDK integration test environment

FROM xuechendi/oneapi-aikit:hydro.ai

WORKDIR /home/vmagent/app
ENV http_proxy "http://child-prc.intel.com:913"
ENV https_proxy "http://child-prc.intel.com:913"
ENV PATH /opt/intel/oneapi/intelpython/latest/condabin:$PATH

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN python -m pip install pytest pyrecdp yapf pylint pylint-json2html 

SHELL ["conda", "run", "-n", "tensorflow", "/bin/bash", "-c"]
RUN python -m pip install pytest

SHELL ["conda", "run", "-n", "pytorch_1.10", "/bin/bash", "-c"]
RUN python -m pip install pytest
ENV LD_PRELOAD /opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/lib/libiomp5.so

SHELL ["/bin/bash", "-c"] 
RUN conda init bash