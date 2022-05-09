# Dockerfile for AIDK integration test environment

FROM xuechendi/oneapi-aikit:hydro.ai

ENV http_proxy "http://child-prc.intel.com:913"
ENV https_proxy "http://child-prc.intel.com:913"
ENV LD_PRELOAD /opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/lib/libiomp5.so

RUN /opt/intel/oneapi/intelpython/latest/bin/pip install -U pyrecdp
RUN /opt/intel/oneapi/intelpython/latest/bin/pip install -U yapf
RUN /opt/intel/oneapi/intelpython/latest/bin/pip install -U pylint
RUN /opt/intel/oneapi/intelpython/latest/bin/pip install -U pylint-json2html
RUN /opt/intel/oneapi/intelpython/latest/bin/pip install -U pytest
RUN /opt/intel/oneapi/intelpython/latest/envs/tensorflow/bin/pip install -U pytest
RUN /opt/intel/oneapi/intelpython/latest/envs/pytorch_1.10/bin/pip install -U pytest
