QuickStart
==========

How to run
----------

* For in-stock model with sigopt

.. code-block:: bash

    SIGOPT_API_TOKEN-${SIGOPT_TOKEN} python run_hydroai.py --data_path ${dataset_path} --model_name [dlrm, wnd, dien, pipeline_test]

* For in-stock model without sigopt

.. code-block:: bash

    python run_hydroai.py --data_path ${dataset_path} --model_name [dlrm, wnd, dien, pipeline_test] --no_sigopt

* For user-define model

.. code-block:: bash

    python run_hydroai.py --data_path ${dataset_path} --model_name udm --executable_python ${python_path} --program ${path to your train.py}

What is the input
-----------------

* in-stock model name or your own model example/example_train.py, current supported in-stock models are: DLRM, WnD, DIEN
* dataset as listed structure:
  * train - folder
  * validate - folder
  * example/metadata.yaml
* modication of conf/hydroai_defaults.conf

Quick Start
-----------

* Prepare docker

.. code-block:: bash

    git clone https://github.com/intel-innersource/frameworks.bigdata.bluewhale.git
    git checkout hydro.ai
    git submodule update --init --recursive
    docker run -it --privileged --network host --device-/dev/dri -v ${dataset_path}:/home/vmagent/app/dataset -v `pwd`:/home/vmagent/app/hydro.ai -w /home/vmagent/app/ docker.io/xuechendi/oneapi-aikit:hydro.ai /bin/bash
    source /opt/intel/oneapi/setvars.sh --ccl-configuration-cpu_icc --force
    #optional - config proxy
    source /home/vmagent/app/hydro.ai/config_proxy
    #optional - start sshd service
    sudo service ssh start
    bash config_passwdless_ssh.sh ${other_train_node}

* Prepare sigopt token

`<https://app.sigopt.com/tokens/info>`_

* launch hydro.ai

.. code-block:: bash

    SIGOPT_API_TOKEN=${SIGOPT_TOKEN} python run_hydroai.py --model_name pipeline_test --data_path /home/vmagent/app/dataset/test_pipeline/
