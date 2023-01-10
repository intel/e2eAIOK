QuickStart
==========

How to run
----------

* For in-stock model with sigopt

.. code-block:: bash

    SIGOPT_API_TOKEN=${SIGOPT_TOKEN} python run_e2eaiok.py --data_path ${dataset_path} --model_name [dlrm, wnd, dien, pipeline_test, twitter_recsys, rnnt, tpcxai09]

* For in-stock model without sigopt

.. code-block:: bash

    python run_e2eaiok.py --data_path ${dataset_path} --model_name [dlrm, wnd, dien, pipeline_test, twitter_recsys, rnnt, tpcxai09] 

* For user-define model

.. code-block:: bash

    python run_e2eaiok.py --data_path ${dataset_path} --model_name udm --executable_python ${python_path} --program ${path to your train.py}

What is the input
-----------------

* in-stock model name or your own model example/example_train.py, current supported in-stock models are: DLRM, WnD, DIEN
* dataset as listed structure:

  * train - folder
  
  * validate - folder
  
  * metadata.yaml - file
  
* modication of conf/e2eaiok_defaults.conf

Pre-processed Data
------------------

Internal copy vsr602://mnt/nvme2/chendi/BlueWhale/dataset

Quick Start
-----------

* Prepare docker

.. code-block:: bash

    git clone https://github.com/intel-innersource/frameworks.bigdata.bluewhale.git
    git checkout e2eaiok
    git submodule update --init --recursive
    docker run -it --privileged --network host --device=/dev/dri --read-only -v ${dataset_path}:/home/vmagent/app/dataset -v `pwd`:/home/vmagent/app/e2eaiok -w /home/vmagent/app/ docker.io/xuechendi/oneapi-aikit:hydro.ai /bin/bash
    source /etc/profile.d/spark-env.sh
    
    # optional - config proxy
    source /home/vmagent/app/e2eaiok/config_proxy
    
    # optional - start sshd service
    sudo service ssh start
    bash config_passwdless_ssh.sh ${other_train_node}
    
* test DLRM    
.. code-block:: bash

    source /opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/.local/env/setvars.sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch_ipex-0.1-py3.7-linux-x86_64.egg/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/intelpython/python3.7/envs/pytorch_mlperf/lib/python3.7/site-packages/torch/lib/

    # prepare data
    # tree ../dataset/criteo/
    # ../dataset/criteo/
    # ├── day_day_count.npz
    # ├── day_fea_count.npz
    # ├── metadata_dlrm_example.yaml
    # ├── model_size.json
    # ├── train
    # │   └── train_data.bin
    # └── valid
    #     └── test_data.bin

    # Use e2eaiok API
    SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm --conf conf/e2eaiok_defaults_dlrm_example.conf

    # Use SDA API
    SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/criteo" --model_name dlrm --conf conf/e2eaiok_defaults_dlrm_example.conf

* Test WnD
.. code-block:: bash

    source /opt/intel/oneapi/setvars.sh --force
    # prepare data
    # tree -d 1 ../dataset/outbrain/
    # 1 [error opening dir]
    # ../dataset/outbrain/
    # ├── meta
    # │   └── transformed_metadata
    # ├── train
    # └── valid

    # change hosts and eth in conf/e2eaiok_defaults_wnd_example.conf
    # iface: ${eth0}
    # hosts:
    #   - ${host_name}

    # Use e2eaiok API
    SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/outbrain" --model_name wnd --conf conf/e2eaiok_defaults_wnd_example.conf

    # Use SDA API
    SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/outbrain" --model_name wnd --conf conf/e2eaiok_defaults_wnd_example.conf
    
 * Test DIEN
.. code-block:: bash

    source /opt/intel/oneapi/setvars.sh --force
    # dataset layout
    #  tree ../dataset/amazon_reviews/
    # ../dataset/amazon_reviews/
    # ├── cat_voc.pkl
    # ├── meta.yaml
    # ├── mid_voc.pkl
    # ├── train
    # │   └── local_train_splitByUser
    # ├── uid_voc.pkl
    # └── valid
    #     └── local_test_splitByUser

    # Use e2eaiok API
    SIGOPT_API_TOKEN=${TOKEN} python run_e2eaiok.py --data_path "/home/vmagent/app/dataset/amazon_reviews" --model_name dien

    # Use SDA API
    SIGOPT_API_TOKEN=${TOKEN} python SDA/SDA.py --data_path "/home/vmagent/app/dataset/amazon_reviews" --model_name dien


* Prepare sigopt token

`<https://app.sigopt.com/tokens/info>`_

* launch e2eaiok

.. code-block:: bash

    SIGOPT_API_TOKEN=${SIGOPT_TOKEN} python run_e2eaiok.py --model_name pipeline_test --data_path /home/vmagent/app/dataset/test_pipeline/
