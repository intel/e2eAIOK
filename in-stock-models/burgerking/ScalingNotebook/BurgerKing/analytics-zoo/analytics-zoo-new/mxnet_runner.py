#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import logging
import os
import socket
import subprocess
import time
from contextlib import closing

import mxnet as mx
import ray.services
from dmlc_tracker.tracker import get_host_ip
from mxnet import gluon


class MXNetRunner(object):
    """Manages a MXNet model for training."""

    # Symbolic API doesn't need loss_creator. Loss is defined in model output.
    # Users can specify train_function (not recommended) or use the default one.
    def setup_distributed(self, env, config, train_data, test_data, model_creator,
                          loss_creator=None, metrics_creator=None, train_function=None):
        logging.basicConfig(level=logging.INFO)  # This can print log messages to console.
        self.config = config  # TODO: add check for config keys
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.metrics_creator = metrics_creator
        self.train_function = train_function
        self.is_worker = False
        self.epoch = 0
        env["DMLC_NODE_HOST"] = self.get_node_ip()
        if env["DMLC_ROLE"] == "worker":
            self.is_worker = True

        if self.is_worker:
            os.environ.update(env)
            if "seed" in self.config:
                mx.random.seed(self.config["seed"])
            self.kv = mx.kv.create(self.config["kvstore"])

            import numpy as np
            from functools import reduce

            def get_data_label(partition_data):
                def combine(dict1, dict2):
                    return {key: np.concatenate((value, dict2[key]), axis=0)
                            for (key, value) in dict1.items()}

                data_list = [data['data'] for data in partition_data]
                label_list = [data['label'] for data in partition_data]
                if isinstance(partition_data[0]['data'], dict):
                    data = reduce(lambda dict1, dict2: combine(dict1, dict2), data_list)
                elif isinstance(partition_data[0]['data'], np.ndarray):
                    data = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                                  data_list)
                if isinstance(partition_data[0]['label'], dict):
                    label = reduce(lambda dict1, dict2: combine(dict1, dict2), label_list)
                elif isinstance(partition_data[0]['label'], np.ndarray):
                    label = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                                   label_list)
                return {'data': data, 'label': label}

            # retrieve train data
            train_partition_data = ray.get(train_data[self.kv.rank].get_data())
            train_data_label = get_data_label(train_partition_data)
            self.train_data = mx.io.ResizeIter(
                mx.io.NDArrayIter(data=train_data_label['data'],
                                  label=train_data_label['label'],
                                  batch_size=config["batch_size"],
                                  shuffle=True), size=3268)
            # retrieve val data
            val_partition_data = ray.get(test_data[self.kv.rank].get_data())
            val_data_label = get_data_label(val_partition_data)
            self.val_data = mx.io.NDArrayIter(data=val_data_label['data'],
                                              label=val_data_label['label'],
                                              batch_size=config["batch_size"],
                                              shuffle=True)

            self.model = self.model_creator(self.config)
            if self.loss_creator:
                self.loss = self.loss_creator(self.config)
            if self.val_data:
                assert self.metrics_creator, "Metrics is needed for val data"
                self.metrics = self.metrics_creator(self.config)
            else:
                self.metrics = None
            # For BaseModule, use symbolic API. Otherwise, use imperative API.
            if not isinstance(self.model, mx.module.BaseModule):
                self.trainer = gluon.Trainer(self.model.collect_params(), self.config["optimizer"],
                                             optimizer_params=self.config["optimizer_params"],
                                             kvstore=self.kv)
            else:  # Trainer is not needed for symbolic API.
                self.trainer = None
        else:  # server
            # Need to use the environment on each raylet process for the correct python environment.
            modified_env = os.environ.copy()
            modified_env.update(env)
            subprocess.Popen("python -c 'import mxnet'", shell=True, env=modified_env)

    def step(self):
        """Runs a training epoch and updates the model parameters."""
        self.epoch += 1
        stats = dict()
        stats["epoch"] = self.epoch
        if self.is_worker:
            start_time = time.time()
            if self.train_function:  # User want to specify their own training logic. Not recommended. Not tested.
                self.train_function(self)
            elif self.trainer:  # Imperative API
                self.train_data.reset()
                if self.metrics:
                    self.metrics.reset()  # metrics will accumulate for one batch
                batch_start_time = time.time()
                for i, batch in enumerate(self.train_data):
                    # MXNet treats all CPUs on a single machine as a single device.
                    # So whether you specify cpu(0) or cpu(), MXNet will use all CPU cores on the machine.
                    data = gluon.utils.split_and_load(batch.data[0].astype("float32"), ctx_list=[mx.cpu()],
                                                      batch_axis=0)
                    label = gluon.utils.split_and_load(batch.label[0].astype("float32"), ctx_list=[mx.cpu()],
                                                       batch_axis=0)
                    outputs = []
                    Ls = []
                    from mxnet import autograd as ag
                    with ag.record():
                        for x, y in zip(data, label):
                            z = self.model(x)  # forward
                            L = self.loss(z, y)
                            # store the loss and do backward after we have done forward
                            # on all GPUs for better speed on multiple GPUs.
                            Ls.append(L)
                            outputs.append(z)
                        ag.backward(Ls)
                    self.trainer.step(batch.data[0].shape[0])
                    if self.metrics:
                        self.metrics.update(label, outputs)
                    if "log_interval" in self.config and not (i + 1) % self.config["log_interval"]:
                        # This would print on driver for each pid.
                        print_output = ""
                        print_output += 'Epoch[%d] Batch[%d]  Speed: %f samples/sec %s=%f' % (
                            self.epoch, i, self.config["batch_size"] / (time.time() - batch_start_time), "loss",
                            Ls[0].asnumpy().mean())
                        if self.metrics:
                            names, accs = self.metrics.get()
                            if not isinstance(names, list):
                                names = [names]
                                accs = [accs]
                            for name, acc in zip(names, accs):
                                print_output += ' %s=%f' % (name, acc)
                        print(print_output)
                    batch_start_time = time.time()
                if self.metrics:
                    names, accs = self.metrics.get()
                    if not isinstance(names, list):
                        names = [names]
                        accs = [accs]
                    for name, acc in zip(names, accs):
                        stats[name] = acc
            else:  # Symbolic API
                # TODO: seems no history (i.e. validation accuracy) returned by fit?
                self.model.fit(train_data=self.train_data,
                               num_epoch=1 if "epochs" not in self.config else self.config["epochs"],
                               initializer=self.config["init"],
                               kvstore=self.kv,
                               optimizer=self.config["optimizer"],
                               eval_metric=self.metrics,
                               validation_metric=self.metrics,
                               eval_data=self.val_data,
                               batch_end_callback=None if "log_interval" not in self.config
                               else mx.callback.Speedometer(self.config["batch_size"], self.config["log_interval"]),
                               epoch_end_callback=None if "model" not in self.config
                               else mx.callback.do_checkpoint(self.config["model"]))
            epoch_time = time.time() - start_time
            stats["epoch_time"] = epoch_time
        return stats

    def validate(self):
        # TODO: validate for symbolic API
        stats = dict()
        stats["epoch"] = self.epoch
        if self.is_worker:
            self.metrics.reset()
            self.val_data.reset()
            for batch in self.val_data:
                data = gluon.utils.split_and_load(batch.data[0].astype("float32", copy=False),
                                                  ctx_list=[mx.cpu()], batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0].astype("float32", copy=False),
                                                   ctx_list=[mx.cpu()], batch_axis=0)
                outputs = [self.model(X) for X in data]
                self.metrics.update(label, outputs)
            names, accs = self.metrics.get()
            if not isinstance(names, list):
                names = [names]
                accs = [accs]
            for name, acc in zip(names, accs):
                stats[name] = acc
        return stats

    def shutdown(self):
        """Attempts to shut down the runner."""
        if self.is_worker:
            del self.model
            del self.train_data
            del self.val_data
            del self.kv
            del self.trainer
            del self.loss
            # TODO: also delete downloaded data as well?

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class MXNetTrainer(object):
    def __init__(self,
                 config,
                 train_data,
                 test_data,
                 model_creator,
                 loss_creator=None,
                 metrics_creator=None,
                 train_function=None,
                 # Specify cpu resources for actors so that two actors won't use the same raylet.
                 runner_cpus=None):
        self.config = config
        self.train_data = train_data
        self.test_data = test_data
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.metrics_creator = metrics_creator
        self.train_function = train_function
        self.num_workers = config["num_workers"]
        self.num_servers = config["num_servers"] if "num_servers" in self.config else self.num_workers

        # Generate actor class
        # Add a dummy custom resource for server to diff from worker
        Worker = ray.remote(num_cpus=runner_cpus, resources={"_mxnet_worker": 1})(MXNetRunner) if runner_cpus \
            else ray.remote(MXNetRunner)
        Server = ray.remote(num_cpus=runner_cpus, resources={"_mxnet_server": 1})(MXNetRunner) if runner_cpus \
            else ray.remote(MXNetRunner)

        # Start runners: workers followed by servers
        self.runners = [
            Worker.remote()
            for i in range(self.num_workers)
        ]
        self.runners += [
            Server.remote()
            for i in range(self.num_servers)
        ]

        # Compute URL for initializing distributed setup
        ips = ray.get(
            [runner.get_node_ip.remote() for runner in self.runners])
        ports = ray.get(
            [runner.find_free_port.remote() for runner in self.runners])
        print(ips)
        print(ports)

        env = {
            "DMLC_PS_ROOT_URI": str(get_host_ip()),
            "DMLC_PS_ROOT_PORT": str(find_free_port()),
            "DMLC_NUM_SERVER": str(self.num_servers),
            "DMLC_NUM_WORKER": str(self.num_workers),
        }
        envs = []
        for i in range(self.num_workers):
            current_env = env.copy()
            current_env['DMLC_ROLE'] = 'worker'
            envs.append(current_env)
        for i in range(self.num_servers):
            current_env = env.copy()
            current_env['DMLC_ROLE'] = 'server'
            envs.append(current_env)

        env['DMLC_ROLE'] = 'scheduler'
        modified_env = os.environ.copy()
        modified_env.update(env)
        # Need to contain system env to run bash
        subprocess.Popen("python -c 'import mxnet'", shell=True, env=modified_env)

        ray.get([
            runner.setup_distributed.remote(envs[i], self.config,
                                            train_data.get_partitions(),
                                            test_data.get_partitions(),
                                            self.model_creator,
                                            self.loss_creator,
                                            self.metrics_creator,
                                            self.train_function)
            for i, runner in enumerate(self.runners)
        ])

    def train(self):
        """Runs a training epoch."""
        stats = ray.get([w.step.remote() for w in self.runners])
        return stats

    def validate(self):
        """Evaluates the model on the validation data set."""
        stats = ray.get([w.validate.remote() for w in self.runners])
        return stats

    def shutdown(self):
        """Shuts down runners and releases resources."""
        for runner in self.runners:
            runner.shutdown.remote()
            runner.__ray_terminate__.remote()

# TODO: add model save and restore
