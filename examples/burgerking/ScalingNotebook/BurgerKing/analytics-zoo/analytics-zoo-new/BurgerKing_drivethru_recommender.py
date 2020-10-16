import boto3

import mxnet as mx
import numpy as np

import zoo.xshard.pandas
from mxnet.gluon import nn
from zoo import init_spark_on_yarn, init_spark_on_local
from zoo.ray import RayContext

from mxnet_runner import MXNetTrainer

# Need to know the following global values beforehand to construct the model.
n_plus, n_time, n_bkids, n_weather, n_feels = 522, 167, 126, 35, 20


def prepare_data(df, use_context):
    if use_context:
        data = {'pluids': np.array(df['pluids'].values.tolist(), dtype=int),
                                    'bkidx': df['bkidx'].values,
                                    'timeidx': df['timeidx'].values,
                                    'feels_bucket': df['feelsBucket'].values,
                                    'weatheridx': df['weatheridx'].values}
    else:
        data = {'pluids': np.array(df['pluids'].values.tolist(), dtype=int)}
    label = {'output_label': df['label'].values}
    return {'data': data,'label': label}


def get_model(config):
    class SelfAttention(nn.HybridBlock):
        def __init__(self, att_unit, att_hops, **kwargs):
            super(SelfAttention, self).__init__(**kwargs)
            with self.name_scope():
                self.ut_dense = nn.Dense(att_unit, activation='tanh', flatten=False)
                self.et_dense = nn.Dense(att_hops, activation=None, flatten=False)

        def hybrid_forward(self, F, x):
            # x shape: [batch_size, seq_len, embedding_width]
            # ut shape: [batch_size, seq_len, att_unit]
            ut = self.ut_dense(x)
            # et shape: [batch_size, seq_len, att_hops]
            et = self.et_dense(ut)

            # att shape: [batch_size,  att_hops, seq_len]
            att = F.softmax(F.transpose(et, axes=(0, 2, 1)), axis=-1)
            # output shape [batch_size, att_hops, embedding_width]
            output = F.batch_dot(att, x)

            return output, att

    y_true = mx.symbol.Variable('output_label')
    pluids = mx.symbol.Variable('pluids')
    bkidx = mx.symbol.Variable('bkidx')
    timeidx = mx.symbol.Variable('timeidx')
    feels_bucket = mx.symbol.Variable('feels_bucket')
    weatheridx = mx.symbol.Variable('weatheridx')
    plu_embed = mx.symbol.Embedding(data=pluids, input_dim=n_plus, output_dim=50, name='plu_embed')
    bkidx_embed = mx.symbol.Embedding(data=bkidx, input_dim=n_bkids, output_dim=100, name='bkid_embed')
    time_embed = mx.symbol.Embedding(data=timeidx, input_dim=n_time, output_dim=100, name='time_embed')
    feels_embed = mx.symbol.Embedding(data=feels_bucket, input_dim=n_feels, output_dim=100, name='feels_embed')
    weather_embed = mx.symbol.Embedding(data=weatheridx, input_dim=n_weather, output_dim=100, name='weather_embed')

    # False if use mkl optimized mxnet and would lead to better performance on CLX8280 cluster
    use_stack = True
    if use_stack:
        stacked_rnn_cells = mx.rnn.SequentialRNNCell()
        stacked_rnn_cells.add(mx.rnn.BidirectionalCell(mx.rnn.GRUCell(num_hidden=50, prefix="gru_l"),
                                                       mx.rnn.GRUCell(num_hidden=50, prefix="gru_r")))
        stacked_out, _ = stacked_rnn_cells.unroll(length=5, inputs=plu_embed, merge_outputs=True, layout="NTC")
    else:
        fused_cell = mx.rnn.FusedRNNCell(50, num_layers=1, bidirectional=True, mode="gru", prefix="")
        stacked_out, _ = fused_cell.unroll(length=5, inputs=plu_embed, merge_outputs=True, layout="NTC")

    attention_out, att = SelfAttention(100, 1).hybrid_forward(mx.sym, stacked_out)
    flatten = mx.symbol.flatten(attention_out, "flatten")

    context_features = mx.symbol.broadcast_mul((1 + bkidx_embed + time_embed + weather_embed + feels_embed),
                                               flatten, name='latent_cross')
    ac1 = mx.symbol.Activation(data=context_features, act_type="relu", name="relu1")
    dropout1 = mx.symbol.Dropout(data=ac1, p=0.3, name="dropout1")
    fc1 = mx.symbol.FullyConnected(data=dropout1, num_hidden=int(n_plus), name='fc1')
    rec_model = mx.symbol.SoftmaxOutput(data=fc1, label=y_true, name='output')

    mod = mx.mod.Module(symbol=rec_model,
                        data_names=['pluids', 'bkidx', 'timeidx', 'feels_bucket', 'weatheridx'],
                        label_names=['output_label'],
                        context=[mx.cpu()])
    return mod


def get_metrics(config):
    return 'accuracy'


def create_config():
    config = {
        "seed": 123,
        "num_workers": 2,
        "batch_size": 16000,
        "epochs": 5,
        "kvstore": "dist_sync",
        "init": mx.init.Xavier(rnd_type="gaussian"),
        "optimizer": 'adagrad',
        "log_interval": 500,
        "model_prefix": 'drivethru_attention_d',
    }
    return config


def get_train_test_path():
    session = boto3.Session(
        aws_access_key_id="access_key",
        aws_secret_access_key="secret_access_key",
    )
    s3_client = session.client('s3', verify=False)

    keys = []
    resp = s3_client.list_objects_v2(Bucket='bucket', Prefix='path')
    for obj in resp['Contents']:
        keys.append(obj['Key'])
    files = list(dict.fromkeys(keys))
    train_keys = files[:880]
    print("Train files: ", len(train_keys))
    train_files = ["s3://" + 'bucket/' + file for file in train_keys]
    train_paths = ",".join(train_files)

    test_keys = files[880:]
    print("Test files: ", len(test_keys))
    test_files = ["s3://" + 'bucket/' + file for file in test_keys]
    test_paths = ",".join(test_files)

    return train_paths, test_paths


if __name__ == "__main__":
    config = create_config()

    sc = init_spark_on_yarn(
        hadoop_conf="/opt/work/hadoop-2.7.2/etc/hadoop",
        conda_name="mxnet",
        # 1 executor for ray head node. The remaining executors for raylets.
        # Each executor is given enough cores to be placed on one node.
        # Each MXNetRunner will run in one executor, namely one node.
        num_executor=config["num_workers"],
        executor_cores=44,
        executor_memory="10g",
        driver_memory="2g",
        driver_cores=16,
        extra_executor_memory_for_ray="5g",
        extra_python_lib="mxnet_runner.py")
    ray_ctx = RayContext(sc=sc,
                         object_store_memory="10g",
                         env={"OMP_NUM_THREADS": "22",
                              "KMP_AFFINITY": "granularity=fine,compact,1,0",
                              "AWS_ACCESS_KEY_ID":"access_key",
                              "AWS_SECRET_ACCESS_KEY": "secret_access_key"
                              })
    ray_ctx.init()

    train_path, test_path = get_train_test_path()
    train_shards = zoo.xshard.pandas.read_json(train_path, ray_ctx)
    train_shards.apply(prepare_data, True)
    train_shards.repartition(config["num_workers"])

    test_shards = zoo.xshard.pandas.read_json(test_path, ray_ctx)
    test_shards.apply(prepare_data, True)
    test_shards.repartition(config["num_workers"])

    trainer = MXNetTrainer(config, train_shards, test_shards, get_model, metrics_creator=get_metrics,
                           runner_cpus=22)
    train_stats = trainer.train()
    for stat in train_stats:
        if len(stat.keys()) > 1:  # Worker
            print(stat)
    ray_ctx.stop()
    sc.stop()
