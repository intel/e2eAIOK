import tensorflow as tf
import horovod.tensorflow.keras as hvd

from tf.data.features import FeatureMeta
from tf.data.dataloader import BinDataset

def get_data(train, valid, features):
    train_dataset = BinDataset(train, features, 32)
    eval_dataset = BinDataset(valid, features, 32)
    return train_dataset, eval_dataset

def get_model(features):
    NUMERIC_COLUMNS = features.numerical_keys
    CATEGORICAL_COLUMNS = features.categorical_keys
    categorical_meta = features.categorical_meta

    wide_weighted_outputs = []
    deep_embedding_outputs = []
    numeric_dense_inputs = []
    features = {}

    for col in NUMERIC_COLUMNS+CATEGORICAL_COLUMNS:
        features[col] = tf.keras.Input(shape=(1,),
                                           batch_size=None,
                                           name=col,
                                           dtype=tf.float32 if col in NUMERIC_COLUMNS else tf.int32,
                                           sparse=False)

    for key in CATEGORICAL_COLUMNS:
        wide_weighted_outputs.append(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
            categorical_meta[key]['voc_size'], 1, input_length=1)(features[key])))
        deep_embedding_outputs.append(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
            categorical_meta[key]['voc_size'], categorical_meta[key]['emb_dim'])(features[key])))
    for key in NUMERIC_COLUMNS:
        numeric_dense_inputs.append(features[key])

    categorical_output_contrib = tf.keras.layers.add(wide_weighted_outputs,
                                                     name='categorical_output')
    numeric_dense_tensor = tf.keras.layers.concatenate(
        numeric_dense_inputs, name='numeric_dense')

    dnn = tf.keras.layers.concatenate(numeric_dense_inputs+deep_embedding_outputs)
    for unit_size in [32, 16]:
        dnn = tf.keras.layers.Dense(units=unit_size, activation='relu')(dnn)
        dnn = tf.keras.layers.Dropout(rate=0.2)(dnn)
        dnn = tf.keras.layers.BatchNormalization()(dnn)
    dnn = tf.keras.layers.Dense(units=1)(dnn)
    dnn_model = tf.keras.Model(inputs=features,
                               outputs=dnn)
    linear_output = categorical_output_contrib + tf.keras.layers.Dense(1)(numeric_dense_tensor)

    linear_model = tf.keras.Model(inputs=features,
                                  outputs=linear_output)

    model = tf.keras.experimental.WideDeepModel(
        linear_model, dnn_model, activation='sigmoid')

    return model

# def get_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Concatenate(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(1),
#     ])
#     return model

def train(model, dataset, loss_fn, optimizer):
    for step, (x,y) in enumerate(dataset):
        print(f'step: {step}')
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step == 2:
            break

def main():
    hvd.init()
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(32)

    train_file = '/home/vmagent/app/dataset/criteo/train/train_data.bin'
    eval_file = '/home/vmagent/app/dataset/criteo/valid/test_data.bin'
    features = FeatureMeta('tf/data/criteo_meta.yaml')

    train_dataset, eval_dataset = get_data(train_file, eval_file, features)
    model = get_model(features)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train(model, train_dataset, loss_fn, optimizer)

    model.save('model/tensorflow')

if __name__ == '__main__':
    main()