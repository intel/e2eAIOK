# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Modifications copyright Intel
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

import tensorflow as tf


def wide_deep_model(args, features):
    wide_columns, deep_columns = features.get_feature_columns()
    NUMERIC_COLUMNS = features.numerical_keys
    CATEGORICAL_COLUMNS = features.categorical_keys
    categorical_meta = features.categorical_meta

    wide_weighted_outputs = []
    numeric_dense_inputs = []
    wide_columns_dict = {}
    deep_columns_dict = {}
    features = {}

    for col in wide_columns:
        features[col.key] = tf.keras.Input(shape=(1,),
                                           batch_size=None,
                                           name=col.key,
                                           dtype=tf.float32 if col.key in NUMERIC_COLUMNS else tf.int32,
                                           sparse=False)
        wide_columns_dict[col.key] = col
    for col in deep_columns:
        is_embedding_column = ('key' not in dir(col))
        key = col.categorical_column.key if is_embedding_column else col.key

        if key not in features:
            features[key] = tf.keras.Input(shape=(1,),
                                           batch_size=None,
                                           name=key,
                                           dtype=tf.float32 if col.key in NUMERIC_COLUMNS else tf.int32,
                                           sparse=False)
        deep_columns_dict[key] = col

    for key in wide_columns_dict:
        if key in CATEGORICAL_COLUMNS:
            wide_weighted_outputs.append(tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
                categorical_meta[key]['voc_size'], 1, input_length=1)(features[key])))
        else:
            numeric_dense_inputs.append(features[key])

    categorical_output_contrib = tf.keras.layers.add(wide_weighted_outputs,
                                                     name='categorical_output')
    numeric_dense_tensor = tf.keras.layers.concatenate(
        numeric_dense_inputs, name='numeric_dense')
    deep_columns = list(deep_columns_dict.values())

    dnn = tf.keras.layers.DenseFeatures(feature_columns=deep_columns)(features)
    for unit_size in args.deep_hidden_units:
        dnn = tf.keras.layers.Dense(units=unit_size, activation='relu')(dnn)
        dnn = tf.keras.layers.Dropout(rate=args.deep_dropout)(dnn)
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
