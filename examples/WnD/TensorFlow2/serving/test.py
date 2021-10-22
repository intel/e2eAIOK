import tensorflow as tf
import numpy as np
from zoo.serving.client import InputQueue, OutputQueue
import time


def save_model():
    inputs = {}
    inputs['input1'] = tf.keras.Input(shape=(1,), batch_size=None, name='input1', dtype=tf.float32, sparse=False)
    inputs['input2'] = tf.keras.Input(shape=(1,), batch_size=None, name='input2', dtype=tf.float32, sparse=False)

    dnn = tf.keras.layers.concatenate([inputs['input1'], inputs['input2']])
    dnn = tf.keras.layers.Dense(units=32, activation='relu')(dnn)
    output = tf.keras.layers.Dense(units=1)(dnn)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile()
    tf.keras.models.save_model(model, '/tmp/model')

def predict_model():
    model = tf.keras.models.load_model('/tmp/model')
    inputs = {'input1': tf.convert_to_tensor([[1], [2]]), 'input2': tf.convert_to_tensor([[2], [3]])}
    preds = model.predict(inputs)
    print(preds)

def serving():
    input_api = InputQueue()
    output_api = OutputQueue()
    output_api.dequeue()

    # data = {'input1': np.array([[1], [2]]), 'input2': np.array([[2], [3]])} # cluster serving did not support batch size dim in input
    data = {'input1': np.array([1]), 'input2': np.array([2])}
    input_api.enqueue('test', **data)
    time.sleep(1)
    preds = output_api.dequeue()
    print(preds)
    

if __name__ == '__main__':
    # save_model()
    # predict_model()
    serving()