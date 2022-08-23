import tensorflow as tf

class ModelLoader:
    def __init__(self, model_dir, model_type):
        self.model_dir = model_dir
        self.model_type = model_type

    def load_model(self):
        return tf.keras.models.load_model(self.model_dir)
