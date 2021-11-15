import numpy as np

from keras.optimizer_v2.adam import Adam
from tensorflow import keras
from tensorflow.keras import layers

from models.base import BaseModel


class NN_24_12_softmax(BaseModel):
    def __init__(self, model_options):
        super().__init__()
        self.state_shape = model_options['state_shape']
        self.action_size = model_options['action_size']
        self.learning_rate = float(model_options['learning_rate'])

        self.model = self._create()

    def _create(self):
        """ builds the model using keras"""
        model = keras.Sequential()

        # input shape is of observations
        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        model.add(layers.Dense(24, input_shape=self.state_shape, activation="relu"))
        model.add(layers.Dense(12, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="softmax"))

        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def predict(self, state):
        state_array = np.asarray(state).reshape(1, -1)
        return self.model(state_array).numpy().flatten()

    def fit(self, x, y):
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        history = self.model.train_on_batch(x_array, y_array)
        return history
