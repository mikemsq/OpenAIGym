import numpy as np

from keras.optimizer_v2.adam import Adam
from tensorflow import keras
from tensorflow.keras import layers


class BaseModel:
    def __init__(self):
        pass

    def predict(self, state):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError


class CartPoleModel(BaseModel):
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
        model.add(layers.Dense(24, input_shape=self.state_shape, activation="relu"))
        # add a relu layer
        model.add(layers.Dense(12, activation="relu"))

        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        model.add(layers.Dense(self.action_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def predict(self, state):
        return self.model(state).numpy()

    def fit(self, x, y):
        history = self.model.train_on_batch(x, y)
        return history


class QTableModel(BaseModel):
    def __init__(self, model_options):
        super().__init__()
        self.observation_space = model_options['observation_space']
        self.action_size = model_options['action_size']

        self.model = self._create()

    def _create(self):
        # create a multidimensional Q table

        if type(getattr(self.observation_space, 'spaces', None)) == tuple:
            q_table_shape = tuple(map(lambda x: x.n, self.observation_space.spaces))
        else:
            q_table_shape = (self.observation_space.n, )
        q_table_shape += (self.action_size, )

        # model = np.random.random(q_table_shape) / 1000
        model = np.zeros(q_table_shape) + 1e-7
        return model

    @staticmethod
    def _state2index(state):
        if type(state) == tuple:
            index = tuple(map(lambda i: int(i), state))
        else:
            index = state
        return index

    def predict(self, state):
        index = self._state2index(state)
        return np.copy(self.model[index])

    def fit(self, x, y):
        index = self._state2index(x)
        history = self.model[index] = y
        return history
