import numpy as np

from models.base import BaseModel


class QTable(BaseModel):
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
