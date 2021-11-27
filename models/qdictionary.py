from collections import defaultdict
import numpy as np

from models.base import BaseModel


class QDictionary(BaseModel):
    def __init__(self, model_options):
        super().__init__()
        self.observation_space = model_options['observation_space']
        self.action_size = model_options['action_size']

        self.model = defaultdict(lambda: np.zeros(self.action_size))

    def predict(self, state):
        return np.copy(self.model[state])

    def fit(self, x, y):
        history = None
        for i in range(len(x)):
            history = self.model[x[i]] = y[i]
        return history
