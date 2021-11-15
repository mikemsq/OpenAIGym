class BaseModel:
    def __init__(self):
        pass

    def predict(self, state):
        raise NotImplementedError

    def fit(self, x, y):
        raise NotImplementedError
