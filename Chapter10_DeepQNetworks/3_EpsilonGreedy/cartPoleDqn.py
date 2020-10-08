from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


class DQN():
    def __init__(self, state_shape, num_actions, lr):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr

        state = Input(shape=self.state_shape)
        x = Dense(24)(state)
        x = Activation("relu")(x)
        x = Dense(24)(x)
        x = Activation("relu")(x)
        out = Dense(self.num_actions)(x)
        self.model = Model(inputs=state, outputs=out)
        self.model.compile(loss="mse", optimizer=Adam(lr=self.lr))

    def train(self, states, q_values):
        self.model.fit(states, q_values, verbose=0)

    def predict(self, states):
        return self.model.predict(states)

    def update_model(self, other_model):
        self.model.set_weights(other_model.get_weights())

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)


if __name__ == "__main__":
    d = DQN(state_shape=4, num_actions=2, lr=0.001)
