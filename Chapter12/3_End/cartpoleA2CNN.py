import numpy as np
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


class Actor(Model):
    def __init__(
        self, num_observations: int, num_actions: int, learning_rate: float
    ) -> None:
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()

    def build_model(self) -> Model:
        actor_in = Input(shape=self.num_observations)
        x = Dense(units=24)(actor_in)
        x = Activation("relu")(x)
        x = Dense(self.num_actions)(x)
        actor_out = Activation("softmax")(x)
        model = Model(inputs=actor_in, outputs=actor_out)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
        )
        return model

    def call(self, states: np.ndarray) -> np.ndarray:
        return self.internal_model(states).numpy()

    def fit(self, states: np.ndarray, actions: np.ndarray) -> None:
        self.internal_model.fit(x=states, y=actions, verbose=0)

    def update_model(self, other_model: Model) -> None:
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str) -> None:
        self.internal_model.load_weights(path)

    def save_model(self, path: str) -> None:
        self.internal_model.save_weights(path)


class Critic(Model):
    def __init__(
        self, num_observations: int, num_values: int, learning_rate: float
    ) -> None:
        super().__init__()
        self.num_observations = num_observations
        self.num_values = num_values
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()

    def build_model(self) -> Model:
        critic_in = Input(shape=self.num_observations)
        x = Dense(units=24)(critic_in)
        x = Activation("relu")(x)
        critic_out = Dense(self.num_values)(x)
        model = Model(inputs=critic_in, outputs=critic_out)
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def call(self, states: np.ndarray) -> np.ndarray:
        return self.internal_model(states).numpy()

    def fit(self, states: np.ndarray, values: np.ndarray) -> None:
        self.internal_model.fit(x=states, y=values, verbose=0)

    def update_model(self, other_model: Model) -> None:
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path: str) -> None:
        self.internal_model.load_weights(path)

    def save_model(self, path: str) -> None:
        self.internal_model.save_weights(path)


if __name__ == "__main__":
    actor = Actor(num_observations=4, num_actions=2, learning_rate=0.001)
    actor.internal_model.summary()
    critic = Critic(num_observations=4, num_values=1, learning_rate=0.005)
    critic.internal_model.summary()
