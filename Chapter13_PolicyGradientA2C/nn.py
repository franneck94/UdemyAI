from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class NN(Model):
    def __init__(self, num_observations, num_actions,
                 num_values, lr_actor, lr_critic):
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        state = Input(shape=(num_observations,))
        x = Dense(24)(state)
        x = Activation("relu")(x)

        actor_x = Dense(self.num_actions)(x)
        actor_out = Activation("softmax")(actor_x)
        self.actor = Model(
            inputs=state,
            outputs=actor_out
        )
        self.actor.summary()
        self.actor.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.lr_actor)
        )

        critic_x = Dense(self.num_values)(x)
        critic_out = Activation("linear")(critic_x)
        self.critic = Model(
            inputs=state,
            outputs=critic_out
        )
        self.critic.summary()
        self.critic.compile(
            loss="mse",
            optimizer=Adam(
                learning_rate=self.lr_critic
            )
        )

    # Actor Functions
    def train_actor(self, states, actions):
        self.actor.fit(states, actions, verbose=0)

    def predict_actor(self, states):
        return self.actor.predict(states)

    # Critic Functions
    def train_critic(self, states, values):
        self.critic.fit(states, values, verbose=0)

    def predict_critic(self, states):
        return self.critic.predict(states)
