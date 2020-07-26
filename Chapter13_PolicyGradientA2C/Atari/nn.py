import gym
import numpy as np
import matplotlib.pyplot as plt
import  collections
 
from keras.models import *
from keras.layers import *
from keras.layers import *
from keras.optimizers import *
   
class NN(Model):
    def __init__(self, img_shape, num_actions, num_values, lr_actor, lr_critic):
        super(NN, self).__init__()
        self.img_shape = img_shape
        self.num_actions = num_actions
        self.num_values = num_values
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        img = Input(shape=img_shape)
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4))(img)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation("relu")(x)

        actor_x = Dense(self.num_actions)(x)
        actor_out = Activation("softmax")(actor_x)
        self.actor = Model(inputs=img, outputs=actor_out)
        self.actor.summary()
        self.actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.lr_actor))

        critic_x = Dense(self.num_values)(x)
        critic_out = Activation("linear")(critic_x)
        self.critic = Model(inputs=img, outputs=critic_out)
        self.critic.summary()
        self.critic.compile(loss="mse", optimizer=Adam(lr=self.lr_critic))

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