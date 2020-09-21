from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam


class NN(Model):
    def __init__(self, num_observations, num_actions, num_values, lr_actor, lr_critic):
        super(NN, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
