from time import sleep

import gym
import numpy as np
import random

from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tqdm import tqdm

from queues import RingBuf, ExperienceReplay


def to_gryscale(img):
    return np.mean(img, axis=2).astype('uint8')


def downsample(img):
    return img[::2, ::2]


def crop_image(img):
    return img[30:-10, 6:-6]


class QLearner:
    def __init__(self, env_name='BreakoutDeterministic-v4', preprocess_funcs=[], replay_size=1000000,
                 screen_size=(74, 85), n_state_frames=4, batch_size=32, gamma=0.99):
        self.env = gym.make(env_name)
        self.n_state_frames = n_state_frames
        model_input_size = (*screen_size, n_state_frames)
        self.n_actions = self.env.action_space.n
        self.model = self._get_model(model_input_size, self.n_actions)
        self.memory = ExperienceReplay(replay_size, n_state_frames)
        self.preprocess_funcs = preprocess_funcs
        self.batch_size = batch_size
        self.state = RingBuf(n_state_frames)
        self.gamma = gamma
        self.iteration = None
        self.episode = None
        self.rewards = []

    def _get_model(self, input_size, n_actions):
        """Returns short conv model with mask at the end of the network."""
        screen_input = layers.Input(input_size)
        actions_input = layers.Input(n_actions)

        x = layers.Conv2D(16, (3, 3), padding='same')(screen_input)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), padding='same')(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.ReLU()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(n_actions)(x)
        x = layers.Multiply()([x, actions_input])

        model = models.Model(inputs=[screen_input, actions_input], outputs=x)
        optimizer = optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        model.compile(optimizer, loss='mse')
        return model

    def train(self, n_iterations, plot=True, iteration=1, verbose=True):
        print('Training Started')
        self.iteration = iteration
        self.episode = 1

        for _ in tqdm(range(n_iterations)):
            self.epoch()
            self.iteration += 1

            if verbose:
                self.print_stats()
            if plot:
                self.plot()

    def epoch(self):
        self.env.reset()
        self.set_init_state()
        epoch_rewards = []

        terminate = False
        while not terminate:
            action = self.choose_action()
            new_frame, reward, terminate = self.env_step(action)
            epoch_rewards.append(reward)
            action_mask = self.encode_action(action)
            self.update_memory(action_mask, new_frame, reward, terminate)
            # update state
            self.update_state(new_frame)
            self.episode += 1


            if len(self.memory) >= 32:
                # Sample and fit
                start_states, actions, rewards, next_states, is_terminal = self.memory.sample_batch(32)
                start_states = start_states/255
                next_states = next_states/255
                self.fit_batch(start_states, actions, rewards, next_states, is_terminal)
        print(float(np.mean(epoch_rewards)))
        self.rewards.append(float(np.mean(epoch_rewards)))

    def plot(self):
        plt.plot(np.arange(1, len(self.rewards) + 1, 1), self.rewards)
        plt.show(block=False)

    def env_step(self, action: int):
        """Call env.step and return preprocessed frame of new state."""
        new_frame, reward, terminate, _ = self.env.step(action)
        new_frame = self.preprocess_image(new_frame)
        reward = self.clip_reward(reward)
        return new_frame, reward, terminate

    def update_memory(self, action_mask, new_frame, reward, terminate):
        self.memory.add(self.state.to_list(), action_mask, new_frame, reward, terminate)

    def clip_reward(self, reward):
        return np.sign(reward)

    def encode_action(self, action: int):
        action_mask = np.zeros((self.n_actions, ))
        action_mask[action] = 1
        return action_mask

    def set_init_state(self):
        screen = self.get_current_screen()
        # flush current state with starting screen
        for _ in range(self.n_state_frames):
            self.state.append(screen)

    def get_current_screen(self):
        screen = self.env.render(mode='rgb_array')
        return self.preprocess_image(screen)

    def preprocess_image(self, img):
        for function in self.preprocess_funcs:
            img = function(img)
        return img

    def choose_action(self):
        """Choose action agent will take."""
        epsilon = self.get_epsilon()

        # Choose the action
        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.choose_best_action()
        return action

    def get_epsilon(self):
        return max(0.1, 1 - (self.episode - 1) * 1 / 1000000)

    def choose_best_action(self) -> int:
        state = np.expand_dims(np.swapaxes(self.state.to_list(), 0, 2), 0)
        state = state/255
        prediction = self.model.predict_on_batch([state, np.ones((1, self.n_actions))])
        return int(np.argmax(prediction))

    def update_state(self, screen):
        self.state.append(screen)

    def fit_batch(self, start_states, actions, rewards, next_states,
                  is_terminal):
        """Do one deep Q learning iteration.

        Params:
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

        """
        assert (0 <= start_states.all() <= 1)
        assert (0 <= next_states.all() <= 1)
        if rewards.any():
            a = 1

        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.model.train_on_batch([start_states, actions], actions * Q_values[:, None])

    def save_model(self):
        self.model.save('model')

    def evaluate(self):
        self.env.reset()
        self.set_init_state()

        terminate = False
        while not terminate:
            action = self.choose_best_action()
            self.env.render()
            sleep(0.05)
            new_frame, reward, terminate = self.env_step(action)
            self.update_state(new_frame)

    def print_stats(self):
        print(f'iteration: {self.iteration}, episode: {self.episode}, epsilon: {self.get_epsilon()}')

learner = QLearner(preprocess_funcs=[to_gryscale, crop_image, downsample])

learner.train(10, plot=False)
#
# while True:
#     learner.evaluate()

