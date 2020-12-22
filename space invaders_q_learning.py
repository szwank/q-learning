from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.keras import layers, models, optimizers
from tqdm import tqdm


def to_gryscale(img):
    return np.mean(img, axis=2).astype('uint8')


def downsample(img):
    return img[::2, ::2]


def crop_image(img):
    return img[30:-10, 6:-6]


class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def extend(self, elements):
        for element in elements:
            self.append(element)

    def to_list(self):
        return self[:]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        else:
            if idx >= 0:
                return self.data[(self.start + idx) % len(self.data)]
            else:
                if idx < -len(self.data):
                    raise ValueError("Incorrect index- out of range")
                return self.data[(self.end - idx + 1) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ExperienceReplay:
    def __init__(self, size, n_state_frames):
        """n_state_frames determines how much frames in state."""
        # States are saved as separete frames. Each state consists of n_state_frames.
        self.n_state_frames = n_state_frames

        # Buffer for states needs to be bigger. We store there n_state_frames
        # per state and extra frame as future state
        self.states = RingBuf(size + n_state_frames)
        self.actions = RingBuf(size)
        self.rewards = RingBuf(size)
        self.teminate_state = RingBuf(size)

    def __len__(self):
        return len(self.actions)

    def add(self, state: List[np.array], action: np.array, new_frame: np.array, reward: int,
            terminate: bool):
        # add only new_frame, the rest of them are all alreaty in buffer
        if len(self.states) == 0:
            self.states.extend(state)
        self.states.append(new_frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.teminate_state.append(terminate)

    def sample_batch(self, n):
        """Returns batch of samples."""
        states = []
        actions = []
        rewards = []
        next_states = []
        terminate_state = []
        # remove -2 because -1 is from counting from 0 and -1 is from counting most
        # current frame as first one and foutrue frame as +1 frame
        for i in np.random.randint(self.n_state_frames, len(self.states) - 2, n):
            state, action, reward, next_state, terminate = self.get_sample(i)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminate_state.append(terminate)
        return np.swapaxes(np.array(states), 1, 3), np.array(actions), np.array(rewards), \
               np.swapaxes(np.array(next_states), 1, 3), np.array(terminate_state)

    def get_sample(self, idx):
        state = self.states[idx - self.n_state_frames:idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_state = self.states[idx - self.n_state_frames+1:idx+1]
        terminate = self.teminate_state[idx]

        return np.array(state), action, reward, next_state, terminate


class QLearner:
    def __init__(self, env_name='BreakoutDeterministic-v4', preprocess_funcs=[], replay_size=1000000,
                 screen_size=(74, 85), n_state_frames=4, batch_size=32, gamma=0.99):
        self.env = gym.make(env_name)
        self.n_state_frames = n_state_frames
        model_input_size = (*screen_size, n_state_frames)
        self.n_actions = self.env.action_space.n
        self.model = self._get_model(model_input_size, self.n_actions)
        self.memory = ExperienceReplay(replay_size, n_state_frames)
        self.iteration = 1
        self.preprocess_funcs = preprocess_funcs
        self.batch_size = batch_size
        self.state = RingBuf(n_state_frames)
        self.gamma = gamma
        self.iteration = None

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

    def train(self, n_iterations):
        print('Training Started')
        self.iteration = 1

        for _ in tqdm(range(n_iterations)):
            self.epoch()
            self.iteration += 1

    def epoch(self):
        self.env.reset()
        self.set_init_state()

        terminate = False
        while not terminate:
            action = self.choose_action()
            new_frame, reward, terminate, _ = self.env.step(action)
            action_mask = self.encode_action(action)
            self.update_memory(action_mask, new_frame, reward, terminate)
            # update state
            self.update_state(new_frame)

        # Sample and fit
        batch = self.memory.sample_batch(32)
        self.fit_batch(*batch)

    def update_memory(self, action_mask, new_frame, reward, terminate):
        pp_new_frame = self.preprocess_image(new_frame)
        reward = self.clip_reward(reward)
        self.memory.add(self.state.to_list(), action_mask, pp_new_frame, reward, terminate)

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
        return max(0.1, 1 - (self.iteration - 1) * 1 / 1000000)

    def choose_best_action(self):
        prediction = self.model.predict(self.state.data, np.ones((1, self.n_actions)))
        print(f'prediction: {prediction}')
        return np.argmax(prediction)

    def update_state(self, screen):
        preprocessed_screen = self.preprocess_image(screen)
        self.state.append(preprocessed_screen)

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
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.model.fit(
            [start_states, actions], actions * Q_values[:, None],
            epochs=1, batch_size=len(start_states), verbose=0
        )

    


learner = QLearner(preprocess_funcs=[to_gryscale, crop_image, downsample], replay_size=1000)

learner.train(1)
