import gc
from time import sleep
from typing import List

import gym
import matplotlib
from tensorflow.python.keras.models import clone_model

from networks import get_original_model

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random

from tqdm import tqdm

from preprocessing import to_gryscale, downsample, crop_image
from queues import RingBuf, PrioritizedExperienceReplay, ExperienceReplay


class QLearner:
    def __init__(self, model, env_name='BreakoutDeterministic-v4', preprocess_funcs=[], replay_size=1000000,
                 n_state_frames=4, batch_size=32, gamma=0.99, replay_start_size=50000,
                 final_exploration_frame=1000000, update_between_n_episodes=4, update_network_period=10000,
                 max_game_length=-1):
        """
        Params:
        - model: agent NN model. Model should have two inputs: first one for states (its size
        depend on preprocess_funcs and its order and value of n_state_frames argument) should be for action
        mask(binary vector multiplied by output, used for training). last dimension of first input should
        be equal to n_state_frames. Output of network should be equal to number of possible actions in
        passed environment
        - env_name: name of gym environment(https://github.com/openai/gym/wiki/Table-of-environments) on
        with agent will be trained
        - preprocess_funcs: list of functions used to preprocess state of environment
        - replay_size: size of experience replay buffer
        - n_state_frames: number of states passed at once to network for one forward propagation
        - batch_size: number of states-actions-rewards passed on network train epoch (minibatch size)
        - gamma: gamma parameter in Q-Value equation: QValue[s] = reward + gamma * max(QValue[s+1])
        - replay_start_size: initial size to with experience replay will be filled with random actions
         before training
        - final_exploration_frame: last frame number before epsilon will be equal to 0.1
        - update_between_n_episodes: number of played games after with network will be updated on minibatch
        - update_network_period: number of frames after with target network will be updated with weights
        from online model
        - max_game_length: max number of states before game termination.When game is terminated this
        way last state WILL NOT be set as termination state

        """
        # training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_start_size = replay_start_size
        self.final_exploration_frame = final_exploration_frame
        self.n_state_frames = n_state_frames
        self.n_games_between_update = update_between_n_episodes
        self.update_network_period = update_network_period
        self.max_game_length = max_game_length

        # functional
        self.iteration = None
        self.n_actions_taken = None
        self.trained_on_n_frames = 0
        self.rewards = []
        self.replay_size = replay_size
        self.fig = None
        self.points = None

        # other stuff initialization
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.online_model = model
        self.target_model = clone_model(model)
        self.model_input_shape = model.input_shape
        self.model_state_input_shape = self.model_input_shape[0][1:]
        self.memory = PrioritizedExperienceReplay(self.replay_size, n_state_frames)
        self.preprocess_funcs = preprocess_funcs
        self.state = RingBuf(n_state_frames)

    def train(self, n_frames=1000000, plot=True, iteration=1, render_period=100):
        print('Training Started')
        self.iteration = iteration
        self.n_actions_taken = 0
        fig = None
        rewards = []

        print("Initialization of experience replay")
        self._init_experience_replay()

        update_target_network_after_n_iteration = round(self.update_network_period / self.batch_size)

        with tqdm(total=n_frames) as progress_bar:
            print("Training started")
            while self.trained_on_n_frames < n_frames:

                self.episode()
                self.iteration += 1

                gc.collect()

                progress_bar.update(self.trained_on_n_frames - progress_bar.last_print_n)
                progress_bar.set_description_str(self.get_stats())

                if self.iteration % update_target_network_after_n_iteration == 0:
                    print("Model switched")
                    self.update_target_model_weights()
                    self.save_model()
                if plot:
                    self.plot()
                if self.iteration % render_period == 0:
                    self.visual_evaluate()

                if self.iteration % 10 == 0:
                    n = 10
                    evaluation_score = self.evaluate(n)
                    rewards.append(sum(evaluation_score)/len(evaluation_score))

                    if fig is None:
                        fig, ax = plt.subplots(1, 1)
                        plt.show(block=False)
                        plt.draw()
                        plt.title(f'Average games reward per {n} games- evaluation')
                        plt.xlabel(f'Iteration ({n} games, one update)')
                        plt.ylabel('Average reward')

                        points = plt.plot(np.arange(1, len(rewards) + 1, 1), rewards)

                    points[0].set_data(np.arange(1, len(rewards) + 1, 1), rewards)
                    ax.set_xlim(1, len(rewards) + 1)
                    ax.set_ylim(min(rewards), max(rewards))
                    fig.canvas.draw()

    def _init_experience_replay(self):
        """Fill partially experience replay memory with states-actions by plying the game."""
        with tqdm(total=self.replay_start_size) as progress_bar:
            while len(self.memory) < self.replay_start_size:
                self._play_game(kind='init')
                progress_bar.update(len(self.memory) - progress_bar.last_print_n)

    def episode(self):
        """Simulate on episode of training"""
        games_played = 0
        rewards = []
        while games_played < self.n_games_between_update:
            game_rewards = self._play_game(render=False)

            games_played += 1

            rewards.append(sum(game_rewards))

        self.rewards.append(sum(rewards)/len(rewards))

        self._update_network()

    def _play_game(self, kind: str = 'train', render=False) -> List[int or float]:
        """Play one game until termination state. Returns gained rewards per action."""
        self.reset_environment()

        game_rewards = []
        terminate = False
        game_length = 0

        game_memory = ExperienceReplay(self.replay_size, self.n_state_frames)
        while not terminate and not self._terminate_game(game_length):
            action = self.choose_action()
            new_frame, reward, terminate = self.env_step(action, render)
            game_rewards.append(reward)
            action_mask = self.encode_action(action)
            game_memory.add(self.state.to_list(), action_mask, new_frame, reward, terminate)
            # update state
            self.update_state(new_frame)
            self.n_actions_taken += 1

        self.update_memory(game_memory, kind)

        return game_rewards

    def reset_environment(self):
        state = self.env.reset()
        self.set_init_state(state)

    def set_init_state(self, state):
        # flush current state with starting screen
        for _ in range(self.n_state_frames):
            self.state.append(state)

    def _terminate_game(self, game_length: int) -> bool:
        """Returns false if max_game_length == -1 or game_lenght is less than max_game_length,
        otherwise returns True."""
        return self.max_game_length != -1 and not game_length <= self.max_game_length

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
        return max(0.1, 1 - self.trained_on_n_frames / self.final_exploration_frame)

    def choose_best_action(self) -> int:
        # switch channel axis and add additional dimension(batch size) expected by network
        prediction = self._get_current_state_prediction()
        return int(np.argmax(prediction))

    def _get_current_state_prediction(self):
        state = np.expand_dims(np.moveaxis(self.state.to_list(), 0, len(self.model_state_input_shape)-1), 0)
        return self._get_prediction(state)

    def _get_prediction(self, states):
        return self.online_model.predict_on_batch([states, np.ones((len(states), self.n_actions))])

    def env_step(self, action: int, render=False):
        """Call env.step and return preprocessed frame of new state."""
        if render is True:
            self.env.render()
        new_frame, reward, terminate, _ = self.env.step(action)
        new_frame = self.preprocess_image(new_frame)
        reward = self.clip_reward(reward)
        return new_frame, reward, terminate

    def preprocess_image(self, img):
        for function in self.preprocess_funcs:
            img = function(img)
        return img

    def clip_reward(self, reward):
        return np.sign(reward)

    def encode_action(self, action: int):
        action_mask = np.zeros((self.n_actions,))
        action_mask[action] = 1
        return action_mask

    def update_state(self, screen):
        self.state.append(screen)

    def update_memory(self, temp_memory: ExperienceReplay, kind):
        states, actions, rewards, next_states, terminate_state = self._get_entire_memory(temp_memory)
        next_states = np.array(next_states)

        if kind == 'init':
            errors = rewards
        else:
            action = np.array(actions, dtype=bool)
            states = np.array(states)
            prediction = self._get_prediction(states)[action]

            next_state_value = self.target_model.predict([next_states, np.ones(actions.shape)])
            next_action = self.online_model.predict([next_states, np.ones(actions.shape)])

            errors = np.abs(prediction - (rewards + self.gamma * np.take(next_state_value, np.argmax(next_action, axis=1))))

        self.memory.extend(np.moveaxis(states, 1, len(self.model_state_input_shape)), actions, next_states[..., 0], rewards, terminate_state, errors**2)

    def _get_entire_memory(self, memory):
        states, actions, rewards, next_states, terminate_states = memory.get_all()
        return np.moveaxis(states, 1, len(self.model_state_input_shape)),\
               actions, rewards, np.moveaxis(next_states, 1, len(self.model_state_input_shape)), terminate_states

    def _update_network(self):
        # Sample and fit
        start_states, actions, rewards, next_states, is_terminal = self.sample_memory()
        self.fit_batch(start_states, actions, rewards, next_states, is_terminal)
        self.trained_on_n_frames += self.batch_size

    def sample_memory(self):
        start_states, actions, rewards, next_states, is_terminal = self.memory.sample_batch(self.batch_size)
        return np.moveaxis(start_states, 1, len(self.model_state_input_shape)),\
               actions, rewards, np.moveaxis(next_states, 1, len(self.model_state_input_shape)), is_terminal

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
        next_Q_targets = self.online_model.predict([next_states, np.ones(actions.shape)])
        next_Q_values = self.target_model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        greedy_policy_action = np.argmax(next_Q_targets, axis=1)
        Q_values = rewards + self.gamma * np.take(next_Q_values, greedy_policy_action)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.online_model.train_on_batch([start_states, actions], actions * Q_values[:, None])

    def update_target_model_weights(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def get_stats(self):
        return f'iteration: {self.iteration}, number of actions taken: {self.n_actions_taken}, ' \
               f'epsilon: {self.get_epsilon()}, trained on n frames: {self.trained_on_n_frames}'

    def plot(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
            plt.show(block=False)
            plt.draw()
            plt.title(f'Average games reward per {self.n_games_between_update} games')
            plt.xlabel(f'Iteration ({self.n_games_between_update} games, one update)')
            plt.ylabel('Average reward')

            self.points = plt.plot(np.arange(1, len(self.rewards) + 1, 1), self.rewards)

        self.points[0].set_data(np.arange(1, len(self.rewards) + 1, 1), self.rewards)
        self.ax.set_xlim(1, len(self.rewards) + 1)
        self.ax.set_ylim(min(self.rewards), max(self.rewards))
        self.fig.canvas.draw()

    def save_model(self):
        self.online_model.save('model')

    def visual_evaluate(self):
        self.reset_environment()

        terminate = False
        while not terminate:
            action = self.choose_best_action()
            self.env.render()
            sleep(0.05)
            new_frame, reward, terminate = self.env_step(action)
            self.update_state(new_frame)

    def evaluate(self, n) -> List:
        """Evaluate agent across n episodes. Returns list of game scores."""
        score = []

        for i in range(n):
            self.reset_environment()
            terminate = False

            game_score = 0
            game_length = 0

            while not terminate and not self._terminate_game(game_length):
                action = self.choose_best_action()
                new_frame, reward, terminate = self.env_step(action)
                self.update_state(new_frame)
                game_score += reward
                game_length += 1

            score.append(game_score)
        return score


if __name__ == "__main__":
    learner = QLearner(model=get_original_model((85, 74, 4), 4, 0.00025),
                       preprocess_funcs=[to_gryscale, crop_image, downsample],
                       replay_size=100000,
                       final_exploration_frame=100000,
                       replay_start_size=100)

    learner.train(plot=False)
