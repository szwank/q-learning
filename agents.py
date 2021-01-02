import os

# training it's faster with cpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import random
import gc
from time import sleep
from typing import List

import gym
from LinePlotter import LinePlotter
from networks import get_dense_model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.keras.models import clone_model

from tqdm import tqdm

from queues import RingBuf, ExperienceReplay, PrioritizedExperienceReplay


class DQNAgent:
    def __init__(self, model, environment, preprocess_funcs=[], replay_size=1000000,
                 n_state_frames=4, batch_size=32, gamma=0.99, replay_start_size=50000,
                 final_exploration_frame=1000000, min_eps=0.1, max_eps=1, update_between_n_episodes=4,
                 alfa=2, initial_memory_error=10, skipp_n_states=4, actions_between_update=4):
        """
        Params:
        - model: agent NN model. Model should have two inputs: first one for states (its size
        depend on preprocess_funcs and its order and value of n_state_frames argument) should be for action
        mask(binary vector multiplied by output, used for training). First dimension of first input should
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
        - final_exploration_frame: last frame number before epsilon will be equal to min_eps
        - min_eps: minimum epsilon value
        - max_eps: start value of epsilon
        - update_between_n_episodes: number of played games after with network will be updated on minibatch
        - alfa: parameter used to tell how much more we care about transitions with big errors. When set to 0
        transitions will be sampled uniformly
        - initial_memory_error: value set to prioritized experience replay as initial error value.
        High value ensures each transitions will be seen at least once, but too high can degrade
        variety of transitions on model training
        - skipp_n_states: number of states played between with agent repeats last action on training
        - actions_between_update: number of actions taken between successive model updates

        """
        # training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_start_size = replay_start_size
        self.final_exploration_frame = final_exploration_frame
        self.n_state_frames = n_state_frames
        self.n_games_between_update = update_between_n_episodes
        self.alfa = alfa
        self.initial_memory_error = initial_memory_error
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.skipp_n_states = skipp_n_states
        self.actions_between_update = actions_between_update

        # functional
        self.iteration = None
        self.n_actions_taken = None
        self.trained_on_n_frames = 0
        self.games_scores = []
        self.replay_size = replay_size
        self.fig = None
        self.points = None
        self.score_plotter = LinePlotter(data=[],
                                         title=f'Game score- training (E-greedy policy used)',
                                         x_title=f'Episode (1 game, many updates)',
                                         y_title='Game score')
        self.q_value_plotter = LinePlotter(data=[],
                                           title=f'Average game Q-Value- training (E-greedy policy used)',
                                           x_title=f'Episode (1 game, many updates)',
                                           y_title='Q-Value')
        self.first_q_value_plotter = LinePlotter(data=[],
                                           title=f'First state Q-Value- training (E-greedy policy used)',
                                           x_title=f'Episode (1 game, many updates)',
                                           y_title='Q-Value')
        self.error_plotter = LinePlotter(data=[],
                                           title=f'Average Error- training (E-greedy policy used)',
                                           x_title=f'Network update',
                                           y_title='Average Error')

        # other stuff initialization
        self.env = environment
        self.n_actions = self.env.action_space.n
        self.online_model = model
        self.model_input_shape = model.input_shape
        self.model_state_input_shape = self.model_input_shape[0][1:]
        self.memory = ExperienceReplay(self.replay_size, n_state_frames)
        self.preprocess_funcs = preprocess_funcs
        # We want to store additional game state as feature game state.
        # Feature state is stored as first element.
        self._state = RingBuf(n_state_frames + 1)

    def train(self, n_frames=1000000, plot=True, iteration=1, visual_evaluation_period=100, evaluate_on=10,
              evaluation_period=10, save_model_period=10, render=False):
        print('Training Started')
        self.iteration = iteration
        self.n_actions_taken = 0
        plotter = LinePlotter(data=[],
                              title=f'Average score in {evaluate_on} games- evaluation',
                              x_title='Evaluation number',
                              y_title='Average score')

        print("Initialization of experience replay")
        self._init_experience_replay()

        with tqdm(total=n_frames) as progress_bar:
            print("Training started")
            while self.trained_on_n_frames < n_frames:

                self.episode(render)
                self.iteration += 1

                gc.collect()

                progress_bar.update(self.trained_on_n_frames - progress_bar.last_print_n)
                progress_bar.set_description_str(self.get_stats())

                self.train_utilities(evaluate_on, evaluation_period, plot, plotter, save_model_period,
                                     visual_evaluation_period)

        print(f"Evaluation score on 100 games: {np.mean(self.evaluate(100))}")

    def train_utilities(self, evaluate_on, evaluation_period, plot, plotter, save_model_period,
                        visual_evaluation_period):
        """Call train utilities like plotting values"""
        if self.iteration % save_model_period == 0:
            print("Model saved")
            self.save_model()
        if plot:
            self.score_plotter.plot()
            self.q_value_plotter.plot()
            self.first_q_value_plotter.plot()
            self.error_plotter.plot()
        if self.iteration % visual_evaluation_period == 0:
            self.visual_evaluate()
        if self.iteration % evaluation_period == 0:
            evaluation_score = self.evaluate(evaluate_on)
            plotter.add_data(sum(evaluation_score) / len(evaluation_score))
            plotter.plot()

    def _init_experience_replay(self):
        """Fill partially experience replay memory with states-actions by plying the game.
        Displays progress bar."""
        with tqdm(total=self.replay_start_size) as progress_bar:
            while len(self.memory) < self.replay_start_size:
                self.reset_environment()
                terminate = False

                while not terminate:
                    action = self.env.action_space.sample()
                    reward, terminate = self.env_step(action, render=False)
                    action_mask = self.encode_action(action)
                    self.update_memory(action_mask, reward, terminate)

                progress_bar.update(len(self.memory) - progress_bar.last_print_n)

    def episode(self, render):
        """Run one episode of training"""
        game_score, q_values = self._play_game(render=render, update=True)

        self.score_plotter.add_data(game_score)
        self.q_value_plotter.add_data(np.average(q_values))
        self.first_q_value_plotter.add_data(q_values[0])

    def _play_game(self, render=False, update=False) -> (int or float, List[int or float]):
        """Play one game until termination state. States are added to memory. Returns game score."""
        self.reset_environment()

        game_score = 0
        terminate = False
        game_length = 0
        Q_values = []

        while not terminate:
            action = self.choose_action()
            self.n_actions_taken += 1
            Q_values.append(np.max(self._get_current_state_prediction()))

            while not terminate:
                reward, terminate = self.env_step(action, render)
                game_score += reward
                action_mask = self.encode_action(action)
                self.update_memory(action_mask, reward, terminate)

                game_length += 1

                if game_length % self.skipp_n_states == 0:
                    break

            if update is True and self.n_actions_taken % self.actions_between_update == 0:
                self._update_agent()

        return game_score, Q_values

    def reset_environment(self):
        """Reset gym environment and state field."""
        state = self.env.reset()
        state = self.preprocess_state(state)
        self._reset_state(state)

    def _reset_state(self, state):
        """Flush state field with passed state"""
        # Remember we have additional element for future state we need to flush
        for _ in range(self.n_state_frames + 1):
            self.update_state(state)

    def choose_action(self):
        """Choose action agent will take."""
        epsilon = self.get_epsilon()

        if random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            action = self.choose_best_action()
        return action

    def get_epsilon(self):
        return max(self.min_eps, self.max_eps - self.trained_on_n_frames * self.max_eps / self.final_exploration_frame)

    def choose_best_action(self) -> int:
        """Choose best action according to current policy."""
        prediction = self._get_current_state_prediction()
        return int(np.argmax(prediction))

    def _get_current_state_prediction(self):
        state = np.expand_dims(self.current_state, axis=0)
        return self._get_prediction(state)

    def _get_prediction(self, states):
        """Returns predictions of QValue for all possible actions for passed states."""
        return self.online_model.predict_on_batch([states, np.ones((len(states), self.n_actions))])

    @property
    def current_state(self):
        """Returns current agent state"""
        return self._state[:self.n_state_frames]

    @property
    def feature_state(self):
        """Returns future agent state"""
        return self._state[-self.n_state_frames:]

    def env_step(self, action: int, render=False):
        """Calls env.step and updates agent state. Returns reward and boolean if it's a termination state.
        If passed render as True render and display environment screen"""
        if render is True:
            self.env.render()

        new_frame, reward, terminate, _ = self.env.step(action)
        new_frame = self.preprocess_state(new_frame)
        self.update_state(new_frame)
        reward = self.clip_reward(reward)
        return reward, terminate

    def preprocess_state(self, state):
        """Preprocess state"""
        for function in self.preprocess_funcs:
            state = function(state)
        return state

    def clip_reward(self, reward):
        return np.sign(reward)

    def encode_action(self, action: int):
        """Returns one hot encoded action"""
        action_mask = np.zeros((self.n_actions,))
        action_mask[action] = 1
        return action_mask

    def update_state(self, screen):
        self._state.append(screen)

    def update_memory(self, action_mask, reward, terminate):
        """Add transition to agent memory"""
        self.memory.add(self.current_state, action_mask, self.feature_state, reward, terminate)

    def _update_agent(self):
        """Makes model fit on one minibatch and update errors of prioritized experience memory."""
        start_states, actions, rewards, next_states, is_terminal = self.sample_batch_from_memory()
        errors = self.fit_batch(start_states, actions, rewards, next_states, is_terminal)
        self.error_plotter.add_data(np.average(errors))
        self.trained_on_n_frames += self.batch_size

    def sample_batch_from_memory(self):
        """Returns batch_size samples from memory"""
        return self.memory.sample_batch(self.batch_size)

    def fit_batch(self, start_states, actions, rewards, next_states, is_terminal):
        """Updates model.

        Params:
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

        Returns Q value errors
        """
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.online_model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is reward by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        target_Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.online_model.train_on_batch([start_states, actions], actions * target_Q_values[:, None])

        actions = np.array(actions, dtype=bool)
        new_Q_values = self.online_model.predict([start_states, np.ones(actions.shape)])[actions]
        return target_Q_values - new_Q_values

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

            self.points = plt.plot(np.arange(1, len(self.games_scores) + 1, 1), self.games_scores)

        self.points[0].set_data(np.arange(1, len(self.games_scores) + 1, 1), self.games_scores)
        self.ax.set_xlim(1, len(self.games_scores) + 1)
        self.ax.set_ylim(min(self.games_scores), max(self.games_scores))
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
            reward, terminate = self.env_step(action)

    def evaluate(self, n) -> List:
        """Evaluate agent across n episodes. Returns list of game scores."""
        score = []

        for i in range(n):
            self.reset_environment()
            terminate = False

            game_score = 0
            game_length = 0

            while not terminate:
                action = self.choose_best_action()
                reward, terminate = self.env_step(action)
                game_score += reward
                game_length += 1

            score.append(game_score)
        return score


class DoubleDQNAgent(DQNAgent):
    def __init__(self, transitions_seen_between_updates=10000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_model = clone_model(self.online_model)
        self.n_model_updates = 0
        self.transitions_seen_between_updates = transitions_seen_between_updates

    def update_target_model_weights(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def train_utilities(self, evaluate_on, evaluation_period, plot, plotter, save_model_period,
                        visual_evaluation_period):
        """Run training utilities. Periodically update target model weights."""
        super().train_utilities(evaluate_on, evaluation_period, plot, plotter, save_model_period,
                        visual_evaluation_period)
        # we want to update target_model weight after transitions_seen_between_updates transitions was seen
        if (self.n_model_updates + 1) * self.transitions_seen_between_updates > self.trained_on_n_frames:
            self.update_target_model_weights()
            self.n_model_updates += 1

    def fit_batch(self, start_states, actions, rewards, next_states, is_terminal):
        """Updates model.

        Params:
        - start_states: numpy array of starting states
        - actions: numpy array of one-hot encoded actions corresponding to the start states
        - rewards: numpy array of rewards corresponding to the start states and actions
        - next_states: numpy array of the resulting states corresponding to the start states and actions
        - is_terminal: numpy boolean array of whether the resulting state is terminal

        Returns Q value errors
        """
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.target_model.predict([next_states, np.ones(actions.shape)])
        # The Q values of the terminal states is reward by definition, so override them
        next_Q_values[is_terminal] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        target_Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1)
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        self.online_model.train_on_batch([start_states, actions], actions * target_Q_values[:, None])

        actions = np.array(actions, dtype=bool)
        new_Q_values = self.online_model.predict([start_states, np.ones(actions.shape)])[actions]
        return target_Q_values - new_Q_values


class PrioritizedDQNAgent(DQNAgent):
    def __init__(self, alfa, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replay_size = kwargs.get('replay_size')
        self.memory = PrioritizedExperienceReplay(replay_size, self.n_state_frames)
        self.alfa = alfa

    def _update_agent(self):
        """Makes model fit on one minibatch and update errors of prioritized experience memory."""
        start_states, actions, rewards, next_states, is_terminal, indexes = self.sample_batch_from_memory()
        errors = self.fit_batch(start_states, actions, rewards, next_states, is_terminal)
        self.error_plotter.add_data(np.average(errors))
        self.trained_on_n_frames += self.batch_size

        errors = self.get_memory_error(errors)
        self.memory.update_errors(indexes, errors)

    def get_memory_error(self, model_errors):
        return np.power(np.abs(model_errors), self.alfa)

    def sample_memory(self):
        start_states, actions, rewards, next_states, is_terminal, indexes = self.memory.sample_batch(self.batch_size)
        return start_states, actions, rewards, next_states, is_terminal, indexes




if __name__ == "__main__":
    pass
    # model = load_model('model')
    # learner = DQNAgent(model=model,
    #                    env_name='CartPole-v1',
    #                    replay_size=25000,
    #                    replay_start_size=10000,
    #                    final_exploration_frame=400000,
    #                    batch_size=32,
    #                    n_state_frames=1,
    #                    gamma=0.99,
    #                    initial_memory_error=1,
    #                    update_between_n_episodes=1
    #                    )
    # # print(np.mean(learner.evaluate(100)))
    # learner.visual_evaluate()


