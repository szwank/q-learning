import gym

from agents import DQNAgent
from networks import get_dense_model

environment = gym.make('MountainCar-v0')
# Increasing max episode length helps much car in first phase of training
# Also increasing skipp_n_states helps with this
environment._max_episode_steps = 1000

learner = DQNAgent(model=get_dense_model((1, 2), 3, 0.00025),
                   environment=environment,
                   replay_size=100000,
                   replay_start_size=100000,
                   final_exploration_frame=200000*64,
                   batch_size=64,
                   n_state_frames=1,
                   gamma=0.99,
                   initial_memory_error=1,
                   update_between_n_episodes=1,
                   skipp_n_states=4,
                   actions_between_update=8,
                   # transitions_seen_between_updates=1000
                   )

learner.train(n_frames=200000*64, plot=True, visual_evaluation_period=100000, render=True, evaluate_on=1)