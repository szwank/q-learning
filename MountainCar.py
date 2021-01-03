import gym

from agents import PrioritizedDQNAgent, DQNAgent
from networks import get_dense_model

environment = gym.make('MountainCar-v0')
# Increasing max episode length helps much car in first phase of training
# Also increasing skipp_n_states helps with this
# It's good to note paper(https://arxiv.org/abs/1712.00378) shows short episodes can destabilise learning
# violating the Markov property
environment._max_episode_steps = 1000

learner = PrioritizedDQNAgent(model=get_dense_model((1, 2), 3, 0.00025),
                              environment=environment,
                              replay_size=1000000,
                              replay_start_size=50000,
                              final_exploration_frame=1000000,
                              batch_size=32,
                              n_state_frames=1,
                              gamma=0.99,
                              initial_memory_error=1,
                              update_between_n_episodes=1,
                              skipp_n_states=4,
                              actions_between_update=1,
                              alfa=1.2
                              # transitions_seen_between_updates=1000
                              )

learner.train(n_frames=1000000, plot=True, visual_evaluation_period=100000, render=True, evaluate_on=1)