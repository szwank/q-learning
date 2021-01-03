import gym

from agents import PrioritizedDQNAgent, DQNAgent, AIODQNAgent
from networks import get_dense_model

environment = gym.make('CartPole-v1')
# Increasing max episode length helps much car in first phase of training
# Also increasing skipp_n_states helps with this
# It's good to note paper(https://arxiv.org/abs/1712.00378) shows short episodes can destabilise learning
# violating the Markov property
# environment._max_episode_steps = 1000


def normalise_state(state):
    return (state - environment.observation_space.low) / (environment.observation_space.high - environment.observation_space.low)

learner = AIODQNAgent(model=get_dense_model((1, 4), 2, 0.00025),
                              preprocess_funcs=[normalise_state],
                              environment=environment,
                              replay_size=100000,
                              replay_start_size=5000,
                              final_exploration_frame=1000000,
                              batch_size=32,
                              n_state_frames=1,
                              gamma=0.99,
                              initial_memory_error=1,
                              update_between_n_episodes=1,
                              skipp_n_states=1,
                              actions_between_update=4,
                              alfa=0.6
                              # transitions_seen_between_updates=1000
                              )

learner.train(n_frames=1000000, plot=True, visual_evaluation_period=100000, render=False, evaluate_on=1)