import gym

from agents import PrioritizedDQNAgent, DQNAgent, AIODQNAgent, FullDQNAgent, DoubleDQNAgent
from networks import get_dense_model

environment = gym.make('CartPole-v1')


def normalise_state(state):
    return (state - environment.observation_space.low) / (
            environment.observation_space.high - environment.observation_space.low)


learner = DoubleDQNAgent(model=get_dense_model((1, 4), 2, 0.00025),
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
                         actions_between_update=1,
                         alfa=0.6,
                         transitions_seen_between_updates=10000,
                         )

learner.train(n_frames=5000000, plot=True, visual_evaluation_period=100000, render=False, evaluate_on=5)
