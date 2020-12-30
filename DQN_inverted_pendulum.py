import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from dqn import QLearner
from networks import get_dense_model

learner = QLearner(model=get_dense_model((4, 4), 2, 0.00025),
                   env_name='CartPole-v1',
                   replay_size=100000,
                   replay_start_size=10000,
                   final_exploration_frame=100000,
                   batch_size=64,
                   max_game_length=2000
                   )

learner.train(n_frames=1000000, plot=True, render_period=10)

