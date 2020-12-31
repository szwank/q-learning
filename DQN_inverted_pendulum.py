import os
# training it's faster with cpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from dqn import QLearner
from networks import get_dense_model

learner = QLearner(model=get_dense_model((4, 1), 2, 0.00025/4),
                   env_name='CartPole-v1',
                   replay_size=10000,
                   replay_start_size=10000,
                   final_exploration_frame=32*1000,
                   batch_size=32,
                   max_game_length=475,
                   n_state_frames=1,
                   gamma=0.99,
                   update_network_period=1000
                   )

learner.train(n_frames=32*1000, plot=True, render_period=50)

