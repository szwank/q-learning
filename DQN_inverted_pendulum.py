import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from dqn import QLearner
from networks import get_dense_model

learner = QLearner(model=get_dense_model((4, 4), 2, 0.00025),
                   env_name='CartPole-v1',
                   replay_size=1000000,
                   replay_start_size=50000,
                   final_exploration_frame=1000000,
                   )

learner.train(plot=True)

