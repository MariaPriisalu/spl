
import os
external=True

import tensorflow as tf

from settings import RANDOM_SEED
tf.compat.v1.set_random_seed(RANDOM_SEED)

import numpy as np


from settings import run_settings

from RL.RLmain import RL


##
# Main script- train an RL agent on real or toy case.
#







def main(setting):

    RL().evaluate(setting, viz=False)

if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    setup=run_settings()
    tf.compat.v1.set_random_seed(setup.random_seed_tf)
    np.random.seed(setup.random_seed_np)
    main(setup)
    # setup = run_settings(name_movie="agent_evaluate_goal_agent_old")
    # setup.name_movie="agent_evaluate_goal_agent_old"
    # setup.sem = 0
    # setup.load_weights ="Results/agent/agent_3D_reach_goal_final_2019-05-15-16-22-31.162529/"
    # main(setup)
    #setup = run_settings()
    # setup.sem=0
    # setup.name_movie="agent_evaluate_rgb_agent"
    # setup.load_weights="Results/agent/agent_3D_rgb_reward_for_on_traj_2019-03-10-11-29-56.080025/"
    #main(setup)
