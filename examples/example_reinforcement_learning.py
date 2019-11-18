# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This example shows how to use reinforcement learning to refine an already trained recurrent neural network. It requires
having run example_training.py first.
"""

from molbot import data_processing, reinforcement_learning, rewards
from random import shuffle

import os

# Import the data and parse it
current_dir = os.path.dirname(os.path.realpath(__file__))
model_file = os.path.join(current_dir, "example-save.h5")
data_handler_file = os.path.join(current_dir, "data_proc.pickle")

# Creating the reinforcement learning object
reward_f = rewards.calculate_tpsa_reward
rl = reinforcement_learning.Reinforcement_learning(model_file=model_file,
                                                   data_handler_file=data_handler_file,
                                                   reward_function=reward_f)

# Running the reinforcement learning
rl.train(temperature=0.75, epochs=4, n_train_episodes=15, sigma=60)

# Saving the new model
rl.save("rl_model.h5")