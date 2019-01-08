# Copyright (c) NovaData Solutions LTD. All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

from models import data_processing, reinforcement_learning, rewards
from random import shuffle

import os

# Import the data and parse it
current_dir = os.path.dirname(os.path.realpath(__file__))
in_d = open(current_dir + "/../data/TyrosineproteinkinaseJAK2.csv", 'r')

molecules = []

for line in in_d:
    line_split = line.rstrip().split(",")
    if "First" in line_split[1]:
        continue
    else:
        molecules.append(line_split[-1][1:-1])

shuffle(molecules)

# Creating the reinforcement learning object
model_file = current_dir + "/../data/model.h5"
data_handler_file = current_dir + "/../data/data_proc.pickle"
reward_f = rewards.calculate_tpsa_reward
rl = reinforcement_learning.Rienforcement_learning(model_file=model_file,
                                                   data_handler_file=data_handler_file,
                                                   reward_function=reward_f)

# Running the reinforcement learning
rl.train(temperature=0.75, epochs=4, n_train_episodes=15, sigma=60)

# Saving the new model
rl.save("rl_model.h5")