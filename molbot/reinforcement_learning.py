# Copyright (c) Michael Mazanetz (NovaData Solutions LTD.), Silvia Amabilino (NovaData Solutions LTD.,
# University of Bristol), David Glowacki (University of Bristol). All rights reserved.
# Licensed under the GPL. See LICENSE in the project root for license information.

"""
This module implements a class that can be used for reinforcement learning of the models from the Smiles_generator
class.
"""

import keras.backend as K
from keras import optimizers
from keras.models import load_model
from keras.layers import Activation
from keras.layers import Lambda

import numpy as np
import random

from models import utils
from models import data_processing

class Reinforcement_learning():

    def __init__(self, model_file, data_handler_file, reward_function):
        """

        :param model_file: Name of the file in which the model has been previously saved.
        :type model_file: string
        :param data_handler_file: Name of the file in which the data handler has been previously saved.
        :type data_handler_file: string
        :param reward_function: a function that will evaluate if the predicted smiles are valid and scores them
        :type reward_function: python function that takes as input one-hot encoded smiles strings (i.e. a numpy array of
        shape (n_samples, max_length, n_char) and returns the rewards (a list of length n_valid_smiles) and a list of
        indices of the invalid smiles.
        """

        self._load_model(model_file)
        self._load_data_handler(data_handler_file)
        self.reward_function = reward_function

    def train(self, epochs=5, n_train_episodes=15, temperature=0.75, sigma=60, rl_learning_rate=0.0005):
        """
        This function fits the model using reinforcement learning.


        :param epochs: number of iterations of RL to do
        :type epochs: int
        :param n_train_episodes: number of training episodes to generate in each epoch
        :type n_train_episodes: int
        :param temperature: Temperature factor in the softmax
        :type temperature: positive float
        :param max_length: maximum length of an episode
        :type max_length: int
        :param sigma: parameter that controls how to weight the desirability of a sequence. Explanation https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-017-0235-x equation on bottom right of page 4
        :type sigma: float
        :param rl_learning_rate: learning rate for optimiser in the reinforcement learning algorithm
        :type rl_learning_rate: positive float

        :return: None
        """

        # Checking the inputs
        utils.check_ep(n_train_episodes)
        utils.check_temperature(temperature)
        utils.check_sigma(sigma)
        utils.check_lr(rl_learning_rate)

        # Making the Reinforcement Learning training function
        training_function = self._generate_rl_training_fn(self.agent, sigma, rl_learning_rate)

        # The training function takes as arguments: the state, the action and the reward.
        # These have to be calculated in advance and stored.
        experience = []
        rewards = []

        for ep in range(epochs):
            # This generates some episodes (i.e. smiles)
            experience, rewards = self._rl_episodes(self.agent, self.prior, self.dh, n_train_episodes,
                                                    experience, rewards)

            for _ in range(n_train_episodes):
                # TODO think about removing this loop
                random_n = random.randint(0, len(experience) - 1)
                state = experience[random_n][0]
                prior_loglikelihood = experience[random_n][1]
                reward = experience[random_n][2]

                training_function([state, prior_loglikelihood, reward])

    def save(self, filename='model.h5'):
        """
        This function enables to save the trained model so that then training or predictions can be done at a later stage.

        :param filename: Name of the file in which to save the model.
        :return: None
        """

        self.agent.save(filename, overwrite=True)

    def _load_model(self, filename='model.h5'):
        """
        This function loads a model that has been previously saved.

        :param filename: Name of the file in which the model has been previously saved.
        :type filename: string
        :return: None
        """
        self.agent = load_model(filename)
        self.prior = load_model(filename)

    def _load_data_handler(self, filename="data_proc.pickle"):
        """
        This function loads the data handler that has been previously saved as a pickle.

        :param filename: Name of the file in which the data handler has been previously saved.
        :type filename: string
        :return: None
        """

        self.dh = data_processing.Molecules_processing()
        self.dh.load(filename)

    def _modify_model_for_predictions(self, model, temperature):
        """
        This function modifies the model for predict time by adding a temperature factor to the softmax activation
        function.

        :param model: the model to modify
        :type model: keras model
        :param temperature: temperature that modifies the softmax
        :type temperature: float > 0 and <= 1
        :return: The modified model
        """

        model.pop()
        model.pop()
        model.add(Lambda(lambda x: x / temperature))
        model.add(Activation('softmax'))

        return model

    def _generate_rl_training_fn(self, model_agent, sigma, lr):
        """
        This function extends the model so that Reinforcement Learning can be done.

        :param temperature: the temperature of the softmax parameter
        :type temperature: positive float
        :param sigma: parameter that controls how to weight the desirability of a sequence. Explanation https://jcheminf.biomedcentral.com/track/pdf/10.1186/s13321-017-0235-x equation on bottom right of page 4
        :type sigma: float
        :param lr: learning rate for optimiser in the reinforcement learning algorithm
        :type lr: positive float
        :return: the model and the training function
        :rtype: a keras object and a keras function
        """

        # The first argument is the model input
        hot_encoded_sequence = model_agent.input

        # The probabilities that the agent would assign in each state
        agent_action_prob_placeholder = model_agent.output

        # The log likelihood of a sequence from a prior
        prior_loglikelihood = K.placeholder(shape=(None,), name="prior_loglikelihood")

        # The log likelihood of a sequence from the agent
        individual_action_probability = K.sum(hot_encoded_sequence[:, 1:] * agent_action_prob_placeholder[:, :-1],
                                              axis=-1)
        agent_likelihood = K.prod(individual_action_probability)
        agent_loglikelihood = K.log(agent_likelihood)

        # Reward that the sequence has obtained
        reward_placeholder = K.placeholder(shape=(None,), name="reward")

        # Augmented log-likelihood: prior log lokelihood + sigma * desirability of the sequence
        sigma_k = K.constant(sigma)
        desirability = reward_placeholder
        augmented_likelihood = prior_loglikelihood + sigma_k * desirability

        # Loss function
        loss = K.pow(augmented_likelihood - agent_loglikelihood, 2)

        # Optimiser and updates
        optimiser = optimizers.Adam(lr=lr, clipnorm=3.0)
        updates = optimiser.get_updates(params=model_agent.trainable_weights, loss=loss)

        rl_training_function = K.function(inputs=[hot_encoded_sequence, prior_loglikelihood, reward_placeholder],
                                          outputs=[], updates=updates)

        return rl_training_function

    def _rl_episodes(self, model_agent, model_prior, data_handler, n_episodes, experience, rewards):
        """
        This function takes generates new SMILES using the agent and then calculates their probability using the prior.
        It then calculates the reward of the generated SMILES and adds all information to the experience buffer.

        :param model_agent: the model that will be modified by the RL algorithm
        :type model_agent: Smiles_generator object
        :param model_prior: the original model
        :param data_handler: Smiles_generator object
        :param n_episodes: number of training episodes on which to train
        :type n_episodes: int
        :param max_length: maximum length of generated smiles
        :type max_length: int
        :param experience: contains the generated hot-encoded smiles, their prior probability and their reward.
        :type experience: list of tuples with 3 elements each
        :param rewards: contains the rewards for the generated smiles (Redundant: could be potentially removed)
        :type rewards: list of floats
        :return: updated experience and rewards.
        """

        # Using the agent network to predict a smile
        X = data_handler.get_empty(n_episodes*2)
        hot_pred = self._pred(X=X, model=model_agent, max_length=data_handler.max_size)

        # Calculate the sequence log-likelihood for the prior
        prior_action_prob = model_prior.predict(hot_pred)
        individual_action_probability = np.sum(np.multiply(hot_pred[:, 1:], prior_action_prob[:, :-1]), axis=-1)
        prod_individual_action_prob = np.prod(individual_action_probability, axis=-1)
        sequence_log_likelihood = np.log(prod_individual_action_prob)

        if np.isnan(np.sum(sequence_log_likelihood)) or np.isinf(np.sum(sequence_log_likelihood)):
            print("There are NaNs in the predictions.")
            exit()

        # Calculate the reward for the finished smile
        smiles_predictions = data_handler.onehot_decode(hot_pred)
        new_rewards = self.reward_function(smiles_predictions)

        # If the experience buffer is not full, add as many are needed
        if len(experience) < n_episodes:

            # Sort in order of increaseing reward
            idx_sorted = np.argsort(new_rewards)
            hot_pred = hot_pred[idx_sorted]
            sequence_log_likelihood = sequence_log_likelihood[idx_sorted]
            new_rewards = np.asarray(new_rewards)[idx_sorted]

            # Append the smiles with largest reward first
            for n_ep in range(n_episodes-len(experience)):
                if n_ep+1 > len(new_rewards):
                    break
                expanded_hot_pred = np.expand_dims(hot_pred[-(n_ep+1)], axis=0)
                experience.append((expanded_hot_pred, sequence_log_likelihood[-(n_ep+1)], new_rewards[-(n_ep+1)]))
                rewards.append(new_rewards[-(n_ep+1)])
        else:
            # If the minimum reward is smaller than the reward for the current smile, replace it
            while min(rewards) < max(new_rewards):
                idx_to_pop = np.argmin(rewards)
                idx_to_add = np.argmax(new_rewards)
                del experience[idx_to_pop]
                del rewards[idx_to_pop]
                expanded_hot_pred = np.expand_dims(hot_pred[idx_to_add], axis=0)
                experience.append((expanded_hot_pred, sequence_log_likelihood[idx_to_add], new_rewards[idx_to_add]))
                rewards.append(new_rewards[idx_to_add])
                hot_pred = np.delete(hot_pred, idx_to_add, axis=0)
                sequence_log_likelihood = np.delete(sequence_log_likelihood, idx_to_add, axis=0)
                new_rewards = np.delete(new_rewards, idx_to_add, axis=0)

        return experience, rewards

    def _pred(self, X, model, max_length):
        """
        This function predicts one-hot encoded smiles strings starting from a fragment.

        :param X: One-hot encoded fragment of smile string
        :param model: the keras model to use for prediction
        :param max_length: maximum length of predicted molecules
        :return: predicted one-hot encoded smiles strings
        """

        n_feat = X.shape[-1]
        n_samples = X.shape[0]

        # Predictions
        X_pred = np.zeros((n_samples, max_length, n_feat))
        X_pred[:, :X.shape[1], :] = X

        for i in range(1, max_length):
            prob_distribution = model.predict(X_pred[:, :i, :])

            # Slow step
            for j in range(n_samples):
                idx_out = np.random.choice(np.arange(n_feat), p=prob_distribution[j, -1])
                X_pred[j, i, idx_out] = 1

        return X_pred