"""
This module contains the RNN that learns from SMILES strings and then generates new SMILES.
"""
import numpy as np
import random

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn import __version__
import sklearn.model_selection as modsel

from keras import optimizers
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import TimeDistributed
from keras.callbacks import TensorBoard
from keras.layers import Lambda
from keras.models import load_model
import keras.backend as K

from models import utils

class Smiles_generator(BaseEstimator):

    def __init__(self, tensorboard=False, hidden_neurons_1=256, hidden_neurons_2=256, dropout_1=0.3, dropout_2=0.5,
                 batch_size="auto", epochs=4, learning_rate=0.001):
        """
        This function initialises the parent class common to both Model 1 and 2.

        :param tensorboard: whether to log progress to tensorboard or not
        :type tensorboard: bool
        :param hidden_neurons_1: number of hidden units in the first LSTM
        :type hidden_neurons_1: int
        :param hidden_neurons_2: number of hidden units in the second LSTM
        :type hidden_neurons_2: int
        :param dropout_1: dropout rate in the first LSTM
        :type dropout_1: float
        :param dropout_2:  dropout rate in the 2nd LSTM
        :type dropout_2: float
        :param batch_size: Size of the data set batches to use during training
        :type batch_size: int
        :param epochs: number of iterations of training
        :type epochs: int
        :param smiles: list of smiles strings from which to learn
        :type smiles: list of strings
        :param learning_rate: size of the step taken by the optimiser
        :type learning_rate: float > 0
        """

        self.tensorboard = utils.set_tensorboard(tensorboard)
        self.hidden_neurons_1 = utils.set_hidden_neurons(hidden_neurons_1)
        self.hidden_neurons_2 = utils.set_hidden_neurons(hidden_neurons_2)
        self.dropout_1 = utils.set_dropout(dropout_1)
        self.dropout_2 = utils.set_dropout(dropout_2)
        self.batch_size = utils.set_provisional_batch_size(batch_size)
        self.epochs = utils.set_epochs(epochs)
        self.learning_rate = utils.set_learning_rate(learning_rate)

        self.model = None
        self.loaded_model = None

    def fit(self, X, y):
        """
        This function fits the parameters of the RNN to the data provided.

        :param X: Input one-hot-encoded padded smiles strings
        :type X: np.array of shape (n_samples, max_len, n_char)
        :param y: Output one-hot-encoded padded smiles strings
        :type y: np.array of shape (n_samples, max_len, n_char)

        :return: the estimator object
        """

        # Check the inputs
        X, y = utils.check_X_y(X, y)

        # Construct the model
        model = self._build_model(X.shape[-1])

        # Adjust batch_size based on number of samples provided
        batch_size = utils.set_batch_size(self.batch_size, X.shape[0])

        # Set up tensorboard
        if self.tensorboard:
            tensorboard = TensorBoard(log_dir='./tb', write_graph=True, write_images=False)
            callbacks_list = [tensorboard]
        else:
            callbacks_list = []

        # If the model has never been trained, create a new one, otherwise restart training from a previously trained model
        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            model = self._build_model(X.shape[-1])
        elif not isinstance(self.model, type(None)):
            model = self.model
        elif not isinstance(self.loaded_model, type(None)):
            model = self.loaded_model

        # If there are enough samples, use some as validation data
        if X.shape[0] >= 20:
            X_train, X_val, y_train, y_val = modsel.train_test_split(X, y, test_size=0.05)

            model.fit(X_train, y_train, batch_size=batch_size, verbose=1, epochs=self.epochs,
                        callbacks=callbacks_list, validation_data=(X_val, y_val))
        else:
            model.fit(X, y, batch_size=batch_size, verbose=1, epochs=self.epochs, callbacks=callbacks_list)

        # Update the right variable with the newly trained model
        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            self.model = model
        elif not isinstance(self.model, type(None)):
            self.model = model
        elif not isinstance(self.loaded_model, type(None)):
            self.loaded_model = model

        self._model = model
        self.is_fitted_ = True

        return self

    def fit_with_rl(self, data_handler, epochs, n_train_episodes, temperature, max_length, sigma, rl_learning_rate):
        """
        This function fits the model using reinforcement learning.

        :param data_handler: object of the class Molecules_processing
        :type data_handler: Molecules_processing
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
        if utils.is_none(self.model) and utils.is_none(self.loaded_model):
            raise utils.InputError("Fit with reinforcement learning can only be called after the model has been trained.")

        utils.check_ep(n_train_episodes)
        utils.check_temperature(temperature)
        utils.check_maxlength(max_length)
        utils.check_sigma(sigma)
        utils.check_lr(rl_learning_rate)

        # Creatint a 'prior' and an 'agent' model
        if utils.is_none(self.model):
            model_prior = self._modify_model_for_predictions(self.loaded_prior, temperature)
            model_agent = self._modify_model_for_predictions(self.loaded_model, temperature)  # Note: self.loaded_model will be modified as it is the same object as model_agent
        else:
            # TODO implement reinforcement learning without reloading
            # This requires to be able to make a deep copy of the model or to save it and then reload it
            raise NotImplementedError

        # Making the Reinforcement Learning training function
        training_function = self._generate_rl_training_fn(model_agent, sigma, rl_learning_rate)

        # The training function takes as arguments: the state, the action and the reward.
        # These have to be calculated in advance and stored.
        experience = []
        rewards = []

        for ep in range(epochs):
            # This generates some episodes (i.e. smiles)
            experience, rewards = self._rl_episodes(model_agent, model_prior, data_handler, n_train_episodes,
                                                    max_length, experience, rewards)

            for _ in range(10):
                random_n = random.randint(0, len(experience)-1)
                state = experience[random_n][0]
                prior_loglikelihood = experience[random_n][1]
                reward = experience[random_n][2]

                training_function([state, prior_loglikelihood, reward])

    def predict(self, X, temperature=1.0, max_length=100):
        """
        This function starts from a hot encoded SMILES and predicts the remaining part of the molecule. X needs to
        at least contain a 'G' character, it cannot be empty.

        :param X: hot-encoded SMILES
        :type X: np.array with shape (n_samples, length_smiles, n_char)
        :param temperature: Temperature factor for the modified softmax
        :type temperature: float
        :param max_length: maximum length of a smile string to generate
        :type max_length: int
        :return: hot-encoded SMILES
        :rtype: np.array of shape (n_samples, length_smiles, n_char)
        """

        utils.check_temperature(temperature)
        utils.check_maxlength(max_length)

        if isinstance(self.model, type(None)) and isinstance(self.loaded_model, type(None)):
            raise Exception("The model has not been fit and no saved model has been loaded.\n")
        elif not isinstance(self.model, type(None)):
            model = self._modify_model_for_predictions(self.model, temperature)
        else:
            model = self._modify_model_for_predictions(self.loaded_model, temperature)

        X_pred = self._pred(X, model, max_length)

        return X_pred

    def save(self, filename='model'):
        """
        This function enables to save the trained model so that then training or predictions can be done at a later stage.

        :param filename: Name of the file in which to save the model.
        :return: None
        """
        model_name = filename + ".h5"

        if not isinstance(self.model, type(None)):
            self.model.save(model_name, overwrite=False)
        elif not isinstance(self.loaded_model, type(None)):
            self.loaded_model.save(model_name, overwrite=False)
        else:
            raise utils.InputError("No model to be saved.")

    def load(self, filename='model'):
        """
        This function loads a model that has been previously saved.

        :param filename: Name of the file in which the model has been previously saved.
        :return: None
        """
        model_name = filename + ".h5"

        self.loaded_model = load_model(model_name)
        self.loaded_prior = load_model(model_name)

    def _build_model(self, n_feat):
        """
        This function generates the RNN.

        :return: Keras Sequential object
        """
        model = Sequential()
        # This will output (max_size, n_hidden_1)
        model.add(LSTM(units=self.hidden_neurons_1, input_shape=(None, n_feat), return_sequences=True,
                       dropout=self.dropout_1))
        # This will output (max_size, n_hidden_2)
        model.add(
            LSTM(units=self.hidden_neurons_2, input_shape=(None, self.hidden_neurons_1), return_sequences=True,
                 dropout=self.dropout_2))
        # This will output (max_size, n_feat)
        model.add(TimeDistributed(Dense(n_feat), input_shape=(None, self.hidden_neurons_2)))
        # Modifying softmax with temperature
        model.add(Lambda(lambda x: x / 1))
        model.add(Activation('softmax'))
        optimiser = optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                    amsgrad=False)
        model.compile(loss="categorical_crossentropy", optimizer=optimiser)

        return model

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
        individual_action_probability = K.sum(hot_encoded_sequence[:, 1:] * agent_action_prob_placeholder[:, :-1], axis=-1)
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

    def _pred(self, X, model, max_length):

        n_feat = X.shape[-1]
        n_samples = X.shape[0]

        # Predictions
        X_pred = np.zeros((n_samples, max_length, n_feat))
        X_pred[:, :X.shape[1], :] = X

        for i in range(1, max_length):
            prob_distribution = model.predict(X_pred[:, :i, :])

            # Slow step
            for j in range(n_samples):
                idx_out = np.random.choice(np.arange(n_feat), p=prob_distribution[j, 0])
                X_pred[j, i, idx_out] = 1

        return X_pred

    def _rl_episodes(self, model_agent, model_prior, data_handler, n_episodes, max_length, experience, rewards):
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
        X = data_handler.onehot_decode(["G"]*n_episodes)
        hot_pred, exp = self._pred(X=X, model=model_agent, max_length=max_length)

        # Calculate the sequence log-likelihood for the prior
        prior_action_prob = model_prior.predict(hot_pred)
        individual_action_probability = np.sum(np.multiply(hot_pred[:, 1:], prior_action_prob[:, :-1]), axis=-1)
        prod_individual_action_prob = np.prod(individual_action_probability)
        sequence_log_likelihood = np.log(prod_individual_action_prob)

        # Calculate the reward for the finished smile
        smiles_predictions = data_handler.onehot_decode(hot_pred)
        new_rewards = self._calculate_reward(smiles_predictions)

        # Remove all invalid smiles
        try:
            idx_notnone = np.where(new_rewards != None)
            hot_pred = np.delete(hot_pred, idx_notnone, axis=0)
            sequence_log_likelihood = np.delete(sequence_log_likelihood, idx_notnone, axis=0)
            new_rewards = np.delete(new_rewards, idx_notnone, axis=0)
        except ValueError:
            pass

        # If the experience buffer is not full, add as many are needed
        if len(experience) < n_episodes:
            for n_ep in range(n_episodes-len(experience)):
                # Sort in order of increaseing reward
                idx_sorted = np.argsort(new_rewards)
                hot_pred = hot_pred[idx_sorted]
                sequence_log_likelihood = sequence_log_likelihood[idx_sorted]
                new_rewards = new_rewards[idx_sorted]
                # Append the smiles with largest reward first
                experience.append((hot_pred[-(n_ep+1)], sequence_log_likelihood[-(n_ep+1)], new_rewards[-(n_ep+1)]))
                rewards.append(new_rewards[-(n_ep+1)])
        else:
            # If the minimum reward is smaller than the reward for the current smile, replace it
            while min(rewards) < max(new_rewards):
                idx_to_pop = np.argmin(rewards)
                idx_to_add = np.argmax(new_rewards)
                del experience[idx_to_pop]
                del rewards[idx_to_pop]
                experience.append((hot_pred[idx_to_add], sequence_log_likelihood[idx_to_add], new_rewards[idx_to_add]))
                rewards.append(new_rewards[idx_to_add])
                hot_pred = np.delete(hot_pred, idx_to_add, axis=0)
                sequence_log_likelihood = np.delete(sequence_log_likelihood, idx_to_add, axis=0)
                new_rewards = np.delete(new_rewards, idx_to_add, axis=0)

        return experience, rewards

    def _calculate_reward(self, X_strings):
        """
        This function calculates the reward for a particular molecule.

        :param X_strings: SMILES strings
        :type X_string: list of strings
        :return: the rewards
        :rtype: list of float
        """

        from rdkit.Chem import Descriptors, MolFromSmiles
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.error')

        rewards = []
        for string in X_strings:
            m = MolFromSmiles(string)

            # If the predicted smiles is invalid, give no reward
            if utils.is_none(m):
                rewards.append(None)
                continue

            TPSA = Descriptors.TPSA(m)

            # To obtain molecules mostly with polarity between 90 and 120
            rewards.append(np.exp(-(TPSA - 105) ** 2))

        return rewards

if __name__ == "__main__":

    from models import data_processing

    # Getting the data
    data_file = "/Volumes/Transcend/PhD/NovaData_solutions/dataset/TyrosineproteinkinaseJAK2.csv"
    in_d = open(data_file, "r")

    # Read molecules and activities from CSV file
    molecules = []
    activities = []

    for line in in_d:
        line = line.rstrip()
        line_split = line.split(",")
        molecule_raw = line_split[-1]
        molecule = molecule_raw[1:-1]
        if molecule == "SMI (Canonical)":
            pass
        else:
            molecules.append(molecule)


    # Processing the data
    dh = data_processing.Molecules_processing()
    X = dh.onehot_encode(molecules)

    estimator = Smiles_generator(epochs=1, batch_size=500)
    estimator.fit(X[:500], X[:500])
    hot_pred = estimator.predict(X[:2,:1, :])

    mol_pred = dh.onehot_decode(hot_pred)

    print(mol_pred)
