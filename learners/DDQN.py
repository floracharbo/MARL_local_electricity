import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError

# adapted from https://github.com/iomanker/DQN-DDQN-for-Tensorflow-2.0


class Agent_DDQN():
    """ Tabular Q-learning learner """
    def __init__(self, env, rl, t):
        self.n_agents = env.n_agents
        self.state_dim = len(rl['state_space'])
        # action_dim is discrete number of possible actions
        # per agent per time step
        # dim_actions is the number of decisions to make
        # (e.g. for all agents, for all time steps...)
        # action_dim is how many different possible actions
        # to consider in the Q table overall
        self.action_dim = rl['n_actions'] ** rl['dim_actions']
        self.gamma = rl['DDQN']['gamma']  # 0.90
        for e in ['epsilon0', 'buffer_capacity', 'batch_size']:
            self.__dict__[e] = rl['DDQN'][e]
        for e in ['epsilon_end', 'epsilon_decay_param']:
            self.__dict__[e] = rl['DDQN'][e][t]
        TARGET_UPDATE_CYCLE = 1  # 100

        self.opt = tf.keras.optimizers.Adam(0.01)
        # Discount factor
        self.loss_function = MeanSquaredError()
        self.target_update_cycle = TARGET_UPDATE_CYCLE
        self.memories_nameList = ['state', 'action', 'reward', 'next_state']

        self.enable_DDQN = True
        self.reset()

    def reset(self):
        self.epsilon = self.epsilon0
        self.evaluation_network = self.build_model()
        self.target_network = self.build_model()
        self.training_counter = 0
        self.target_update_counter = 0
        self.memories_dict = {}
        for item_name in self.memories_nameList:
            self.memories_dict[item_name] = None

    def epsilon_update(self):
        # call this for each episode
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (self.epsilon0 - self.epsilon_end) \
                * self.epsilon_decay_param

    def sample_action(self, state, eps_greedy=True):
        # call this at start of each time step
        action_random = np.random.randint(0, self.action_dim)
        state = np.reshape(state, [1, self.state_dim])
        tmp = self.evaluation_network(state)
        action_argmax = tf.argmax(tmp[0])
        rdn_number = np.random.uniform()
        if eps_greedy and rdn_number < self.epsilon:
            action = action_random
        else:
            action = action_argmax

        return action

    def learn(self, state, action, reward, next_state):
        # call this at end of each time step
        for item_name, item in zip(self.memories_nameList,
                                   [state, action, reward, next_state]):
            if self.memories_dict[item_name] is None:
                self.memories_dict[item_name] = [item]
            else:
                self.memories_dict[item_name] = \
                    np.append(self.memories_dict[item_name], item)
        self.training_counter += 1

        # episode_reward += reward

        if self.training_counter >= self.buffer_capacity:
            self.train()
            self.delete_experience()

    def build_model(self):
        model = tf.keras.Sequential([
            InputLayer((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])

        return model

    def train(self):
        # DQN - Experience Replay for Mini-batch
        random_select = np.random.choice(self.buffer_capacity, self.batch_size)
        states = self.memories_dict["state"][random_select]
        rewards = self.memories_dict["reward"][random_select]
        nextStates = self.memories_dict["next_state"][random_select]
        with tf.GradientTape() as tape:
            q_eval_arr = self.evaluation_network(states)
            q_eval = tf.reduce_max(q_eval_arr, axis=1)
            print("q_eval: {}".format(q_eval))
            if self.enable_DDQN:
                # Double Deep Q-Network
                q_values = self.evaluation_network(nextStates)
                q_values_actions = tf.argmax(q_values, axis=1)
                target_q_values = self.target_network(nextStates)
                indice = tf.stack([range(self.batch_size),
                                   q_values_actions], axis=1)
                discount_factor = tf.gather_nd(target_q_values, indice)
            else:
                # Deep Q-Network
                target_q_values = self.target_network(nextStates)
                discount_factor = tf.reduce_max(target_q_values, axis=1)

            # Q function
            q_target = rewards + self.gamma * discount_factor
            print("q_target: {}".format(q_target))
            loss = self.loss_function(q_eval, q_target)

        gradients_of_network = tape.gradient(
            loss, self.evaluation_network.trainable_variables)
        self.opt.apply_gradients(
            zip(gradients_of_network,
                self.evaluation_network.trainable_variables))
        self.target_update_counter += 1
        # DQN - Frozen update
        if self.target_update_counter % self.target_update_cycle == 0:
            self.target_network.set_weights(
                self.evaluation_network.get_weights())

    def delete_experience(self):
        for item_name in self.memories_nameList:
            self.memories_dict[item_name] = None
        self.training_counter = 0
