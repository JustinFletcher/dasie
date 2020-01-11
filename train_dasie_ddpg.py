"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here:
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import argparse
from datetime import datetime
import pprint as pp
import os
import random
from collections import deque


tf.compat.v1.disable_eager_execution()

class ReplayBuffer(object):

    """
     A simple replay buffer, borrowed with gratitude from Patrick Emami.
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound,
                 learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.compat.v1.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()


        with tf.name_scope("actor_target"):
            self.target_network_params = tf.compat.v1.trainable_variables()[
                                         len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights

        with tf.name_scope("actor_target_update"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) +
                    tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
                self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
                self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                                 shape=(None, self.s_dim),
                                                                 name="observation")


        with tf.name_scope("actor"):
            with tf.name_scope("hidden_layer_400"):
                n_hidden = 400
                W = tf.Variable(tf.random.normal((self.s_dim, n_hidden)))
                b = tf.Variable(tf.random.normal((n_hidden,)))
                x = tf.matmul(observation_batch_placeholder, W) + b
                # net = tflearn.layers.normalization.batch_normalization(net)
                x = tf.nn.relu(x)
            with tf.name_scope("hidden_layer_300"):
                n_hidden = 300
                W = tf.Variable(tf.random.normal((400, n_hidden)))
                b = tf.Variable(tf.random.normal((n_hidden,)))
                x = tf.matmul(x, W) + b
                # net = tflearn.layers.normalization.batch_normalization(net)
                x = tf.nn.relu(x)
            with tf.name_scope("hidden_layer_1"):
                n_hidden = 1
                W = tf.Variable(tf.random.uniform((300, n_hidden), minval=-0.003, maxval=0.003))
                b = tf.Variable(tf.random.uniform((n_hidden,)))
                x = tf.matmul(x, W) + b
                # net = tflearn.layers.normalization.batch_normalization(net)
                output_batch = tf.nn.tanh(x)

        # Scale output to -action_bound to action_bound
        scaled_output_batch = tf.multiply(output_batch, self.action_bound)
        return observation_batch_placeholder, output_batch, scaled_output_batch

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
                self.inputs: inputs,
                self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
                self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
                self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma,
                 num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.compat.v1.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        with tf.name_scope("critic_target"):
            self.target_network_params = tf.compat.v1.trainable_variables()[(len(
                self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization

        with tf.name_scope("actor_target_update"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(
                    tf.multiply(self.network_params[i], self.tau) \
                    + tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        # self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.loss = tf.math.reduce_mean(tf.pow((self.predicted_q_value - self.out), 2))
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):

        with tf.name_scope("critic"):
            action_batch_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                                shape=(None, self.a_dim),
                                                                name="action")
            observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                                     shape=(None, self.s_dim),
                                                                     name="observation")

            with tf.name_scope("hidden_layer_observation_400"):
                n_hidden = 400
                W = tf.Variable(tf.random.normal((self.s_dim, n_hidden)))
                b = tf.Variable(tf.random.normal((n_hidden,)))
                x_o = tf.matmul(observation_batch_placeholder, W) + b
                # net = tflearn.layers.normalization.batch_normalization(net)
                x_o = tf.nn.relu(x_o)

            with tf.name_scope("hidden_layer_observation_300"):
                n_hidden = 300
                W_o1 = tf.Variable(tf.random.normal((400, n_hidden)))
                b_o1 = tf.Variable(tf.random.normal((n_hidden,)))
                x_o = tf.matmul(x_o, W_o1 + b_o1)
                # net = tflearn.layers.normalization.batch_normalization(net)
                # x_o = tf.nn.relu(x_o)

            # Add the action tensor in the 2nd hidden layer
            # Use two temp layers to get the corresponding weights and biases

            with tf.name_scope("hidden_layer_action_300"):
                n_hidden = 300
                W_a0 = tf.Variable(tf.random.normal((self.a_dim, n_hidden)))
                b_a0 = tf.Variable(tf.random.normal((n_hidden,)))
                x_a = tf.matmul(action_batch_placeholder, W_a0) + b_a0
                # net = tflearn.layers.normalization.batch_normalization(net)
                # x_a = tf.nn.relu(x_a)


            # net = tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b,
            #         activation='relu')
            x = x_a + x_o
            # x = tf.matmul(x_o, W_o1) + b_o1 + tf.matmul(action_batch_placeholder, W_a0) + b_a0

            # linear layer connected to 1 output representing Q(s,a)
            # Weights are init to Uniform[-3e-3, 3e-3]
            # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            # out = tflearn.fully_connected(net, 1, weights_init=w_init)
            #

            # with tf.name_scope("hidden_layer_merged_300"):
            #     n_hidden = 300
            #     W = tf.Variable(tf.random.uniform((300, n_hidden), minval=-0.003, maxval=0.003))
            #     b = tf.Variable(tf.random.uniform((n_hidden,)))
            #     x = tf.matmul(x, W) + b
            x = tf.nn.relu(x)
            with tf.name_scope("hidden_layer_merged_1"):
                n_hidden = 1
                W = tf.Variable(tf.random.uniform((300, n_hidden), minval=-0.003, maxval=0.003))
                b = tf.Variable(tf.random.uniform((n_hidden,)))
                output_batch = tf.matmul(x, W) + b
                # net = tflearn.layers.normalization.batch_normalization(net)
                # output_batch = tf.nn.tanh(x)
                # output_batch = tf.nn.tanh(x)
        return observation_batch_placeholder, action_batch_placeholder, output_batch

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
                self.inputs: inputs,
                self.action: action,
                self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
                self.inputs: inputs,
                self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
                self.target_inputs: inputs,
                self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
                self.inputs: inputs,
                self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(
            size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu,
                                                                      self.sigma)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.compat.v1.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, flags, actor, critic, actor_noise):
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()


    if flags.baseline:

        logdir_path = os.path.join(flags.logdir, "baseline")

    else:

        logdir_path = os.path.join(flags.logdir, flags.run_name)

    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(logdir_path, sess.graph)
    writer.flush()


    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(flags.replay_buffer_size, flags.random_seed)

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)


    for i in range(0):

        state = env.reset()
        action = env.action_space.sample()
        new_observation, reward, terminal, info = env.step(action)

        replay_buffer.add(np.reshape(state, (actor.s_dim,)),
                          np.reshape(action, (actor.a_dim,)),
                          reward,
                          terminal,
                          np.reshape(new_observation, (actor.s_dim,)))


    for i in range(flags.num_episodes):

        observation = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(flags.max_episode_steps):

            if  flags.render_env:
                env.render()

            # Consult the actor model for an action.
            a = actor.predict(np.reshape(observation, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(observation, (actor.s_dim,)),
                              np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > flags.batch_size:

                (observation_batch,
                 action_batch,
                 reward_batch,
                 terminal_batch,
                 next_observation_batch) = replay_buffer.sample_batch(flags.batch_size)

                # Calculate targets
                target_q = critic.predict_target(next_observation_batch,
                                                 actor.predict_target(next_observation_batch))

                y_i = list()
                for k in range(flags.batch_size):
                    if terminal_batch[k]:
                        y_i.append(reward_batch[k])
                    else:
                        y_i.append(reward_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                        observation_batch,
                        action_batch,
                        np.reshape(y_i, (flags.batch_size, 1)))



                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(observation_batch)
                grads = critic.action_gradients(observation_batch, a_outs)
                actor.train(observation_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            observation = s2
            ep_reward += r

            if terminal:

                tf.summary.scalar("ep_reward",
                                  ep_reward,
                                  step=i)
                tf.summary.scalar("ep_ave_max_q",
                                  ep_ave_max_q / float(j),
                                  step=i)
                tf.summary.scalar("ep_max_q",
                                  ep_ave_max_q,
                                  step=i)

                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(
                    int(ep_reward), i, (ep_ave_max_q / float(j))))
                break


def main(flags):
    with tf.compat.v1.Session() as sess:

        env = gym.make(flags.env)
        np.random.seed(flags.random_seed)
        # tf.set_random_seed(int(args['random_seed']))
        env.seed(flags.random_seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess,
                             state_dim,
                             action_dim,
                             action_bound,
                             flags.actor_learning_rate,
                             flags.target_mixing_factor,
                             flags.batch_size)

        critic = CriticNetwork(sess,
                               state_dim,
                               action_dim,
                               flags.critic_learning_rate,
                               flags.target_mixing_factor,
                               flags.discount_factor,
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))


        train(sess, env, flags, actor, critic, actor_noise)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for training.')

    # agent parameters
    # parser.add_argument('--actor-lr', help='actor network learning rate',
    #                     default=0.0001)
    # parser.add_argument('--critic-lr', help='critic network learning rate',
    #                     default=0.001)
    # parser.add_argument('--gamma', help='discount factor for critic updates',
    #                     default=0.99)
    # parser.add_argument('--tau', help='soft target update parameter',
    #                     default=0.001)
    # parser.add_argument('--buffer-size', help='max size of the replay buffer',
    #                     default=1000000)
    # parser.add_argument('--minibatch-size',
    #                     help='size of minibatch for minibatch-SGD',
    #                     default=64)
    #
    # # run parameters
    # parser.add_argument('--env',
    #                     help='choose the gym env- tested on {Pendulum-v0}',
    #                     default='Pendulum-v0')
    # # parser.add_argument('--env',
    # #                     help='choose the gym env- tested on {Pendulum-v0}',
    # #                     default='MountainCarContinuous-v0')
    #
    #
    # parser.add_argument('--random-seed', help='random seed for repeatability',
    #                     default=1234)
    # parser.add_argument('--max-episodes',
    #                     help='max num of episodes to do while training',
    #                     default=50000)
    # parser.add_argument('--max-episode-len', help='max length of 1 episode',
    #                     default=1000)
    # parser.add_argument('--render-env', help='render the gym env',
    #                     action='store_true')
    # parser.add_argument('--use-gym-monitor', help='record gym results',
    #                     action='store_true')
    # parser.add_argument('--monitor-dir',
    #                     help='directory for storing gym results',
    #                     default='./results/gym_ddpg')
    # parser.add_argument('--summary-dir',
    #                     help='directory for storing tensorboard info',
    #                     default='./results/tf_ddpg')


    parser.add_argument('--random_seed', type=int,
                        default=1234,
                        help='A random seed for repeatability.')

    parser.add_argument('--env',
                        help='A gym environment name string.',
                        default='Pendulum-v0')

    parser.add_argument('--max_episode_steps', type=int,
                        default=1000,
                        help='Steps per episode limit.')

    parser.add_argument('--logdir', type=str,
                        default=".\\temp\\",
                        help='The directory to which summaries are written.')

    parser.add_argument('--run_name', type=str,
                        default="run_" + str(datetime.timestamp(datetime.now())),
                        help='The name of this run')

    parser.add_argument('--num_episodes', type=int,
                        default=50000,
                        help='Number of episodes to run.')

    parser.add_argument('--replay_buffer_size', type=int,
                        default=1000000,
                        help='Number of experience tuples in replay buffer.')

    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='Number of experiences in a batch.')

    parser.add_argument('--discount_factor', type=float,
                        default=0.99,
                        help='The discount factor on future rewards, gamma.')

    parser.add_argument('--target_mixing_factor', type=float,
                        default=0.001,
                        help='The update weight for the target networks.')

    parser.add_argument('--actor_learning_rate', type=float,
                        default=0.0001,
                        help='.')

    parser.add_argument('--critic_learning_rate', type=float,
                        default=0.001,
                        help='.')

    parser.add_argument('--render_env', action='store_true',
                        default=False,
                        help='If provided, render the environment.')

    parser.add_argument('--baseline', action='store_true',
                        default=False,
                        help='If provided, do not train the model.')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    # cli_main(parsed_flags)
    #
    # parser.set_defaults(render_env=False)
    # parser.set_defaults(use_gym_monitor=True)
    #
    # args = vars(parser.parse_args())
    #
    # pp.pprint(args)

    main(parsed_flags)