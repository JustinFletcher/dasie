"""
A pure TensorFlow implementation of DDPG.

http://arxiv.org/pdf/1509.02971v2.pdf

Author: Justin Fletcher
"""

import os
import time
import random
import argparse
from datetime import datetime
from collections import deque

import gym
import numpy as np
import tensorflow as tf


# First, prevent TensorFlow from foisting filthy eager execution upon us.
tf.compat.v1.disable_eager_execution()

class ReplayBuffer(object):

    """
     A simple replay buffer, borrowed with gratitude from Patrick Emami.
    """

    def __init__(self, buffer_size=1024, random_seed=123):
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



def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.compat.v1.name_scope('summaries'):
        mean = tf.reduce_mean(input_tensor=var)
        tf.compat.v1.summary.scalar('mean', mean)
        with tf.compat.v1.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(input_tensor=var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(input_tensor=var))
        tf.compat.v1.summary.histogram('histogram', var)

class ActorNetwork(object):
    """

    """

    def __init__(self, sess, state_dim, action_dim, action_bound,
                 learning_rate, tau, batch_size):

        """
        This constructor builds a graph corresponding to an actor model in the
        DDPG actor-critic framework. This model is coupled to the OpenAi gym
        environment provided.

        :param sess: a tf.Session, in which the graph will be built.
        :param state_dim: an int, specifying the size of the action space.
        :param action_dim:
        :param action_bound:
        :param learning_rate:
        :param tau:
        :param batch_size:
        """
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size


        # First, define a top level scope to hold all actor ops.
        with tf.name_scope("actor"):

            # TODO: Replace with function-as-interface structure.
            # Next, build the model using the local model creation function.
            with tf.name_scope("actor_model"):
                self.inputs, self.out, self.scaled_out = self._create_actor_network()

            # Parse the trainable variables in the main model.
            self.network_params = tf.compat.v1.trainable_variables()

            # Target Network
            with tf.name_scope("actor_model_target"):
                self.target_inputs, self.target_out, self.target_scaled_out = self._create_actor_network()


            self.target_network_params = tf.compat.v1.trainable_variables()[
                                         len(self.network_params):]

            # Next, build the graph for updating the target parameters.
            with tf.name_scope("actor_target_update_ops"):

                # Start by making a list to store the update ops.
                self.update_target_network_params_ops = list()

                # TODO: Make this a zip over main, target param pairs. ID w/ scope.
                # Then iterate over each target network parameter, and...
                for i in range(len(self.target_network_params)):

                    # ...scale the corresponding main parameter and...
                    scaled_param = tf.multiply(self.network_params[i],
                                               self.tau)

                    # ...scale this target parameter. Then, ...
                    scaled_target = tf.multiply(self.target_network_params[i],
                                                1. - self.tau)

                    # ...blend the parameters by adding them, and...
                    target_param = scaled_param + scaled_target

                    # ...add an assign overwriting the prior target parameter.
                    update_op = self.target_network_params[i].assign(
                        target_param)

                    # Finally, add the update op to a list to be run later.
                    self.update_target_network_params_ops.append(update_op)

        # This gradient will be provided by the critic network
        self.action_gradient = tf.compat.v1.placeholder(tf.float32,
                                                        [None, self.a_dim])

        with tf.name_scope("policy_gradients"):

            # Compute the gradient of the actor output (action) wrt parameters.
            # Scale the gradient my the gradient of value (Q) wrt actions.
            # The result is called a policy gradient; but it's unnormalized...
            self.unnormalized_actor_gradients = tf.gradients(self.scaled_out,
                                                             self.network_params,
                                                             -self.action_gradient)

            # ...so normalize each gradient by dividing by batch size.
            self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size),
                                            self.unnormalized_actor_gradients))

        with tf.name_scope("actor_optimizer"):

            # Build an op that applies the policy gradients to the model.
            self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate). \
                apply_gradients(zip(self.actor_gradients, self.network_params))


        self.summaries = tf.compat.v1.summary.merge_all()
        # And store the total number of parameters.
        # TODO: Remove this.
        self.num_trainable_vars = len(
                self.network_params) + len(self.target_network_params)

    def _create_actor_network(self):
        observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                                 shape=(None, self.s_dim),
                                                                 name="observation")


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
            variable_summaries(W)
            b = tf.Variable(tf.random.normal((n_hidden,)))
            variable_summaries(b)
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
        return self.sess.run([self.optimize, self.summaries], feed_dict={
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
        self.sess.run(self.update_target_network_params_ops)

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

        with tf.name_scope("critic"):

            with tf.name_scope("critic_model"):

                self.inputs, self.action, self.out = self.create_critic_network()

            self.network_params = tf.compat.v1.trainable_variables()[num_actor_vars:]

            with tf.name_scope("critic_target"):

                # Target Network
                self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

                self.target_network_params = tf.compat.v1.trainable_variables()[(len(
                    self.network_params) + num_actor_vars):]

            # Next, build the graph for updating the target parameters.
            with tf.name_scope("critic_target_update_ops"):

                # Start by making a list to store the update ops.
                self.update_target_network_params_ops = list()

                # TODO: Make this a zip over main, target param pairs. ID w/ scope.
                # Then iterate over each target network parameter, and...
                for i in range(len(self.target_network_params)):

                    # ...scale the corresponding main parameter and...
                    scaled_param = tf.multiply(self.network_params[i],
                                               self.tau)

                    # ...scale this target parameter. Then, ...
                    scaled_target = tf.multiply(self.target_network_params[i],
                                                1. - self.tau)

                    # ...blend the parameters by adding them, and...
                    target_param = scaled_param + scaled_target

                    # ...add an assign that overwrites the prior target parameter.
                    update_op = self.target_network_params[i].assign(target_param)

                    # Finally, add the update op to a list to be run later.
                    self.update_target_network_params_ops.append(update_op)

        # Create a placeholder to accept the Q value estimate a runtime.
        self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])

        with tf.name_scope("mean_squared_bellman_error_loss"):

            # Now, define a loss op computing batch MSE between target and Q.
            self.loss = tf.math.reduce_mean(tf.pow((self.predicted_q_value - self.out), 2))

        with tf.name_scope("critic_optimizer"):

            self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.name_scope("action_gradients"):

            # Get the gradient of the net w.r.t. the action.
            # For each action in the minibatch (i.e., for each x in xs),
            # this will sum up the gradients of each critic output in the minibatch
            # w.r.t. that action. Each output is independent of all
            # actions except for one.
            self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):

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
        self.sess.run(self.update_target_network_params_ops)


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.6, theta=.15, dt=1e-2, x0=None):
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

    summary_vars = list()

    episode_reward = tf.Variable(0.)
    summary_vars.append(episode_reward)
    tf.summary.scalar("Reward", episode_reward)

    episode_ave_max_q = tf.Variable(0.)
    summary_vars.append(episode_ave_max_q)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_ops = tf.compat.v1.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, flags, actor, critic, actor_noise, summary_ops):
    """
    This function realizes the DDPG training loop against an OpenAI Gym
    environment, using an ActorModel and CriticModel.

    :param sess: a tf.Session, in which the main graph is built.
    :param env: a gym.Environment to train the models against.
    :param flags: a namespace containing user-specified flags.
    :param actor: an ActorModel built in sess against env.
    :param critic: a CriticModel built in sess against env.
    :param actor_noise: a callable which returns action-sized arrays.
    :return: None
    """


    # First, if this a baseline run, set the logdir and prevent learning.
    if flags.baseline:

        logdir_path = os.path.join(flags.logdir, "baseline")

    else:

        logdir_path = os.path.join(flags.logdir, flags.run_name)

    # TODO: Either use or deprecate this.
    # summary_ops, summary_vars = build_summaries()

    # Begin training by intializing the graph.
    sess.run(tf.compat.v1.global_variables_initializer())

    # Create a summary writer, and flush it to write the inital graph.
    writer = tf.compat.v1.summary.FileWriter(logdir_path, sess.graph)

    # writer.flush()
    # writer = tf.summary.create_file_writer(logdir_path)

    # Run the target network update ops, since everything's initialized.
    actor.update_target_network()
    critic.update_target_network()

    # Create a replay buffer to store observations.
    replay_buffer = ReplayBuffer(flags.replay_buffer_size, flags.random_seed)

    # Iterate through a user-specified number of random uniform samples.
    for _ in range(flags.num_random_samples):

        # Reset the enviroment, get a random action, and take it; then...
        state = env.reset()
        action = env.action_space.sample()
        new_observation, reward, terminal, info = env.step(action)

        # TODO: Consider random tree search to get high-value trajectories.

        # Add that random interaction to the buffer.
        replay_buffer.add(np.reshape(state, (actor.s_dim,)),
                          np.reshape(action, (actor.a_dim,)),
                          reward,
                          terminal,
                          np.reshape(new_observation, (actor.s_dim,)))

    episode_running_times = list()

    # Enter the main training loop.
    for i in range(flags.num_episodes):

        episode_start_time = time.time()

        # Refresh all episode bookkeeping variables.
        ep_reward = 0
        ep_ave_max_q = 0

        # Reset the state.
        observation = env.reset()

        train_step_run_times = list()
        step_run_times = list()

        # For the user-specified number of steps...
        for j in range(flags.max_episode_steps):

            step_start_time = time.time()

            if flags.render_env:
                env.render()

            # Consult the actor model for an action.
            a = actor.predict(np.reshape(observation, (1, actor.s_dim))) + actor_noise()

            new_observation, reward, terminal, info = env.step(a[0])

            # print(terminal)

            replay_buffer.add(np.reshape(observation, (actor.s_dim,)),
                              np.reshape(a, (actor.a_dim,)), reward,
                              terminal, np.reshape(new_observation, (actor.s_dim,)))

            train_step_start_time = time.time()

            # If we've stored at least the required number of batches.
            if replay_buffer.size() > (flags.min_batches * flags.batch_size):

                # Sample a random batch of experience.
                (observation_batch,
                 action_batch,
                 reward_batch,
                 terminal_batch,
                 next_observation_batch) = replay_buffer.sample_batch(flags.batch_size)

                # Compute the Q values implied by the actor and critic, then...
                target_q = critic.predict_target(next_observation_batch,
                                                 actor.predict_target(next_observation_batch))

                # ...discount the Q value, and compute the target Q-value.
                terminal_vector = [1. - float(t) for t in terminal_batch]
                # y_i = reward_batch + (terminal_vector * target_q) * flags.discount_factor

                target_q_value_batch = list()
                for reward_i, terminal_i, target_q_i in zip(reward_batch,
                                                            terminal_vector,
                                                            target_q):
                    y_i = reward_i + (terminal_i * target_q_i * flags.discount_factor)

                    target_q_value_batch.append(y_i)

                # Train the critic, minimizing the mean squared Bellman error.
                predicted_q_value, _ = critic.train(
                        observation_batch,
                        action_batch,
                        np.reshape(target_q_value_batch, (flags.batch_size, 1)))

                # Store the maximum of the predicted Q values.
                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient.
                policy_action_batch = actor.predict(observation_batch)
                grads = critic.action_gradients(observation_batch,
                                                policy_action_batch)


                _, actor_summaries = actor.train(observation_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            observation = new_observation
            ep_reward += reward

            # TODO: compute observation-action covariance for episode traj.
            # TODO: compute action variance across episode traj.

            episode_end_time = time.time() - episode_start_time
            episode_running_times.append(episode_end_time)

            step_end_time = time.time() - step_start_time
            step_run_times.append(step_end_time)

            train_step_end_time = time.time() - train_step_start_time
            train_step_run_times.append(train_step_end_time)

            if terminal:
                writer.add_summary(actor_summaries, i)
                #
                # with writer.as_default():
                #     tf.summary.scalar("ep_reward",
                #                       ep_reward,
                #                       step=i)
                #     tf.summary.scalar("ep_ave_max_q",
                #                       ep_ave_max_q / float(j),
                #                       step=i)
                #     tf.summary.scalar("ep_max_q",
                #                       ep_ave_max_q,
                #                       step=i)
                #     tf.summary.scalar("ep_run_time",
                #                       episode_end_time,
                #                       step=i)
                #     tf.summary.scalar("step_run_time",
                #                       np.mean(step_run_times),
                #                       step=i)
                #     tf.summary.scalar("train_step_run_time",
                #                       np.mean(train_step_run_times),
                #                       step=i)

                # writer.write()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Episode Time: {:.4f} | Step Time: {:.5f} | Train Step Time: {:.5f}'.format(
                        int(ep_reward),
                        i,
                        (ep_ave_max_q / float(j)),
                        episode_end_time,
                        np.mean(step_run_times),
                        np.mean(train_step_run_times)))
                break


def main(flags):

    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Register our custom DASIE environment.
    gym.envs.registration.register(
        id='Dasie-v0',
        entry_point='dasie_gym_env.dasie:DasieEnv',
        max_episode_steps=flags.max_episode_steps,
        reward_threshold=flags.reward_threshold,
    )
    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:

        # Build a gym environment; pass the CLI flags to the constructor as kwargs.
        env = gym.make(flags.env, **vars(flags))
        np.random.seed(flags.random_seed)
        tf.compat.v1.set_random_seed(flags.random_seed)
        env.seed(flags.random_seed)

        print(env.action_space)
        action_dim = env.action_space.shape[0]
        print(action_dim)
        action_bound = env.action_space.high

        state_dim = env.observation_space.shape[0]
        print(state_dim)

        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        # Build an actor model in this sess for this env.
        actor = ActorNetwork(sess,
                             state_dim,
                             action_dim,
                             action_bound,
                             flags.actor_learning_rate,
                             flags.target_mixing_factor,
                             flags.batch_size)

        # Build an critic model in this sess for this env.
        critic = CriticNetwork(sess,
                               state_dim,
                               action_dim,
                               flags.critic_learning_rate,
                               flags.target_mixing_factor,
                               flags.discount_factor,
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        summary_ops = tf.compat.v1.summary.merge_all()

        train(sess, env, flags, actor, critic, actor_noise, summary_ops)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for training.')

    # Set arguments and their default values
    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--reward_threshold', type=float,
                        default=25.0,
                        help='Max reward per episode.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1234,
                        help='A random seed for repeatability.')

    parser.add_argument('--env',
                        type=str,
                        help='A gym environment name string.',
                        default='Pendulum-v0')

    parser.add_argument('--max_episode_steps',
                        type=int,
                        default=1000,
                        help='Steps per episode limit.')

    parser.add_argument('--logdir',
                        type=str,
                        default=".\\temp\\",
                        help='The directory to which summaries are written.')

    parser.add_argument('--run_name',
                        type=str,
                        default="run_" + str(datetime.timestamp(datetime.now())),
                        help='The name of this run')

    parser.add_argument('--num_episodes',
                        type=int,
                        default=50000,
                        help='Number of episodes to run.')

    parser.add_argument('--num_random_samples',
                        type=int,
                        default=0,
                        help='Number of uniform env samples to prepend.')

    parser.add_argument('--replay_buffer_size',
                        type=int,
                        default=1000000,
                        help='Number of experience tuples in replay buffer.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Number of experiences in a batch.')

    parser.add_argument('--min_batches',
                        type=int,
                        default=1,
                        help="The minimum number of batches needed to train.")

    parser.add_argument('--discount_factor',
                        type=float,
                        default=0.99,
                        help='The discount factor (gamma) on future rewards.')

    parser.add_argument('--target_mixing_factor',
                        type=float,
                        default=0.001,
                        help='The update weight (tau) for target networks.')

    parser.add_argument('--actor_learning_rate',
                        type=float,
                        default=0.0001,
                        help='The learning rate for the actor optimizer')

    parser.add_argument('--critic_learning_rate',
                        type=float,
                        default=0.001,
                        help='The learning rate for the critic optimizer.')

    parser.add_argument('--render_env',
                        action='store_true',
                        default=False,
                        help='If provided, render the environment.')

    parser.add_argument('--baseline',
                        action='store_true',
                        default=False,
                        help='If provided, do not train the model.')


    ############################ DASIE FLAGS ##################################
    parser.add_argument('--extended_object_image_file', type=str,
                        help='Filename of image to convolve PSF with (if none, PSF returned)')

    ### Telescope / pupil-plane setup ###

    # For now, passing in telescope setup pkl overrides any CLI arguments relating to
    # telescope setup.  I tried a bunch of strategies to make it possible to have the best
    # of both worlds: with a loadable setup where CLI args would override specific values,
    # but it was ugly not matter what strategy I tried based on current code structure
    parser.add_argument('--telescope_setup_pkl', type=str,
                        help='.pkl file containing dict passed into MultiAperturePSFSampler (overrides CLI telescope arguments)')

    parser.add_argument('--num_apertures', type=int,
                        default=15,
                        help='Number of apertures in ELF annulus')

    parser.add_argument('--telescope_radius', type=float,
                        default=1.25,
                        help='Distance from telescope center to aperture centers (meters)')

    parser.add_argument('--subaperture_radius', type=float,
                        default=None,
                        help='Radius of each sub-aperture (default is maximal filling) (meters)')

    parser.add_argument('--spider_width', type=float,
                        default=None,
                        help='Width of spider (default is no spider) (meters)')

    parser.add_argument('--spider_angle', type=float,
                        default=None,
                        help='Spider orientation angle (0-90) (default is random) (degrees)')

    parser.add_argument('--pupil_plane_resolution', type=int,
                        default=2 ** 8,
                        help='Resolution of pupil plane simulation')

    parser.add_argument('--piston_actuate_scale', type=float,
                        default=1e-7,
                        help='Sub-aperture piston actuation scale (meters)')

    parser.add_argument('--tip_tilt_actuate_scale', type=float,
                        default=1e-7,
                        help='Sub-aperture tip and tilt actuation scale (microns/meter)~=(radians)')

    ### Focal-plane setup ###
    parser.add_argument('--filter_central_wavelength', type=float,
                        default=1e-6,
                        help='Central wavelength of focal-plane observation (meters)')

    parser.add_argument('--filter_psf_extent', type=float,
                        default=4.0,
                        help='Angular extent of simulated PSF (arcsec)')

    parser.add_argument('--filter_psf_resolution', type=int,
                        default=2 ** 8,
                        help='Resolution of simulated PSF (this and extent set pixel scale for extended image convolution)')

    parser.add_argument('--filter_fractional_bandwidth', type=float,
                        default=0.05,
                        help='Fractional bandwidth of filter')

    parser.add_argument('--filter_bandwidth_samples', type=int,
                        default=3,
                        help='Number of pupil-planes used to simulate bandwidth (1 = monochromatic)')

    ### Atmosphere setup ###
    parser.add_argument('--atmosphere_type', type=str,
                        default="none",
                        help='Atmosphere type: "none" (default), "single" layer, or "multi" layer')

    parser.add_argument('--atmosphere_fried_paramater', type=float,
                        default=0.25,
                        help='Fried paramater, r0 @ 550nm (maters)')

    parser.add_argument('--atmosphere_outer_scale', type=float,
                        default=200.0,
                        help='Atmosphere outer-scale (maters)')

    # !!! Note: Doesn't currentoly work with multi-layer atmospheres, stuck at 10m/s
    parser.add_argument('--atmosphere_velocity', type=float,
                        default=10.0,
                        help='Atmosphere velocity (maters/second)')

    # !!! Breaks render right now, but should work for simulation...
    parser.add_argument('--enable_atmosphere_scintilation', action='store_true',
                        default=False,
                        help='Simulate atmospheric scintilation in multi-layer atmosphere')

    ### Object flux and detector noise ###

    # In order to get photon noise (and have read noise make sense)
    # we need to specify photon flux integrated over the length of our exposures
    # (photons/m^2).
    # This can map onto observable magnitudes latter with less assumptions up front
    parser.add_argument('--integrated_photon_flux', type=float,
                        help='Total number of photons/m^2 from FOV (Default: None = no noise)')

    # This dpeneds on integrated_photon_flux being specified
    # Not sure that a reasonable default for this is, but there should be *some*
    parser.add_argument('--read_noise', type=float,
                        default=10.0,
                        help='Scaler giving the rms read noise (counts) (Only used when integrated_photon_flux specified)')

    ### Deformable mirror approximation of PTT actuation ###
    parser.add_argument('--dm_actuator_num', type=int,
                        help='Number of DM actuators on a side (Default: None = no DM approximation of PTT)')

    parser.add_argument('--dm_actuator_spacing', type=float,
                        default=0.1125,
                        help='pupil-plane spacing of actuators in meters')

    ### Simulation setup ###
    parser.add_argument('--step_time_granularity', type=float,
                        default=0.01,
                        help='The time granularity of DASIE step (seconds)')

    parser.add_argument('--tip_phase_error_scale', type=float,
                        default=0.01,
                        help='The initial tip alignment std.')

    parser.add_argument('--tilt_phase_error_scale', type=float,
                        default=0.01,
                        help='The initial tilt alignment std.')

    parser.add_argument('--piston_phase_error_scale', type=float,
                        default=0.01,
                        help='The initial piston alignment std.')


    parser.add_argument('--num_steps', type=int,
                        default=500,
                        help='Number of steps to run.')

    parser.add_argument('--simulated_inference_latency', type=float,
                        default=0.025,
                        help='The latency caused by the model in secs.')

    parser.add_argument('--simulated_command_transmission_latency', type=float,
                        default=0.030,
                        help='The latency caused by command transfer in secs.')

    parser.add_argument('--simulated_actuation_latency', type=float,
                        default=0.005,
                        help='The latency caused by actuation in secs.')

    parser.add_argument('--silence', action='store_true',
                        default=False,
                        help='If provided, be quiet.')

    parser.add_argument('--dasie_version', type=str,
                        default="test",
                        help='Which version of the DASIE sim do we use?')
    ###########################################################################

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()


    main(parsed_flags)