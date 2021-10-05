"""

A from-scratch implementation of DDPG, which is trained to optimally control
the Distributed Aperture System for Interferometric Exploitation (DASIE).

Author: Justin Fletcher


"""

import os
import random
import argparse
import numpy as np
from collections import deque

from datetime import datetime

import gym
import tensorflow as tf

tf.autograph.set_verbosity(3)


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
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ActorModel(tf.keras.Model):
    def __init__(self, env_action_space):
        super(ActorModel, self).__init__()

        self.env_action_space = env_action_space
        self.action_bound = env_action_space.high

        init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)

        self.dense_400 = tf.keras.layers.Dense(400, activation=None,
                                               input_shape=(None, 3, 1,))
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()
        self.dense_300 = tf.keras.layers.Dense(300,
                                               activation=None)
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.relu_2 = tf.keras.layers.ReLU()
        self.dense_1 = tf.keras.layers.Dense(1,
                                             kernel_initializer=init,
                                             bias_initializer=init,
                                             activation='tanh')

        self.mul = tf.keras.layers.Multiply()

    def call(self, observation):

        x = self.dense_400(observation)
        # x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.dense_300(x)
        # x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.dense_1(x)
        x = x * self.action_bound

        return x


class CriticModel(tf.keras.Model):
    def __init__(self, env_action_space):
        super(CriticModel, self).__init__()

        self.env_action_space = env_action_space


        init = tf.keras.initializers.RandomUniform(minval=-0.003,
                                                   maxval=0.003,
                                                   seed=None)


        self.obs_dense_1 = tf.keras.layers.Dense(400,
                                                 activation=None,
                                                 input_shape=(None, 3,))
        self.obs_bn = tf.keras.layers.BatchNormalization()
        self.obs_relu = tf.keras.layers.ReLU()
        self.obs_dense_2 = tf.keras.layers.Dense(300,
                                                 activation=None,
                                                 use_bias=False)


        self.act_dense_1 = tf.keras.layers.Dense(300,
                                                 activation=None)

        self.merge_relu = tf.keras.layers.ReLU()
        self.merge_dense = tf.keras.layers.Dense(1,
                                                 kernel_initializer=init,
                                                 bias_initializer=init,
                                                 activation=None)

        self.add = tf.keras.layers.Add()

    def call(self, observation, action):
        x_s = self.obs_dense_1(observation)
        # x_s = self.obs_bn(x_s)
        x_s = self.obs_relu(x_s)
        x_s = self.obs_dense_2(x_s)

        x_a = self.act_dense_1(action)

        # x = tf.matmul(x_o, W_o1) + tf.matmul(action_batch_placeholder, W_a0) + b_a0
        x = self.add([x_s, x_a])
        x = self.merge_relu(x)
        x = self.merge_dense(x)

        return x

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

def cli_main(flags):

    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # # Register our custom DASIE environment.
    # gym.envs.registration.register(
    #     id='Dasie-v0',
    #     entry_point='dasie_gym_env.dasie:DasieEnv',
    #     max_episode_steps=flags.max_episode_steps,
    #     reward_threshold=flags.reward_threshold,
    # )
    #
    # # Build a gym environment; pass the CLI flags to the constructor as kwargs.
    # env = gym.make('Dasie-v0', **vars(flags))

    env = gym.make("Pendulum-v0")

    # Randomly initialize a critic network, Q(s, a)...
    critic_model = CriticModel(env.action_space)

    # ...and randomly initialize an actor network mu(s).
    actor_model = ActorModel(env.action_space)

    # Duplicate the critic target network, Q'(s, a)...
    target_critic_model = CriticModel(env.action_space)
    target_critic_model.set_weights(critic_model.get_weights())

    # ...and duplicate the target actor network mu'(s).
    target_actor_model = ActorModel(env.action_space)
    target_actor_model.set_weights(actor_model.get_weights())

    # Initialize an empty replay buffer.
    replay_buffer = ReplayBuffer(flags.replay_buffer_size)

    if flags.baseline:

        writer = tf.summary.create_file_writer(os.path.join(flags.logdir,
                                                            "baseline"))

    else:

        writer = tf.summary.create_file_writer(os.path.join(flags.logdir,
                                                            flags.run_name))

    critic_optimizer = tf.keras.optimizers.Adam(flags.critic_learning_rate)

    actor_optimizer = tf.keras.optimizers.Adam(flags.actor_learning_rate)

    # Iterate over the number of desired episodes.
    for i_episode in range(flags.num_episodes):

        # Reset the environment, and get an initial observation...
        print("observation space:")
        print(env.observation_space.sample())
        print("action space:")
        print(env.action_space.sample())

        episode_rewards = list()

        observation = env.reset()
        observation = np.expand_dims(np.squeeze(observation), 0)

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space.shape[0]))

        # Now, iterate over the desired number of steps. In each step...
        for t in range(flags.max_episode_steps):

            global_step = (flags.max_episode_steps * i_episode) + t

            if flags.render_env:

                env.render()

            # Query an action from the actor model, then...
            action = actor_model(observation) + actor_noise()


            # ...take that action, and parse the state.
            new_observation, reward, terminal, info = env.step(action)

            with writer.as_default():
                tf.summary.scalar("instantaneous_reward",
                                  np.squeeze(reward),
                                  step=global_step)

            if not flags.silence:

                print("############################################")
                print("Running episode %s." % str(i_episode))
                print("Running step %s." % str(t))
                print("observation input:")
                print(observation)
                print("action output:")
                print(action)
                print("reward:")
                print(reward)
                print("terminal:")
                print(terminal)
                print("############################################")

            # Track variables.
            episode_rewards.append(np.squeeze(reward))


            # ...store this state in the replay buffer...
            # (Dimensions: Fresh squeezed and filtered!)
            replay_buffer.add(np.squeeze(observation),
                              np.squeeze(action.numpy()),
                              np.squeeze(reward),
                              np.squeeze(np.int(terminal)),
                              np.squeeze(new_observation))

            # ...and set the new observation for the next time step.
            observation = np.expand_dims(np.squeeze(new_observation), 0)

            # If the environment says we're done, stop this episode. Else...
            if terminal:
                print("Episode finished after {} timesteps".format(t + 1))
                break

            # Check if our replay buffer has at least one batch to train.
            if replay_buffer.size() >= flags.batch_size:

                if not flags.baseline:

                    # Now, sample a batch on which to train the models.
                    (observation_batch,
                     action_batch,
                     reward_batch,
                     terminal_batch,
                     new_observation_batch) = replay_buffer.sample_batch(flags.batch_size)

                    ###############################################################
                    # Start: Build the critic optimizer.                          #
                    ###############################################################

                    target_action = target_actor_model(new_observation_batch)

                    # print(target_action)
                    #
                    # # ...and reshape the output for future use.
                    # target_action = tf.expand_dims(tf.squeeze(target_action), 0)
                    # print(target_action)
                    #
                    # die

                    # Given the next state & inferred action, get the inferred Q.
                    target_critic_value = target_critic_model(new_observation_batch, target_action)

                    # Now, discount the Q value, and compute the target Q-value.
                    target_value = reward_batch + ([1 - t for t in terminal_batch] * target_critic_value) * flags.discount_factor

                    # ...and compute the MSE. Finally, normalize.
                    # action_batch = tf.expand_dims(tf.squeeze(action_batch), 0)

                    action_batch = tf.reshape(tf.squeeze(action_batch), [flags.batch_size, 1])

                    with tf.GradientTape() as tape:

                        # critic model dies here, but not above... action batches look different
                        # print(target_action)
                        # print(action_batch)
                        #
                        # die
                        current_step_q = critic_model(observation_batch, action_batch)

                        bellman_loss = tf.keras.losses.MSE(current_step_q, target_value)

                    value_gradients = tape.gradient(bellman_loss, critic_model.trainable_variables)
                    critic_optimizer.apply_gradients(zip(value_gradients, critic_model.trainable_variables))

                    # critic_optimizer.minimize(bellman_loss, critic_model.trainable_variables

                    with writer.as_default():

                        tf.summary.scalar("current_step_bellman_loss_mean",
                                          np.mean(bellman_loss.numpy()),
                                          step=global_step)
                        tf.summary.scalar("current_step_q_mean",
                                          np.mean(current_step_q.numpy()),
                                          step=global_step)
                        tf.summary.scalar("current_step_target_value_mean",
                                          np.mean(target_value.numpy()),
                                          step=global_step)
                        tf.summary.scalar("current_step_bellman_loss_var",
                                          np.var(bellman_loss.numpy()),
                                          step=global_step)
                        tf.summary.scalar("current_step_q_var",
                                          np.var(current_step_q.numpy()),
                                          step=global_step)
                        tf.summary.scalar("current_step_target_value_var",
                                          np.var(target_value.numpy()),
                                          step=global_step)

                    ###############################################################
                    # End: Build the critic optimizer.                            #
                    ###############################################################

                    ###############################################################
                    # Start: Build the actor optimizer.                           #
                    ###############################################################

                    with tf.GradientTape() as tape:

                        policy_action_batch = actor_model(observation_batch)

                        q_estimate_batch = critic_model(observation_batch, policy_action_batch)

                    # Take the gradient of the policy action with respect to policy params.
                    # unnormalized_policy_gradients = tape.gradient(q_estimate_batch, actor_model.trainable_variables)
                    # Normalize the policy gradients.
                    # policy_gradients = [-g / flags.batch_size for g in unnormalized_policy_gradients]
                    action_gradients = tape.gradient(q_estimate_batch, policy_action_batch)
                    # print(action_gradients)
                    #
                    # self.unnormalized_actor_gradients = tf.gradients(
                    #         self.scaled_out, self.network_params, -self.action_gradient)
                    # self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size),
                    #                                 self.unnormalized_actor_gradients))

                    with tf.GradientTape() as tape:

                        policy_action_batch = actor_model(observation_batch)

                    actor_gradients = tape.gradient(policy_action_batch, actor_model.trainable_variables)

                    unnormalized_policy_gradients = list()
                    for action_gradient, actor_gradient in zip(action_gradients, actor_gradients):

                        unnormalized_policy_gradient = tf.math.multiply(action_gradient, actor_gradient)
                        unnormalized_policy_gradients.append(unnormalized_policy_gradient)

                    # policy_gradients = list(map(lambda x: tf.math.divide(x, flags.batch_size), unnormalized_policy_gradients))

                    # policy_gradients = tape.gradient(q_estimate_batch, actor_model.trainable_variables)
                    policy_gradients = [tf.math.divide(g, flags.batch_size) for g in unnormalized_policy_gradients]

                    # print(policy_gradients)
                    #
                    # die
                    #
                    #
                    # # And finally apply them.
                    # actor_optimizer.apply_gradients(zip(policy_gradients, actor_model.trainable_variables))
                    actor_optimizer.apply_gradients(zip(policy_gradients, actor_model.trainable_variables))

                    var_updates = list()
                    for var, target_var in zip(actor_model.get_weights(), target_actor_model.get_weights()):
                        var_updates.append((flags.target_mixing_factor * target_var) + ((1 - flags.target_mixing_factor) * var))

                    target_actor_model.set_weights(var_updates)

                    var_updates = list()
                    for (var, target_var) in zip(critic_model.get_weights(),  target_critic_model.get_weights()):
                        var_updates.append((flags.target_mixing_factor * target_var) + ((1 - flags.target_mixing_factor) * var))

                    target_critic_model.set_weights(var_updates)
                    # target_critic_model = critic_model
                    #
                    # target_actor_model = actor_model
                    ###############################################################
                    # End: Build the actor optimizer.                             #
                    ###############################################################

                    with writer.as_default():

                        tf.summary.scalar("actor_model_param_sum",
                                          np.sum([np.sum(v.numpy()) for v in actor_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("actor_model_param_mean",
                                          np.mean([np.mean(v.numpy()) for v in actor_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("actor_model_param_var",
                                          np.var([np.var(v.numpy()) for v in actor_model.trainable_variables]),
                                          step=global_step)

                        tf.summary.scalar("target_actor_model_param_sum",
                                          np.sum([np.sum(v.numpy()) for v in target_actor_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("target_actor_model_param_mean",
                                          np.mean([np.mean(v.numpy()) for v in target_actor_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("target_actor_model_param_var",
                                          np.var([np.var(v.numpy()) for v in target_actor_model.trainable_variables]),
                                          step=global_step)

                        tf.summary.scalar("critic_model_param_sum",
                                          np.sum([np.sum(v.numpy()) for v in critic_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("critic_model_param_mean",
                                          np.mean([np.mean(v.numpy()) for v in critic_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("critic_model_param_var",
                                          np.var([np.var(v.numpy()) for v in critic_model.trainable_variables]),
                                          step=global_step)

                        tf.summary.scalar("target_critic_model_param_sum",
                                          np.sum([np.sum(v.numpy()) for v in target_critic_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("target_critic_model_param_mean",
                                          np.mean([np.mean(v.numpy()) for v in target_critic_model.trainable_variables]),
                                          step=global_step)
                        tf.summary.scalar("target_critic_model_param_var",
                                          np.var([np.var(v.numpy()) for v in target_critic_model.trainable_variables]),
                                          step=global_step)
        # if terminal:


        # End episode loop.

        with writer.as_default():

            tf.summary.scalar("episode_reward_sum", np.sum(episode_rewards), step=i_episode)
            tf.summary.scalar("episode_reward_mean", np.mean(episode_rewards), step=i_episode)
            tf.summary.scalar("episode_reward_var", np.var(episode_rewards), step=i_episode)
    # Once all episodes are complete, close the environment.
    env.close()




if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--num_apertures', type=int,
                        default=9,
                        help='Number of apertures to simulate.')

    parser.add_argument('--reward_threshold', type=float,
                        default=25.0,
                        help='Max reward per episode.')

    parser.add_argument('--simulated_inference_latency', type=float,
                        default=0.025,
                        help='The latency caused by the model in secs.')

    parser.add_argument('--simulated_command_transmission_latency', type=float,
                        default=0.030,
                        help='The latency caused by command transfer in secs.')

    parser.add_argument('--simulated_actuation_latency', type=float,
                        default=0.005,
                        help='The latency caused by actuation in secs.')

    parser.add_argument('--simulation_time_granularity', type=float,
                        default=0.001,
                        help='The time granularity of DASIE sim in secs.')

    parser.add_argument('--tip_phase_error_scale', type=float,
                        default=0.01,
                        help='The initial tip alignment std.')

    parser.add_argument('--tilt_phase_error_scale', type=float,
                        default=0.01,
                        help='The initial tilt alignment std.')

    parser.add_argument('--piston_phase_error_scale', type=float,
                        default=0.01,
                        help='The initial piston alignment std.')

    parser.add_argument('--silence', action='store_true',
                        default=False,
                        help='If provided, be quiet.')

    parser.add_argument('--dasie_version', type=str,
                        default="test",
                        help='Which version of the DASIE sim do we use?')

    ###########################################################################
    #                               New Arguments                             #
    ###########################################################################

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
                        default=0.999,
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
    cli_main(parsed_flags)