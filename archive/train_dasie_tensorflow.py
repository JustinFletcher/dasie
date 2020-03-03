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

import gym
import tensorflow as tf

tf.autograph.set_verbosity(3)


class ActorModel(tf.keras.Model):
    def __init__(self, env_action_space):
        super(ActorModel, self).__init__()

        self.env_action_space = env_action_space


    def call(self, observation):


        # x = tf.keras.layers.Flatten(input_shape=(None, 1, 3))(observation)
        x = tf.keras.layers.Dense(400, activation=None, input_shape=(None, 3, 1,))(observation)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(300, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(1, activation='tanh')(x)


        return x


class CriticModel(tf.keras.Model):
    def __init__(self, env_action_space):
        super(CriticModel, self).__init__()


        self.env_action_space = env_action_space

        self.obs_dense_1 = tf.keras.layers.Dense(400, activation=None, input_shape=(None, 3, 1,))
        self.obs_bn = tf.keras.layers.BatchNormalization()
        self.obs_relu = tf.keras.layers.ReLU()
        self.obs_dense_2 = tf.keras.layers.Dense(300, activation=None)

        self.act_dense_1 = tf.keras.layers.Dense(300, activation=None, input_shape=(None, 8, 1))

        self.merge_relu = tf.keras.layers.ReLU()
        self.merge_dense = tf.keras.layers.Dense(1, activation=None)

    def call(self, observation, action):
        x_s = self.obs_dense_1(observation)
        x_s = self.obs_bn(x_s)
        x_s = self.obs_relu(x_s)
        x_s = self.obs_dense_2(x_s)

        x_a = self.act_dense_1(action)

        x = x_s + x_a
        x = self.merge_relu(x)
        x = self.merge_dense(x)

        return x

def mix_trainable_variables(model_a, model_b, mixing_factor, mix_locally=True):
    mixed_var_values = list()

    for i, (a_var, b_var) in enumerate(zip(model_a.trainable_variables,
                                           model_b.trainable_variables)):
        mixed_var = (mixing_factor * a_var) + ((1 - mixing_factor) * b_var)

        if mix_locally:

            model_b.trainable_variables[i] = mixed_var

        mixed_var_values.append(mixed_var)

    return (mixed_var_values)

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

    # with tf.compat.v1.Session() as sess:

    env = gym.make("Pendulum-v0")

    # TODO Randomly initialize a critic network, Q(s, a)...
    critic_model = CriticModel(env.action_space)

    # TODO ...and randomly initialize an actor network mu(s).
    actor_model = ActorModel(env.action_space)

    # TODO Randomly initialize a critic target network, Q'(s, a)...
    target_critic_model = CriticModel(env.action_space)

    target_critic_model.set_weights(critic_model.get_weights())

    # TODO ...and randomly initialize a target actor network mu'(s).
    target_actor_model = ActorModel(env.action_space)
    target_actor_model.set_weights(actor_model.get_weights())



    #
    # action_batch_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, env.action_space.shape[0]), name="action")
    # reward_batch_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="reward")
    # observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]), name="observation")
    # new_observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]), name="new_observation")

    # #######################################################################
    # # Start: Build the critic optimizer.                                  #
    # #######################################################################
    #
    #
    # # Compute the target value for the observation, then...
    # target_value = reward_batch_placeholder + flags.discount_factor * target_critic_model(new_observation_batch_placeholder,
    #                                                                                       target_actor_model(new_observation_batch_placeholder))
    #
    # # ...get the predicted value from the model...
    # q_value = critic_model(observation_batch_placeholder, action_batch_placeholder)
    #
    # # ...and compute the MSE. Finally, normalize.
    # bellman_loss = tf.reduce_mean((target_value - q_value) ** 2) / flags.batch_size
    #
    # print("##################### DEBUG #######################")
    # print(critic_model.trainable_variables)
    # print("##################### DEBUG #######################")
    #
    #
    # # Create an optimizer op for this loss...
    # optimize_critic = tf.keras.optimizers.Adam.minimize(bellman_loss, critic_model.trainable_variables)
    #
    # #######################################################################
    # # End: Build the critic optimizer.                                    #
    # #######################################################################
    #
    #
    # #######################################################################
    # # Start: Build the actor optimizer.                                  #
    # #######################################################################
    #
    # # First, compute the gradient of the critic function wrt action.
    # policy_action_batch = actor_model(observation_batch_placeholder)
    #
    # # Take the gradient of predicted value wrt the policy action.
    # critic_action_gradient = tf.gradients(critic_model(observation_batch_placeholder, policy_action_batch),
    #                                       policy_action_batch)
    #
    # # Take the gradient of the policy action with respect to policy params.
    #
    #
    # unnormalized_policy_gradients = tf.gradients(policy_action_batch, actor_model.trainable_variables, tf.compat.v1.negative(critic_action_gradient))
    #
    #
    #
    # # Normalize the policy gradients.
    # policy_gradients = list(map(lambda x: tf.div(x, flags.batch_size), unnormalized_policy_gradients))
    #
    # optimize_actor = tf.keras.optimizers.Adam.apply_gradients(zip(policy_gradients, actor_model.trainable_variables))
    # #######################################################################
    # # End: Build the actor optimizer.                                    #
    # #######################################################################

    # Initialize an empty replay buffer.
    replay_buffer = ReplayBuffer(flags.replay_buffer_size)

    # Iterate over the number of desired episodes.
    for i_episode in range(flags.num_episodes):

        # Reset the environment, and get an initial observation...
        print("observation space:")
        print(env.observation_space.sample())
        print("action space:")
        print(env.action_space.sample())

        observation = env.reset()
        observation = np.expand_dims(np.squeeze(observation), 0)

        # TODO ...and establish an exploration noise distribution.

        # Now, iterate over the desired number of steps. In each step...
        for t in range(flags.max_episode_steps):


            if flags.render_env:

                env.render()

            # Query an action from the actor model, then...
            action = actor_model(observation)
            # ...take that action, and parse the state.
            new_observation, reward, terminal, info = env.step(action)

            if not flags.silence:

                print("##################### DEBUG #######################")
                print("Running episode %s." % str(i_episode))
                print("Running step %s." % str(t))
                print("observation input:")
                print(observation)
                print("action output:")
                print(action)
                print("reward:")
                print(reward)
                print("##################### DEBUG #######################")

            # If the environment says we're done, stop this episode. Else...
            if terminal:
                print("Episode finished after {} timesteps".format(t + 1))
                break

            # ...store this state in the replay buffer...
            # (Dimensions: Fresh squeezed and filtered!)
            replay_buffer.add(np.squeeze(observation),
                              np.squeeze(action.numpy()),
                              np.squeeze(reward),
                              np.squeeze(terminal),
                              np.squeeze(new_observation))

            # ...and set the new observation for the next time step.
            observation = np.expand_dims(np.squeeze(new_observation), 0)

            print("new obs")

            print(new_observation)
            # Check if our replay buffer has at least one batch to train.
            if replay_buffer.size() >= flags.batch_size:

                # Now, sample a batch on which to train the models.
                (observation_batch,
                 action_batch,
                 reward_batch,
                 terminal_batch,
                 new_observation_batch) = replay_buffer.sample_batch(flags.batch_size)

                # print(observation_batch)
                # print(action_batch)
                # print(reward_batch)
                # print(terminal_batch)
                # print(new_observation_batch)



                # # Construct a feed dict and apply an optimizer step.
                # feed_dict={action_batch_placeholder: action_batch,
                #            reward_batch_placeholder: reward_batch,
                #            observation_batch_placeholder: observation_batch,
                #            new_observation_batch_placeholder: new_observation_batch}
                #
                # sess.run(optimize_critic, feed_dict=feed_dict)
                # sess.run(optimize_actor, feed_dict=feed_dict)

                #######################################################################
                # Start: Build the critic optimizer.                                  #
                #######################################################################

                target_action = target_actor_model(new_observation_batch)
                target_action = np.expand_dims(np.squeeze(target_action), 0)

                # Compute the target value for the observation, then...
                target_critic_value = target_critic_model(new_observation_batch, target_action)
                target_value = reward_batch + flags.discount_factor * target_critic_value


                action_batch = np.expand_dims(np.squeeze(action_batch), 0)

                # ...get the predicted value from the model...
                # q_value = critic_model(observation_batch, action_batch)

                # ...and compute the MSE. Finally, normalize.
                bellman_loss = lambda: tf.reduce_mean((target_value - critic_model(observation_batch, action_batch)) ** 2) / flags.batch_size
                # print("##################### DEBUG #######################")
                # print()
                # print("##################### DEBUG #######################")


                # Create an optimizer op for this loss...

                optimizer = tf.keras.optimizers.Adam()
                optimizer.minimize(loss=bellman_loss, var_list=critic_model.trainable_variables)
                #######################################################################
                # End: Build the critic optimizer.                                    #
                #######################################################################


                #######################################################################
                # Start: Build the actor optimizer.                                  #
                #######################################################################

                # Take the gradient of predicted value wrt the policy action.
                with tf.GradientTape(persistent=True) as tape:

                    # First, compute the gradient of the critic function wrt action.
                    policy_action_batch = tf.expand_dims(tf.squeeze(actor_model(observation_batch)), 0)

                    critic_value_batch = critic_model(observation_batch, policy_action_batch)

                critic_action_gradients = tape.gradient(critic_value_batch, policy_action_batch)

                # Take the gradient of the policy action with respect to policy params.
                unnormalized_policy_gradients = tape.gradient(policy_action_batch, actor_model.trainable_variables, tf.compat.v1.negative(critic_action_gradients))

                # Normalize the policy gradients.
                policy_gradients = list(map(lambda x: tf.div(x, flags.batch_size), unnormalized_policy_gradients))

                # And finally apply them.
                actor_optimizer = tf.keras.optimizers.Adam()
                actor_optimizer.apply_gradients(zip(policy_gradients, actor_model.trainable_variables))

                #######################################################################
                # End: Build the actor optimizer.                                    #
                #######################################################################

                # Finally, update the target models.
                mix_trainable_variables(actor_model, target_actor_model, flags.target_mixing_factor)
                mix_trainable_variables(critic_model, target_critic_model, flags.target_mixing_factor)
    # Once all episodes are complete, close the environment.
    env.close()




if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--max_episode_steps', type=int,
                        default=64,
                        help='Steps per episode limit.')

    parser.add_argument('--num_episodes', type=int,
                        default=100,
                        help='Number of episodes to run.')

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

    parser.add_argument('--replay_buffer_size', type=int,
                        default=2028,
                        help='Number of experience tuples in replay buffer.')

    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='Number of experiences in a batch.')

    parser.add_argument('--discount_factor', type=float,
                        default=0.1,
                        help='The discount factor on future rewards, gamma.')

    parser.add_argument('--target_mixing_factor', type=float,
                        default=0.1,
                        help='The update weight for the target networks.')


    parser.add_argument('--render_env', action='store_true',
                        default=False,
                        help='If provided, render the environment.')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)