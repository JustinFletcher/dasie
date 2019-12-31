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



class ActorModel(tf.keras.Model):
  def __init__(self, env_action_space):
    super(ActorModel, self).__init__()

    self.env_action_space = env_action_space
    self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.d2 = tf.keras.layers.Dense(env_action_space.size, activation='softmax')

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
model = ActorModel()

# with tf.GradientTape() as tape:
#   logits = model(images)
#   loss_value = loss(logits, labels)
# grads = tape.gradient(loss_value, model.trainable_variables)
# optimizer.apply_gradients(zip(grads, model.trainable_variables))


# class ActorModel(object):
#
#     def __init__(self, env_action_space):
#
#         self.env_action_space = env_action_space
#
#
#     def __call__(self, *args, **kwargs):
#
#         # TODO Return the output node for an input placeholder, given.
#
#         return self.env_action_space.sample()

class CriticModel(object):

    def __init__(self):

        parameters = list()

    def __call__(self, *args, **kwargs):
        # TODO Return the output node for an input placeholder, given.

        return 0

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

    with tf.compat.v1.Session() as sess:

        env = gym.make("Pendulum-v0")

        # TODO Randomly initialize a critic network, Q(s, a)...
        critic_model = CriticModel()

        # TODO ...and randomly initialize an actor network mu(s).
        actor_model = ActorModel(env.action_space)

        # TODO Randomly initialize a critic target network, Q'(s, a)...
        target_critic_model = CriticModel()

        # TODO ...and randomly initialize a target actor network mu'(s).
        target_actor_model = ActorModel(env.action_space)


        #######################################################################
        # Start: Build the critic optimizer.                                  #
        #######################################################################

        action_batch_placeholder = tf.compat.v1.placeholder(tf.float32, name="action")
        reward_batch_placeholder = tf.compat.v1.placeholder(tf.float32, name="reward")
        observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32, name="observation")
        new_observation_batch_placeholder = tf.compat.v1.placeholder(tf.float32, name="new_observation")

        # Compute the target value for the observation, then...
        target_value = reward_batch_placeholder + flags.discount_factor * target_critic_model(new_observation_batch_placeholder,
                                                                                              target_actor_model(new_observation_batch_placeholder))

        # ...get the predicted value from the model...
        q_value = critic_model(observation_batch_placeholder, action_batch_placeholder)

        # ...and compute the MSE. Finally, normalize.
        bellman_loss = tf.reduce_mean((target_value - q_value) ** 2) / flags.batch_size

        # Create an optimizer op for this loss...
        optimize_critic = tf.train.AdamOptimizer().minimize(bellman_loss)

        #######################################################################
        # End: Build the critic optimizer.                                    #
        #######################################################################


        #######################################################################
        # Start: Build the actor optimizer.                                  #
        #######################################################################

        # Now, update the policy model.
        # First, compute the gradient of the critic function wrt action.
        critic_action_gradient = tf.gradients(critic_model(observation_batch_placeholder, actor_model(observation_batch_placeholder)),
                                              actor_model(observation_batch_placeholder))
        # actor_grad = tf.gradients(actor_model(observation_batch_placeholder),
        #                           actor_model.parameters)


        # Magic????
        unnormalized_actor_gradients = tf.gradients(actor_model(observation_batch_placeholder), actor_model.parameters, -critic_action_gradient)
        policy_gradients = list(map(lambda x: tf.div(x, flags.batch_size), unnormalized_actor_gradients))
        optimize_actor = tf.train.AdamOptimizer().apply_gradients(zip(policy_gradients, actor_model.parameters))
        #######################################################################
        # End: Build the actor optimizer.                                    #
        #######################################################################


        # Initialize an empty replay buffer.
        replay_buffer = ReplayBuffer(flags.replay_buffer_size)

        # Iterate over the number of desired episodes.
        for i_episode in range(flags.num_episodes):

            # Reset the environment, and get an initial observation...
            observation = env.reset()

            # TODO ...and establish an exploration noise distribution.

            # Now, iterate over the desired number of steps. In each step...
            for t in range(flags.num_steps):

                if not flags.silence:

                    print("Running step %s." % str(t))

                if flags.render_env:

                    env.render()

                # Query an action from the actor model, then...
                action = actor_model(observation)

                # ...take that action, and parse the state.
                new_observation, reward, terminal, info = env.step(action)

                # If the environment says we're done, stop this episode. Else...
                if terminal:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

                # ...store this state in the replay buffer...
                replay_buffer.add(observation,
                                  action,
                                  reward,
                                  terminal,
                                  new_observation)

                # ...and set the new observation for the next time step.
                observation = new_observation


                # Check if our replay buffer has at least one batch to train.
                if replay_buffer.size() >= flags.batch_size:

                    # Now, sample a batch on which to train the models.
                    (observation_batch,
                     action_batch,
                     reward_batch,
                     terminal_batch,
                     new_observation_batch) = replay_buffer.sample_batch(flags.batch_size)

                    # Construct a feed dict and apply an optimizer step.
                    feed_dict={action_batch_placeholder: action_batch,
                               reward_batch_placeholder: reward_batch,
                               observation_batch_placeholder: observation_batch,
                               new_observation_batch_placeholder: new_observation_batch}

                    sess.run(optimize_critic, feed_dict=feed_dict)
                    sess.run(optimize_actor, feed_dict=feed_dict)


                    # Finally, update the target models.
                    target_actor_model.parameters = (flags.target_mixing_factor * actor_model.parameters) + ((1 - flags.target_mixing_factor) * target_actor_model.parameters)
                    target_critic_model.parameters = (flags.target_mixing_factor * critic_model.parameters) + ((1 - flags.target_mixing_factor) * target_critic_model.parameters)


    # Once all episodes are complete, close the environment.
    env.close()




if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--phase_simulation_resolution', type=int,
                        default=2 ** 11,
                        help='Size of simulated aperture image.')

    parser.add_argument('--max_episode_steps', type=int,
                        default=10000,
                        help='Steps per episode limit.')

    parser.add_argument('--num_episodes', type=int,
                        default=2,
                        help='Number of episodes to run.')

    parser.add_argument('--num_apertures', type=int,
                        default=9,
                        help='Number of apertures to simulate.')

    parser.add_argument('--reward_threshold', type=float,
                        default=25.0,
                        help='Max reward per episode.')

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
                        default=128,
                        help='Number of experiences in a batch.')


    parser.add_argument('--discount_factor', type=float,
                        default=0.1,
                        help='The discount factor on future rewards, gamma.')


    parser.add_argument('--render_env', action='store_true',
                        default=False,
                        help='If provided, render the environment.')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)