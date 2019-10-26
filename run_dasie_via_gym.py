"""
This script runs a simulated Distributed Aperture System for Interferometric
Exploitation via the OpenAI gym interface.

Author: Justin Fletcher
Date: 20 October 2019
"""

import os
import argparse

import gym

# TODO: Move this whole script down a level.

def cli_main(flags):
    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Register our custom DASIE environment.
    gym.envs.registration.register(
        id='Dasie-v0',
        entry_point='dasie_gym_env.dasie:DasieEnv',
        max_episode_steps=flags.max_episode_steps,
        reward_threshold=flags.reward_threshold,
    )

    # Build a gym environment; pass the CLI flags to the constructor as kwargs.
    env = gym.make('Dasie-v0', **vars(flags))

    # Iterate over the number of desired episodes.
    for i_episode in range(flags.num_episodes):

        # Reset the environment...
        observation = env.reset()

        # ..and iterate over the desired number of steps. In each step...
        for t in range(flags.num_steps):

            # ...show the environment to the caller...
            env.render()
            print(observation)

            # ...get a random action...
            action = env.action_space.sample()

            # ...take that action, and parse the state.
            observation, reward, done, info = env.step(action)

            # If the environment says we're done, stop this episode.
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    # Once all episodes are complete, close the environment.
    env.close()

if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--num_episodes', type=int,
                        default=20,
                        help='Number of episodes to run.')

    parser.add_argument('--max_episode_steps', type=int,
                        default=20,
                        help='Steps per episode limit.')

    parser.add_argument('--reward_threshold', type=float,
                        default=25.0,
                        help='Max reward per episode.')

    parser.add_argument('--num_steps', type=int,
                        default=100,
                        help='Number of steps to run.')

    parser.add_argument('--is_recurrent', action='store_true',
                        default=False,
                        help='Should we use a recurrent (Convolutional LSTM) '
                             'variant of the model')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


