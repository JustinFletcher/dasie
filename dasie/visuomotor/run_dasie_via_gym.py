"""
This script runs a simulated Distributed Aperture System for Interferometric
Exploitation via the OpenAI gym interface.

Author: Justin Fletcher, Ian Cunnyngham
Date: 20 October 2019
"""

import os
import argparse

import gym

# Function which adds all the arugments for running a multi-aperture simulation to argparser instance
# TODO: this is a horrible worst practice that needs to be removed.
# from env_model.simulate_multi_aperture import add_multi_aperture_telescope_args


# TODO: Move this whole script down a level.

def cli_main(flags):

    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Register our custom DASIE environment.
    gym.envs.registration.register(
        id='Dasie-v0',
        entry_point='env_model.dasie:DasieEnv',
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

            if not flags.silence:

                print("Running step %s." % str(t))

            # ...show the environment to the caller...
            if not flags.no_render:
                env.render()

            # ...get a random action...
            action = env.action_space.sample()

            # ...take that action, and parse the state.
            observation, reward, done, info = env.step(action)

            # TODO: Model goes here. action = f(observation)
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
    
    # Wasn't sure if defaulting to rendering or not made more sense
    # Keeps my example commands simpler  this way...
    parser.add_argument('--no_render', type=str,
                    default=False,
                    help='Disable environment render function')
    
    
    ### Add all the arguments for Multi-Aperture Telescope simulator
    # parser = add_multi_aperture_telescope_args(parser)
    
    
    ### Gym simulation setup ###
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
    
    parser.add_argument('--max_episode_steps', type=int,
                        default=10000,
                        help='Steps per episode limit.')

    parser.add_argument('--num_episodes', type=int,
                        default=1,
                        help='Number of episodes to run.')

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

    parser.add_argument('--silence', action='store_true',
                        default=False,
                        help='If provided, be quiet.')

    parser.add_argument('--dasie_version', type=str,
                        default="test",
                        help='Which version of the DASIE sim do we use?')


    parser.add_argument('--render_frequency',
                        type=int,
                        default=1,
                        help='Render gif this frequently, in steps.')


    parser.add_argument('--filter_psf_resolution', type=int,
                        default=2 ** 8,
                        help='Resolution of simulated PSF (this and extent set pixel scale for extended image convolution)')



    ############################ DASIE FLAGS ##################################
    parser.add_argument('--extended_object_image_file', type=str,
                        default=".\\resources\\sample_image.png",
                        help='Filename of image to convolve PSF with (if none, PSF returned)')

    parser.add_argument('--extended_object_distance', type=str,
                        default=None,
                        help='Distance in meters to the extended object.')

    parser.add_argument('--extended_object_extent', type=str,
                        default=None,
                        help='Extent in meters of the extended object image.')

    parser.add_argument('--observation_window_size',
                        type=int,
                        default=2**1,
                        help='Number of frames input to the model.')

    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


