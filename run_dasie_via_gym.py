"""
This script runs a simulated Distributed Aperture System for Interferometric
Exploitation via the OpenAI gym interface.

Author: Justin Fletcher, Ian Cunnyngham
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
    # Keeps my example commands ismpler this way...
    parser.add_argument('--no_render', type=str,
                    default=False,
                    help='Disable environment render function')
    
    ### Extended object setup ###
    # Must match focal-plane resolution if noise is provided
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
                        default=.25,
                        help='Fried paramater, r0 @ 550nm (maters)')
    
    parser.add_argument('--atmosphere_outer_scale', type=float,
                        default=200,
                        help='Atmosphere outer-scale (maters)')
    
    # !!! Note: Doesn't currentoly work with multi-layer atmospheres, stuck at 10m/s
    parser.add_argument('--atmosphere_velocity', type=float,
                        default=10,
                        help='Atmosphere velocity (maters/second)')
    
    # !!! Breaks render right now, but should work for simulation...
    parser.add_argument('--enable_atmosphere_scintilation', action='store_true',
                        default=False,
                        help='Simulate atmospheric scintilation in multi-layer atmosphere')
    
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

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


