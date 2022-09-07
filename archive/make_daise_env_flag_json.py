"""


Author: Justin Fletcher
"""

import os
import json
import time
import random
import datetime
import argparse
from datetime import datetime
from collections import deque


from miss_utilities.utils import save_as_json




def main(flags):

    env_dict = dict()


    env_dict["env_kwargs"] = flags.__dict__
    env_dict["env_name"] = 'Dasie-v0'

    print("Saving: \n")
    print(flags.__dict__)
    print("To: %s" % flags.json_file_path)
    save_as_json(flags.json_file_path, env_dict)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for training.')

    ############################ JSON FLAGS ###################################
    parser.add_argument('--json_file_path', type=str,
                        default=".\\dasie.json",
                        help='Path and name of the DASIE JSON.')

    ############################ DASIE KWARGS #################################
    parser.add_argument('--extended_object_image_file', type=str,
                        default=None,
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