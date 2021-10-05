"""
A library for the computation of DASIE MTF ensembles.

Author: Justin Fletcher
"""

import os
import math
import time
import copy
import json
import math
import hcipy
import codecs
import joblib
import datetime
import argparse
import itertools

import pandas as pd
import numpy as np

from decimal import Decimal

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulate_multi_aperture import SimulateMultiApertureTelescope

from multi_aperture_psf import MultiAperturePSFSampler

def cosine_similarity(u, v):
    """

    :param u: Any np.array matching u in shape (and semantics probably)
    :param v: Any np.array matching u in shape (and semantics probably)
    :return: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    """
    u = u.flatten()
    v = v.flatten()

    return (np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v)))

def generalized_gaussian(X, mu, alpha, beta):

    """

    :param X: A 1d array onto which the gaussian wil be projected
    :param mu: The mean of the gaussian.
    :param alpha:
    :param beta:
    :return: A vector with shape matching X.
    """

    return (beta / (2 * alpha * math.gamma(1/beta))) * np.exp(-(np.abs(X - mu) / alpha) ** beta)

@np.vectorize
def plane_2d(x, y, x_0, y_0, slope_x, slope_y, height):
    return ((x - x_0) * slope_x) + ((y - y_0) * slope_y) + height

@np.vectorize
def generalized_gaussian_2d(u, v, mu_u, mu_v, alpha, beta):

    scale_constant = (beta / (2 * alpha * math.gamma(1 / beta)))

    value = np.exp(-(((u - mu_u)**2 + (v - mu_v)**2) / alpha) ** beta)

    z = scale_constant * value

    return z

def aperture_function_2d(X, Y, mu_u, mu_v, alpha, beta, tip, tilt, piston):

    generalized_gaussian_2d_sample = generalized_gaussian_2d(X, Y, mu_u, mu_v, alpha, beta)

    plane_2d_sample = plane_2d(X, Y, mu_u, mu_v, tip, tilt, piston)

    aperture_sample = generalized_gaussian_2d_sample * plane_2d_sample

    return aperture_sample

def main(flags):

    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(".", "logs", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Read the target image only once.
    perfect_image = plt.imread('sample_image.png')
    perfect_image = perfect_image / np.max(perfect_image)
    perfect_image_spectrum = np.fft.fft2(perfect_image)
    perfect_image_flipped = np.fliplr(np.flipud(perfect_image))

    # Set DASIE parameters.
    num_apertures = flags.num_subapertures
    num_apertures = 15

    pupil_plane_meters = 4.5
    telescope_aperture_centroid_diameter_meters = 1.5
    radius_to_centroid_pupil_fraction = telescope_aperture_centroid_diameter_meters / pupil_plane_meters

    # Set simulation parameters
    spatial_quantization = 256
    alpha = 0.0045
    beta = 100.0
    radius = radius_to_centroid_pupil_fraction

    # Establish the simulation mesh grid.
    x = np.linspace(-1.0, 1.0, spatial_quantization)
    y = np.linspace(-1.0, 1.0, spatial_quantization)
    X, Y = np.meshgrid(x, y)

    # Construct the pupil plan by adding independent aperture functions.
    pupil_plane = np.zeros((spatial_quantization, spatial_quantization))

    for aperture_num in range(num_apertures):

        # Set a random t/t/p. In the future, these will be variables initialized to 0.0.
        # tip = np.random.uniform(0.0, 6.0)
        # tilt = np.random.uniform(0.0, 6.0)
        # piston = np.random.uniform(0.0, 6.0)
        tip = 0.0
        tilt = 0.0
        piston = 1.0

        rotation = (aperture_num + 1) / num_apertures
        mu_u = radius * np.cos((2 * np.pi) * rotation)
        mu_v = radius * np.sin((2 * np.pi) * rotation)

        pupil_plane += aperture_function_2d(X, Y, mu_u, mu_v, alpha, beta, tip, tilt, piston)

    pupil_plane = pupil_plane / np.max(pupil_plane)

    # Compute the PSF from the pupil plane.
    # psf = np.abs(np.fft.fft2(pupil_plane))
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_plane)))) ** 2

    # Compute the OTF, which is the Fourier transform of the PSF.
    otf = np.fft.fft2(psf)

    # Compute the MTF, which is the real component of the OTF
    mtf = np.abs(otf)

    # Apply the MTF to the object plane spectrum to get the image plane spectrum.
    distributed_aperture_image_spectrum = perfect_image_spectrum * mtf

    # Compute the image plane image.
    distributed_aperture_image = np.abs((np.fft.fft2(distributed_aperture_image_spectrum)))
    distributed_aperture_image = distributed_aperture_image / np.max(distributed_aperture_image)

    # Measure the cosine similarity between the object plane and image plane images.
    distributed_aperture_image_cosine_similarity = cosine_similarity(distributed_aperture_image, perfect_image_flipped)
    print("Cosine similarity: " + str(distributed_aperture_image_cosine_similarity))


    # If requested, save the plot.
    if flags.save_plot:

        plt.matshow(pupil_plane)
        # plt.matshow(np.log(psf))
        # plt.matshow(np.log((mtf)))
        # plt.matshow(distributed_aperture_image)

        run_id = None
        if run_id:

            fig_path = os.path.join(save_dir, run_id + '.png')
            plt.savefig(fig_path)

        else:

            fig_path = os.path.join('./', 'tmp.png')
            plt.savefig(fig_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for training.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1234,
                        help='A random seed for repeatability.')

    parser.add_argument('--plot_periodicity',
                        type=int,
                        default=100,
                        help='Number of epochs to wait before plotting.')

    parser.add_argument('--num_subapertures',
                        type=int,
                        default=15,
                        help='Number of DASIE subapertures.')

    parser.add_argument('--distributed_aperture_diameter_start',
                        type=float,
                        default=1.0,
                        help='Diameter of the distributed aperture system in meters.')

    parser.add_argument('--filter_psf_extent',
                        type=float,
                        default=2.0,
                        help='Angular extent of simulated PSF (arcsec)')

    parser.add_argument('--monolithic_aperture_diameter_start',
                        type=float,
                        default=1.0,
                        help='Diameter of the monolithic aperture system in meters.')

    parser.add_argument('--distributed_aperture_diameter_stop',
                        type=float,
                        default=30.0,
                        help='Diameter of the distributed aperture system in meters.')

    parser.add_argument('--monolithic_aperture_diameter_stop',
                        type=float,
                        default=30.0,
                        help='Diameter of the monolithic aperture system in meters.')

    parser.add_argument('--aperture_diameter_num',
                        type=int,
                        default=64,
                        help='Number of linspaced aperture values to simulate')

    parser.add_argument('--num_actuation_states',
                        type=int,
                        default=2**14,
                        help='Number of possible actuation states.')

    parser.add_argument('--actuation_scale',
                        type=float,
                        default=0.001,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--initial_temperature',
                        type=float,
                        default= 1.0,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--mtf_scale',
                        type=int,
                        default=256,
                        help='Square array size for MFT.')

    parser.add_argument('--ensemble_size',
                        type=int,
                        default=2,
                        help='Number of samples to take.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=100000000,
                        help='Number annealing steps to run.')

    parser.add_argument("--show_plot", action='store_true',
                        default=False,
                        help="Show the plot?")

    parser.add_argument("--save_plot", action='store_true',
                        default=False,
                        help='Save the plot?')


    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()


    main(parsed_flags)