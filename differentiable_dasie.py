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

import tensorflow as tf



# First, prevent TensorFlow from foisting filthy eager execution upon us.
tf.compat.v1.disable_eager_execution()

def cosine_similarity(u, v):
    """
    :param u: Any np.array matching u in shape (and semantics probably)
    :param v: Any np.array matching u in shape (and semantics probably)
    :return: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    """

    u = tf.reshape(u, [-1])
    v = tf.reshape(v, [-1])
    u = tf.cast(u, tf.float64)
    v = tf.cast(v, tf.float64)

    projection = tf.tensordot(u, v, axes=1)

    norm_product = tf.math.multiply(tf.norm(u), tf.norm(v))

    cos_sim = tf.math.divide(projection, norm_product)

    return cos_sim


def generalized_gaussian(X, mu, alpha, beta):
    """
    :param X: A 1d array onto which the gaussian wil be projected
    :param mu: The mean of the gaussian.
    :param alpha:
    :param beta:
    :return: A vector with shape matching X.
    """

    return (beta / (2 * alpha * math.gamma(1 / beta))) * np.exp(
        -(np.abs(X - mu) / alpha) ** beta)


def plane_2d(x, y, x_0, y_0, slope_x, slope_y, height):
    return ((x - x_0) * slope_x) + ((y - y_0) * slope_y) + height


@np.vectorize
def generalized_gaussian_2d(u, v, mu_u, mu_v, alpha, beta):
    scale_constant = (beta / (2 * alpha * tf.exp(tf.math.lgamma((1 / beta)))))
    # print(scale_constant)
    #
    # scale_constant = (beta / (2 * alpha * math.gamma(1 / beta)))
    # print(scale_constant)
    exponent = -(((u - mu_u) ** 2 + (v - mu_v) ** 2) / alpha) ** beta

    # exponent = tf.math.maximum(exponent, -1e-4)
    # print("##########")
    # print(u)
    # print(v)
    # print(mu_u)
    # print(mu_v)
    # print(alpha)
    # print(beta)
    # print(exponent)
    # print("##########")

    value = tf.exp(exponent)

    z = scale_constant * value

    return z

def tensor_generalized_gaussian_2d(T):
    u, v, mu_u, mu_v, alpha, beta = T
    # beta_real = tf.abs(beta)


    # TODO: This is horrible, but works around tf.math.lgamma not supporting real valued complex datatypes.
    u = tf.cast(u, dtype=tf.float64)
    v = tf.cast(v, dtype=tf.float64)
    mu_u = tf.cast(mu_u, dtype=tf.float64)
    mu_v = tf.cast(mu_v, dtype=tf.float64)
    alpha = tf.cast(alpha, dtype=tf.float64)
    beta = tf.cast(beta, dtype=tf.float64)
    # complex_number = tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64))

    # exponent = tf.cast(tf.math.lgamma((1 / tf.cast(beta, dtype=tf.float64))), dtype=tf.complex64)
    # exponent = tf.math.lgamma((1 / tf.cast(beta, dtype=tf.float64)))
    scale_constant = beta / (2 * alpha * tf.exp(tf.math.lgamma((1 / beta))))
    # print(scale_constant)
    #
    # scale_constant = (beta / (2 * alpha * math.gamma(1 / beta)))
    # print(scale_constant)
    exponent = -(((u - mu_u) ** 2 + (v - mu_v) ** 2) / alpha) ** beta

    # exponent = tf.math.maximum(exponent, -1e-4)
    # print("##########")
    # print(u)
    # print(v)
    # print(mu_u)
    # print(mu_v)
    # print(alpha)
    # print(beta)
    # print(exponent)
    # print("##########")

    value = tf.exp(exponent)

    z = scale_constant * value

    # TODO: This is horrible, but works around tf.math.lgamma not supporting real valued complex datatypes.
    z = tf.cast(z, dtype=tf.complex128)

    return z


@np.vectorize
def circle_mask(X, Y, x_center, y_center, radius):
    r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
    return r < radius


def aperture_function_2d(X, Y, mu_u, mu_v, alpha, beta, tip, tilt, piston):

    print("Starting aperture function.")
    # generalized_gaussian_2d_sample = tf.vectorized_map(generalized_gaussian_2d, X, Y, mu_u, mu_v, alpha, beta)
    T_mu_u = tf.ones_like(X) * mu_u
    T_mu_v = tf.ones_like(X) * mu_v
    T_alpha = tf.ones_like(X) * alpha
    T_beta = tf.ones_like(X) * beta
    T = (X, Y, T_mu_u, T_mu_v, T_alpha, T_beta)
    generalized_gaussian_2d_sample = tf.vectorized_map(tensor_generalized_gaussian_2d, T)

    # generalized_gaussian_2d_sample = generalized_gaussian_2d(X, Y, mu_u, mu_v,
    #                                                          alpha, beta)

    # Subsistitute simple circular mask function
    # Warning!  Ugly alpha fudge factor!
    # subaperture_radius = alpha*4
    # generalized_gaussian_2d_sample = circle_mask(X, Y, mu_u, mu_v, subaperture_radius)

    print("getting plane.")
    plane_2d_sample = plane_2d(X, Y, mu_u, mu_v, tip, tilt, piston)

    # print(g.gradient(plane_2d_sample, ttp_variables))
    # die

    # The piston tip and tilt are encoded as the phase-angle of pupil plane
    print("generating phase angle field.")
    plane_2d_field = tf.exp(plane_2d_sample)
    # plane_2d_field = tf.exp(1.j * plane_2d_sample)

    print("multiplying.")
    generalized_gaussian_2d_sample = generalized_gaussian_2d_sample
    aperture_sample = plane_2d_field * generalized_gaussian_2d_sample

    print(aperture_sample)

    print("Ending aperture function.")
    return aperture_sample

class DASIEModel(object):

    def __init__(self, num_apertures=15):

        ttp_variables = list()
        model_parameters = list()

        # Build the variables this model will need.
        self.build_variables()


        with tf.GradientTape(persistent=True) as self.tape:

            ttp_variables = list()
            model_parameters = list()

            # Use a constant real-valued complex number as a gradient stop to the
            # imaginary part of the t/t/p variables.
            for aperture_num in range(num_apertures):
                # Construct the variables wrt which we differentiate.
                tip_variable_name = str(aperture_num) + "_tip"
                tip_parameter = tf.Variable(0.0,
                                            dtype=tf.float64,
                                            name=tip_variable_name)
                model_parameters.append(tip_parameter)
                tip = tf.complex(tip_parameter,
                                 tf.constant(0.0, dtype=tf.float64))
                # microns / meter (not far off from microradian tilt)
                tilt_variable_name = str(aperture_num) + "_tilt"

                tilt_parameter = tf.Variable(0.0,
                                             dtype=tf.float64,
                                             name=tilt_variable_name)
                model_parameters.append(tilt_parameter)
                tilt = tf.complex(tilt_parameter,
                                  tf.constant(0.0, dtype=tf.float64))
                # microns
                piston_variable_name = str(aperture_num) + "_piston"

                piston_parameter = tf.Variable(0.001,
                                               dtype=tf.float64,
                                               name=piston_variable_name)
                model_parameters.append(piston_parameter)
                piston = tf.complex(piston_parameter,
                                    tf.constant(0.0, dtype=tf.float64))

                ttp_variables.append([tip, tilt, piston])

            # ttp_variables = list()
            #
            # # Use a constant real-valued complex number as a gradient stop to the
            # # imaginary part of the t/t/p variables.
            # for aperture_num in range(num_apertures):
            #     # Construct the variables wrt which we differentiate.
            #     tip_variable_name = str(aperture_num) + "_tip"
            #     tip = tf.complex(tf.Variable(0.0,
            #                                   dtype=tf.float64,
            #                                   name=tip_variable_name),
            #                      tf.constant(0.0, dtype=tf.float64))
            #     # microns / meter (not far off from microradian tilt)
            #     tilt_variable_name = str(aperture_num) + "_tilt"
            #     tilt = tf.complex(tf.Variable(0.0,
            #                                   dtype=tf.float64,
            #                                   name=tilt_variable_name),
            #                       tf.constant(0.0, dtype=tf.float64))
            #     # microns
            #     piston_variable_name = str(aperture_num) + "_piston"
            #     piston = tf.complex(tf.Variable(0.001,
            #                                   dtype=tf.float64,
            #                                   name=piston_variable_name),
            #                         tf.constant(0.0, dtype=tf.float64))
            #
            #     ttp_variables.append([tip, tilt, piston])

            print("####START#####")

            # Read the target image only once.
            perfect_image = plt.imread('sample_image.png')
            perfect_image = perfect_image / np.max(perfect_image)
            perfect_image_spectrum = np.fft.fft2(perfect_image)
            perfect_image_flipped = np.fliplr(np.flipud(perfect_image))

            # Set DASIE simulation parameters.
            spatial_quantization = 256
            beta = 10.0
            #     radius = 0.81
            #     alpha = 0.025

            radius = 1.25  # meters
            # radius = tf.complex(tf.constant(radius, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
            alpha = np.pi * radius / num_apertures / 4  # Completely BS alpha scaling factor again...

            filter_wavelength_micron = 1.0
            #     field_of_view_arcsec = 15.0    # Big FOV makes pupil-function easier to see
            field_of_view_arcsec = 4.0

            # Pupil-plane scaling factor
            # 4.848 microradians / arcsec
            pupil_extent = filter_wavelength_micron * spatial_quantization / (
                        4.848 * field_of_view_arcsec)

            # Establish the simulation mesh grid.
            x = np.linspace(-pupil_extent / 2, pupil_extent / 2, spatial_quantization)
            y = np.linspace(-pupil_extent / 2, pupil_extent / 2, spatial_quantization)
            X, Y = np.meshgrid(x, y)
            X = tf.complex(tf.constant(X), tf.constant(0.0, dtype=tf.float64))
            Y = tf.complex(tf.constant(Y), tf.constant(0.0, dtype=tf.float64))

            # Construct the pupil plan by adding independent aperture functions.
            # pupil_plane = np.zeros((spatial_quantization, spatial_quantization),
            #                        dtype=np.complex)

            pupil_plane = tf.zeros((spatial_quantization, spatial_quantization), dtype=tf.complex128)

            for aperture_num in range(num_apertures):

                print("Building aperture number %d." % aperture_num)

                # Parse out the tip, tilt, and piston variables for this aperture.
                tip, tilt, piston = ttp_variables[aperture_num]

                # Piston (in microns) encoded in phase angle
                pison_phase = 2 * np.pi * piston / filter_wavelength_micron

                # Tip and tilt (micron/meter) ~= (microradians of tilt)
                tip_phase = 2 * np.pi * tip / filter_wavelength_micron
                tilt_phase = 2 * np.pi * tilt / filter_wavelength_micron

                rotation = (aperture_num + 1) / num_apertures
                # complex_type_rotation = tf.complex(tf.constant(rotation, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64))
                mu_u = radius * tf.cos((2 * np.pi) * rotation)
                mu_v = radius * tf.sin((2 * np.pi) * rotation)
                mu_u = tf.cast(mu_u, dtype=tf.complex128)
                mu_v = tf.cast(mu_v, dtype=tf.complex128)
                # mu_u = radius * tf.cos((2 * np.pi) * rotation)
                # mu_v = radius * tf.sin((2 * np.pi) * rotation)

                pupil_plane += aperture_function_2d(X,
                                                    Y,
                                                    mu_u,
                                                    mu_v,
                                                    alpha,
                                                    beta,
                                                    tip_phase,
                                                    tilt_phase,
                                                    pison_phase)


            # pupil_plane = tf.cast(pupil_plane, tf.complex128)

            # Basically the following is the loss function of the pupil plane.
            # The pupil plan here can be thought of an estimator, parameterized by
            # t/t/p values, of the true image.

            # Compute the PSF from the pupil plane.
            # TODO: ask for advice, should I NOT be taking the ABS here?
            # psf = np.abs(np.fft.fft2(pupil_plane))
            # psf = tf.abs(
            #     tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(pupil_plane)))) ** 2
            psf = tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(pupil_plane)))) ** 2

            # Compute the OTF, which is the Fourier transform of the PSF.
            otf = tf.signal.fft2d(tf.cast(psf, tf.complex128))

            # Compute the mtf, which is the real component of the OTF
            # np_mtf = np.absolute(otf)
            mtf = tf.math.abs(otf)
            # mtf = tf.math.abs(otf * tf.complex(tf.constant(1.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64)))

            # Note: tf and np handle muls of complex and reals differently!
            # np: comp = real * comp
            # np_distributed_aperture_image_spectrum = np_mtf * perfect_image_spectrum
            # tf: real = real * comp
            distributed_aperture_image_spectrum = tf.cast(mtf, dtype=tf.complex128) * perfect_image_spectrum

            # print(np_distributed_aperture_image_spectrum)
            # print(distributed_aperture_image_spectrum)
            # distributed_aperture_image_spectrum = np_distributed_aperture_image_spectrum
            # die

            # Original... when distributed_aperture_image_spectrum is np this works, when its tf it doesn't...
            # distributed_aperture_image = tf.abs(tf.signal.fft2d(tf.cast(distributed_aperture_image_spectrum, tf.complex128)))
            # Modified...
            distributed_aperture_image = tf.abs(tf.signal.fft2d(distributed_aperture_image_spectrum))

            self.distributed_aperture_image = distributed_aperture_image
            self.mtf = mtf
            self.otf = otf
            self.pupil_plane = pupil_plane
            self.perfect_image_flipped = perfect_image_flipped
            self.perfect_image = perfect_image
            self.perfect_image_spectrum = perfect_image_spectrum
            self.ttp_variables = ttp_variables
            self.psf = psf
            self.model_parameters = model_parameters
            self.distributed_aperture_image_cosine_similarity = cosine_similarity(distributed_aperture_image, perfect_image_flipped)

    def __call__(self):

        return(self.distributed_aperture_image)


    def get_grads(self):

        grads = self.tape.gradient(self.distributed_aperture_image_cosine_similarity, self.model_parameters)

        return(grads)

    def optimize(self):

        grads = self.tape.gradient(self.distributed_aperture_image_cosine_similarity, self.model_parameters)
        for parameter_num in range(len(self.model_parameters)):
            self.model_parameters[parameter_num].assign_add(grads[parameter_num])

    def plot(self):

        # plt.matshow(pupil_plane)
        # plt.matshow(np.log(psf))
        # plt.matshow(np.log((mtf)))
        # plt.matshow(distributed_aperture_image)

        # Ian's alternative plots
        plt.figure(figsize=[12, 4])
        plt.subplot(141)
        # Plot phase angle
        plt.imshow(np.angle(self.pupil_plane), cmap='twilight_shifted')
        plt.colorbar()
        # Overlay aperture mask
        plt.imshow(np.abs(self.pupil_plane), cmap='Greys', alpha=.2)

        plt.subplot(142)
        # Plot log10(psf)
        plt.imshow(np.log10(self.psf), vmin=-3, cmap='inferno')
        plt.colorbar()

        plt.subplot(143)
        # Plot log10(psf)
        plt.imshow(self.distributed_aperture_image, vmin=-3, cmap='inferno')
        plt.colorbar()

        plt.subplot(144)
        # Plot log10(psf)
        # plt.imshow(np.abs(check_image), vmin=-3, cmap='inferno')
        plt.imshow(self.perfect_image_flipped, vmin=-3, cmap='inferno')
        plt.colorbar()

        run_id = None
        if run_id:

            fig_path = os.path.join(save_dir, run_id + '.png')
            plt.savefig(fig_path)

        else:

            fig_path = os.path.join('./', 'tmp.png')
            plt.savefig(fig_path)

@tf.function
def optimize_dasie_parameters(dasie_model,
                              optimizer):

    with tf.GradientTape(persistent=True) as g:

        distributed_aperture_image = dasie_model()
        perfect_image_flipped = dasie_model.perfect_image_flipped
        distributed_aperture_image_cosine_similarity = cosine_similarity(distributed_aperture_image, perfect_image_flipped)
        print("Cosine similarity: " + str(distributed_aperture_image_cosine_similarity))
        dasie_loss = distributed_aperture_image_cosine_similarity

    # grads = dasie_model.tape.gradient(dasie_model.distributed_aperture_image_cosine_similarity, dasie_model.ttp_variables)
    grads = g.gradient(dasie_model.distributed_aperture_image_cosine_similarity, dasie_model.model_parameters)
    print(grads)
    die

    # Now we put the train step here because TF2 made bad choices...
    if optimizer:

        optimizer.apply_gradients(zip(grads, dasie_model.model_parameters))
        # print(dasie_model.ttp_variables)

        # for aperture_num in range(len(dasie_model.ttp_variables)):
        #     dasie_model.ttp_variables[aperture_num] += grads[aperture_num]

        # print(dasie_model.ttp_variables)
        # die
        # print("Does this print fast or slow?")
    else:

        print("No optimizer provided.")

    return dasie_loss

def main(flags):


    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(".", "logs", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Define the optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    dasie_model = DASIEModel()

    print("Start grads sanity check.")
    # print(dasie_model.get_grads())
    print("End grads sanity check.")

    print("##########################################")
    print(dasie_model.model_parameters)
    print("##########################################")
    dasie_model.optimize()
    # print(dasie_model.get_grads())
    print(dasie_model.model_parameters)
    print("##########################################")
    die

    optimization_steps = 8
    for optimization_step in range(optimization_steps):

        print("optimization step %d." % optimization_step)
        loss_value = optimize_dasie_parameters(dasie_model,
                                               optimizer=optimizer)

        print("loss: %f: " % loss_value)

        # ttp_variables = ttp_variables + grads

    dasie_model.plot()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='provide arguments for training.')

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
                        default=2 ** 14,
                        help='Number of possible actuation states.')

    parser.add_argument('--actuation_scale',
                        type=float,
                        default=0.001,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--initial_temperature',
                        type=float,
                        default=1.0,
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