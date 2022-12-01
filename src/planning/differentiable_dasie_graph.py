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
import glob
import codecs
import joblib
import datetime
import argparse
import itertools

import pandas as pd
import numpy as np

from decimal import Decimal

# TODO: Refactor this import.
from dataset_generator import DatasetGenerator

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Tentative.
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import tensorflow as tf
# TODO: Implement TF probability.
# import tensorflow_probability as tfp


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

def circle_mask(X, Y, x_center, y_center, radius):
    r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
    return r < radius

def zernike_0(T):

    z = 1.0
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_1(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)

    z = 2 * cartesian_radial * tf.math.sin(cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_2(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = 2 * cartesian_radial* tf.math.cos(cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_3(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(6 * cartesian_radial**2) * tf.math.sin(2 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z


def zernike_4(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(3.0, dtype=tf.float64))  * (cartesian_radial - 1.0)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_5(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(6.0, dtype=tf.float64)) * (cartesian_radial ** 2) * tf.math.cos(2 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_6(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(8.0, dtype=tf.float64)) * (cartesian_radial ** 3.0) * tf.math.sin(3.0 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_7(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(8.0, dtype=tf.float64)) * ((3.0 * (cartesian_radial**3)) - (2.0 * cartesian_radial)) * tf.math.sin(cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z


def zernike_8(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(8.0, dtype=tf.float64)) * ((3.0 * (cartesian_radial**3)) - (2.0 * cartesian_radial)) * tf.math.cos(cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_9(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(8.0, dtype=tf.float64)) * (cartesian_radial ** 3) * tf.math.cos(3.0 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z


def zernike_10(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(10.0, dtype=tf.float64)) * (cartesian_radial ** 4) * tf.math.sin(4.0 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_11(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(10.0, dtype=tf.float64)) * ((4.0 * (cartesian_radial ** 4)) - (3.0 * (cartesian_radial ** 2))) * tf.math.sin(2.0 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_12(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(5.0, dtype=tf.float64)) * ((6.0 * (cartesian_radial ** 4)) - (6.0 * (cartesian_radial ** 2)) + 1.0)
    z = tf.cast(z, dtype=tf.complex128)

    return z

def zernike_13(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(10.0, dtype=tf.float64)) * ((4.0 * (cartesian_radial ** 4)) - (3.0 * (cartesian_radial ** 2))) * tf.math.cos(2.0 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)
    return z

def zernike_14(T):

    # Unpack the input tensor.
    u, v, mu_u, mu_v, aperture_radius, subaperture_radius = T

    u_field = tf.cast(u, dtype=tf.float64) - mu_u
    v_field = tf.cast(v, dtype=tf.float64) - mu_v
    cartesian_radial = tf.math.sqrt(u_field**2 + v_field**2) / subaperture_radius
    cartesian_radial = tf.math.minimum(cartesian_radial, 1.0)
    cartesian_azimuth = tf.math.atan2(v_field, u_field)
    z = tf.math.sqrt(tf.constant(10.0, dtype=tf.float64)) * (cartesian_radial ** 4) * tf.math.cos(4 * cartesian_azimuth)
    z = tf.cast(z, dtype=tf.complex128)

    return z


def select_zernike_function(term_number):

    if term_number == 0:

        function_name = zernike_0

    elif term_number == 1:

        function_name = zernike_1

    elif term_number == 2:

        function_name = zernike_2

    elif term_number == 3:

        function_name = zernike_3

    elif term_number == 4:

        function_name = zernike_4

    elif term_number == 5:

        function_name = zernike_5

    elif term_number == 6:

        function_name = zernike_6

    elif term_number == 7:

        function_name = zernike_7

    elif term_number == 8:

        function_name = zernike_8

    elif term_number == 9:

        function_name = zernike_9

    elif term_number == 10:

        function_name = zernike_10

    elif term_number == 11:

        function_name = zernike_11

    elif term_number == 12:

        function_name = zernike_12

    elif term_number == 13:

        function_name = zernike_13

    elif term_number == 14:

        function_name = zernike_14

    else:
        raise ValueError("You provided a Zernike coefficient for a term (" \
                         + str(term_number) +") that is not supported by this \
                         library. Limit your terms to [0, 14].")

    return function_name

def init_zernike_coefficients(num_zernike_indices=1,
                              zernike_init_type="constant",
                              zernike_debug=False,
                              debug_nonzero_coefficient=None):

    # If a zernike debug is indicated...
    if zernike_debug:

        # ...make just one coefficient non-zero per subap.
        zernike_coefficients = [1.0] * num_zernike_indices

        for i in range(len(zernike_coefficients)):

            if i != debug_nonzero_coefficient:

                zernike_coefficients[i] = 0.0

    # If we're not debugging, then initialize the coefficients as specified.
    else:

        zernike_coefficients = list()

        for _ in range(num_zernike_indices):

            if zernike_init_type == "constant":

                zernike_coefficient = 1.0

            elif zernike_init_type == "np.random.uniform":

                zernike_coefficient = np.random.uniform(0.0, 1.0)

            else:

                raise ValueError("You provided --zernike_init_type of %s, but \
                                 only 'constant' and 'np.random.uniform' are \
                                 supported")

            zernike_coefficients.append(zernike_coefficient)

    return zernike_coefficients


def zernike_aperture_function_2d(X,
                                 Y,
                                 mu_u,
                                 mu_v,
                                 aperture_radius,
                                 subaperture_radius,
                                 zernike_coefficients):

    print("-Starting Zernike aperture function.")
    tensor_zernike_2d_sample = None
    tensor_zernike_2d_sample_initialized  = False

    # Build tensors upon which to perform a vectorized map.
    T_mu_u = tf.ones_like(X) * mu_u
    T_mu_v = tf.ones_like(X) * mu_v
    T_aperture_radius = tf.ones_like(X) * aperture_radius
    T_subaperture_radius = tf.ones_like(X) * subaperture_radius
    T_X = tf.complex(tf.constant(X), tf.constant(0.0, dtype=tf.float64))
    T_Y = tf.complex(tf.constant(Y), tf.constant(0.0, dtype=tf.float64))
    T = (T_X, T_Y, T_mu_u, T_mu_v, T_aperture_radius, T_subaperture_radius)

    # Iterate over each zernike term-coefficient pair, adding its contribution.
    for term_number, zernike_coefficient in enumerate(zernike_coefficients):

        print("--Building Zernike Term " + str(term_number) + ".")

        # Select the function corresponding to this term number.
        zernike_term_function = select_zernike_function(term_number=term_number)

        # Either initialize the aperture disk, or add this contribution to it.
        if tensor_zernike_2d_sample_initialized:
            tensor_zernike_2d_sample += zernike_coefficient * tf.vectorized_map(zernike_term_function, T)
        else:
            tensor_zernike_2d_sample = zernike_coefficient * tf.vectorized_map(zernike_term_function, T)
            tensor_zernike_2d_sample_initialized = True

        print("--End of Zernike Term.")

    # Apply a circle mask to set all non-aperture pixels to 0.0.
    print("-Masking subaperture.")
    pupil_mask = circle_mask(X, Y, mu_u, mu_v, subaperture_radius)
    pupil_mask = tf.cast(tf.constant(pupil_mask), dtype=tf.complex128)
    tensor_masked_zernike_2d_sample = tensor_zernike_2d_sample * pupil_mask

    # The piston tip and tilt are encoded as the phase-angle of pupil plane
    print("-Generating phase angle field.")
    # TODO: Reinstate after debug? Talk to Ryan: Why should I do this?
    tensor_zernike_2d_field = tensor_masked_zernike_2d_sample
    # tensor_zernike_2d_field = tf.exp(tensor_masked_zernike_2d_sample)
    # tensor_zernike_2d_field = tensor_zernike_2d_field * pupil_mask

    print("-Ending aperture function.")
    return tensor_zernike_2d_field

class DASIEModel(object):

    def __init__(self,
                 sess,
                 train_dataset,
                 valid_dataset,
                 batch_size,
                 loss_name="mse",
                 learning_rate=1.0,
                 num_apertures=15,
                 spatial_quantization=256,
                 image_x_scale=256,
                 image_y_scale=256,
                 supaperture_radius_meters=None,
                 num_exposures=1,
                 recovery_model_filter_scale=16,
                 diameter_meters=2.5,
                 num_zernike_indices=1,
                 zernike_debug=False,
                 hadamard_image_formation=True,
                 writer=None):

        self.learning_rate = learning_rate
        self.num_apertures = num_apertures
        self.batch_size = batch_size
        self.spatial_quantization = spatial_quantization
        self.image_x_scale = image_x_scale
        self.image_y_scale = image_y_scale
        self.sess = sess
        self.writer = writer
        self.loss_name = loss_name
        self.num_exposures = num_exposures


        train_iterator = train_dataset.get_iterator()
        self.train_iterator_handle = sess.run(train_iterator.string_handle())

        valid_iterator = valid_dataset.get_iterator()
        self.valid_iterator_handle = sess.run(valid_iterator.string_handle())

        self.handle = tf.compat.v1.placeholder(tf.string, shape=[])

        # Abstract specific iterators as only their types.
        iterator_output_types = train_iterator.output_types
        iterator = tf.compat.v1.data.Iterator.from_string_handle(self.handle,
                                                                 iterator_output_types)
        dataset_batch = iterator.get_next()

        self.dataset_batch = dataset_batch

        with tf.name_scope("dasie_model"):

            # TODO: Split model to create intermediary placeholder.
            self.inputs, self.output_images = self._build_dasie_model(
                inputs=dataset_batch,
                spatial_quantization=spatial_quantization,
                num_apertures=self.num_apertures,
                radius_meters=diameter_meters / 2,
                supaperture_radius_meters=supaperture_radius_meters,
                num_exposures=num_exposures,
                recovery_model_filter_scale=recovery_model_filter_scale,
                num_zernike_indices=num_zernike_indices,
                zernike_debug=zernike_debug,
                hadamard_image_formation=hadamard_image_formation,
                )

        with tf.name_scope("dasie_recovery_inference"):
            # TODO: make inference not random by loading weights.
            self.inference_batch = tf.compat.v1.placeholder(tf.float64, shape=[1, num_exposures, spatial_quantization, spatial_quantization])
            self.recovered_image = self._build_recovery_model(self.inference_batch,
                                                              filter_scale=recovery_model_filter_scale)

            # Compute the Loss.

            # Compute the metrics.

            # Build the optimization operation.

    def _image_model(self,
                     hadamard_image_formation=False,
                     mtf=None,
                     psf=None):

        with tf.name_scope("image_plane_model"):

            if hadamard_image_formation:
                complex_mtf = tf.cast(mtf, dtype=tf.complex128)
                image_spectrum = self.perfect_image_spectrum * complex_mtf
                image = tf.abs(tf.signal.fft2d(image_spectrum))
            else:
                image = tf.nn.conv2d(
                    # tf.squeeze(self.perfect_image, axis=-1),
                    self.perfect_image,
                    tf.expand_dims(tf.expand_dims(psf, -1), -1),
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    data_format='NHWC'
                )
                image = tf.squeeze(image, axis=-1)

            image = image / tf.reduce_max(image)

        return image

    def _build_geometric_optics(self, pupil_plane):

        # Compute the PSF from the pupil plane.
        with tf.name_scope("psf_model"):
            shifted_pupil_plane = tf.signal.ifftshift(pupil_plane)
            shifted_pupil_spectrum = tf.signal.fft2d(shifted_pupil_plane)
            psf = tf.abs(tf.signal.fftshift(shifted_pupil_spectrum)) ** 2

        # Compute the OTF, which is the Fourier transform of the PSF.
        with tf.name_scope("otf_model"):

            otf = tf.signal.fft2d(tf.cast(psf, tf.complex128))

        # Compute the mtf, which is the real component of the OTF.
        with tf.name_scope("mtf_model"):

            mtf = tf.math.abs(otf)

        return (psf, otf, mtf)

    def _build_zernike_coefficient_variables(self,
                                             zernike_coefficients,
                                             trainable=True):

        # Make TF Variables for each subaperture Zernike coeff.
        zernike_coefficients_variables = list()

        for (i, zernike_coefficient) in enumerate(zernike_coefficients):

            # Construct the tf.Variable for this coefficient.
            variable_name = "zernike_coefficient_" + str(i)

            # Build a TF Variable around this tensor.
            real_var = tf.Variable(zernike_coefficient,
                                   dtype=tf.float64,
                                   name=variable_name,
                                   trainable=trainable)

            # Construct a complex tensor from real Variable.
            imag_constant = tf.constant(0.0, dtype=tf.float64)
            variable = tf.complex(real_var, imag_constant)

            # Append the final variable to the list.
            zernike_coefficients_variables.append(variable)

        return(zernike_coefficients_variables)

    def _apply_noise(self,
                     image,
                     gaussian_mean=1e-5,
                     poisson_mean_arrival=4e-5):

        # TODO: Implement Gaussian and Poisson process noise.
        # Apply the reparameterization trick kingma2014autovariational
        gaussian_dist = tfp.distributions.Normal(loc=tf.zeros_like(image),
                                                 scale=tf.ones_like(image))

        gaussian_sample = tfp.distributions.Sample(gaussian_sample)
        gaussian_noise = image + (gaussian_mean ** 2) * gaussian_sample

        # Apply the score-gradient trick williams1992simple

        rate = image / poisson_mean_arrival
        p = tfp.distributions.Poisson(rate=rate, validate_args=True)
        sampled = tfp.monte_carlo.expectation(f=lambda z: z,
                                              samples=p.sample(1),
                                              log_prob=p.log_prob,
                                              use_reparameterization=False)
        poisson_noise = sampled * poisson_mean_arrival

        noisy_image = gaussian_noise + poisson_noise

        return noisy_image

    def _build_dasie_model(self,
                           inputs=None,
                           num_apertures=15,
                           num_exposures=1,
                           radius_meters=1.25,
                           supaperture_radius_meters=None,
                           field_of_view_arcsec=4.0,
                           spatial_quantization=256,
                           filter_wavelength_micron=1.0,
                           recovery_model_filter_scale=16,
                           num_zernike_indices=1,
                           zernike_debug=False,
                           hadamard_image_formation=True):

        # TODO: Externalize
        zernike_init_type = "np.random.uniform"

        # TODO: Externalize.
        lock_dm_values = False
        if lock_dm_values:
            dm_trainable = False
        else:
            dm_trainable = True

        # Build object plane image batch tensor objects.
        # TODO: Refactor "perfect_image" to "object_batch" everywhere.
        batch_shape = (self.image_x_scale, self.image_y_scale)
        if inputs is not None:
            self.perfect_image = inputs
        else:
            self.perfect_image = tf.compat.v1.placeholder(tf.float64,
                                                          shape=batch_shape,
                                                          name="object_batch")

        with tf.name_scope("image_spectrum_model"):

            self.perfect_image_spectrum = tf.signal.fft2d(
                tf.cast(tf.squeeze(self.perfect_image, axis=-1),
                        dtype=tf.complex128))

        # TODO: Modularize physics stuff.
        # Start: physics stuff.
        # Compute the pupil extent: 4.848 microradians / arcsec
        # pupil_extend = [m] * [count] / ([microradians / arcsec] * [arcsec])
        # pupil_extent = [count] [micrometers] / [microradian]
        # pupil_extent = [micrometers] / [microradian]
        # pupil_extent = [micrometers] / [microradian]
        # pupil_extent = [meters] / [radian]
        # TODO: Ask Ryan for help: What are these units?
        pupil_extent = filter_wavelength_micron * spatial_quantization / (4.848 * field_of_view_arcsec)
        self.pupil_extent = pupil_extent
        print("pupil_extent=" + str(pupil_extent))
        self.phase_scale = 2 * np.pi / filter_wavelength_micron
        # Compute the subaperture pixel extent.
        self.pupil_meters_per_pixel = radius_meters / spatial_quantization
        self.subaperture_size_pixels = int(supaperture_radius_meters // self.pupil_meters_per_pixel)


        # Build the simulation mesh grid.
        # TODO: Verify these physical coordinates; clarify pupil vs radius.
        x = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        y = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        self.pupil_dimension_x = x
        self.pupil_dimension_y = y
        X, Y = np.meshgrid(x, y)
        # End: Physics stuff.

        self.psfs = list()
        self.otfs = list()
        self.mtfs = list()
        self.distributed_aperture_images = list()

        # For each exposure, build the pupil function for that exposure.
        self.pupil_planes = list()
        for exposure_num in range(num_exposures):
            with tf.name_scope("exposure_" + str(exposure_num)):

                # Build the pupil plane quantization grid for this exposure.
                pupil_plane = tf.zeros((spatial_quantization,
                                        spatial_quantization),
                                       dtype=tf.complex128)

                # Build the model of the pupil plane, using the Variables.
                with tf.name_scope("pupil_plane_model"):

                    for aperture_num in range(num_apertures):

                        print("Building aperture number %d." % aperture_num)

                        # Compute the subap centroid cartesian coordinates.
                        # TODO: correct radius to place the edge, rather than the center, at radius
                        rotation = (aperture_num + 1) / self.num_apertures
                        mu_u = radius_meters * np.cos((2 * np.pi) * rotation)
                        mu_v = radius_meters * np.sin((2 * np.pi) * rotation)

                        # Build the variables for this subaperture.
                        with tf.name_scope("subaperture_"+ str(aperture_num)):

                            # Initialize the coefficients for this subaperture.
                            subap_zernike_coeffs = init_zernike_coefficients(
                                num_zernike_indices=num_zernike_indices,
                                zernike_init_type=zernike_init_type,
                                zernike_debug=zernike_debug,
                                debug_nonzero_coefficient=aperture_num
                            )

                            # Build TF Variables around the coefficients.
                            subap_zernike_coefficients_vars = self._build_zernike_coefficient_variables(subap_zernike_coeffs,
                                                                                                        trainable=dm_trainable)
                            # Render this subaperture on the pupil plane grid.
                            # TODO: Ryan: Here's where I can set the phase scale to physical units. Should I?
                            # pupil_plane += self.phase_scale * zernike_aperture_function_2d(X,
                            #                                                                Y,
                            #                                                                mu_u,
                            #                                                                mu_v,
                            #                                                                radius_meters,
                            #                                                                supaperture_radius_meters,
                            #                                                                subap_zernike_coefficients_variables,
                            #                                                                )


                            pupil_plane += zernike_aperture_function_2d(X,
                                                                        Y,
                                                                        mu_u,
                                                                        mu_v,
                                                                        radius_meters,
                                                                        supaperture_radius_meters,
                                                                        subap_zernike_coefficients_vars,
                                                                        )

                # This pupil plane is complete, now add it to the list.
                self.pupil_planes.append(pupil_plane)


                psf, otf, mtf = self._build_geometric_optics(pupil_plane)

                # Store the psf, otf, and mtf tensors for later evaluation.
                self.psfs.append(psf)
                self.otfs.append(otf)
                self.mtfs.append(mtf)

                distributed_aperture_image_plane = self._image_model(
                    hadamard_image_formation=hadamard_image_formation,
                    psf=psf,
                    mtf=mtf)

                with tf.name_scope("sensor_model"):

                    # # TODO: Implement Gaussian and Poisson process noise.
                    # distributed_aperture_image = self._apply_noise(distributed_aperture_image_plane)
                    distributed_aperture_image = distributed_aperture_image_plane


                # Finally, add the image from this pupil to the list.
                self.distributed_aperture_images.append(distributed_aperture_image)

        # Now, construct a monolithic aperture of the same radius.
        with tf.name_scope("monolithic_aperture"):

            with tf.name_scope("pupil_plane"):

                self.monolithic_pupil_plane = zernike_aperture_function_2d(X,
                                                                           Y,
                                                                           0.0,
                                                                           0.0,
                                                                           radius_meters,
                                                                           radius_meters,
                                                                           zernike_coefficients=[0.001],
                                                                           )

            # Compute the PSF from the pupil plane.
            with tf.name_scope("psf_model"):
                self.monolithic_psf = tf.math.abs(tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(self.monolithic_pupil_plane)))) ** 2

            # Compute the OTF, which is the Fourier transform of the PSF.
            with tf.name_scope("otf_model"):
                self.monolithic_otf = tf.signal.fft2d(tf.cast(self.monolithic_psf, tf.complex128))

            # Compute the mtf, which is the real component of the OTF.
            with tf.name_scope("mtf_model"):
                self.monolithic_mtf = tf.math.abs(self.monolithic_otf)


            self.monolithic_aperture_image = self._image_model(hadamard_image_formation=hadamard_image_formation,
                                                               psf=self.monolithic_psf,
                                                               mtf=self.monolithic_mtf)

        with tf.name_scope("distributed_aperture_image_recovery"):

            # Combine the ensemble of images with the restoration function.
            self.recovered_image = self._build_recovery_model(self.distributed_aperture_images,
                                                              filter_scale=recovery_model_filter_scale)

        self.perfect_image_flipped = tf.reverse(tf.reverse(tf.squeeze(self.perfect_image, axis=-1), [-1]), [1])
        self.image_mse = tf.reduce_mean((self.recovered_image - self.perfect_image_flipped) ** 2)

        with tf.name_scope("dasie_loss"):

            if self.loss_name == "mse":
                loss = self.image_mse
            if self.loss_name == "mae":
                loss = tf.reduce_mean(tf.math.abs(self.recovered_image - self.perfect_image_flipped))
            if self.loss_name == "l2":
                loss = tf.math.sqrt(tf.math.reduce_sum((self.recovered_image - self.perfect_image_flipped) ** 2))
            if self.loss_name == "cos":
                loss = -cosine_similarity(self.recovered_image,
                                          self.perfect_image_flipped)

            self.loss = loss

        with tf.name_scope("dasie_metrics"):

            self.monolithic_aperture_image_mse = tf.reduce_mean((self.monolithic_aperture_image - self.perfect_image_flipped) ** 2)
            self.distributed_aperture_image_mse = tf.reduce_mean((self.recovered_image - self.perfect_image_flipped)**2)
            self.da_mse_mono_mse_ratio = self.distributed_aperture_image_mse / self.monolithic_aperture_image_mse
            # TODO: Implement SSIM
            # TODO: Implement PSNR

            # Add some instrumentation for ttp.
            # tips = list()
            # tilts = list()
            # pistons = list()
            # for (tip, tilt, piston) in phase_variables:
            #     tips.append(tip)
            #     tilts.append(tilt)
            #     pistons.append(piston)
            # with self.writer.as_default():
            #     tf.summary.histogram("tip", tips)
            #     tf.summary.histogram("tilt", tilts)
            #     tf.summary.histogram("piston", pistons)


        # I wonder if this works...
        with self.writer.as_default():

            # TODO: refactor all these endpoints to name *_batch.
            tf.summary.scalar("in_graph_loss", self.loss)
            tf.summary.scalar("monolithic_aperture_image_mse", self.monolithic_aperture_image_mse)
            tf.summary.scalar("distributed_aperture_image_mse", self.distributed_aperture_image_mse)
            tf.summary.scalar("da_mse_mono_mse_ratio", self.da_mse_mono_mse_ratio)
            # tf.compat.v1.summary.scalar("v1_test", self.loss)
        with tf.name_scope("dasie_optimizer"):
            # Build an op that applies the policy gradients to the model.
            self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


        output_batch = self.recovered_image

        with tf.compat.v1.Graph().as_default():
            tf.summary.scalar("debug_metric", 0.5)

        # self.summaries = tf.compat.v1.summary.all_v2_summary_ops()
        # self.v1_summaries = tf.compat.v1.summary.merge_all()

        return self.perfect_image, output_batch


    def _build_recovery_model(self,
                              distributed_aperture_images_batch,
                              filter_scale):
        """

        :param distributed_aperture_images_batch: a batch of DASIE images.
        :param filter_scale: the smallest filter scale to use.
        :return:
        """
        with tf.name_scope("recovery_model"):

            # Stack the the images in the ensemble to form a batch of inputs.
            distributed_aperture_images_batch = tf.stack(distributed_aperture_images_batch, axis=-1)

            with tf.name_scope("recovery_feature_extractor"):
                input = distributed_aperture_images_batch
                # down_l0 conv-c15-k7-s1-LRelu input
                down_l0 = self.conv_block(input,
                                    input_channels=self.num_exposures,
                                    output_channels=filter_scale,
                                    kernel_size=7,
                                    stride=1,
                                    activation="LRelu")
                #

                # down_l0 conv-c15-k7-s1-LRelu down_l0
                down_l0 = self.conv_block(down_l0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=7,
                                    stride=1,
                                    activation="LRelu")
                #

                # down_l1 conv-c30-k5-s2-LRelu down_l0
                down_l1_0 = self.conv_block(down_l0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale * 2,
                                    kernel_size=5,
                                    stride=2,
                                    activation="LRelu")
                #

                # down_l1 conv-c30-k3-s1-LRelu down_l1
                down_l1 = self.conv_block(down_l1_0,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # down_l1 conv-c30-k3-s1-LRelu down_l1
                down_l1 = self.conv_block(down_l1,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # down_l2 conv-c60-k5-s2-LRelu down_l1
                down_l2 = self.conv_block(down_l1_0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale * 4,
                                    kernel_size=5,
                                    stride=2,
                                    activation="LRelu")
                #

                # down_l2 conv-c60-k3-s1-LRelu down_l2
                down_l2 = self.conv_block(down_l2,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #


                # down_l2 conv-c60-k3-s1-LRelu down_l2
                down_l2 = self.conv_block(down_l2,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #
                # End of downsample and pre-skip.


                # conv_l2_k0 conv-c60-k3-s1-LRelu down_l2
                conv_l2_k0 = self.conv_block(down_l2,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k1 conv-c60-k3-s1-LRelu conv_l2_k0
                conv_l2_k1 = self.conv_block(conv_l2_k0,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k2 conv-c60-k3-s1-LRelu Concat([conv_l2_k0, conv_l2_k1])
                conv_l2_k2 = self.conv_block(tf.concat([conv_l2_k0, conv_l2_k1], axis=-1),
                                    input_channels=filter_scale * 8,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k3 conv-c60-k3-s1-LRelu conv_l2_k2
                conv_l2_k3 = self.conv_block(conv_l2_k2,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k4 conv-c60-k3-s1-LRelu conv_l2_k3
                conv_l2_k4 = self.conv_block(conv_l2_k3,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #


                # conv_l2_k5 conv-c60-k3-s1-LRelu conv_l2_k4
                conv_l2_k5 = self.conv_block(conv_l2_k4,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #
                # End of bottom pipe.


                # Mid-resolution.
                # conv_l1_k0 conv-c30-k3-s1-LRelu down_l1
                conv_l1_k0 = self.conv_block(down_l1,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l1_k1 conv-c30-k3-s1-LRelu conv_l1_k0
                conv_l1_k1 = self.conv_block(conv_l1_k0,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #
                # conv_l1_k2 conv-c30-k3-s1-LRelu Concat([conv_l1_k0, conv_l1_k1])
                conv_l1_k2 = self.conv_block(tf.concat([conv_l1_k0, conv_l1_k1], axis=-1),
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l1_k3 conv-c30-k3-s1-LRelu conv_l1_k2
                conv_l1_k3 = self.conv_block(conv_l1_k2,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #


                # conv_l1_k4 conv-c30-k3-s1-LRelu conv_l1_k3
                conv_l1_k4 = self.conv_block(conv_l1_k3,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l1_k5 conv-c30-k3-s1-LRelu conv_l1_k4
                conv_l1_k5 = self.conv_block(conv_l1_k4,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # up_l2 convT-c30-k2-s2-LRelu conv_l2_k5
                # modded to: up_l2 convT-c60-k2-s2-LRelu conv_l2_k5
                # pull the bottom pipe up.
                up_l2 = self.convT_block(conv_l2_k5,
                                    input_downsample_factor=4,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=2,
                                    stride=2,
                                    activation="LRelu")
                #

                # conv_l1_k6 conv-c30-k3-s1-LRelu Concat([up_l2, conv_l1_k5])
                # modded to: input 60
                conv_l1_k6 = self.conv_block(tf.concat([up_l2, conv_l1_k5], axis=-1),
                                    input_channels=filter_scale * 6,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l1_k7 conv-c30-k3-s1-LRelu conv_l1_k6
                conv_l1_k7 = self.conv_block(conv_l1_k6,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #
                # End of mid-resolution pipe.

                # High Resolution.
                # conv_l0_k0 conv-c15-k3-s1-LRelu down_l0
                conv_l0_k0 = self.conv_block(down_l0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #
                # conv_l0_k1 conv-c15-k3-s1-LRelu conv_l0_k0
                conv_l0_k1 = self.conv_block(conv_l0_k0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l0_k2 conv-c15-k3-s1-LRelu Concat([conv_l1_k0, conv_l0_k1])c
                # halv the input size = conv_l1_k0
                # This is wrong in tseng2021neural! The sizes don't match!
                # Modded to: conv_l0_k2 conv-c15-k3-s1-LRelu Concat([conv_l0_k0, conv_l0_k1])
                # Then I moved the whole concat down to be consistent with fig 5
                conv_l0_k2 = self.conv_block(conv_l0_k1,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l0_k3 conv-c15-k3-s1-LRelu conv_l0_k2
                conv_l0_k3 = self.conv_block(conv_l0_k2,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l0_k4 conv-c15-k3-s1-LRelu conv_l0_k3
                # Move the skip connection here to be consistent with Fig 5.
                conv_l0_k4 = self.conv_block(tf.concat([conv_l0_k0, conv_l0_k3], axis=-1),
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #


                # conv_l0_k5 conv-c15-k3-s1-LRelu conv_l0_k4
                conv_l0_k5 = self.conv_block(conv_l0_k4,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # up_l1 convT-c15-k2-s2-LRelu conv_l1_k5
                # modded to: up_l1 convT-c30-k2-s2-LRelu conv_l1_k5
                up_l1 = self.convT_block(conv_l1_k5,
                                    input_downsample_factor=2,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=2,
                                    stride=2,
                                    activation="LRelu")
                #


                # conv_l0_k6 conv-c15-k3-s1-LRelu Concat([up_l1, conv_l0_k5])
                conv_l0_k6 = self.conv_block(tf.concat([up_l1, conv_l0_k5], axis=-1),
                                    input_channels=filter_scale * 3,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l0_k7 conv-c15-k3-s1-LRelu conv_l0_k6
                conv_l0_k7 = self.conv_block(conv_l0_k6,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")

            with tf.name_scope("recovery_feature_propagator"):

                # fp_l0 Feature Propagator (PSF_1x, conv_l0_k7)
                # TODO: Implement feature propagator.
                fp_l0 = conv_l0_k7
                # fp_l1 Feature Propagator (PSF_2x, conv_l1_k7)
                # TODO: Implement feature propagator.
                fp_l1 = conv_l1_k7
                # fp_l2 Feature Propagator (PSF_4x, conv_l2_k5)
                # TODO: Implement feature propagator.
                fp_l2 = conv_l2_k5

            with tf.name_scope("recovery_decoder"):
                # conv_l0_k0 conv-c30-k5-s1-LRelu fp_l0
                conv_l0_k0 = self.conv_block(fp_l0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=5,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l0_k1 conv-c30-k5-s1-LRelu conv_l0_k0
                conv_l0_k1 = self.conv_block(conv_l0_k0,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=5,
                                    stride=1,
                                    activation="LRelu")
                #
                # down_l0 conv-c30-k5-s2-LRelu conv_l0_k1
                down_l0 = self.conv_block(conv_l0_k1,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale * 2,
                                    kernel_size=5,
                                    stride=2,
                                    activation="LRelu")
                #

                # conv_l1_k0 conv-c60-k3-s1-LRelu Concat([fp_l1, down_l0])
                conv_l1_k0 = self.conv_block(tf.concat([fp_l1, down_l0], axis=-1),
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #
                # conv_l1_k1 conv-c60-k3-s1-LRelu conv_l1_k0
                conv_l1_k1 = self.conv_block(conv_l1_k0,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # down_l1 conv-c60-k3-s2-LRelu conv_l1_k1
                down_l1 = self.conv_block(conv_l1_k1,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=2,
                                    activation="LRelu")
                #

                # conv_l2_k0 conv-c120-k3-s1-LRelu Concat([fp_l2, down_l1])
                # Modded to 60.
                conv_l2_k0 = self.conv_block(tf.concat([fp_l2, down_l1], axis=-1),
                                    input_channels=filter_scale * 8,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k1 conv-c120-k3-s1-LRelu conv_l2_k0
                # Report error - this is never used, even in the paper.
                conv_l2_k1 = self.conv_block(conv_l2_k0,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k2 conv-c120-k3-s1-LRelu Concat([conv_l2_k0, fp_l2, down_l1])
                # NOTE: This was wrong in the tseng2021neural! conv_l2_k0 -> conv_l2_k1
                conv_l2_k2 = self.conv_block(tf.concat([conv_l2_k1, fp_l2, down_l1], axis=-1),
                                             # change to 10 if breaks
                                    input_channels=filter_scale * 12,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l2_k3 conv-c120-k3-s1-LRelu conv_l2_k2
                # modded to: conv_l2_k3 conv-c60-k3-s1-LRelu conv_l2_k2
                conv_l2_k3 = self.conv_block(conv_l2_k2,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # up_l2 convT-c60-k2-s2-LRelu conv_l2_k3
                # modded to 60 input
                up_l2 = self.convT_block(conv_l2_k3,
                                    input_downsample_factor=4,
                                    input_channels=filter_scale * 4,
                                    output_channels=filter_scale * 4,
                                    kernel_size=2,
                                    stride=2,
                                    activation="LRelu")
                #

                # conv_l1_k2 conv-c60-k3-s1-LRelu Concat([conv_l1_k1, up_l2])
                # modded to 30.
                conv_l1_k2 = self.conv_block(tf.concat([conv_l1_k1, up_l2], axis=-1),
                                    input_channels=filter_scale * 8,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # conv_l1_k3 conv-c60-k3-s1-LRelu conv_l1_k2
                conv_l1_k3 = self.conv_block(conv_l1_k2,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=3,
                                    stride=1,
                                    activation="LRelu")
                #

                # up_l1 convT-c60-k2-s2-LRelu conv_l2_k3
                # NOTE: This was wrong in the tseng2021neural! conv_l2_k3 -> conv_l1_k3
                up_l1 = self.convT_block(conv_l1_k3,
                                    input_downsample_factor=2,
                                    input_channels=filter_scale * 2,
                                    output_channels=filter_scale * 2,
                                    kernel_size=2,
                                    stride=2,
                                    activation="LRelu")
                #

                # conv_l0_k2 conv-c30-k5-s1-LRelu Concat([conv_l0_k1, up_l1])
                conv_l0_k2 = self.conv_block(tf.concat([conv_l0_k1, up_l1], axis=-1),
                                    input_channels=filter_scale * 3,
                                    output_channels=filter_scale,
                                    kernel_size=5,
                                    stride=1,
                                    activation="LRelu")
                #

                # Output RGB conv_l0_k2
                # NOTE: This was underspecified in tseng2021neural!
                conv_l0_k3 = self.conv_block(conv_l0_k2,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=2,
                                    stride=1,
                                    activation="LRelu")
                #
                conv_l0_k4 = self.conv_block(conv_l0_k3,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=2,
                                    stride=1,
                                    activation="LRelu")
                conv_l0_k5 = self.conv_block(conv_l0_k4,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=2,
                                    stride=1,
                                    activation="LRelu")
                conv_l0_k6 = self.conv_block(conv_l0_k5,
                                    input_channels=filter_scale,
                                    output_channels=filter_scale,
                                    kernel_size=2,
                                    stride=1,
                                    activation="LRelu")
                conv_l0_k7 = self.conv_block(conv_l0_k6,
                                    input_channels=filter_scale,
                                    output_channels=1,
                                    kernel_size=2,
                                    stride=1,
                                    activation="LRelu")
                conv_l0_k0 = self.conv_block(conv_l0_k7,
                                    input_channels=1,
                                    output_channels=1,
                                    kernel_size=2,
                                    stride=1,
                                    activation="LRelu")

            # Remove the now irrelevant channel dim.
            recovered_image_batch = tf.squeeze(conv_l0_k0)

        return recovered_image_batch

    def convT_block(self,
                   input_feature_map,
                   input_downsample_factor,
                   input_channels,
                   output_channels=1,
                   kernel_size=2,
                   stride=1,
                   activation="LRelu",
                   name=None):

        if not name:

            name = "convT-c" + str(output_channels) + "-k" + str(kernel_size) + "-s" + str(stride) + "-" + activation

        with tf.name_scope(name):

            # Initialize the filter variables as he2015delving.
            he_relu_init_std = np.sqrt(2 / (input_channels * (kernel_size**2)))
            filters = tf.Variable(tf.random.normal((kernel_size,
                                                    kernel_size,
                                                    input_channels,
                                                    output_channels),
                                                   stddev=he_relu_init_std,
                                                   dtype=tf.float64))

            # Encode the strides for TensorFlow, and build the conv graph.
            strides = [1, stride, stride, 1]

            # Given the base quantization, div by downsample, mul by stride.
            print(self.image_x_scale)
            output_x_scale = (self.image_x_scale // input_downsample_factor) * stride
            output_y_scale = (self.image_y_scale // input_downsample_factor) * stride
            output_shape = (self.batch_size,
                            output_x_scale,
                            output_y_scale,
                            output_channels)

            conv_output = tf.nn.conv2d_transpose(input_feature_map,
                                                 filters,
                                                 output_shape,
                                                 strides,
                                                 padding="SAME",
                                                 data_format='NHWC',
                                                 dilations=None,
                                                 name=name)

            # Apply an activation function.
            output_feature_map = tf.nn.leaky_relu(conv_output, alpha=0.02)

        return output_feature_map

    def conv_block(self,
                   input_feature_map,
                   input_channels,
                   output_channels=1,
                   kernel_size=2,
                   stride=1,
                   activation="LRelu",
                   name=None):

        if not name:

            name = "conv-c" + str(output_channels) + "-k" + str(kernel_size) + "-s" + str(stride) + "-" + activation

        with tf.name_scope(name):

            # Initialize the filter variables as he2015delving.
            he_relu_init_std = np.sqrt(2 / (input_channels * (kernel_size**2)))
            filters = tf.Variable(tf.random.normal((kernel_size,
                                                    kernel_size,
                                                    input_channels,
                                                    output_channels),
                                                   stddev=he_relu_init_std,
                                                   dtype=tf.float64))

            # Encode the strides for TensorFlow, and build the conv graph.
            strides = [1, stride, stride, 1]
            conv_output = tf.nn.conv2d(input_feature_map,
                                       filters,
                                       strides,
                                       padding="SAME",
                                       data_format='NHWC',
                                       dilations=None,
                                       name=None)

            # Apply an activation function.
            output_feature_map = tf.nn.leaky_relu(conv_output, alpha=0.02)

        return output_feature_map


    def plot(self, show_plot=False, logdir=None, step=None):

        # Create the directory for the plots
        step_plot_dir = os.path.join(logdir, 'step_' + str(step) + '_plots')
        if not os.path.exists(step_plot_dir):
            os.makedirs(step_plot_dir)

        def save_and_close_current_plot(logdir, plot_name="default"):
            fig_path = os.path.join(logdir, str(plot_name) + '.png')
            plt.gcf().set_dpi(600)
            plt.savefig(fig_path)
            plt.close()

        # Do a single sess.run to get all the values from a single batch.
        (pupil_planes,
         psfs,
         mtfs,
         distributed_aperture_images,
         perfect_image_flipped,
         perfect_image_spectrum,
         perfect_image,
         recovered_image,
         monolithic_pupil_plane,
         monolithic_psf,
         monolithic_mtf,
         monolithic_aperture_image
         ) = self.sess.run([self.pupil_planes,
                            self.psfs,
                            self.mtfs,
                            self.distributed_aperture_images,
                            self.perfect_image_flipped,
                            self.perfect_image_spectrum,
                            self.perfect_image,
                            self.recovered_image,
                            self.monolithic_pupil_plane,
                            self.monolithic_psf,
                            self.monolithic_mtf,
                            self.monolithic_aperture_image],
                           feed_dict={self.handle: self.valid_iterator_handle})

        # These are actually batches, so just take the first element.
        perfect_image_flipped = perfect_image_flipped[0]
        perfect_image_spectrum = perfect_image_spectrum[0]
        monolithic_aperture_image = monolithic_aperture_image[0]
        recovered_image = np.squeeze(recovered_image[0])

        # Iterate over each element of the ensemble from the DA system.
        for i, (pupil_plane,
                psf,
                mtf,
                distributed_aperture_image) in enumerate(zip(pupil_planes,
                                                             psfs,
                                                             mtfs,
                                                             distributed_aperture_images)):

            # These are actually batches, so just take the first one.
            distributed_aperture_image = distributed_aperture_image[0]

            # Plot phase angle
            left = self.pupil_dimension_x[0]
            right = self.pupil_dimension_x[-1]
            bottom = self.pupil_dimension_y[0]
            top = self.pupil_dimension_y[-1]
            # plt.imshow(np.angle(pupil_plane),
            #            cmap='twilight_shifted',
            #            extent=[left,right,bottom,top])
            # Overlay aperture mask
            plt.imshow(np.real(pupil_plane), cmap='inferno',
                       extent=[left, right, bottom, top])
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="pupil_plane_" + str(i))

            # Plot phase angle
            left = self.pupil_dimension_x[0]
            right = self.pupil_dimension_x[-1]
            bottom = self.pupil_dimension_y[0]
            top = self.pupil_dimension_y[-1]
            # Overlay aperture mask
            ax1 = plt.subplot(1, 2, 1)
            ax1.set_title('np.imag')
            plt.imshow(np.imag(pupil_plane), cmap='Greys',
                       extent=[left,right,bottom,top])
            plt.colorbar()

            ax2 = plt.subplot(1, 2, 2)
            plt.imshow(np.real(pupil_plane), cmap='Greys',
                       extent=[left,right,bottom,top])
            ax2.set_title('np.real')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="raw_pupil_plane_" + str(i))

            plt.imshow(np.log10(psf), cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="log_psf_" + str(i))

            plt.imshow(np.log10(mtf), cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="log_mtf_" + str(i))

            plt.imshow(distributed_aperture_image, cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="da_image_" + str(i))


        plt.imshow(recovered_image, cmap='inferno')
        plt.colorbar()

        save_and_close_current_plot(step_plot_dir,
                                    plot_name="recovered_image")

        # Plot phase angle
        left = self.pupil_dimension_x[0]
        right = self.pupil_dimension_x[-1]
        bottom = self.pupil_dimension_y[0]
        top = self.pupil_dimension_y[-1]
        # plt.imshow(np.angle(monolithic_pupil_plane), cmap='twilight_shifted',
        #            extent=[left, right, bottom, top])
        # plt.colorbar()

        # Overlay aperture mask
        plt.imshow(np.abs(monolithic_pupil_plane), cmap='inferno',
                   extent=[left, right, bottom, top])
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="monolithic_pupil_plane")

        plt.imshow(np.log10(monolithic_psf), cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_monolithic_psf")

        plt.imshow(np.log10(monolithic_mtf), cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_monolithic_mtf")

        plt.imshow(monolithic_aperture_image, cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="monolithic_aperture_image")

        plt.imshow(perfect_image_flipped, cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="object")

        plt.imshow(np.log10(np.abs(perfect_image_spectrum)), cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_object_spectrum")


    def train(self):

        return self.sess.run([self.loss,
                              self.monolithic_aperture_image_mse,
                              self.distributed_aperture_image_mse,
                              self.da_mse_mono_mse_ratio,
                              self.optimize],
                             feed_dict={self.handle: self.train_iterator_handle})

    def validate(self):

        return self.sess.run([self.loss,
                              self.monolithic_aperture_image_mse,
                              self.distributed_aperture_image_mse,
                              self.da_mse_mono_mse_ratio,
                              ],
                             feed_dict={self.handle: self.valid_iterator_handle})

    def recover(self, images):

        return self.sess.run([self.recovered_image
                              ],
                             feed_dict={self.handle: self.valid_iterator_handle})

def train(sess,
          dasie_model,
          train_dataset,
          valid_dataset,
          num_steps=1,
          plot_periodicity=1,
          writer=None,
          step_update=None,
          all_summary_ops=None,
          writer_flush=None,
          logdir=None,
          save_plot=False,
          show_plot=False,
          results_dict=None):


    # Build the initializers for the required datasets.
    train_dataset_initializer = train_dataset.get_initializer()
    valid_dataset_initializer = valid_dataset.get_initializer()

    # If no results dict is provided, make a blank one.
    if not results_dict:

        results_dict = dict()

    results_dict["results"] = dict()

    # Initialize all the metrics we want to populate during training.
    results_dict["results"]["train_loss_list"] = list()
    results_dict["results"]["train_dist_mse_list"] = list()
    results_dict["results"]["train_mono_mse_list"] = list()
    results_dict["results"]["train_mse_ratio_list"] = list()
    results_dict["results"]["valid_loss_list"] = list()
    results_dict["results"]["valid_dist_mse_list"] = list()
    results_dict["results"]["valid_mono_mse_list"] = list()
    results_dict["results"]["valid_mse_ratio_list"] = list()
    results_dict["results"]["train_epoch_time_list"] = list()

    sess.run(tf.compat.v1.global_variables_initializer())

    # Enter the main training loop.
    for i in range(num_steps):

        print("Beginning Epoch %d" % i)
        tf.summary.experimental.set_step(i)

        # If requested, plot the model status.
        if save_plot:
            if (i % plot_periodicity) == 0:

                sess.run(valid_dataset_initializer)
                print("Plotting...")
                dasie_model.plot(logdir=logdir,
                                 show_plot=show_plot,
                                 step=i)
                print("Plotting completed.")

        # Initialize the training dataset iterator to prepare for training.
        print("Training...")
        sess.run(train_dataset_initializer)

        # Initialize the training display metrics.
        train_loss = 0.0
        train_monolithic_aperture_image_mse = 0.0
        train_distributed_aperture_image_mse = 0.0
        train_da_mse_mono_mse_ratio = 0.0
        train_steps = 0.0

        # Run training steps until the iterator is exhausted.
        start_time = time.time()
        try:
            while True:

                # Execute one gradient update and get our tracked results.
                print("Starting train step %d..." % train_steps)

                step_start_time = time.time()
                (step_train_loss,
                 step_train_monolithic_aperture_image_mse,
                 step_train_distributed_aperture_image_mse,
                 step_train_da_mse_mono_mse_ratio,
                 _) = dasie_model.train()
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                # Increment all of our metrics.
                # TODO: Eventually refactor to summaries.
                train_loss += step_train_loss
                train_distributed_aperture_image_mse += step_train_distributed_aperture_image_mse
                train_monolithic_aperture_image_mse += step_train_monolithic_aperture_image_mse
                train_da_mse_mono_mse_ratio += step_train_da_mse_mono_mse_ratio
                train_steps += 1.0
                print("...step_train_loss = %f..." % step_train_loss)
                print("...step_train_da_mse_mono_mse_ratio = %f..." % step_train_da_mse_mono_mse_ratio)
                print("...train step %d complete in %f sec." % (train_steps, step_time))

        # OutOfRangeError indicates we've finished the iterator, so report out.
        except tf.errors.OutOfRangeError:

            end_time = time.time()
            train_epoch_time = end_time - start_time
            mean_train_loss = train_loss / train_steps
            mean_train_distributed_aperture_image_mse = train_distributed_aperture_image_mse / train_steps
            mean_train_monolithic_aperture_image_mse = train_monolithic_aperture_image_mse / train_steps
            mean_train_da_mse_mono_mse_ratio = train_da_mse_mono_mse_ratio / train_steps

            results_dict["results"]["train_loss_list"].append(mean_train_loss)
            results_dict["results"]["train_dist_mse_list"].append(mean_train_distributed_aperture_image_mse)
            results_dict["results"]["train_mono_mse_list"].append(mean_train_monolithic_aperture_image_mse)
            results_dict["results"]["train_mse_ratio_list"].append(mean_train_da_mse_mono_mse_ratio)
            results_dict["results"]["train_epoch_time_list"].append(train_epoch_time)

            print("Train Loss: %f" % mean_train_loss)
            print("Train DA MSE: %f" % mean_train_distributed_aperture_image_mse)
            print("Epoch %d Training Complete." % i)

            pass

        # Initialize the validation dataset iterator to prepare for validation.
        print("Validating...")
        sess.run(valid_dataset_initializer)

        # Initialize the validation display metrics.
        valid_loss = 0.0
        valid_monolithic_aperture_image_mse = 0.0
        valid_distributed_aperture_image_mse = 0.0
        valid_da_mse_mono_mse_ratio = 0.0
        valid_steps = 0.0

        # Validate by looping an calling validate batches, until...
        try:
            while True:
                # Execute one gradient update step.
                (step_valid_loss,
                 step_valid_monolithic_aperture_image_mse,
                 step_valid_distributed_aperture_image_mse,
                 step_valid_da_mse_mono_mse_ratio) = dasie_model.validate()

                # Increment all of our metrics.
                # TODO: Eventually refactor to summaries.
                valid_loss += step_valid_loss
                valid_distributed_aperture_image_mse += step_valid_distributed_aperture_image_mse
                valid_monolithic_aperture_image_mse += step_valid_monolithic_aperture_image_mse
                valid_da_mse_mono_mse_ratio += step_valid_da_mse_mono_mse_ratio
                valid_steps += 1.0

        # ...there are no more validate batches.
        except tf.errors.OutOfRangeError:

            # Compute the epoch results.
            mean_valid_loss = valid_loss / valid_steps
            mean_valid_distributed_aperture_image_mse = valid_distributed_aperture_image_mse / valid_steps
            mean_valid_monolithic_aperture_image_mse = valid_monolithic_aperture_image_mse / valid_steps
            mean_valid_da_mse_mono_mse_ratio = valid_da_mse_mono_mse_ratio / valid_steps

            # Store the epoch results.
            results_dict["results"]["valid_loss_list"].append(mean_valid_loss)
            results_dict["results"]["valid_dist_mse_list"].append(mean_valid_distributed_aperture_image_mse)
            results_dict["results"]["valid_mono_mse_list"].append(mean_valid_monolithic_aperture_image_mse)
            results_dict["results"]["valid_mse_ratio_list"].append(mean_valid_da_mse_mono_mse_ratio)

            print("Validation Loss: %f" % mean_valid_loss)
            print("Validation DA MSE: %f" % mean_valid_distributed_aperture_image_mse)

            print("Epoch %d Validation Complete." % i)
            pass

        # Write the results dict for this epoch.
        json_file = os.path.join(logdir, "results_" + str(i) + ".json")
        json.dump(results_dict, open(json_file, 'w'))
        # data = json.load(open("file_name.json"))

        # TODO: Dump recovery model weights and Zernike plan here.

        # TODO: Refactor to report at the step scale for training.
        # Execute the summary writer ops to write their values.
        sess.run(valid_dataset_initializer)
        feed_dict = {dasie_model.handle: dasie_model.valid_iterator_handle}
        sess.run(all_summary_ops, feed_dict=feed_dict)
        sess.run(step_update, feed_dict=feed_dict)
        sess.run(writer_flush, feed_dict=feed_dict)

def speedplus_parse_function(example_proto):
    """
    This is the first step of the generator/augmentation chain. Reading the
    raw file out of the TFRecord is fairly straight-forward, though does
    require some simple fixes. For instance, the number of bounding boxes
    needs to be padded to some upper bound so that the tensors are all of
    the same shape and can thus be batched.

    :param example_proto: Example from a TFRecord file
    :return: The raw image and padded bounding boxes corresponding to
    this TFRecord example.
    """
    # Define how to parse the example
    features = {
        "image_raw": tf.io.VarLenFeature(dtype=tf.string),
        "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        "height": tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    # Parse the example
    features_parsed = tf.io.parse_single_example(serialized=example_proto,
                                                 features=features)
    width = tf.cast(features_parsed["width"], tf.int32)
    height = tf.cast(features_parsed["height"], tf.int32)

    # filename = tf.cast(
    #     tf.sparse.to_dense(features_parsed["filename"], default_value=""),
    #     tf.string,
    # )

    image = tf.sparse.to_dense(features_parsed["image_raw"], default_value="")
    image = tf.io.decode_raw(image, tf.uint8)

    image = tf.reshape(image, [width, height, 1])
    image = tf.cast(image, tf.float64)

    return image

def main(flags):


    # beta = 32.0
    # TODO: Externalize
    subaperture_spacing_meters = 0.1

    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    dir_name = timestamp + "_" + str(os.getpid())
    save_dir = os.path.join(".", "logs", dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # TODO: Document how distance to the target is quantified implicitly.
    # TODO: Document how extent of the target is quantified implicitly.
    # Compute the scaling factor from meters to alpha for a GG PDF over meters.
    # alpha = np.log(-np.log(epsilon)) / np.log(beta) * ap_radius_meters
    # alpha = alpha / (flags.num_subapertures)
    # epsilon = 1e-15
    ap_radius_meters = (flags.aperture_diameter_meters / 2)
    subap_radius_meters = ((ap_radius_meters * np.sin(np.pi / flags.num_subapertures)) - (subaperture_spacing_meters / 2)) / (1 + np.sin(np.pi / flags.num_subapertures))
    # meters_to_alpha = 1 /  (2 * (np.log(-np.log(epsilon)) / np.log(beta)))
    # subap_alpha = subap_radius_meters * meters_to_alpha / 2
    # monolithic_alpha = flags.aperture_diameter_meters * meters_to_alpha * 2


    # Map our dataset name to relative locations and parse functions.
    if flags.dataset_name == "speedplus":
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "speedplus_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "speedplus_tfrecords", "valid")


    elif flags.dataset_name == "inria_holiday":
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "inria_holiday_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "inria_holiday_tfrecords", "valid")

    elif flags.dataset_name == "speedplus_one":
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "speedplus_one_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "speedplus_one_tfrecords", "valid")

    else:
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "onesat_example_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "onesat_example_tfrecords", "valid")

    # Set the crop size to the spatial quantization scale.
    if flags.crop:
        crop_size = flags.spatial_quantization
    else:
        crop_size = None

    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:

        print("\n\n\n\n\n\n\n\n\n Session Created \n\n\n\n\n\n\n")

        # Set all our seeds.
        np.random.seed(flags.random_seed)
        tf.compat.v1.set_random_seed(flags.random_seed)

        # Make summary management variables.
        step = tf.Variable(0, dtype=tf.int64)
        step_update = step.assign_add(1)
        tf.summary.experimental.set_step(step)
        writer = tf.summary.create_file_writer(save_dir)

        # Build our datasets.
        train_dataset = DatasetGenerator(train_data_dir,
                                         parse_function=parse_function,
                                         augment=False,
                                         shuffle=False,
                                         crop_size=crop_size,
                                         batch_size=flags.batch_size,
                                         num_threads=2,
                                         buffer=32,
                                         encoding_function=None,
                                         cache_dataset_memory=False,
                                         cache_dataset_file=False,
                                         cache_path="")

        # We create a tf.data.Dataset object wrapping the valid dataset here.
        valid_dataset = DatasetGenerator(valid_data_dir,
                                         parse_function=parse_function,
                                         augment=False,
                                         shuffle=False,
                                         crop_size=crop_size,
                                         batch_size=flags.batch_size,
                                         num_threads=2,
                                         buffer=32,
                                         encoding_function=None,
                                         cache_dataset_memory=False,
                                         cache_dataset_file=False,
                                         cache_path="")

        # Get the image shapes stored during dataset construction.
        image_x_scale = train_dataset.image_shape[0]
        image_y_scale = train_dataset.image_shape[1]


        # Manual debug here, to diagnose data problems.
        plot_data = False
        if plot_data:

            for i in range(16):
                train_iterator = train_dataset.get_iterator()
                train_dataset_batch = train_iterator.get_next()
                train_dataset_initializer = train_dataset.get_initializer()
                sess.run(train_dataset_initializer)

                valid_iterator = valid_dataset.get_iterator()
                valid_dataset_batch = valid_iterator.get_next()
                valid_dataset_initializer = valid_dataset.get_initializer()
                sess.run(valid_dataset_initializer)

                for j in range(2):

                    np_train_dataset_batch = sess.run(train_dataset_batch)
                    np_valid_dataset_batch = sess.run(valid_dataset_batch)

                    plt.subplot(241)
                    plt.imshow(np_train_dataset_batch[0])
                    plt.subplot(242)
                    plt.imshow(np_train_dataset_batch[1])
                    plt.subplot(243)
                    plt.imshow(np_train_dataset_batch[2])
                    plt.subplot(244)
                    plt.imshow(np_train_dataset_batch[3])

                    plt.subplot(245)
                    plt.imshow(np_valid_dataset_batch[0])
                    plt.subplot(246)
                    plt.imshow(np_valid_dataset_batch[1])
                    plt.subplot(247)
                    plt.imshow(np_valid_dataset_batch[2])
                    plt.subplot(248)
                    plt.imshow(np_valid_dataset_batch[3])
                    plt.show()

        # Build a DA model. Inputs: n p/t/t tensors. Output: n image tensors.
        dasie_model = DASIEModel(sess,
                                 batch_size=flags.batch_size,
                                 train_dataset=train_dataset,
                                 valid_dataset=valid_dataset,
                                 num_exposures=flags.num_exposures,
                                 spatial_quantization=flags.spatial_quantization,
                                 image_x_scale=image_x_scale,
                                 image_y_scale=image_y_scale,
                                 learning_rate=flags.learning_rate,
                                 diameter_meters=flags.aperture_diameter_meters,
                                 num_apertures=flags.num_subapertures,
                                 supaperture_radius_meters=subap_radius_meters,
                                 recovery_model_filter_scale=flags.recovery_model_filter_scale,
                                 loss_name=flags.loss_name,
                                 writer=writer,
                                 num_zernike_indices=flags.num_zernike_indices,
                                 hadamard_image_formation=flags.hadamard_image_formation,
                                 zernike_debug=flags.zernike_debug)

        # Merge all the summaries from the graphs, flush and init the nodes.
        all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        writer_flush = writer.flush()
        sess.run([writer.init(), step.initializer])

        base_results_dict = vars(flags)

        # Optimize the DASIE model parameters.
        train(sess,
              dasie_model,
              train_dataset,
              valid_dataset,
              num_steps=flags.num_steps,
              plot_periodicity=flags.plot_periodicity,
              writer=writer,
              step_update=step_update,
              all_summary_ops=all_summary_ops,
              writer_flush=writer_flush,
              logdir=save_dir,
              save_plot=flags.save_plot,
              show_plot=flags.show_plot,
              results_dict=base_results_dict)



if __name__ == '__main__':

    # TODO: I need to enable a test of negligable, random, and learned articulations to measure validation set reconstructions.

    parser = argparse.ArgumentParser(
        description='provide arguments for training.')

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')


    parser.add_argument('--logdir',
                        type=str,
                        default=".\\logs\\",
                        help='The directory to which summaries are written.')

    parser.add_argument('--run_name',
                        type=str,
                        default=datetime.datetime.today().strftime('%Y%m%d_%H%M%S'),
                        help='The name of this run')

    parser.add_argument('--loss_name',
                        type=str,
                        default="mae",
                        help='The loss function used.')

    parser.add_argument('--num_steps',
                        type=int,
                        default=4096,
                        help='The number of optimization steps to perform..')

    parser.add_argument('--random_seed',
                        type=int,
                        default=np.random.randint(0, 2048),
                        help='A random seed for repeatability.')

    parser.add_argument('--plot_periodicity',
                        type=int,
                        default=64,
                        help='Number of epochs to wait before plotting.')

    parser.add_argument('--num_subapertures',
                        type=int,
                        default=15,
                        help='Number of DASIE subapertures.')


    parser.add_argument('--num_zernike_indices',
                        type=int,
                        default=1,
                        help='Number of Zernike terms to simulate.')

    parser.add_argument('--aperture_diameter_meters',
                        type=float,
                        default=2.5,
                        help='Diameter of DA and mono apertures in meters.')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='The size of the optimizer step.')

    parser.add_argument('--spatial_quantization',
                        type=int,
                        default=256,
                        help='Quantization of all images.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Number of perfect images per batch.')

    parser.add_argument('--num_exposures',
                        type=int,
                        default=1,
                        help='The number of sequential frames to model.')

    parser.add_argument("--show_plot", action='store_true',
                        default=False,
                        help="Show the plot?")

    parser.add_argument("--save_plot", action='store_true',
                        default=False,
                        help='Save the plot?')

    parser.add_argument("--crop", action='store_true',
                        default=False,
                        help='If true, crop images to spatial_quantization.')

    parser.add_argument("--hadamard_image_formation", action='store_true',
                        default=False,
                        help='If true, use MTF, image spectrum product, else \
                              use PSF convolution.')

    parser.add_argument('--dataset_root', type=str,
                        default="..\\data",
                        help='Path to a directory hol ding all datasetss.')

    parser.add_argument('--dataset_name', type=str,
                        default="speedplus",
                        help='Path to the train data TFRecords directory.')

    parser.add_argument('--object_plane_scale', type=float,
                        default=1.0,
                        help='The angular scale/pixel of the object plane.')

    parser.add_argument('--recovery_model_filter_scale',
                        type=int,
                        default=16,
                        help='Base filter size for recovery model.')

    parser.add_argument("--zernike_debug", action='store_true',
                        default=False,
                        help="If true, each subaperture is constrained such \
                              that only the Zernike coefficient with the same \
                              index as the subaperture index is none-zero.")




    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)