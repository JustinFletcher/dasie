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

from recovery_models import RecoveryModel


# First, prevent TensorFlow from foisting filthy eager execution upon us.
tf.compat.v1.disable_eager_execution()

# TODO: Externalize.
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

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


def set_kwargs_default(key, value, kwargs):
    kwargs[key] = kwargs.get(key, value)
    return (kwargs[key])

class DASIEModel(object):

    def __init__(self,
                 sess,
                 train_dataset=None,
                 valid_dataset=None,
                 # batch_size=2,
                 # loss_name="mse",
                 # learning_rate=1.0,
                 # num_apertures=15,
                 # spatial_quantization=256,
                 # image_x_scale=256,
                 # image_y_scale=256,
                 # subaperture_radius_meters=None,
                 # num_exposures=1,
                 # recovery_model_filter_scale=16,
                 # diameter_meters=2.5,
                 # num_zernike_indices=1,
                 # zernike_debug=False,
                 # hadamard_image_formation=True,
                 # writer=None,
                 **kwargs):

        self.sess = sess
        # First, set the kwargs. This is needed to persist kwargs in a save.
        # TODO: refactor constructor for this approach.
        # self.learning_rate = learning_rate
        # self.learning_rate = kwargs.get('learning_rate', 1.0)
        self.learning_rate = set_kwargs_default('learning_rate', 1.0, kwargs)
        # self.num_apertures = num_apertures
        # self.num_apertures = kwargs.get('num_apertures', 15)
        self.num_apertures = set_kwargs_default('num_apertures', 15, kwargs)
        # self.batch_size = batch_size
        # self.batch_size = kwargs.get('batch_size', 2)
        self.batch_size = set_kwargs_default('batch_size', 2, kwargs)
        # self.spatial_quantization = spatial_quantization
        # self.spatial_quantization = kwargs.get('spatial_quantization', 256)
        self.spatial_quantization = set_kwargs_default('spatial_quantization', 256, kwargs)
        # self.image_x_scale = image_x_scale
        # self.image_x_scale = kwargs.get('image_x_scale', 256)
        self.image_x_scale = set_kwargs_default('image_x_scale', 256, kwargs)
        # self.image_y_scale = image_y_scale
        # self.image_y_scale = kwargs.get('image_y_scale', 256)
        self.image_y_scale = set_kwargs_default('image_y_scale', 256, kwargs)
        # self.writer = writer
        # self.writer = kwargs.get('writer', None)
        self.writer = set_kwargs_default('writer', None, kwargs)
        # self.loss_name = loss_name
        # self.loss_name = kwargs.get('loss_name', "mse")
        self.loss_name = set_kwargs_default('loss_name', "mse", kwargs)
        # self.num_exposures = num_exposures
        # self.num_exposures = kwargs.get('num_exposures', 1)
        self.num_exposures = set_kwargs_default('num_exposures', 1, kwargs)
        # self.subaperture_radius_meters = subaperture_radius_meters
        # self.subaperture_radius_meters = kwargs.get('subaperture_radius_meters',
        #                                             None)
        self.subaperture_radius_meters = set_kwargs_default('subaperture_radius_meters', None, kwargs)
        print(self.subaperture_radius_meters)
        # die
        # self.diameter_meters = diameter_meters
        # self.diameter_meters = kwargs.get('diameter_meters', 2.5)
        self.diameter_meters = set_kwargs_default('diameter_meters', 2.5, kwargs)
        # self.recovery_model_filter_scale = recovery_model_filter_scale
        # self.recovery_model_filter_scale = kwargs.get('recovery_model_filter_scale',
        #                                               16)
        self.recovery_model_filter_scale = set_kwargs_default('recovery_model_filter_scale', 16, kwargs)
        # self.num_zernike_indices = num_zernike_indices
        # self.num_zernike_indices = kwargs.get('num_zernike_indices', 15)
        self.num_zernike_indices = set_kwargs_default('num_zernike_indices', 15, kwargs)
        # self.zernike_debug = zernike_debug
        # self.zernike_debug = kwargs.get('zernike_debug', False)
        self.zernike_debug = set_kwargs_default('zernike_debug', False, kwargs)
        # self.hadamard_image_formation = hadamard_image_formation
        # self.hadamard_image_formation = kwargs.get('hadamard_image_formation',
        #                                            True)
        self.hadamard_image_formation = set_kwargs_default('hadamard_image_formation', True, kwargs)

        # Persist the kwargs to enable model saving and recovery.
        self.kwargs = kwargs

        self.radius_meters = self.diameter_meters / 2

        # TODO: Make this optional by providing some other dataset batch.
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

            # TODO: Look in here and see if this will build without the dataset
            self._build_dasie_model(
                inputs=dataset_batch,
                spatial_quantization=self.spatial_quantization,
                num_apertures=self.num_apertures,
                radius_meters=self.radius_meters,
                subaperture_radius_meters=self.subaperture_radius_meters,
                num_exposures=self.num_exposures,
                recovery_model_filter_scale=self.recovery_model_filter_scale,
                num_zernike_indices=self.num_zernike_indices,
                zernike_debug=self.zernike_debug,
                hadamard_image_formation=self.hadamard_image_formation,
                )

        with tf.name_scope("distributed_aperture_image_recovery_model"):

            # Combine the ensemble of images with the restoration function.
            self.recovered_image = self._build_recovery_model(
                self.distributed_aperture_images,
                filter_scale=self.recovery_model_filter_scale)


        with tf.name_scope("dasie_loss"):

            # First, add some bookeeping nodes.
            self.perfect_image_flipped = tf.reverse(
                tf.reverse(tf.squeeze(self.perfect_image, axis=-1), [-1]), [1])
            self.image_mse = tf.reduce_mean(
                (self.recovered_image - self.perfect_image_flipped) ** 2)

            # Then build the selected loss function.
            if self.loss_name == "mse":
                loss = self.image_mse
            if self.loss_name == "mae":
                loss = tf.reduce_mean(tf.math.abs(
                    self.recovered_image - self.perfect_image_flipped))
            if self.loss_name == "l2":
                loss = tf.math.sqrt(tf.math.reduce_sum((self.recovered_image - self.perfect_image_flipped) ** 2))
            if self.loss_name == "cos":
                loss = -cosine_similarity(self.recovered_image,
                                          self.perfect_image_flipped)

            self.loss = loss

        with tf.name_scope("dasie_metrics"):

            self.monolithic_aperture_image_mse = tf.reduce_mean((self.monolithic_aperture_image - self.perfect_image_flipped) ** 2)
            self.distributed_aperture_image_mse = tf.reduce_mean(
                (self.recovered_image - self.perfect_image_flipped) ** 2)
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
            # TODO: clean this up.
            # I wonder if this works...
            with self.writer.as_default():

                # TODO: refactor all these endpoints to name *_batch.
                tf.summary.scalar("in_graph_loss", self.loss)
                tf.summary.scalar("monolithic_aperture_image_mse",
                                  self.monolithic_aperture_image_mse)
                tf.summary.scalar("distributed_aperture_image_mse",
                                  self.distributed_aperture_image_mse)
                tf.summary.scalar("da_mse_mono_mse_ratio",
                                  self.da_mse_mono_mse_ratio)
                # tf.compat.v1.summary.scalar("v1_test", self.loss)

            with tf.compat.v1.Graph().as_default():
                tf.summary.scalar("debug_metric", 0.5)
            # self.summaries = tf.compat.v1.summary.all_v2_summary_ops()
            # self.v1_summaries = tf.compat.v1.summary.merge_all()

        with tf.name_scope("dasie_optimizer"):
            # Build an op that applies the policy gradients to the model.
            self.optimize = tf.compat.v1.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)


        self.inputs = self.perfect_image
        self.output_images = self.recovered_image



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
                           subaperture_radius_meters=None,
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

        # TODO: Make the distributed aperture optical model a separate method.

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
        self.subaperture_size_pixels = int(subaperture_radius_meters // self.pupil_meters_per_pixel)


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
        self.plan = dict()
        for exposure_num in range(num_exposures):
            self.plan[exposure_num] = dict()
            with tf.name_scope("exposure_" + str(exposure_num)):


                # Build the pupil plane quantization grid for this exposure.
                pupil_plane = tf.zeros((spatial_quantization,
                                        spatial_quantization),
                                       dtype=tf.complex128)

                # Build the model of the pupil plane, using the Variables.
                with tf.name_scope("pupil_plane_model"):

                    for aperture_num in range(num_apertures):
                        self.plan[exposure_num][aperture_num] = dict()

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
                            self.plan[exposure_num][aperture_num] = subap_zernike_coefficients_vars
                            # Render this subaperture on the pupil plane grid.
                            # TODO: Ryan: Here's where I can set the phase scale to physical units. Should I?
                            # pupil_plane += self.phase_scale * zernike_aperture_function_2d(X,
                            #                                                                Y,
                            #                                                                mu_u,
                            #                                                                mu_v,
                            #                                                                radius_meters,
                            #                                                                subaperture_radius_meters,
                            #                                                                subap_zernike_coefficients_variables,
                            #                                                                )


                            pupil_plane += zernike_aperture_function_2d(X,
                                                                        Y,
                                                                        mu_u,
                                                                        mu_v,
                                                                        radius_meters,
                                                                        subaperture_radius_meters,
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



    def _build_recovery_model(self,
                              distributed_aperture_images_batch,
                              filter_scale):
        """

        :param distributed_aperture_images_batch: a batch of DASIE images.
        :param filter_scale: the smallest filter scale to use.
        :return:
        """

        # Stack the the images in the ensemble to form a batch of inputs.
        distributed_aperture_images_batch = tf.stack(distributed_aperture_images_batch, axis=-1)

        recovery_model = RecoveryModel()

        return recovery_model.build(distributed_aperture_images_batch,
                                    filter_scale,
                                    self.num_exposures,
                                    self.image_x_scale,
                                    self.image_y_scale,
                                    self.batch_size)

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

    def save(self, save_file_path):
        """
        This function saves a dictionary comprising all weights, kwargs, and
        the git hash so that any model trained from the same commit can be
        restored.
        :return: None
        """
        # var = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]
        # print([v for v in tf.compat.v1.trainable_variables()])
        # checkpoint_path = os.path.join(logdir, "ckpt")
        # checkpoint = tf.train.Checkpoint(model=self.output_images)
        # save_path = checkpoint.save(checkpoint_path)
        # checkpoint = tf.train.Checkpoint(model=self.output_images,
        #                                  zernikes=self.plan)

        save_dict = dict()
        for key, value in self.kwargs.items():
            if is_jsonable(value):
                save_dict[key] = value

        for v in tf.compat.v1.trainable_variables():
            save_dict[v.name] = self.sess.run(v)

        # json.dump(save_dict, open(json_file, 'w'))
        json.dump(save_dict, open(save_file_path, 'w'), cls=NpEncoder)

        return None


    def restore(self, restore_file_path):
        """
        This function loads a dictionary comprising all weights and kwargs,
        enabling their use to restore the saved model if the model is the same.
        :return: None
        """

        # TODO: Implement.
        restore_dict = json.load(restore_file_path)
        for v in tf.compat.v1.trainable_variables():
            # v.value = restore_dict["variables"][v.name]
            v.load(restore_dict["variables"][v.name], self.sess)

        return None

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
