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

import numpy as np
import pandas as pd
import tensorflow as tf


# TODO: Externalize and join.
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
                         + str(term_number) + ") that is not supported by \
                         this library. Limit your terms to [0, 14].")

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

            elif zernike_init_type == "np.random.normal":

                zernike_coefficient = np.random.normal(0.1, 0.01)

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

    # print("-Starting Zernike aperture function.")

    print("Building aperture number %d, term:" % aperture_num, end="", flush=True)
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

        print(str(term_number) + ", ", end="", flush=True)

        # Select the function corresponding to this term number.
        zernike_term_function = select_zernike_function(term_number=term_number)

        # Either initialize the aperture disk, or add this contribution to it.
        if tensor_zernike_2d_sample_initialized:
            tensor_zernike_2d_sample += zernike_coefficient * tf.vectorized_map(zernike_term_function, T)
        else:
            tensor_zernike_2d_sample = zernike_coefficient * tf.vectorized_map(zernike_term_function, T)
            tensor_zernike_2d_sample_initialized = True

    # Map the zernike domain from [-1, 1] to [0, 1], per term
    # Each term can contribute as much as -1, and has a range of size 2.
    # TODO: Ryan, talk with me about this. It changes the mapping from Zernike
    #       coefficients to Z in a way that might be problematic on the bench.
    num_terms = len(zernike_coefficients)
    tensor_zernike_2d_sample = (tensor_zernike_2d_sample + num_terms) / (2 * num_terms)

    # Apply a circle mask to set all non-aperture pixels to 0.0.
    # print("-Masking subaperture.")
    pupil_mask = circle_mask(X, Y, mu_u, mu_v, subaperture_radius)
    pupil_mask = tf.cast(tf.constant(pupil_mask), dtype=tf.complex128)
    tensor_masked_zernike_2d_sample = tensor_zernike_2d_sample * pupil_mask

    # TODO: Reinstate after debug? Talk to Ryan: Why should I do this?

    # The piston tip and tilt are encoded as the phase-angle of pupil plane
    # print("-Generating phase angle field.")
    # Normalize the zernike field so that it may be returned as
    # zernike_min = tf.cast(tf.math.reduce_min(tf.math.abs(tensor_masked_zernike_2d_sample)), dtype=tf.complex128)
    # zernike_max = tf.cast(tf.math.reduce_max(tf.math.abs(tensor_masked_zernike_2d_sample)), dtype=tf.complex128)
    # zernike_range = (zernike_max - zernike_min)
    # tensor_zernike_2d_field = (tensor_masked_zernike_2d_sample - zernike_min) / zernike_range
    # tensor_zernike_2d_field = tf.exp(tensor_masked_zernike_2d_sample)
    # tensor_zernike_2d_field = tensor_zernike_2d_field * pupil_mask

    tensor_zernike_2d_field = tensor_masked_zernike_2d_sample

    print(".")
    return tensor_zernike_2d_field
