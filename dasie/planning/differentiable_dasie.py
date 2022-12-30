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

import numpy as np
import pandas as pd
import tensorflow as tf
# TODO: Implement TF probability.
import tensorflow_probability as tfp

from decimal import Decimal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Tentative.
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from hcipy import *

# TODO: Refactor this import.
import atmosphere
from atmosphere import *
import zernike
from zernike import *
from dataset_generator import DatasetGenerator
from recovery_models import RecoveryModel


# First, prevent TensorFlow from foisting filthy eager execution upon us.
tf.compat.v1.disable_eager_execution()

# TODO: Externalize.
def ssim(x_batch, y_batch):

    x_batch = x_batch / tf.math.reduce_max(x_batch)
    y_batch = y_batch / tf.math.reduce_max(y_batch)
    ssim_batch = tf.image.ssim(
        x_batch,
        y_batch,
        max_val=1.0,
        filter_size=4,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
    )

    return tf.math.reduce_mean(ssim_batch)


# TODO: Externalize.
def psnr(x_batch, y_batch):

    x_batch = x_batch / tf.math.reduce_max(x_batch)
    y_batch = y_batch / tf.math.reduce_max(y_batch)
    psnr_batch = tf.image.psnr(x_batch, y_batch, max_val=1.0)
    return tf.math.reduce_mean(psnr_batch)


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

# TODO: Externalize
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

# TODO: Externalize
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

# TODO: Externalize
def circle_mask(X, Y, x_center, y_center, radius):
    r = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
    return r < radius

# TODO: Externalize
def set_kwargs_default(key, value, kwargs):
    kwargs[key] = kwargs.get(key, value)
    return (kwargs[key])

class DASIEModel(object):

    def __init__(self,
                 sess,
                 train_dataset=None,
                 valid_dataset=None,
                 **kwargs):

        # A session is the only positional (i.e., required) parameter.
        self.sess = sess

        # We set the kwargs outside of the signature, simplifying persistence.
        self.learning_rate = set_kwargs_default(
            'learning_rate', 1.0, kwargs)

        self.num_apertures = set_kwargs_default(
            'num_apertures', 15, kwargs)

        self.batch_size = set_kwargs_default(
            'batch_size', 2, kwargs)

        self.spatial_quantization = set_kwargs_default(
            'spatial_quantization', 256, kwargs)

        self.image_x_scale = set_kwargs_default(
            'image_x_scale', 256, kwargs)

        self.image_y_scale = set_kwargs_default(
            'image_y_scale', 256, kwargs)

        self.writer = set_kwargs_default(
            'writer', None, kwargs)

        self.loss_name = set_kwargs_default(
            'loss_name', "mse", kwargs)

        self.num_exposures = set_kwargs_default(
            'num_exposures', 1, kwargs)

        self.subaperture_radius_meters = set_kwargs_default(
            'subaperture_radius_meters', None, kwargs)

        self.edge_padding_factor = set_kwargs_default(
            'edge_padding_factor', 0.1, kwargs)

        self.diameter_meters = set_kwargs_default(
            'diameter_meters', 2.5, kwargs)

        self.recovery_model_filter_scale = set_kwargs_default(
            'recovery_model_filter_scale', 16, kwargs)

        self.num_zernike_indices = set_kwargs_default(
            'num_zernike_indices', 15, kwargs)

        self.zernike_debug = set_kwargs_default(
            'zernike_debug', False, kwargs)

        self.hadamard_image_formation = set_kwargs_default(
            'hadamard_image_formation', True, kwargs)

        self.subap_area = set_kwargs_default(
            'subap_area', None, kwargs)

        self.mono_ap_area = set_kwargs_default(
            'mono_ap_area', (np.pi * (self.diameter_meters / 2)) ** 2, kwargs)

        self.mono_to_dist_aperture_ratio = set_kwargs_default(
            'mono_to_dist_aperture_ratio', None, kwargs)

        self.object_plane_extent_meters = set_kwargs_default(
            'object_plane_extent_meters', 1.0, kwargs)

        self.object_distance_meters = set_kwargs_default(
            'object_distance_meters', 1000000.0, kwargs)

        self.zernike_init_type = set_kwargs_default(
            'zernike_init_type', "np.random.uniform", kwargs)

        # TODO: Not in use.
        self.filter_wavelength_micron = set_kwargs_default(
            'filter_wavelength_micron', 1.0, kwargs)

        self.sensor_gaussian_mean = set_kwargs_default(
            'sensor_gaussian_mean', 1e-5, kwargs)

        self.sensor_poisson_mean_arrival = set_kwargs_default(
            'sensor_poisson_mean_arrival', 4e-5, kwargs)

        self.dm_stroke_microns = set_kwargs_default(
            'dm_stroke_microns', 8.0, kwargs)

        # TODO: Not in use.
        self.focal_extent_meters = set_kwargs_default(
            'focal_extent_meters', 0.1, kwargs)

        self.r0_mean = set_kwargs_default(
            'r0_mean', 0.020, kwargs)

        self.r0_std = set_kwargs_default(
            'r0_std', 0.0, kwargs)

        self.outer_scale_mean = set_kwargs_default(
            'outer_scale_mean', 2000.0, kwargs)

        self.outer_scale_std = set_kwargs_default(
            'outer_scale_std', 0.0, kwargs)

        self.inner_scale_mean = set_kwargs_default(
            'inner_scale_mean', 0.008, kwargs)

        self.inner_scale_std = set_kwargs_default(
            'inner_scale_std', 0.0, kwargs)

        # TODO: Not in use.
        self.greenwood_time_constant_sec_mean = set_kwargs_default(
            'greenwood_time_constant_sec_mean', 1.0, kwargs)

        # TODO: Not in use.
        self.greenwood_time_constant_sec_std = set_kwargs_default(
            'greenwood_time_constant_sec_std', 0.0, kwargs)

        self.greenwood_time_constant_sec_std = set_kwargs_default(
            'greenwood_time_constant_sec_std', 0.0, kwargs)

        self.effective_focal_length_meters = set_kwargs_default(
            'effective_focal_length_meters', 726.0 , kwargs)

        self.recovery_model_type = set_kwargs_default(
            'recovery_model_type', "tseng2021neural", kwargs)

        self.example_image_index = set_kwargs_default(
            'example_image_index', 0, kwargs)


        # Store a reference field to kwargs to enable model saving & recovery.
        self.kwargs = kwargs

        # For convenience, precompute some physical parameters.
        self.radius_meters = self.diameter_meters / 2

        with tf.name_scope("object_model"):

            # If no dataset is provided, we set the batch operation to None.
            if (train_dataset is None) or (valid_dataset is None):

                self.dataset_batch = None

            # Otherwise, build a string handle iterator operation.
            else:

                train_iterator = train_dataset.get_iterator()
                self.train_iterator_handle = sess.run(
                    train_iterator.string_handle())

                valid_iterator = valid_dataset.get_iterator()
                self.valid_iterator_handle = sess.run(
                    valid_iterator.string_handle())

                self.handle = tf.compat.v1.placeholder(tf.string, shape=[])

                # Abstract specific iterators as only their types.
                iterator_output_types = train_iterator.output_types
                iterator = tf.compat.v1.data.Iterator.from_string_handle(
                    self.handle,
                    iterator_output_types)
                dataset_batch = iterator.get_next()

                self.dataset_batch = dataset_batch

        with tf.name_scope("dasie_model"):

            # TODO: remove superfluous property arguments.
            self._build_dasie_model(
                inputs=self.dataset_batch,
                spatial_quantization=self.spatial_quantization,
                num_apertures=self.num_apertures,
                radius_meters=self.radius_meters,
                subaperture_radius_meters=self.subaperture_radius_meters,
                num_exposures=self.num_exposures,
                num_zernike_indices=self.num_zernike_indices,
                zernike_debug=self.zernike_debug,
                hadamard_image_formation=self.hadamard_image_formation,
                zernike_init_type=self.zernike_init_type,
                )

        with tf.name_scope("distributed_aperture_image_recovery_model"):

            # Combine the ensemble of images with the restoration function.
            self.recovered_image = self._build_recovery_model(
                self.distributed_aperture_images,
                filter_scale=self.recovery_model_filter_scale
            )

        with tf.name_scope("dasie_loss"):

            # First, add some bookeeping nodes.
            self.flipped_object_batch = tf.reverse(
                tf.reverse(
                    tf.squeeze(
                        self.object_batch,
                        axis=-1
                    ),
                    [-1]
                ),
                [1]
            )
            self.image_mse = tf.reduce_mean(
                (self.recovered_image - self.flipped_object_batch) ** 2
            )

            # Then build the selected loss function.
            if self.loss_name == "mse":
                loss = self.image_mse

            if self.loss_name == "mae":
                loss = tf.reduce_mean(
                    tf.math.abs(
                        self.recovered_image - self.flipped_object_batch
                    )
                )

            if self.loss_name == "l2":
                loss = tf.math.sqrt(
                    tf.math.reduce_sum(
                        (self.recovered_image - self.flipped_object_batch) ** 2
                    )
                )

            if self.loss_name == "cos":
                loss = -cosine_similarity(
                    self.recovered_image,
                    self.flipped_object_batch
                )

            self.loss = loss

        with tf.name_scope("dasie_metrics"):

            # Compute MSE.
            self.monolithic_aperture_image_mse = tf.reduce_mean(
                (self.monolithic_aperture_image - self.flipped_object_batch) ** 2)
            self.distributed_aperture_image_mse = tf.reduce_mean(
                (self.recovered_image - self.flipped_object_batch) ** 2)
            self.da_mse_mono_mse_ratio = self.distributed_aperture_image_mse / self.monolithic_aperture_image_mse

            # Compute SSIM.
            self.monolithic_aperture_image_ssim = ssim(
                self.monolithic_aperture_image,
                self.flipped_object_batch
            )
            self.distributed_aperture_image_ssim  = ssim(
                self.recovered_image,
                self.flipped_object_batch
            )
            self.da_ssim_mono_ssim_ratio = self.distributed_aperture_image_ssim / self.monolithic_aperture_image_ssim

            # Compute PSNR.
            self.monolithic_aperture_image_psnr = psnr(
                self.monolithic_aperture_image,
                self.flipped_object_batch
            )
            self.distributed_aperture_image_psnr = psnr(
                self.recovered_image,
                self.flipped_object_batch
            )
            self.da_psnr_mono_psnr_ratio = self.distributed_aperture_image_psnr / self.monolithic_aperture_image_psnr

            if self.writer:
                with self.writer.as_default():

                    # TODO: refactor all these endpoints to name *_batch.
                    tf.summary.scalar("in_graph_loss", self.loss)
                    tf.summary.scalar("monolithic_aperture_image_mse",
                                      self.monolithic_aperture_image_mse)
                    tf.summary.scalar("distributed_aperture_image_mse",
                                      self.distributed_aperture_image_mse)
                    tf.summary.scalar("da_mse_mono_mse_ratio",
                                      self.da_mse_mono_mse_ratio)
                    tf.summary.scalar("monolithic_aperture_image_ssim",
                                      self.monolithic_aperture_image_ssim)
                    tf.summary.scalar("distributed_aperture_image_ssim",
                                      self.distributed_aperture_image_ssim)
                    tf.summary.scalar("da_ssim_mono_ssim_ratio",
                                      self.da_ssim_mono_ssim_ratio)
                    tf.summary.scalar("monolithic_aperture_image_psnr",
                                      self.monolithic_aperture_image_psnr)
                    tf.summary.scalar("distributed_aperture_image_psnr",
                                      self.distributed_aperture_image_psnr)
                    tf.summary.scalar("da_psnr_mono_psnr_ratio",
                                      self.da_psnr_mono_psnr_ratio)
                # tf.compat.v1.summary.scalar("v1_test", self.loss)

            with tf.compat.v1.Graph().as_default():
                tf.summary.scalar("debug_metric", 0.5)
            # self.summaries = tf.compat.v1.summary.all_v2_summary_ops()
            # self.v1_summaries = tf.compat.v1.summary.merge_all()

        with tf.name_scope("dasie_optimizer"):
            # Build an op that applies the policy gradients to the model.
            self.optimize = tf.compat.v1.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)

        # Finally, build externally accessible references to operators.
        self.inputs = self.object_batch
        self.output_images = self.recovered_image

    def _image_model(self,
                     hadamard_image_formation=False,
                     mtf=None,
                     psf=None):

        with tf.name_scope("image_plane_model"):

            if hadamard_image_formation:
                complex_mtf = tf.cast(mtf, dtype=tf.complex128)
                image_spectrum = self.object_spectrum_batch * complex_mtf
                image = tf.abs(tf.signal.fft2d(image_spectrum))
            else:
                image = tf.nn.conv2d(
                    # tf.squeeze(self.object_batch, axis=-1),
                    self.object_batch,
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
            pupil_spectrum = tf.signal.fft2d(pupil_plane)
            shifted_pupil_spectrum = tf.signal.fftshift(pupil_spectrum)
            psf = tf.abs(shifted_pupil_spectrum) ** 2


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

        # Strip any imaginary image component.
        image = tf.math.abs(image)

        # Apply the reparameterization trick from kingma2014autovariational.
        gaussian_dist = tfp.distributions.Normal(loc=tf.zeros_like(image),
                                                 scale=tf.ones_like(image))

        gaussian_sample = tfp.distributions.Sample(gaussian_dist).sample()
        gaussian_noise = image + (gaussian_mean ** 2) * gaussian_sample

        # Apply the score-gradient trick from williams1992simple.
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
                           num_zernike_indices=1,
                           zernike_debug=False,
                           hadamard_image_formation=True,
                           zernike_init_type="np.random.uniform"):

        # TODO: Externalize.
        lock_dm_values = False
        if lock_dm_values:
            dm_trainable = False
        else:
            dm_trainable = True



        # Build object plane image batch tensor objects.
        if inputs is not None:
            self.object_batch = inputs
        else:
            shape = (self.batch_size,
                     self.image_x_scale,
                     self.image_y_scale,
                     1)
            self.object_batch = tf.compat.v1.placeholder(tf.float64,
                                                          shape=shape,
                                                          name="object_batch")

        with tf.name_scope("image_spectrum_model"):

            self.object_spectrum_batch = tf.signal.fft2d(
                tf.cast(
                    tf.squeeze(
                        self.object_batch,
                        axis=-1
                    ),
                    dtype=tf.complex128
                )
            )

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
        pupil_extent = (2 * radius_meters) * (1 + self.edge_padding_factor)
        self.pupil_extent = pupil_extent
        print("pupil_extent=" + str(pupil_extent))
        # This converts radians to radians to meters, in filter wavelengths.
        # self.phase_scale = 2 * np.pi / filter_wavelength_micron
        self.phase_scale_wavelengths = self.dm_stroke_microns / filter_wavelength_micron

        focal_extent = self.focal_extent_meters

        # Compute the subaperture pixel extent.
        self.pupil_meters_per_pixel = radius_meters / spatial_quantization
        self.subaperture_size_pixels = int(
            subaperture_radius_meters // self.pupil_meters_per_pixel
        )


        # Build the simulation mesh grid.
        # TODO: Verify these physical coordinates; clarify pupil vs radius.
        u = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        v = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        self.pupil_dimension_u = u
        self.pupil_dimension_v = v
        pupil_grid_u, pupil_grid_v = np.meshgrid(u, v)

        # TODO: this grid is never used right now. It probably should be.
        x = np.linspace(-focal_extent/2, focal_extent/2, spatial_quantization)
        y = np.linspace(-focal_extent/2, focal_extent/2, spatial_quantization)
        self.focal_dimension_x = x
        self.focal_dimension_y = y
        focal_grid_u, focal_grid_v = np.meshgrid(x, y)


        # End: Physics stuff.

        # Object properties to store intermediary objects.
        self.optics_only_pupil_planes = list()
        self.optics_only_psfs = list()
        self.optics_only_otfs = list()
        self.optics_only_mtfs = list()
        self.psfs = list()
        self.otfs = list()
        self.mtfs = list()
        self.distributed_aperture_image_planes = list()
        self.atmosphere_phase_screens = list()
        self.da_post_atmosphere_image_planes = list()
        self.distributed_aperture_images = list()
        self.pupil_planes = list()
        self.plan = dict()

        # For each exposure, build the pupil function for that exposure.
        for exposure_num in range(num_exposures):

            # Prepare a dict-valued key-value pair for this exposure.
            self.plan[exposure_num] = dict()
            with tf.name_scope("exposure_" + str(exposure_num)):

                print("Building exposure number %d." % exposure_num)

                pupil_size = (spatial_quantization,
                               spatial_quantization)

                # Build the pupil plane quantization grid for this exposure.
                optics_only_pupil_plane = tf.zeros(pupil_size,
                                                   dtype=tf.complex128)

                r0 = np.random.normal(
                    self.r0_mean,
                    self.r0_std
                )

                outer_scale = np.random.normal(
                    self.outer_scale_mean,
                    self.outer_scale_std
                )

                inner_scale = np.random.normal(
                    self.inner_scale_mean,
                    self.inner_scale_std
                )

                self.effective_dm_update_rate_hz = 1000
                exposure_interval_sec = 1 / self.effective_dm_update_rate_hz

                greenwood_time_constant_sec = np.random.normal(
                    self.greenwood_time_constant_sec_mean,
                    self.greenwood_time_constant_sec_std
                )

                # TODO: relate exposure_interval_sec and greenwood_time_constant_sec to set atmosphere_sample_scale
                atmosphere_sample_scale = 1.0

                # Build a static phase grid for reuse in this ensemble.
                static_phase_grid = make_von_karman_phase_grid(
                    r0,
                    spatial_quantization,
                    self.pupil_extent,
                    outer_scale,
                    inner_scale)

                # Initialize base sample matrices for atmosphere evolution.
                gaussian_dist = tfp.distributions.Normal(
                    loc=tf.zeros(pupil_size, dtype=tf.float64),
                    scale=tf.ones(pupil_size, dtype=tf.float64)
                )
                real_sample = tfp.distributions.Sample(
                    gaussian_dist).sample()
                img_sample = tfp.distributions.Sample(
                    gaussian_dist).sample()

                # Build the model of the pupil plane, using the Variables.
                with tf.name_scope("pupil_plane_model"):

                    for aperture_num in range(num_apertures):
                        self.plan[exposure_num][aperture_num] = dict()

                        print("Building aperture number %d." % aperture_num)

                        # Compute the subap centroid cartesian coordinates.
                        rotation = (aperture_num + 1) / self.num_apertures
                        edge_radius = radius_meters - subaperture_radius_meters
                        mu_u = edge_radius * np.cos((2 * np.pi) * rotation)
                        mu_v = edge_radius * np.sin((2 * np.pi) * rotation)

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
                            subap_zernike_coefficients_vars = self._build_zernike_coefficient_variables(
                                subap_zernike_coeffs,
                                trainable=dm_trainable
                            )

                            # Add Zernike Variables to the bookeeeping dict.
                            self.plan[exposure_num][aperture_num] = subap_zernike_coefficients_vars

                            # Render this subaperture on the pupil plane grid.
                            optics_only_pupil_plane += zernike_aperture_function_2d(
                                pupil_grid_u,
                                pupil_grid_v,
                                mu_u,
                                mu_v,
                                radius_meters,
                                subaperture_radius_meters,
                                subap_zernike_coefficients_vars
                            )

                    # TODO: Ryan, is this appropriate?
                    # Standardize the pupil plane, then physically scale it.
                    zernike_min = tf.cast(
                        tf.math.reduce_min(
                            tf.math.real(
                                optics_only_pupil_plane
                            )
                        ),
                        dtype=tf.complex128
                    )
                    zernike_max = tf.cast(
                        tf.math.reduce_max(
                            tf.math.real(
                                optics_only_pupil_plane
                            )
                        ),
                        dtype=tf.complex128
                    )
                    zernike_range = (zernike_max - zernike_min)
                    optics_only_pupil_plane = (optics_only_pupil_plane - zernike_min) / zernike_range
                    optics_only_pupil_plane = optics_only_pupil_plane * self.phase_scale_wavelengths


                    # This pupil plane is complete, now add it to the list.
                    self.optics_only_pupil_planes.append(optics_only_pupil_plane)

                    # Build the optics-only transfer functions.
                    (optics_only_psf,
                     optics_only_otf,
                     optics_only_mtf) = self._build_geometric_optics(
                        optics_only_pupil_plane
                    )

                    # Store the psf, otf, and mtf tensors for later use.
                    self.optics_only_psfs.append(optics_only_psf)
                    self.optics_only_otfs.append(optics_only_otf)
                    self.optics_only_mtfs.append(optics_only_mtf)

                    # For this exposure, develop an atmosphere.
                    with tf.name_scope("atmosphere_model"):

                        # Take a step on a random walk, evolving the atmosphere.
                        real_evo_sample = tfp.distributions.Sample(
                            gaussian_dist
                        ).sample()

                        img_evo_sample = tfp.distributions.Sample(
                            gaussian_dist
                        ).sample()

                        real_sample += atmosphere_sample_scale * real_evo_sample
                        img_sample += atmosphere_sample_scale * img_evo_sample

                        # Build and store the phase screen in radians.
                        phase_screen_radians = make_phase_screen_radians(
                            self.pupil_extent,
                            spatial_quantization,
                            static_phase_grid,
                            real_sample,
                            img_sample
                        )

                        # Convert the phase screen to wavelength-base-microns.
                        # phase_screen_mircons = (filter_wavelength_micron / (2 * np.pi)) * phase_screen_radians
                        # self.atmosphere_phase_screens.append(phase_screen_mircons)
                        phase_screen_wavelengths = (filter_wavelength_micron / (2 * np.pi)) * phase_screen_radians
                        self.atmosphere_phase_screens.append(phase_screen_wavelengths)


                        # Apply the micron displacements to the pupil.
                        # TODO: Clean this mess.
                        phase_screen_wavelenghts_masked = phase_screen_wavelengths * tf.cast(tf.math.greater(tf.math.abs(optics_only_pupil_plane), tf.zeros_like(tf.math.abs(optics_only_pupil_plane))), dtype=tf.float64)
                        da_post_atmosphere_pupil_plane = tf.cast(phase_screen_wavelenghts_masked, dtype=tf.complex128) + optics_only_pupil_plane

                        self.pupil_planes.append(
                            da_post_atmosphere_pupil_plane
                        )

                    # Finally, produce the full-path transfer functions.
                    psf, otf, mtf = self._build_geometric_optics(
                        da_post_atmosphere_pupil_plane
                    )

                    # Store the psf, otf, and mtf tensors for later evaluation.
                    self.psfs.append(psf)
                    self.otfs.append(otf)
                    self.mtfs.append(mtf)

                    distributed_aperture_image_plane = self._image_model(
                        hadamard_image_formation=hadamard_image_formation,
                        psf=psf,
                        mtf=mtf)

                    self.distributed_aperture_image_planes.append(
                        distributed_aperture_image_plane
                    )

                with tf.name_scope("sensor_model"):

                    # Apply Gaussian and Poisson process noise.
                    distributed_aperture_image = self._apply_noise(
                        distributed_aperture_image_plane,
                        gaussian_mean=self.sensor_gaussian_mean,
                        poisson_mean_arrival=self.sensor_poisson_mean_arrival
                    )

                # TODO: Hack because norm intensity is sometimes 2, breaking stuff.
                distributed_aperture_image = distributed_aperture_image / tf.reduce_max(distributed_aperture_image)

                # Finally, add the image from this pupil to the list.
                self.distributed_aperture_images.append(
                    distributed_aperture_image
                )

        # Now, construct a monolithic aperture of the same radius.
        with tf.name_scope("monolithic_aperture"):

            with tf.name_scope("pupil_plane"):

                self.optics_only_monolithic_pupil_plane = zernike_aperture_function_2d(
                    pupil_grid_u,
                    pupil_grid_v,
                    0.0,
                    0.0,
                    radius_meters,
                    radius_meters,
                    zernike_coefficients=[1.0],
                )

            # Produce the full-path transfer functions.
            (self.optics_only_monolithic_psf,
             self.optics_only_monolithic_otf,
             self.optics_only_monolithic_mtf) = self._build_geometric_optics(
                self.optics_only_monolithic_pupil_plane
            )

            self.optics_only_monolithic_aperture_image_plane = self._image_model(
                hadamard_image_formation=hadamard_image_formation,
                psf=self.optics_only_monolithic_psf,
                mtf=self.optics_only_monolithic_mtf
            )

            with tf.name_scope("optics_only_sensor_model"):
                # Apply Gaussian and Poisson process noise.
                self.optics_only_monolithic_aperture_image = self._apply_noise(
                    self.optics_only_monolithic_aperture_image_plane,
                    gaussian_mean=self.sensor_gaussian_mean,
                    poisson_mean_arrival=self.sensor_poisson_mean_arrival
                )


            # TODO: Hack because norm intensity is sometimes 2, breaking stuff.
            self.optics_only_monolithic_aperture_image = self.optics_only_monolithic_aperture_image  / tf.reduce_max(self.optics_only_monolithic_aperture_image)
            with tf.name_scope("atmosphere_model"):

                # Produce a pupil mask and multiply it by the phase screen.
                masekd_phase_screen_wavelenghts= tf.cast(
                    tf.math.greater(
                        tf.math.abs(
                            self.optics_only_monolithic_pupil_plane
                        ),
                        tf.zeros_like(
                            tf.math.abs(
                                self.optics_only_monolithic_pupil_plane
                            )
                        )
                    ),
                    dtype=tf.float64
                ) * phase_screen_wavelengths
                self.mono_post_atmosphere_pupil_plane = tf.cast(
                    masekd_phase_screen_wavelenghts,
                    dtype=tf.complex128
                ) + self.optics_only_monolithic_pupil_plane

            (self.monolithic_psf,
             self.monolithic_otf,
             self.monolithic_mtf) = self._build_geometric_optics(
                self.mono_post_atmosphere_pupil_plane
            )

            self.monolithic_aperture_image_plane = self._image_model(
                hadamard_image_formation=hadamard_image_formation,
                psf=self.monolithic_psf,
                mtf=self.monolithic_mtf
            )

            with tf.name_scope("sensor_model"):
                # Apply Gaussian and Poisson process noise.
                self.monolithic_aperture_image = self._apply_noise(
                    self.monolithic_aperture_image_plane,
                    gaussian_mean=self.sensor_gaussian_mean,
                    poisson_mean_arrival=self.sensor_poisson_mean_arrival
                )

            # TODO: Hack because norm intensity is sometimes 2, breaking stuff.
            self.monolithic_aperture_image = self.monolithic_aperture_image  / tf.reduce_max(self.monolithic_aperture_image)


    def _build_recovery_model(self,
                              distributed_aperture_images_batch,
                              filter_scale):
        """

        :param distributed_aperture_images_batch: a batch of DASIE images.
        :param filter_scale: the smallest filter scale to use.
        :return:
        """

        # Stack the the images in the ensemble to form a batch of inputs.
        distributed_aperture_images_batch = tf.stack(
            distributed_aperture_images_batch,
            axis=-1)

        recovery_model = RecoveryModel(distributed_aperture_images_batch,
                                       filter_scale,
                                       self.num_exposures,
                                       self.image_x_scale,
                                       self.image_y_scale,
                                       self.batch_size,
                                       model_type=self.recovery_model_type)

        return recovery_model.recovered_image_batch

    def plot(self, logdir=None, step=None):

        # Create the directory for the plots
        step_plot_dir = os.path.join(logdir, 'step_' + str(step) + '_plots')
        if not os.path.exists(step_plot_dir):
            os.makedirs(step_plot_dir)

        def save_and_close_current_plot(logdir, plot_name="default", dpi=600):
            fig_path = os.path.join(logdir, str(plot_name) + '.png')
            plt.gcf().set_dpi(dpi)
            plt.savefig(fig_path)
            plt.close()


        # Do a single sess.run to get all the values from a single batch.
        (pupil_planes,
         psfs,
         mtfs,
         atmosphere_phase_screens,
         optics_only_pupil_planes,
         optics_only_psfs,
         optics_only_mtfs,
         distributed_aperture_images,
         flipped_object_batch,
         object_spectrum_batch,
         object_batch,
         recovered_image,
         optics_only_monolithic_pupil_plane,
         optics_only_monolithic_psf,
         optics_only_monolithic_mtf,
         optics_only_monolithic_aperture_image,
         mono_post_atmosphere_pupil_plane,
         monolithic_psf,
         monolithic_mtf,
         monolithic_aperture_image
         ) = self.sess.run([self.pupil_planes,
                            self.psfs,
                            self.mtfs,
                            self.atmosphere_phase_screens,
                            self.optics_only_pupil_planes,
                            self.optics_only_psfs,
                            self.optics_only_mtfs,
                            self.distributed_aperture_images,
                            self.flipped_object_batch,
                            self.object_spectrum_batch,
                            self.object_batch,
                            self.recovered_image,
                            self.optics_only_monolithic_pupil_plane,
                            self.optics_only_monolithic_psf,
                            self.optics_only_monolithic_mtf,
                            self.optics_only_monolithic_aperture_image,
                            self.mono_post_atmosphere_pupil_plane,
                            self.monolithic_psf,
                            self.monolithic_mtf,
                            self.monolithic_aperture_image],
                           feed_dict={self.handle: self.valid_iterator_handle})

        # These are actually batches, so just take the first element.
        flipped_object_example = flipped_object_batch[self.example_image_index]
        object_spectrum_example = object_spectrum_batch[self.example_image_index]
        monolithic_aperture_image = monolithic_aperture_image[self.example_image_index]
        optics_only_monolithic_aperture_image = optics_only_monolithic_aperture_image[self.example_image_index]
        recovered_image = np.squeeze(recovered_image[self.example_image_index])

        # Iterate over each element of the ensemble from the DA system.
        for i, (optics_only_pupil_plane,
                optics_only_psf,
                optics_only_mtf,
                pupil_plane,
                psf,
                mtf,
                atmosphere_phase_screen,
                distributed_aperture_image) in enumerate(zip(optics_only_pupil_planes,
                                                             optics_only_psfs,
                                                             optics_only_mtfs,
                                                             pupil_planes,
                                                             psfs,
                                                             mtfs,
                                                             atmosphere_phase_screens,
                                                             distributed_aperture_images)):

            # These are actually batches, so just take the first one.
            distributed_aperture_image = distributed_aperture_image[0]

            # Build pupil extent helper variables .
            left = self.pupil_dimension_u[0]
            right = self.pupil_dimension_u[-1]
            bottom = self.pupil_dimension_v[0]
            top = self.pupil_dimension_v[-1]
            pupil_extent = [left, right, bottom, top]
            # plt.imshow(np.angle(pupil_plane),
            #            cmap='twilight_shifted',
            #            extent=[left,right,bottom,top])
            # Overlay aperture mask
            plt.imshow(np.real(pupil_plane), cmap='inferno',
                       extent=pupil_extent)
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="pupil_plane_" + str(i))

            # Plot phase angle
            # left = self.pupil_dimension_x[0]
            # right = self.pupil_dimension_x[-1]
            # bottom = self.pupil_dimension_y[0]
            # top = self.pupil_dimension_y[-1]
            # Overlay aperture mask
            # ax1 = plt.subplot(1, 2, 1)
            # ax1.set_title('np.imag')
            # plt.imshow(np.imag(pupil_plane), cmap='Greys',
            #            extent=[left, right, bottom, top])
            # plt.colorbar()
            #
            # ax2 = plt.subplot(1, 2, 2)
            # plt.imshow(np.real(pupil_plane), cmap='Greys',
            #            extent=[left, right, bottom, top])
            # ax2.set_title('np.real')
            # plt.colorbar()
            # save_and_close_current_plot(step_plot_dir,
            #                             plot_name="raw_pupil_plane_" + str(i))

            plt.imshow(np.log10(psf), cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="log_psf_" + str(i))

            plt.imshow(np.log10(mtf), cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="log_mtf_" + str(i))

            plt.imshow(atmosphere_phase_screen,
                       cmap='inferno',
                       extent=pupil_extent)
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="atmosphere_phase_screen_" + str(i))

            plt.imshow(np.real(optics_only_pupil_plane),
                       cmap='inferno',
                       extent=pupil_extent)
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="optics_only_pupil_plane_" + str(i))

            plt.imshow(np.log10(optics_only_psf), cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="log_optics_only_psf_" + str(i))

            plt.imshow(np.log10(optics_only_mtf), cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="log_optics_only_mtf_" + str(i))

            plt.imshow(distributed_aperture_image, cmap='inferno')
            plt.colorbar()
            save_and_close_current_plot(step_plot_dir,
                                        plot_name="da_image_" + str(i))

        plt.imshow(recovered_image, cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="recovered_image")

        # Overlay aperture mask
        plt.imshow(abs(mono_post_atmosphere_pupil_plane),
                   cmap='inferno',
                   extent=pupil_extent)
        plt.colorbar()
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="mono_post_atmosphere_pupil_plane"
        )

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

        plt.imshow(optics_only_monolithic_aperture_image, cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="monolithic_aperture_image")

        plt.imshow(np.abs(optics_only_monolithic_pupil_plane),
                   cmap='inferno',
                   extent=pupil_extent)
        plt.colorbar()
        plt.xlabel('u [m]')
        plt.ylabel('v [m]')
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="optics_only_monolithic_pupil_plane",
            dpi=1200
        )

        plt.imshow(np.log10(optics_only_monolithic_psf),
                   cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_optics_only_monolithic_psf")

        plt.imshow(np.log10(optics_only_monolithic_mtf), cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_optics_only_monolithic_mtf")


        plt.imshow(flipped_object_example, cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="object")

        plt.imshow(np.log10(np.abs(object_spectrum_example)), cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_object_spectrum")

        # End of standard plots, now we do some calibration/sanity checks.
        def crop_center(img, crop_x, crop_y):
            y, x = img.shape
            start_x = x // 2 - crop_x // 2
            start_y = y // 2 - crop_y // 2
            return img[start_y:start_y + crop_y, start_x:start_x + crop_x]

        center_chip = crop_center(optics_only_monolithic_psf, 128, 128)
        norm_center_chip = center_chip / np.max(center_chip)
        plt.imshow(np.log10(norm_center_chip), cmap='inferno')
        plt.colorbar()
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="log_optics_only_monolithic_psf_chip"
        )

        # Line plot
        center_line = crop_center(
            optics_only_monolithic_psf,
            1,
            self.spatial_quantization
        )
        norm_center_line = center_line / np.max(center_line)
        plt.plot(norm_center_line)
        plt.ylabel('Normalised intensity [I]')
        plt.xlabel('Focal plane distance [pixels]')
        plt.yscale('log')
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="log_optics_only_monolithic_psf_line"
        )

        # Line plot zoom
        center_line = crop_center(
            optics_only_monolithic_psf,
            1,
            64
        )
        norm_center_line = center_line / np.max(center_line)
        plt.plot(norm_center_line)
        plt.ylabel('Normalised intensity [I]')
        plt.xlabel('Focal plane distance [pixels]')
        plt.yscale('log')
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="log_optics_only_monolithic_psf_line_zoom"
        )

        pupil_diameter = self.diameter_meters  # m
        # pupil_diameter = 6.5  # m
        wavelength_meters = 1e-6 * self.filter_wavelength_micron  # m
        # wavelength = 750e-9 # m
        # pupil_grid = make_pupil_grid(self.spatial_quantization,
        #                              diameter=pupil_diameter)
        # telescope_pupil_generator = make_magellan_aperture()
        # telescope_pupil = telescope_pupil_generator(pupil_grid)

        pupil_grid = make_pupil_grid(
            self.spatial_quantization,
            (1 + self.edge_padding_factor) * pupil_diameter
        )
        oversampling_factor = 1
        aperture_circ = evaluate_supersampled(
            circular_aperture(pupil_diameter),
            pupil_grid,
            oversampling_factor
        )

        wavefront = Wavefront(aperture_circ, wavelength_meters)

        # wavefront = Wavefront(telescope_pupil, wavelength)
        # q is the number of pixels per diffraction width.
        # num_airy is half size (ie. radius) of the image in the number of diffraction widths
        # TODO: set q and num_airy from properties.
        focal_grid = make_focal_grid(q=4,
                                     num_airy=16,
                                     pupil_diameter=pupil_diameter,
                                     focal_length=self.effective_focal_length_meters,
                                     reference_wavelength=wavelength_meters)
        prop = FraunhoferPropagator(pupil_grid,
                                    focal_grid,
                                    focal_length=self.effective_focal_length_meters)

        focal_image = prop.forward(wavefront)

        # hcipy pupil
        imshow_field(aperture_circ, cmap='inferno')
        plt.colorbar()
        plt.xlabel('u [m]')
        plt.ylabel('v [m]')
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="hcipy_circular_pupil"
        )

        # hcipy circular psf line plot.
        psf = focal_image.intensity
        psf_shape = psf.grid.shape
        slicefoc = psf.shaped[:, psf_shape[0] // 2]
        slicefoc_normalised = slicefoc / psf.max()
        plt.plot(focal_grid.x.reshape(psf_shape)[0, :] * 1e6,
                 slicefoc_normalised)
        plt.xlabel('Focal plane distance [$\mu m$]')
        plt.ylabel('Normalised intensity [I]')
        plt.yscale('log')
        plt.title('hcipy circular telescope PSF')
        # plt.xlim(-10, 10)
        # plt.ylim(5e-6, 2)
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="hcipy_circular_line"
        )

        # hcipy focal psf
        imshow_field(
            np.log10(focal_image.intensity / focal_image.intensity.max()),
            vmin=-5,
            grid_units=1e-6,
            cmap = 'inferno'
        )
        plt.title('Log Normalized Intensity - HCIpy')
        plt.xlabel('Focal plane distance [um]')
        plt.ylabel('Focal plane distance [um]')
        plt.colorbar()
        save_and_close_current_plot(
            step_plot_dir,
            plot_name="hcipy_circular_focal_psf"
        )

        plt.imshow(np.log10(optics_only_monolithic_psf / np.max(optics_only_monolithic_psf)),
                   cmap='inferno',
                   vmin=-5,)
        plt.title('Log Normalized Intensity - DASIE')
        plt.xlabel('Focal plane distance [pixels]')
        plt.ylabel('Focal plane distance [pixels]')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="log_normal_optics_only_monolithic_psf")

        # Direct 2d comparison plot
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Log Normalized Intensity - HCIpy')
        imshow_field(
            np.log10(focal_image.intensity / focal_image.intensity.max()),
            vmin=-5,
            grid_units=1e-6,
            cmap = 'inferno'
        )
        plt.xlabel('Focal plane distance [um]')
        plt.ylabel('Focal plane distance [um]')
        plt.colorbar()

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Log Normalized Intensity - DASIE')
        plt.imshow(np.log10(optics_only_monolithic_psf / np.max(optics_only_monolithic_psf)),
                   cmap='inferno',
                   vmin=-5,)
        plt.xlabel('Focal plane distance [pixels]')
        plt.ylabel('Focal plane distance [pixels]')
        plt.colorbar()
        save_and_close_current_plot(step_plot_dir,
                                    plot_name="focal_psf_compare")

        # Direct line comparison plot
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('Log Normalized Intensity - HCIpy')
        psf = focal_image.intensity
        psf_shape = psf.grid.shape
        slicefoc = psf.shaped[:, psf_shape[0] // 2]
        slicefoc_normalised = slicefoc / psf.max()
        plt.plot(focal_grid.x.reshape(psf_shape)[0, :] * 1e6,
                 slicefoc_normalised)
        plt.xlabel('Focal plane distance [$\mu m$]')
        plt.ylabel('Normalised intensity [I]')
        plt.yscale('log')
        plt.title('hcipy circular telescope PSF')
        # plt.xlim(-10, 10)
        # plt.ylim(5e-6, 2)


        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('Log Normalized Intensity - DASIE')
        center_line = crop_center(
            optics_only_monolithic_psf,
            1,
            self.spatial_quantization
        )
        norm_center_line = center_line / np.max(center_line)
        plt.plot(norm_center_line)
        plt.ylabel('Normalised intensity [I]')
        plt.xlabel('Focal plane distance [pixels]')
        plt.yscale('log')

        save_and_close_current_plot(step_plot_dir,
                                    plot_name="focal_psf_line_compare")

        #
        # ax3 = plt.subplot(1, 3, 3)
        # plt.plot(norm_center_line - slicefoc_normalised)
        # plt.ylabel('Residual Normalised Intensity  [I]')
        # plt.xlabel('Focal plane distance [pixels]')
        # plt.yscale('log')

        # PSF profile residuals.

    def save(self, save_file_path):
        """
        This function saves a dictionary comprising all weights, kwargs, and
        the git hash so that any model trained from the same commit can be
        restored.
        :return: None
        """

        save_dict = dict()
        save_dict["kwargs"] = dict()
        for key, value in self.kwargs.items():
            if is_jsonable(value):
                save_dict["kwargs"][key] = value

        save_dict["variables"] = dict()
        print(
            "Found %i variables to save." %
            len(tf.compat.v1.trainable_variables())
        )

        variable_names = [v.name for v in tf.compat.v1.trainable_variables()]
        variable_values = self.sess.run(tf.compat.v1.trainable_variables())
        for name, value in zip(variable_names, variable_values):
            save_dict["variables"][name] = value

        # json.dump(save_dict, open(json_file, 'w'))
        json.dump(save_dict, open(save_file_path, 'w'), cls=NpEncoder)

        return None

    def restore(self, restore_file_path):
        """
        This function loads a dictionary comprising all weights and kwargs,
        enabling their use to restore the saved model if the model is the same.
        :return: None
        """

        restore_dict = json.load(open(restore_file_path, 'r'))
        for v in tf.compat.v1.trainable_variables():

            v.load(restore_dict["variables"][v.name], self.sess)

        return None

    def train(self):

        return self.sess.run(
            [self.loss,
             self.monolithic_aperture_image_mse,
             self.distributed_aperture_image_mse,
             self.da_mse_mono_mse_ratio,
             self.monolithic_aperture_image_ssim,
             self.distributed_aperture_image_ssim,
             self.da_ssim_mono_ssim_ratio,
             self.monolithic_aperture_image_psnr,
             self.distributed_aperture_image_psnr,
             self.da_psnr_mono_psnr_ratio,
             self.optimize],
            feed_dict={self.handle: self.train_iterator_handle})

    def validate(self):

        return self.sess.run(
            [self.loss,
             self.monolithic_aperture_image_mse,
             self.distributed_aperture_image_mse,
             self.da_mse_mono_mse_ratio,
             self.monolithic_aperture_image_ssim,
             self.distributed_aperture_image_ssim,
             self.da_ssim_mono_ssim_ratio,
             self.monolithic_aperture_image_psnr,
             self.distributed_aperture_image_psnr,
             self.da_psnr_mono_psnr_ratio,
            ],
            feed_dict={self.handle: self.valid_iterator_handle})

    def recover(self, images):

        feed_dict = dict()
        for (var, image) in zip(self.distributed_aperture_images, images):
            feed_dict[var] = np.expand_dims(image, axis=0)

        return self.sess.run([self.recovered_image],
                             feed_dict=feed_dict)
