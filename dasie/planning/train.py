import os
import math
import time
import copy
import json
import math
import glob
import codecs
import joblib
import psutil
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

import zernike
from plot_dasie_performance import *
from recovery_models import RecoveryModel
from differentiable_dasie import DASIEModel


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
    results_dict["results"]["train_dist_ssim_list"] = list()
    results_dict["results"]["train_mono_ssim_list"] = list()
    results_dict["results"]["train_ssim_ratio_list"] = list()
    results_dict["results"]["train_dist_psnr_list"] = list()
    results_dict["results"]["train_mono_psnr_list"] = list()
    results_dict["results"]["train_psnr_ratio_list"] = list()
    results_dict["results"]["valid_loss_list"] = list()
    results_dict["results"]["valid_dist_mse_list"] = list()
    results_dict["results"]["valid_mono_mse_list"] = list()
    results_dict["results"]["valid_mse_ratio_list"] = list()
    results_dict["results"]["valid_dist_ssim_list"] = list()
    results_dict["results"]["valid_mono_ssim_list"] = list()
    results_dict["results"]["valid_ssim_ratio_list"] = list()
    results_dict["results"]["valid_dist_psnr_list"] = list()
    results_dict["results"]["valid_mono_psnr_list"] = list()
    results_dict["results"]["valid_psnr_ratio_list"] = list()
    results_dict["results"]["train_epoch_time_list"] = list()

    sess.run(tf.compat.v1.global_variables_initializer())

    # Enter the main training loop.
    for i in range(num_steps):

        # Start the epoch with a model save, then validate, and finally train.
        print("Beginning Epoch %d" % i)
        tf.summary.experimental.set_step(i)

        # print("Epoch %d Model Restoring." % i)
        # dasie_model.restore(save_file_path)

        # print("Epoch %d Model Restored." % i)

        # First, plot the model status if requested
        print("Epoch %d Plots Plotting." % i)
        if save_plot:
            if (i % plot_periodicity) == 0:

                sess.run(valid_dataset_initializer)
                print("Plotting...")
                dasie_model.plot(logdir=logdir,
                                 step=i)

                if i > 0:
                    plot_dasie_performance(
                        logdir=logdir,
                        step=i,
                    )
                print("Plotting completed.")
            else:

                print("No plots this epoch.")


        print("Epoch %d Plots Plotted." % i)

        # Initialize the validation dataset iterator to prepare for validation.
        print("Epoch %d Validation Beginning..." % i)
        sess.run(valid_dataset_initializer)

        # Initialize the validation display metrics.
        valid_loss = 0.0
        valid_monolithic_aperture_image_mse = 0.0
        valid_distributed_aperture_image_mse = 0.0
        valid_da_mse_mono_mse_ratio = 0.0
        valid_monolithic_aperture_image_ssim = 0.0
        valid_distributed_aperture_image_ssim = 0.0
        valid_da_ssim_mono_ssim_ratio = 0.0
        valid_monolithic_aperture_image_psnr = 0.0
        valid_distributed_aperture_image_psnr = 0.0
        valid_da_psnr_mono_psnr_ratio = 0.0

        valid_steps = 0.0

        # Validate by looping an calling validate batches, until...
        try:
            while True:
                # Execute one validation step.
                (step_valid_loss,
                 step_valid_monolithic_aperture_image_mse,
                 step_valid_distributed_aperture_image_mse,
                 step_valid_da_mse_mono_mse_ratio,
                 step_valid_monolithic_aperture_image_ssim,
                 step_valid_distributed_aperture_image_ssim,
                 step_valid_da_ssim_mono_ssim_ratio,
                 step_valid_monolithic_aperture_image_psnr,
                 step_valid_distributed_aperture_image_psnr,
                 step_valid_da_psnr_mono_psnr_ratio) = dasie_model.validate()

                # Increment all of our metrics.
                # TODO: Eventually refactor to summaries.
                valid_loss += step_valid_loss
                valid_distributed_aperture_image_mse += step_valid_distributed_aperture_image_mse
                valid_monolithic_aperture_image_mse += step_valid_monolithic_aperture_image_mse
                valid_da_mse_mono_mse_ratio += step_valid_da_mse_mono_mse_ratio
                valid_distributed_aperture_image_ssim += step_valid_distributed_aperture_image_ssim
                valid_monolithic_aperture_image_ssim += step_valid_monolithic_aperture_image_ssim
                valid_da_ssim_mono_ssim_ratio += step_valid_da_ssim_mono_ssim_ratio
                valid_distributed_aperture_image_psnr += step_valid_distributed_aperture_image_psnr
                valid_monolithic_aperture_image_psnr += step_valid_monolithic_aperture_image_psnr
                valid_da_psnr_mono_psnr_ratio += step_valid_da_psnr_mono_psnr_ratio
                valid_steps += 1.0
                print("Validation step %d MSE_{m/d}=%f" % (int(valid_steps), (1 / step_valid_da_mse_mono_mse_ratio)))

        # ...there are no more validate batches.
        except tf.errors.OutOfRangeError:

            # Compute the epoch results.
            mean_valid_loss = valid_loss / valid_steps
            mean_valid_distributed_aperture_image_mse = valid_distributed_aperture_image_mse / valid_steps
            mean_valid_monolithic_aperture_image_mse = valid_monolithic_aperture_image_mse / valid_steps
            mean_valid_da_mse_mono_mse_ratio = valid_da_mse_mono_mse_ratio / valid_steps
            mean_valid_distributed_aperture_image_ssim = valid_distributed_aperture_image_ssim / valid_steps
            mean_valid_monolithic_aperture_image_ssim = valid_monolithic_aperture_image_ssim / valid_steps
            mean_valid_da_ssim_mono_ssim_ratio = valid_da_ssim_mono_ssim_ratio / valid_steps
            mean_valid_distributed_aperture_image_psnr = valid_distributed_aperture_image_psnr / valid_steps
            mean_valid_monolithic_aperture_image_psnr = valid_monolithic_aperture_image_psnr / valid_steps
            mean_valid_da_psnr_mono_psnr_ratio = valid_da_psnr_mono_psnr_ratio / valid_steps

            # Store the epoch results.
            results_dict["results"]["valid_loss_list"].append(mean_valid_loss)
            results_dict["results"]["valid_dist_mse_list"].append(mean_valid_distributed_aperture_image_mse)
            results_dict["results"]["valid_mono_mse_list"].append(mean_valid_monolithic_aperture_image_mse)
            results_dict["results"]["valid_mse_ratio_list"].append(mean_valid_da_mse_mono_mse_ratio)
            results_dict["results"]["valid_dist_ssim_list"].append(mean_valid_distributed_aperture_image_ssim)
            results_dict["results"]["valid_mono_ssim_list"].append(mean_valid_monolithic_aperture_image_ssim)
            results_dict["results"]["valid_ssim_ratio_list"].append(mean_valid_da_ssim_mono_ssim_ratio)
            results_dict["results"]["valid_dist_psnr_list"].append(mean_valid_distributed_aperture_image_psnr)
            results_dict["results"]["valid_mono_psnr_list"].append(mean_valid_monolithic_aperture_image_psnr)
            results_dict["results"]["valid_psnr_ratio_list"].append(mean_valid_da_psnr_mono_psnr_ratio)

            print("Validation Loss: %f" % mean_valid_loss)
            print("Validation DA MSE: %f" % mean_valid_distributed_aperture_image_mse)
            print("Validation DA SSIM: %f" % mean_valid_distributed_aperture_image_ssim)
            print("Validation DA PSNR: %f" % mean_valid_distributed_aperture_image_psnr)
            pass

        print("Epoch %d Validation Complete." % i)

        print("Epoch %d Results Saving." % i)
        # Write the results dict for this epoch.
        json_file = os.path.join(logdir, "results_" + str(i) + ".json")
        json.dump(results_dict, open(json_file, 'w'))
        # data = json.load(open("file_name.json"))
        print("Epoch %d Results Saved." % i)

        print("Epoch %d Model Saving." % i)
        save_file_path = os.path.join(logdir, "model_save_" + str(i) + ".json")
        dasie_model.save(save_file_path)
        print("Epoch %d Model Saved." % i)

        # TODO: Refactor to report at the step scale for training.
        # Execute the summary writer ops to write their values.
        sess.run(valid_dataset_initializer)
        feed_dict = {dasie_model.handle: dasie_model.valid_iterator_handle}
        sess.run(all_summary_ops, feed_dict=feed_dict)
        sess.run(step_update, feed_dict=feed_dict)
        sess.run(writer_flush, feed_dict=feed_dict)


        # Initialize the training dataset iterator to prepare for training.
        print("Epoch %d Training Beginning." % i)
        sess.run(train_dataset_initializer)

        # Initialize the training display metrics.
        train_loss = 0.0
        train_monolithic_aperture_image_mse = 0.0
        train_distributed_aperture_image_mse = 0.0
        train_da_mse_mono_mse_ratio = 0.0
        train_monolithic_aperture_image_ssim = 0.0
        train_distributed_aperture_image_ssim = 0.0
        train_da_ssim_mono_ssim_ratio = 0.0
        train_monolithic_aperture_image_psnr = 0.0
        train_distributed_aperture_image_psnr = 0.0
        train_da_psnr_mono_psnr_ratio = 0.0
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
                 step_train_monolithic_aperture_image_ssim,
                 step_train_distributed_aperture_image_ssim,
                 step_train_da_ssim_mono_ssim_ratio,
                 step_train_monolithic_aperture_image_psnr,
                 step_train_distributed_aperture_image_psnr,
                 step_train_da_psnr_mono_psnr_ratio,
                 _) = dasie_model.train()

                print("Memory usage = " + str(psutil.Process(os.getpid()).memory_info().rss / 1000000000) + " GB. ")

                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                # Increment all of our metrics.
                # TODO: Eventually refactor to summaries.
                train_loss += step_train_loss
                train_distributed_aperture_image_mse += step_train_distributed_aperture_image_mse
                train_monolithic_aperture_image_mse += step_train_monolithic_aperture_image_mse
                train_da_mse_mono_mse_ratio += step_train_da_mse_mono_mse_ratio
                train_distributed_aperture_image_ssim += step_train_distributed_aperture_image_ssim
                train_monolithic_aperture_image_ssim += step_train_monolithic_aperture_image_ssim
                train_da_ssim_mono_ssim_ratio += step_train_da_ssim_mono_ssim_ratio
                train_distributed_aperture_image_psnr += step_train_distributed_aperture_image_psnr
                train_monolithic_aperture_image_psnr += step_train_monolithic_aperture_image_psnr
                train_da_psnr_mono_psnr_ratio += step_train_da_psnr_mono_psnr_ratio
                train_steps += 1.0
                print("...step_train_loss = %f..." % step_train_loss)
                print("...step_train_da_mse_mono_mse_ratio = %f..." % step_train_da_mse_mono_mse_ratio)
                print("...step_train_da_ssim_mono_ssim_ratio = %f..." % step_train_da_ssim_mono_ssim_ratio)
                print("...step_train_da_psnr_mono_psnr_ratio = %f..." % step_train_da_psnr_mono_psnr_ratio)
                print("...train step %d complete in %f sec." % (train_steps, step_time))

        # OutOfRangeError indicates we've finished the iterator, so report out.
        except tf.errors.OutOfRangeError:

            end_time = time.time()
            train_epoch_time = end_time - start_time
            mean_train_loss = train_loss / train_steps
            mean_train_distributed_aperture_image_mse = train_distributed_aperture_image_mse / train_steps
            mean_train_monolithic_aperture_image_mse = train_monolithic_aperture_image_mse / train_steps
            mean_train_da_mse_mono_mse_ratio = train_da_mse_mono_mse_ratio / train_steps
            mean_train_distributed_aperture_image_ssim = train_distributed_aperture_image_ssim / train_steps
            mean_train_monolithic_aperture_image_ssim = train_monolithic_aperture_image_ssim / train_steps
            mean_train_da_ssim_mono_ssim_ratio = train_da_ssim_mono_ssim_ratio / train_steps
            mean_train_distributed_aperture_image_psnr = train_distributed_aperture_image_psnr / train_steps
            mean_train_monolithic_aperture_image_psnr = train_monolithic_aperture_image_psnr / train_steps
            mean_train_da_psnr_mono_psnr_ratio = train_da_psnr_mono_psnr_ratio / train_steps

            results_dict["results"]["train_loss_list"].append(mean_train_loss)
            results_dict["results"]["train_dist_mse_list"].append(mean_train_distributed_aperture_image_mse)
            results_dict["results"]["train_mono_mse_list"].append(mean_train_monolithic_aperture_image_mse)
            results_dict["results"]["train_mse_ratio_list"].append(mean_train_da_mse_mono_mse_ratio)
            results_dict["results"]["train_dist_ssim_list"].append(mean_train_distributed_aperture_image_ssim)
            results_dict["results"]["train_mono_ssim_list"].append(mean_train_monolithic_aperture_image_ssim)
            results_dict["results"]["train_ssim_ratio_list"].append(mean_train_da_ssim_mono_ssim_ratio)
            results_dict["results"]["train_dist_psnr_list"].append(mean_train_distributed_aperture_image_psnr)
            results_dict["results"]["train_mono_psnr_list"].append(mean_train_monolithic_aperture_image_psnr)
            results_dict["results"]["train_psnr_ratio_list"].append(mean_train_da_psnr_mono_psnr_ratio)
            results_dict["results"]["train_epoch_time_list"].append(train_epoch_time)

            print("Mean Train Loss: %f" % mean_train_loss)
            print("Mean Train DA MSE: %f" % mean_train_distributed_aperture_image_mse)
            print("Mean Train DA SSIM: %f" % mean_train_distributed_aperture_image_ssim)
            print("Mean Train DA PSNR: %f" % mean_train_distributed_aperture_image_psnr)

            pass

        print("Epoch %d Training Complete." % i)


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

def inaturalist_parse_function(example_proto):
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
        "id": tf.io.VarLenFeature(dtype=tf.string),
        "name": tf.io.VarLenFeature(dtype=tf.string),
        "common_name": tf.io.VarLenFeature(dtype=tf.string),
        "supercategory": tf.io.VarLenFeature(dtype=tf.string),
        "kingdom": tf.io.VarLenFeature(dtype=tf.string),
        "phylum": tf.io.VarLenFeature(dtype=tf.string),
        "class": tf.io.VarLenFeature(dtype=tf.string),
        "order": tf.io.VarLenFeature(dtype=tf.string),
        "family": tf.io.VarLenFeature(dtype=tf.string),
        "genus": tf.io.VarLenFeature(dtype=tf.string),
        "specific_epithet": tf.io.VarLenFeature(dtype=tf.string),
        "image_dir_name": tf.io.VarLenFeature(dtype=tf.string),

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
    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    dir_name = timestamp + "_" + str(os.getpid())
    save_dir = os.path.join(flags.logdir, dir_name)
    os.makedirs(save_dir, exist_ok=True)

    if not flags.atmosphere:
        flags.r0_std = 0.0
        flags.r0_mean = 1000000.0

    # Physical computations from flags.
    # TODO: Fix this.
    ap_radius_meters = (flags.aperture_diameter_meters / 2)
    # subap_radius_meters = (2 * (((ap_radius_meters) * np.sin(np.pi / flags.num_subapertures)) - ((flags.subaperture_spacing_meters / 2))) / (1 + np.sin(np.pi / flags.num_subapertures))) / (1.0 + flags.edge_padding_factor)
    subap_radius_meters = ((((ap_radius_meters) * np.sin(np.pi / flags.num_subapertures)) - ((flags.subaperture_spacing_meters / 2))) / (1 + np.sin(np.pi / flags.num_subapertures)))
    subap_area = np.pi * (subap_radius_meters ** 2)
    total_subap_area = subap_area * flags.num_subapertures
    mono_ap_area = np.pi * (ap_radius_meters ** 2)
    mono_to_dist_aperture_ratio = mono_ap_area / total_subap_area
    object_distance_meters = flags.object_plane_extent_meters / (np.tan((flags.field_of_view_degrees / 2)  * (np.pi / 180)))

    base_results_dict = vars(flags)
    base_results_dict["mono_to_dist_aperture_ratio"] = mono_to_dist_aperture_ratio
    base_results_dict["ap_radius_meters"] = ap_radius_meters
    base_results_dict["subap_radius_meters"] = subap_radius_meters
    base_results_dict["object_scale_meters"] = flags.object_plane_extent_meters
    base_results_dict["object_distance_meters"] = object_distance_meters

    print("mono_to_dist_aperture_ratio = " + str(mono_to_dist_aperture_ratio))
    print("object_distance_meters = " + str(object_distance_meters))
    print("flags.object_plane_extent_meters = " + str(flags.object_plane_extent_meters))
    print("flags.field_of_view_degrees = " + str(flags.field_of_view_degrees))
    # TODO: Append all other derived optical information here!

    # Map our dataset name to relative locations and parse functions.
    if flags.dataset_name == "speedplus":

        print("Selected dataset is speedplus.")
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "speedplus_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "speedplus_tfrecords", "valid")
        example_image_index = 0

    elif flags.dataset_name == "speedplus_synthetic":

        print("Selected dataset is speedplus_synthetic.")
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "speedplus_synthetic_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "speedplus_synthetic_tfrecords", "valid")
        example_image_index = 2

    elif flags.dataset_name == "inria_holiday":

        print("Selected dataset is inria_holiday.")
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(flags.dataset_root, "inria_holiday_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "inria_holiday_tfrecords", "valid")
        example_image_index = 0


    elif flags.dataset_name == "speedplus_one":
        parse_function = speedplus_parse_function
        print("Selected dataset is speedplus_one.")
        train_data_dir = os.path.join(flags.dataset_root, "speedplus_one_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "speedplus_one_tfrecords", "valid")
        example_image_index = 0


    elif flags.dataset_name == "speedplus_overfit":
        parse_function = speedplus_parse_function
        print("Selected dataset is speedplus_overfit.")
        train_data_dir = os.path.join(flags.dataset_root, "speedplus_overfit_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "speedplus_overfit_tfrecords", "valid")
        example_image_index = 0


    elif flags.dataset_name == "usaf1951":
        parse_function = speedplus_parse_function
        print("Selected dataset is usaf1951.")
        train_data_dir = os.path.join(flags.dataset_root, "usaf1951_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "usaf1951_tfrecords", "valid")
        example_image_index = 0


    elif flags.dataset_name == "inaturalist":
        parse_function = inaturalist_parse_function
        print("Selected dataset is inaturalist.")
        train_data_dir = os.path.join(flags.dataset_root, "inaturalist_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "inaturalist_tfrecords", "valid")
        example_image_index = 0


    else:
        parse_function = speedplus_parse_function
        print("Selected dataset is the default, onesat.")
        train_data_dir = os.path.join(flags.dataset_root, "onesat_example_tfrecords", "train")
        valid_data_dir = os.path.join(flags.dataset_root, "onesat_example_tfrecords", "valid")
        example_image_index = 0



    # Set the crop size to the spatial quantization scale.
    if flags.crop:
        crop_size = flags.spatial_quantization
    else:
        crop_size = None

    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:

        print("\n\n\n\n\n\n\n\n\n Session Created \n\n\n\n\n\n\n\n\n")

        # Set all our seeds.
        np.random.seed(flags.random_seed)
        tf.compat.v1.set_random_seed(flags.random_seed)

        # Make summary management variables.
        step = tf.Variable(0, dtype=tf.int64)
        step_update = step.assign_add(1)
        tf.summary.experimental.set_step(step)
        writer = tf.summary.create_file_writer(save_dir)

        print("\n\n\n\n\n\n\n\n\n Building Dataset... \n\n\n\n\n\n\n\n\n")
        # Build our datasets.,
        #             cache_dataset_memory=flags.cache_dataset_memory
        train_dataset = DatasetGenerator(train_data_dir,
                                         parse_function=parse_function,
                                         augment=False,
                                         shuffle=True,
                                         crop_size=crop_size,
                                         batch_size=flags.batch_size,
                                         num_threads=flags.num_dataset_threads,
                                         buffer=flags.dataset_buffer_len,
                                         encoding_function=None,
                                         cache_dataset_memory=flags.cache_dataset_memory,
                                         cache_dataset_file=False,
                                         cache_path="",
                                         max_elements=flags.max_dataset_elements)

        # We create a tf.data.Dataset object wrapping the valid dataset here.
        valid_dataset = DatasetGenerator(valid_data_dir,
                                         parse_function=parse_function,
                                         augment=False,
                                         shuffle=False,
                                         crop_size=crop_size,
                                         batch_size=flags.batch_size,
                                         num_threads=flags.num_dataset_threads,
                                         buffer=flags.dataset_buffer_len,
                                         encoding_function=None,
                                         cache_dataset_memory=flags.cache_dataset_memory,
                                         cache_dataset_file=False,
                                         cache_path="",
                                         max_elements=flags.max_dataset_elements)


        print("\n\n\n\n\n\n\n\n\n Dataset Built... \n\n\n\n\n\n\n\n\n")

        # Get the image shapes stored during dataset construction.
        image_x_scale = train_dataset.image_shape[0]
        image_y_scale = train_dataset.image_shape[1]

        # Manual debug here, to diagnose data problems.
        plot_data = False
        if plot_data:

            for i in range(16):

                # Generate the iterator for the train dataset.
                train_iterator = train_dataset.get_iterator()
                train_dataset_batch = train_iterator.get_next()
                train_dataset_initializer = train_dataset.get_initializer()
                sess.run(train_dataset_initializer)

                # Generate the iterator for the validation dataset.
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

        # Build a DA model.
        dasie_model = DASIEModel(
            sess,
            writer=writer,
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
            subaperture_radius_meters=subap_radius_meters,
            edge_padding_factor=flags.edge_padding_factor,
            recovery_model_filter_scale=flags.recovery_model_filter_scale,
            loss_name=flags.loss_name,
            num_zernike_indices=flags.num_zernike_indices,
            hadamard_image_formation=flags.hadamard_image_formation,
            zernike_debug=flags.zernike_debug,
            ap_radius_meters=ap_radius_meters,
            subap_radius_meters=subap_radius_meters,
            subap_area=subap_area,
            total_subap_area=total_subap_area,
            mono_ap_area=mono_ap_area,
            mono_to_dist_aperture_ratio=mono_to_dist_aperture_ratio,
            object_plane_extent_meters=flags.object_plane_extent_meters,
            object_distance_meters=object_distance_meters,
            zernike_init_type=flags.zernike_init_type,
            filter_wavelength_micron=flags.filter_wavelength_micron,
            sensor_gaussian_mean=flags.sensor_gaussian_mean,
            sensor_poisson_mean_arrival=flags.sensor_poisson_mean_arrival,
            dm_stroke_microns=flags.dm_stroke_microns,
            focal_extent_meters=flags.focal_extent_meters,
            r0_mean=flags.r0_mean,
            r0_std=flags.r0_std,
            outer_scale_mean=flags.outer_scale_mean,
            outer_scale_std=flags.outer_scale_std,
            inner_scale_mean=flags.inner_scale_mean,
            inner_scale_std=flags.inner_scale_std,
            greenwood_time_constant_sec_mean=flags.greenwood_time_constant_sec_mean,
            greenwood_time_constant_sec_std=flags.greenwood_time_constant_sec_std,
            effective_focal_length_meters=flags.effective_focal_length_meters,
            recovery_model_type=flags.recovery_model_type,
            example_image_index=example_image_index,
            plan_diversity_regularization=flags.plan_diversity_regularization,
            plan_diversity_alpha=flags.plan_diversity_alpha,
            atmosphere=flags.atmosphere
        )



        # Merge all the summaries from the graphs, flush and init the nodes.
        all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
        writer_flush = writer.flush()
        sess.run([writer.init(), step.initializer])

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

    parser.add_argument('--subaperture_spacing_meters',
                        type=float,
                        default=0.07,
                        help='Meters of space between subapertures.')

    parser.add_argument('--num_zernike_indices',
                        type=int,
                        default=1,
                        help='Number of Zernike terms to simulate.')

    parser.add_argument('--edge_padding_factor',
                        type=float,
                        default=0.1,
                        help='Factor to expand the pupil simulation grid.')

    parser.add_argument('--aperture_diameter_meters',
                        type=float,
                        default=3.0,
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

    parser.add_argument('--num_dataset_threads',
                        type=int,
                        default=2,
                        help='Number dataset read threads.')

    parser.add_argument('--dataset_buffer_len',
                        type=int,
                        default=4,
                        help='Size of dataset read buffer in examples.')

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

    parser.add_argument("--cache_dataset_memory", action='store_true',
                        default=False,
                        help='If true, cache the dataset in memory.')

    parser.add_argument("--hadamard_image_formation", action='store_true',
                        default=False,
                        help='If true, use MTF, image spectrum product, else \
                              use PSF convolution.')

    parser.add_argument('--dataset_root', type=str,
                        default="..\\data",
                        help='Path to a directory holding all datasets.')

    parser.add_argument('--dataset_name', type=str,
                        default="speedplus",
                        help='Path to the train data TFRecords directory.')

    parser.add_argument('--object_plane_extent_meters', type=float,
                        default=1.0,
                        help='The size of the object plane in meters.')


    parser.add_argument("--plan_diversity_regularization", action='store_true',
                        default=False,
                        help='If true, apply a similarity penalty to loss.')

    parser.add_argument('--plan_diversity_alpha', type=float,
                        default=0.5,
                        help='A balance parameter.')

    # One arcminute = 0.0166667. 67 arcminutes is 0.0186111.
    # 0.0001145 ~= 1m at 1 Mm (LEO)
    parser.add_argument('--field_of_view_degrees',
                        type=float,
                        default=0.001173,
                        help='The FOV of the optical system in degrees.')

    parser.add_argument('--recovery_model_filter_scale',
                        type=int,
                        default=16,
                        help='Base filter size for recovery model.')

    parser.add_argument('--max_dataset_elements',
                        type=int,
                        default=None,
                        help='If provided, limit the number of elements.')

    parser.add_argument('--recovery_model_type',
                        type=str,
                        default="tseng2021neural",
                        help='The type of recovery model to train.')



    parser.add_argument("--zernike_debug", action='store_true',
                        default=False,
                        help="If true, each subaperture is constrained such \
                              that only the Zernike coefficient with the same \
                              index as the subaperture index is none-zero.")

    parser.add_argument("--zernike_init_type",  type=str,
                        default="np.random.uniform",
                        help="An np function signature to use with default \
                              arguments to init the zernike plan variables.")

    parser.add_argument('--filter_wavelength_micron', type=float,
                        default=1.0,
                        help='The wavelength of monochromatic light used \
                              in this simulation.')

    parser.add_argument('--sensor_gaussian_mean',
                        type=float,
                        default=1e-5,
                        help='The mean for the Gaussian sensor noise.')

    parser.add_argument('--sensor_poisson_mean_arrival',
                        type=float,
                        default=4e-5,
                        help='The Poisson mean arrival time for sensor noise.')

    parser.add_argument('--dm_stroke_microns',
                        type=float,
                        default=4.0,
                        help='The full stroke of the modeled DM in microns')

    parser.add_argument('--focal_extent_meters',
                        type=float,
                        default=0.004096,
                        help='The extent of the square focal plane in meters.')

    parser.add_argument('--r0_mean',
                        type=float,
                        default=0.20,
                        help='The mean of the normal distribution of r0.')

    parser.add_argument('--atmosphere',
                        action='store_true',
                        default=False,
                        help='If false, sets r0 mean and std to 0.0.')

    parser.add_argument('--r0_std',
                        type=float,
                        default=0.0,
                        help='The std of the normal distribution of r0.')

    parser.add_argument('--outer_scale_mean',
                        type=float,
                        default=2000.0,
                        help='The std of the normal distribution of outer \
                              scale for the Von Karman atmosphere model. \
                              see: https://arxiv.org/ftp/arxiv/papers/ \
                              1112/1112.6033.pdf')

    parser.add_argument('--outer_scale_std',
                        type=float,
                        default=0.0,
                        help='The std of the normal distribution of outer \
                              scale for the Von Karman atmosphere model. \
                              see: https://arxiv.org/ftp/arxiv/papers/ \
                              1112/1112.6033.pdf')

    parser.add_argument('--inner_scale_mean',
                        type=float,
                        default=0.008,
                        help='The mean of the normal distribution of inner \
                              scale for the Von Karman atmosphere model. \
                              see: https://arxiv.org/ftp/arxiv/papers/ \
                              1112/1112.6033.pdf')

    parser.add_argument('--inner_scale_std',
                        type=float,
                        default=0.0,
                        help='The std of the normal distribution of inner \
                              scale for the Von Karman atmosphere model. \
                              see: https://arxiv.org/ftp/arxiv/papers/ \
                              1112/1112.6033.pdf')

    parser.add_argument('--greenwood_time_constant_sec_mean',
                        type=float,
                        default=1.0,
                        help='The mean of the normal distribution of the \
                              Greenwood time constant.')

    parser.add_argument('--greenwood_time_constant_sec_std',
                        type=float,
                        default=0.0,
                        help='The std of the normal distribution of the \
                              Greenwood time constant.')

    parser.add_argument('--effective_focal_length_meters',
                        type=float,
                        default=200.0,
                        help='The effective focal length in meters.')


    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)