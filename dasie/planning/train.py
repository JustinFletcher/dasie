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
    results_dict["results"]["valid_loss_list"] = list()
    results_dict["results"]["valid_dist_mse_list"] = list()
    results_dict["results"]["valid_mono_mse_list"] = list()
    results_dict["results"]["valid_mse_ratio_list"] = list()
    results_dict["results"]["train_epoch_time_list"] = list()

    sess.run(tf.compat.v1.global_variables_initializer())

    # Enter the main training loop.
    for i in range(num_steps):

        # Start the epoch with a model save, then validate, and finally train.
        print("Beginning Epoch %d" % i)
        tf.summary.experimental.set_step(i)

        # First, plot the model status if requested
        print("Epoch %d Plots Plotting." % i)

        if save_plot:
            if (i % plot_periodicity) == 0:

                sess.run(valid_dataset_initializer)
                print("Plotting...")
                dasie_model.plot(logdir=logdir,
                                 show_plot=show_plot,
                                 step=i)
                print("Plotting completed.")

        print("Epoch %d Plots Plotted." % i)

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


        print("Epoch %d Model Saving." % i)

        # TODO: Dump recovery model weights and Zernike plan here.
        dasie_model.save(logdir=logdir,)

        print("Epoch %d Model Saved." % i)

        # Write the results dict for this epoch.
        json_file = os.path.join(logdir, "results_" + str(i) + ".json")
        json.dump(results_dict, open(json_file, 'w'))
        # data = json.load(open("file_name.json"))

        # TODO: Refactor to report at the step scale for training.
        # Execute the summary writer ops to write their values.
        sess.run(valid_dataset_initializer)
        feed_dict = {dasie_model.handle: dasie_model.valid_iterator_handle}
        sess.run(all_summary_ops, feed_dict=feed_dict)
        sess.run(step_update, feed_dict=feed_dict)
        sess.run(writer_flush, feed_dict=feed_dict)


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
    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    dir_name = timestamp + "_" + str(os.getpid())
    save_dir = os.path.join(".", "logs", dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # Compute the scaling factor from meters to alpha for a GG PDF over meters.
    # alpha = np.log(-np.log(epsilon)) / np.log(beta) * ap_radius_meters
    # alpha = alpha / (flags.num_subapertures)
    # epsilon = 1e-15
    ap_radius_meters = (flags.aperture_diameter_meters / 2)
    subap_radius_meters = ((ap_radius_meters * np.sin(np.pi / flags.num_subapertures)) - (flags.subaperture_spacing_meters / 2)) / (1 + np.sin(np.pi / flags.num_subapertures))
    # meters_to_alpha = 1 /  (2 * (np.log(-np.log(epsilon)) / np.log(beta)))
    # subap_alpha = subap_radius_meters * meters_to_alpha / 2
    # monolithic_alpha = flags.aperture_diameter_meters * meters_to_alpha * 2
    subap_area = np.pi * (subap_radius_meters ** 2)
    total_subap_area = subap_area * flags.num_subapertures
    mono_ap_area = np.pi * (ap_radius_meters ** 2)
    mono_to_dist_aperture_ratio = mono_ap_area / total_subap_area
    print(mono_to_dist_aperture_ratio)

    # TODO: [Ryan and Matthew] I could use help here.
    # TODO: Document how distance to the target is quantified implicitly.
    # TODO: Document how extent of the target is quantified implicitly.
    base_results_dict = vars(flags)
    base_results_dict["mono_to_dist_aperture_ratio"] = mono_to_dist_aperture_ratio
    base_results_dict["ap_radius_meters"] = ap_radius_meters
    base_results_dict["subap_radius_meters"] = subap_radius_meters
    # TODO: Append all other derived optical information here!

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


    parser.add_argument('--subaperture_spacing_meters',
                        type=float,
                        default=0.1,
                        help='Meters of space between subapertures.')


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