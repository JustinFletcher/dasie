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

# mlflow experiment tracking and pyrallis config manager
from pathlib import Path
from recorder import Recorder
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union
import pyrallis
from cfg import TrainConfig


def train(
    sess,
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
    results_dict=None,
    recorder=None,
):

    # Build the initializers for the required datasets.
    train_dataset_initializer = train_dataset.get_initializer()
    valid_dataset_initializer = valid_dataset.get_initializer()

    # Create the checkpoint root dir for mlflow model saving
    ckpt_root = recorder.root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

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

        # print("Epoch %d Model Restoring." % i)
        # dasie_model.restore(save_file_path)
        # print("Epoch %d Model Restored." % i)

        # First, plot the model status if requested
        print("Epoch %d Plots Plotting." % i)

        if save_plot:
            if (i % plot_periodicity) == 0:

                sess.run(valid_dataset_initializer)
                print("Plotting...")
                dasie_model.plot(logdir=logdir, show_plot=show_plot, step=i)
                print("Plotting completed.")

        print("Epoch %d Plots Plotted." % i)

        # Initialize the validation dataset iterator to prepare for validation.
        print("Epoch %d Validation Beginning..." % i)
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

            # log validation epoch metrics to mlflow
            val_metrics = {
                "valid_dist_mse": mean_valid_distributed_aperture_image_mse,
                "valid_mono_mse": mean_valid_monolithic_aperture_image_mse,
                "valid_mse_ratio": mean_valid_da_mse_mono_mse_ratio,
            }
            recorder.log_metrics(val_metrics, step=i)

            print("Validation Loss: %f" % mean_valid_loss)
            print("Validation DA MSE: %f" % mean_valid_distributed_aperture_image_mse)
            pass

        print("Epoch %d Validation Complete." % i)
 
        # Save the model to mlflow
        if i % recorder.cfg.log.save_freq == 0:
            print("Epoch %d Model Saving." % i)
            ckpt_path = ckpt_root / f"model_save_{i}.json"
            dasie_model.save(ckpt_path)
            print("Epoch %d Model Saved." % i)

        # Save the model and results in logdir (w/o recorder)
     
        # print("Epoch %d Results Saving." % i)
        # # Write the results dict for this epoch.
        # json_file = os.path.join(logdir, "results_" + str(i) + ".json")
        # json.dump(results_dict, open(json_file, 'w'))
        # # data = json.load(open("file_name.json"))
        # print("Epoch %d Results Saved." % i)

        # print("Epoch %d Model Saving." % i)
        # save_file_path = os.path.join(logdir, "model_save_" + str(i) + ".json")
        # dasie_model.save(save_file_path)
        # print("Epoch %d Model Saved." % i)

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
        train_steps = 0.0

        # Run training steps until the iterator is exhausted.
        start_time = time.time()
        try:
            while True:
            # (MP) debug/testing
            # while train_steps < 10:

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

                # log train metrics to mlflow according to train_freq (step snapshots, non-aggregated)
                if train_steps % recorder.cfg.log.train_freq == 0:
                    train_step_metrics = {
                        "train_step_dist_mse": step_train_distributed_aperture_image_mse,
                        "train_step_mono_mse": step_train_monolithic_aperture_image_mse,
                        "train_step_mse_ratio": step_train_da_mse_mono_mse_ratio,
                    }
                    recorder.log_metrics(train_step_metrics, step=int(train_steps))

            # (MP) debug/testing
            # raise StopIteration()

        # OutOfRangeError indicates we've finished the iterator, so report out.
        # except (tf.errors.OutOfRangeError, StopIteration):
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

            print("Mean Train Loss: %f" % mean_train_loss)
            print("Mean Train DA MSE: %f" % mean_train_distributed_aperture_image_mse)

            # log train epoch metrics to mlflow
            train_metrics = {
                "train_dist_mse": mean_train_distributed_aperture_image_mse,
                "train_mono_mse": mean_train_monolithic_aperture_image_mse,
                "train_mse_ratio": mean_train_da_mse_mono_mse_ratio,
            }
            recorder.log_metrics(train_metrics, step=i)

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
    features_parsed = tf.io.parse_single_example(
        serialized=example_proto, features=features
    )
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


def main():
    # parse CLI args or yaml config
    cfg = pyrallis.parse(config_class=TrainConfig)

    # beta = 32.0
    # Set the GPUs we want the script to use/see
    gpus = ",".join([str(i) for i in cfg.gpu_list])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Create mlflow recorder and create the specified experiment.
    recorder = Recorder(cfg)
    recorder.create_experiment()

    # Compute the scaling factor from meters to alpha for a GG PDF over meters.
    # alpha = np.log(-np.log(epsilon)) / np.log(beta) * ap_radius_meters
    # alpha = alpha / (cfg.num_subapertures)
    # epsilon = 1e-15
    ap_radius_meters = cfg.aperture_diameter_meters / 2
    subap_radius_meters = (
        (ap_radius_meters * np.sin(np.pi / cfg.num_subapertures))
        - (cfg.subaperture_spacing_meters / 2)
    ) / (1 + np.sin(np.pi / cfg.num_subapertures))
    # meters_to_alpha = 1 /  (2 * (np.log(-np.log(epsilon)) / np.log(beta)))
    # subap_alpha = subap_radius_meters * meters_to_alpha / 2
    # monolithic_alpha = cfg.aperture_diameter_meters * meters_to_alpha * 2
    subap_area = np.pi * (subap_radius_meters**2)
    total_subap_area = subap_area * cfg.num_subapertures
    mono_ap_area = np.pi * (ap_radius_meters**2)
    mono_to_dist_aperture_ratio = mono_ap_area / total_subap_area
    print(mono_to_dist_aperture_ratio)

    # TODO: [Ryan and Matthew] I could use help here.
    # TODO: Document how distance to the target is quantified implicitly.
    # TODO: Document how extent of the target is quantified implicitly.
    derived_optical_params = {
                "mono_to_dist_aperture_ratio": mono_to_dist_aperture_ratio, 
                "ap_radius_meters": ap_radius_meters,
                "subap_radius_meters": subap_radius_meters,
            }
    # TODO: Append all other derived optical information here!

    # Map our dataset name to relative locations and parse functions.
    if cfg.data.dataset_name == "speedplus":
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(cfg.data.dataset_root, "speedplus_tfrecords", "train")
        valid_data_dir = os.path.join(cfg.data.dataset_root, "speedplus_tfrecords", "valid")

    elif cfg.data.dataset_name == "inria_holiday":
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(cfg.data.dataset_root, "inria_holiday_tfrecords", "train")
        valid_data_dir = os.path.join(cfg.data.dataset_root, "inria_holiday_tfrecords", "valid")

    elif cfg.data.dataset_name == "speedplus_one":
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(cfg.data.dataset_root, "speedplus_one_tfrecords", "train")
        valid_data_dir = os.path.join(cfg.data.dataset_root, "speedplus_one_tfrecords", "valid")

    else:
        parse_function = speedplus_parse_function
        train_data_dir = os.path.join(cfg.data.dataset_root, "onesat_example_tfrecords", "train")
        valid_data_dir = os.path.join(cfg.data.dataset_root, "onesat_example_tfrecords", "valid")

    # Set the crop size to the spatial quantization scale.
    if cfg.crop:
        crop_size = cfg.spatial_quantization
    else:
        crop_size = None

    # Start mlflow run, handling interruptions through the context manager.
    # (Starting run at this point in order to use mlflow run dir as `save_dir` for tfevent files)
    with recorder.start_run():
        # log source files to mlflow
        recorder.log_files()
        # log all TrainConfig parameters for this run to mlflow
        recorder.log_run_params()
        # also log some derived params
        recorder.log_params(derived_optical_params)
        # Set directory for exporting artifacts (tfeventsfiles, images, etc)
        save_dir = str(recorder.root)

        # Begin by creating a new session.
        with tf.compat.v1.Session() as sess:

            print("\n\n\n\n\n\n\n\n\n Session Created \n\n\n\n\n\n\n\n\n")

            # Set all our seeds.
            np.random.seed(cfg.random_seed)
            tf.compat.v1.set_random_seed(cfg.random_seed)

            # Make summary management variables.
            step = tf.Variable(0, dtype=tf.int64)
            step_update = step.assign_add(1)
            tf.summary.experimental.set_step(step)
            writer = tf.summary.create_file_writer(save_dir)

            print("\n\n\n\n\n\n\n\n\n Building Dataset... \n\n\n\n\n\n\n\n\n")
            # Build our datasets.
            train_dataset = DatasetGenerator(
                train_data_dir,
                parse_function=parse_function,
                augment=False,
                shuffle=False,
                crop_size=crop_size,
                batch_size=cfg.batch_size,
                num_threads=2,
                buffer=32,
                encoding_function=None,
                cache_dataset_memory=False,
                cache_dataset_file=False,
                cache_path="",
            )

            # We create a tf.data.Dataset object wrapping the valid dataset here.
            valid_dataset = DatasetGenerator(
                valid_data_dir,
                parse_function=parse_function,
                augment=False,
                shuffle=False,
                crop_size=crop_size,
                batch_size=cfg.batch_size,
                num_threads=2,
                buffer=32,
                encoding_function=None,
                cache_dataset_memory=False,
                cache_dataset_file=False,
                cache_path="",
            )
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
                batch_size=cfg.batch_size,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                num_exposures=cfg.num_exposures,
                spatial_quantization=cfg.spatial_quantization,
                image_x_scale=image_x_scale,
                image_y_scale=image_y_scale,
                learning_rate=cfg.learning_rate,
                diameter_meters=cfg.aperture_diameter_meters,
                num_apertures=cfg.num_subapertures,
                subaperture_radius_meters=subap_radius_meters,
                recovery_model_filter_scale=cfg.recovery_model_filter_scale,
                loss_name=cfg.loss_name,
                num_zernike_indices=cfg.num_zernike_indices,
                hadamard_image_formation=cfg.hadamard_image_formation,
                zernike_debug=cfg.zernike_debug,
                recorder=recorder,
            )

            # Merge all the summaries from the graphs, flush and init the nodes.
            all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
            writer_flush = writer.flush()
            sess.run([writer.init(), step.initializer])

            # Optimize the DASIE model parameters.
            train(
                sess,
                dasie_model,
                train_dataset,
                valid_dataset,
                num_steps=cfg.num_steps,
                plot_periodicity=cfg.log.plot_freq,
                writer=writer,
                step_update=step_update,
                all_summary_ops=all_summary_ops,
                writer_flush=writer_flush,
                logdir=save_dir,
                save_plot=cfg.log.save_plot,
                show_plot=cfg.log.show_plot,
                #results_dict=base_results_dict,
                results_dict=None,
                recorder=recorder,
            )

            # stop mlflow run, exit gracefully
            recorder.end_run()


if __name__ == "__main__":

    # TODO: I need to enable a test of negligable, random, and learned articulations to measure validation set reconstructions.
    
    # CLI args are passed into main and parsed with pyrallis, even though the behavior is hidden here.
    main()