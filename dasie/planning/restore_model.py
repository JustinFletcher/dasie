import os
import json
import argparse


import astropy
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import image
import shutil

from dataset_generator import DatasetGenerator

from differentiable_dasie import DASIEModel

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


def save_and_close_current_plot(logdir, plot_name="default", dpi=600):
    fig_path = os.path.join(logdir, str(plot_name) + '.png')
    plt.gcf().set_dpi(dpi)
    plt.savefig(fig_path)
    plt.close()

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def read_fits(filepath):
    """Reads simple 1-hdu FITS file into a numpy arrays
    Parameters
    ----------
    filepath : str
        Filepath to read the array from
    """
    a = astropy.io.fits.getdata(filepath)
    a = a.astype(np.uint16)

    return a


def read_png(filepath):
    """
    Reads a png file into a numpy arrays
    Parameters
    ----------
    filepath : str
        Filepath to a jpg which we wil read into an array
    """
    png_data = image.imread(filepath, format="png")
    png_data = convert(png_data, 0, 255, np.uint8)
    return png_data


def read_jpg(filepath):
    """
    Reads a jpg file into a numpy arrays
    Parameters
    ----------
    filepath : str
        Filepath to a jpg which we wil read into an array
    """
    jpg_data = image.imread(filepath, format="jpg")
    jpg_data = convert(jpg_data, 0, 255, np.uint8)
    return jpg_data


def write_fits(image, filepath):
    """
    Writes a FITS to filepath, give an np.array image.
    Parameters
    :param image: np.array representing an image.
    :param filepath: a complete filepath, at which to write image as FITS
    :return: None
    """
    raise NotImplementedError("Justin didn't implement FITS writing because \
                               he's lazy. Follow this Exception to fix.")

    return


def write_jpg(image, filepath):
    """
    Writes a jpg to filepath, give an np.array image.
    Parameters
    :param image: np.array representing an image.
    :param filepath: a complete filepath, at which to write image as jpg
    :return: None
    """
    print(filepath)
    matplotlib.image.imsave(filepath, image)

    return


def write_png(image, filepath):
    """
    Writes a png to filepath, give an np.array image.
    Parameters
    :param image: np.array representing an image.
    :param filepath: a complete filepath, at which to write image as png
    :return: None
    """
    matplotlib.image.imsave(filepath, image)

    return


def main(flags):



    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:

        print("Restoring Recovery Model.")

        # Parse the kwargs that produced the model, and set batch size to 1.
        restore_dict = json.load(open(flags.dasie_model_save_file, 'r'))
        restore_dict["kwargs"]["batch_size"] = 1
        if not flags.output_file_path:
            output_file_path = os.path.join(".",
                                            "logs",
                                            "inference",
                                            flags.dataset_name)

        else:
            output_file_path = flags.output_file_path

        os.makedirs(output_file_path, exist_ok=True)
        num_dirs = len(next(os.walk(output_file_path))[1])
        print(num_dirs)
        output_file_path = os.path.join(output_file_path, str(num_dirs))
        os.makedirs(output_file_path, exist_ok=True)

        # shutil.copyfile(flags.dasie_model_save_file, output_file_path)


        # Map our dataset name to relative locations and parse functions.
        if flags.dataset_name == "speedplus":

            print("Selected dataset is speedplus.")
            parse_function = speedplus_parse_function
            train_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_tfrecords", "valid")

        elif flags.dataset_name == "speedplus_synthetic":

            print("Selected dataset is speedplus_synthetic.")
            parse_function = speedplus_parse_function
            train_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_synthetic_tfrecords",
                                          "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_synthetic_tfrecords",
                                          "valid")

        elif flags.dataset_name == "inria_holiday":

            print("Selected dataset is inria_holiday.")
            parse_function = speedplus_parse_function
            train_data_dir = os.path.join(flags.dataset_root,
                                          "inria_holiday_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "inria_holiday_tfrecords", "valid")


        elif flags.dataset_name == "speedplus_one":
            parse_function = speedplus_parse_function
            print("Selected dataset is speedplus_one.")
            train_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_one_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_one_tfrecords", "valid")


        elif flags.dataset_name == "speedplus_overfit":
            parse_function = speedplus_parse_function
            print("Selected dataset is speedplus_overfit.")
            train_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_overfit_tfrecords",
                                          "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "speedplus_overfit_tfrecords",
                                          "valid")


        elif flags.dataset_name == "usaf1951":
            parse_function = speedplus_parse_function
            print("Selected dataset is usaf1951.")
            train_data_dir = os.path.join(flags.dataset_root,
                                          "usaf1951_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "usaf1951_tfrecords", "valid")


        elif flags.dataset_name == "inaturalist":
            parse_function = inaturalist_parse_function
            print("Selected dataset is inaturalist.")
            train_data_dir = os.path.join(flags.dataset_root,
                                          "inaturalist_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "inaturalist_tfrecords", "valid")


        elif flags.dataset_name == "inaturalist_micro":
            parse_function = inaturalist_parse_function
            print("Selected dataset is inaturalist.")
            train_data_dir = os.path.join(flags.dataset_root,
                                          "inaturalist_micro_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "inaturalist_micro_tfrecords", "valid")


        else:
            parse_function = speedplus_parse_function
            print("Selected dataset is the default, onesat.")
            train_data_dir = os.path.join(flags.dataset_root,
                                          "onesat_example_tfrecords", "train")
            valid_data_dir = os.path.join(flags.dataset_root,
                                          "onesat_example_tfrecords", "valid")

        dataset = DatasetGenerator(valid_data_dir,
                                   parse_function=parse_function,
                                   augment=False,
                                   shuffle=False,
                                   crop_size=restore_dict["kwargs"]["spatial_quantization"],
                                   batch_size=flags.batch_size,
                                   encoding_function=None,
                                   cache_dataset_file=False,
                                   cache_path="",)
        dataset_iterator = dataset.get_iterator()
        dataset_initializer = dataset.get_initializer()

        sess.run(dataset_initializer)

        # Instantiate a new model with the same kwargs.
        dasie_model = DASIEModel(sess,
                                 dataset,
                                 dataset,
                                 **restore_dict["kwargs"])

        # Restore the weights.
        dasie_model.restore(flags.dasie_model_save_file)
        print("Model Restored.")

        left = dasie_model.pupil_dimension_u[0]
        right = dasie_model.pupil_dimension_u[-1]
        bottom = dasie_model.pupil_dimension_v[0]
        top = dasie_model.pupil_dimension_v[-1]
        pupil_extent = [left, right, bottom, top]

        left = dasie_model.focal_dimension_x[0] * 1e6
        right = dasie_model.focal_dimension_x[-1] * 1e6
        bottom = dasie_model.focal_dimension_y[0] * 1e6
        top = dasie_model.focal_dimension_y[-1] * 1e6
        focal_extent = [left, right, bottom, top]

        object_extent = [-dasie_model.object_plane_extent_meters / 2,
                         dasie_model.object_plane_extent_meters / 2,
                         -dasie_model.object_plane_extent_meters / 2,
                         dasie_model.object_plane_extent_meters / 2]


        [flipped_object_example_batch,
         recovered_image_batch,
         monolithic_aperture_image_batch] =  dasie_model.infer()

        for b in range(flags.num_batches):
            for n, (flipped_object_example,
                    recovered_image,
                    monolithic_aperture_image) in enumerate(zip(flipped_object_example_batch,
                                                                recovered_image_batch,
                                                                monolithic_aperture_image_batch)):

                recovered_image = np.squeeze(recovered_image)

                plt.imshow(np.flipud(np.fliplr(flipped_object_example)),
                           cmap=flags.cmap,
                           extent=object_extent)
                plt.xlabel('Object Plane Distance [$m$]')
                plt.ylabel('Object Plane Distance [$m$]')
                plt.colorbar()
                save_and_close_current_plot(
                    flags.output_file_path,
                    plot_name="object_" + str(b) + "_" + str(n),
                    dpi=dpi)

                plt.imshow(np.flipud(np.fliplr(monolithic_aperture_image)),
                           cmap=flags.cmap,
                           extent=focal_extent)
                plt.xlabel('Pupil Plane Distance [$m$]')
                plt.ylabel('Pupil Plane Distance [$m$]')
                plt.colorbar()
                save_and_close_current_plot(
                    flags.output_file_path,
                    plot_name="monolithic_aperture_image_" + str(b) + "_" + str(n),
                    dpi=dpi)

                plt.imshow(np.flipud(np.fliplr(recovered_image)),
                           cmap=flags.cmap,
                           extent=focal_extent)
                plt.xlabel('Focal Plane Distance [$\mu m$]')
                plt.ylabel('Focal Plane Distance [$\mu m$]')
                plt.colorbar()
                save_and_close_current_plot(
                    flags.output_file_path,
                    plot_name="recovered_image_" + str(b) + "_" + str(n),
                    dpi=dpi
                )

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='provide arguments for training.')

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--dasie_model_save_file',
                        type=str,
                        default=os.path.join(".",
                                             "dasie",
                                             "resources",
                                             "model_save_0.json"),
                        help='The save file for the model to load.')

    parser.add_argument('--dataset_name', type=str,
                        default="inaturalist_micro",
                        help='Name of the dataset.')


    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--output_file_path',
                        type=str,
                        default=None,
                        help='The name of the output image file.')

    parser.add_argument('--dataset_root', type=str,
                        default=os.path.join("..", "data"),
                        help='Path to a directory holding all datasets.')

    parser.add_argument('--dpi', type=int,
                        default=600,
                        help='Path to a directory holding all datasets.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Size of the mini-batch for inference.')

    parser.add_argument('--num_batches',
                        type=int,
                        default=1,
                        help='Number of batches to run inference against.')

    parser.add_argument('--cmap',
                        type=str,
                        default="grays",
                        help='Path to a directory holding all datasets.')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)
