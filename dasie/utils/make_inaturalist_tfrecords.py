"""
A new old TFRecord builder.
Author: Justin Fletcher
"""

import os

import json

import shutil

import argparse

import numpy as np

import astropy.io.fits

import tensorflow as tf

from matplotlib import image

from matplotlib import pyplot as plt

from itertools import islice, zip_longest


def read_fits(filepath):
    """Reads simple 1-hdu FITS file into a numpy arrays
    Parameters
    ----------
    filepath : str
        Filepath to read the array from
    """
    a = astropy.io.fits.getdata(filepath)
    a = astropy.io.jpg.getdata()
    a = a.astype(np.uint16)

    return a


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def read_png(filepath):
    """
    Reads a jpg file into a numpy arrays
    Parameters
    ----------
    filepath : str
        Filepath to a jpg which we wil read into an array
    """
    jpg_data = image.imread(filepath, format="png")

    jpg_data = convert(jpg_data, 0, 255, np.uint8)
    # plt.imshow(jpg_data)
    # plt.show()
    # die
    # jpg_data = jpg_data / np.max(jpg_data)
    #
    # jpg_data = jpg_data.astype(np.uint8)

    # plt.imshow(jpg_data)
    # plt.show()
    # die
    return jpg_data

def read_jpg(filepath):
    """
    Reads a jpg file into a numpy arrays
    Parameters
    ----------
    filepath : str
        Filepath to a jpg which we wil read into an array
    """
    jpg_data = image.imread(filepath, format="jpg")

    # Check shape of image and convert rgb and rgba to greyscale.
    # TODO: externalize this flag.
    greyscale = True
    if greyscale:
        r, g, b = jpg_data[:, :, 0], jpg_data[:, :, 1], jpg_data[:, :, 2]
        jpg_data = 0.2989 * r + 0.5870 * g + 0.1140 * b

    jpg_data = convert(jpg_data, 0, 255, np.uint8)
    # plt.imshow(jpg_data)
    # plt.show()
    # die
    # jpg_data = jpg_data / np.max(jpg_data)
    #
    # jpg_data = jpg_data.astype(np.uint8)

    # plt.imshow(jpg_data)
    # plt.show()
    # die
    return jpg_data


def _int64_feature(value):

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _floats_feature(value):

    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def crop_center(img, crop_x, crop_y):
    y, x = img.shape
    start_x = x // 2 - crop_x // 2
    start_y = y // 2 - crop_y // 2
    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]

def build_speedplus_tf_example(example):

    (image_path) = example

    # Read in the files for this example
    image = read_jpg(image_path)

    # image = crop_center(image, 512, 512)

    image_height = image.shape[1]
    image_width = image.shape[0]
    # Ada: You'll need to read from annotation_path to get, e.g., class labels.

    # Create the features for this example
    # Ada: Note here how I'm serializing the image and then binarizing it.
    features = {
        "image_raw": _bytes_feature([image.tostring()]),
        "height": _int64_feature([image_height]),
        "width": _int64_feature([image_width]),
        # Ada: Annotation mappings go here - I just don't need any right now...
        # "class_label": _int64_feature(class_label)
    }


    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return(example)

def build_imagenet_tf_example(example):

    # TODO: inlcude synset or class annotations.
    (image_path) = example

    # Read in the files for this example
    image = read_jpg(image_path)
    image_height = image.shape[1]
    image_width = image.shape[0]


    # Ada: You'll need to read from annotation_path to get, e.g., class labels.

    # Create the features for this example
    # Ada: Note here how I'm serializing the image and then binarizing it.
    features = {
        "image_raw": _bytes_feature([image.tostring()]),
        "height": _int64_feature([image_height]),
        "width": _int64_feature([image_width]),
        # Ada: Annotation mappings go here - I just don't need any right now...
        # "class_label": _int64_feature(class_label)
    }


    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return(example)



def group_list(ungrouped_list, group_size, padding=None):

    # Magic, probably. I literally don't remember how I made this...
    grouped_list = zip_longest(*[iter(ungrouped_list)] * group_size,
                               fillvalue=padding)

    return(grouped_list)


def make_clean_dir(directory):

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def get_immediate_subdirectories(a_dir):
    """
    Shift+CV from SO
    """
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


def build_speedplus_dataset(datapath):

    # We're going to make a list of filepaths as a template.
    examples = list()

    # Iterate over each subdirectory, each of which is a data subdomain.
    # Ada: This may or may not be applicable to your problem.
    for directory_name in get_immediate_subdirectories(datapath):

        # Ada: Note that this implies that --datadir will have a set of...
        # ...subdirs which will contain a subdir named "images". You are not...
        # ...bound to this convention at all - it's dataset-specific.
        collection_path = os.path.join(datapath, directory_name)
        image_dir_path = os.path.join(collection_path, "images")
        # Ada: Here, I've commented this out for my work. You'll need it.
        # annotation_dir_path = os.path.join(collection_path, "Annotations")

        # Parse the lists of images and annotations, an co-sort them for zip.
        image_paths = sorted(os.listdir(image_dir_path))
        # Ada: I've commented this for my work, but you'll need it.
        # annotation_paths = sorted(os.listdir(annotation_dir_path))
        # Ada: I'm adding a hack in here to avoid breaking the pattern for you.
        # annotation_paths = [None] * len(image_paths)

        # Now that we've built two lists of labeled examples, zip them up...
        for image_path in image_paths:

            # Get first image and annotation and write to file path.
            # Ada: Here, I've commented this out for my work. You'll need it.
            # example = (os.path.join(image_dir_path, image_path),
            #            os.path.join(annotation_dir_path, annotation_path))

            # ...joining each as a tuple and appending them to our list.
            example = os.path.join(image_dir_path, image_path)
            examples.append(example)

    return(examples)


def partition_examples(examples, splits_dict):

    # TODO: Print examples here and see if they are sequential...

    # Create a dict to hold examples.
    partitions = dict()

    # Store the total number of examples.
    num_examples = len(examples)

    # Iterate over the items specifying the partitions.
    for (split_name, split_fraction) in splits_dict.items():

        # Compute the size of this partition.
        num_split_examples = int(split_fraction * num_examples)

        # Pop the next partition elements.
        partition_examples = examples[:num_split_examples]
        examples = examples[num_split_examples:]

        # Map this partitions list of examples to this partition name.
        partitions[split_name] = partition_examples

    return(partitions)


def partition_examples_by_file(examples, split_file_dir):
    # Create a dict to hold examples.
    partitions = dict()

    # Need to read in the splits files
    dir_contents = list()
    for split_file in os.listdir(split_file_dir):
        if split_file.endswith(".txt"):
            dir_contents.append(split_file)

    for split_file_name in dir_contents:

        # Get the name of this split (remove the extension)
        split_name = split_file_name.split(".")[0]

        # Pull the file contents into memory
        split_file_path = os.path.join(split_file_dir, split_file_name)
        fp = open(split_file_path, "r")
        file_contents = fp.readlines()
        fp.close()

        # Remove the end line character
        file_contents = [line[:-1] for line in file_contents]

        # Gotta convert the weird way these are written in the split files
        # to something that looks like an actual path
        # (they are written as "collect dir"_"file name" for some reason)
        split_paths = list()
        for line in file_contents:
            new_path = os.path.join(line.split("_")[0],
                                    "_".join(line.split("_")[1:]))
            split_paths.append(new_path)

        # Now check and see which examples belong in this split
        split_examples = []
        for example in examples:
            full_dir, file_name = os.path.split(example[0])
            full_dir, _ = os.path.split(full_dir)
            _, collect_dir = os.path.split(full_dir)
            example_path = os.path.join(collect_dir, file_name)
            if example_path in split_paths:
                split_examples.append(example)

        # Save this split away in our return dictionary
        print("Saving partition " + str(split_name) +
              " with " + str(len(split_examples)) + " examples.")
        partitions[split_name] = split_examples
    return partitions


def create_tfrecords(data_dir,
                     output_dir,
                     tfrecords_name="tfrecords",
                     examples_per_tfrecord=1,
                     datapath_to_examples_fn=build_speedplus_dataset,
                     tf_example_builder_fn=build_speedplus_tf_example,
                     partition_examples_fn=partition_examples,
                     splits_dict={"data": 1.0}):
    """
    Given an input data directory, process that directory into examples. Group
    those examples into groups to write to a dir.
    """

    # Ada: The following is nice practical example of function-as-interface...
    # ...notice how some function names are symbolic, rather than constant.

    # TODO: Throw exception if interface functions aren't given.

    # Map the provided data directory to a list of tf.Examples.
    examples = datapath_to_examples_fn(data_dir)

    # Use the provided split dictionary to partition the example as a dict.
    partitioned_examples = partition_examples_fn(examples, splits_dict)

    # Iterate over each partition building the TFRecords.
    for (split_name, split_examples) in partitioned_examples.items():

        print("Writing partition %s w/ %d examples." % (split_name,
                                                        len(split_examples)))

        # Build a clean directory to store this partitions TFRecords.
        partition_output_dir = os.path.join(output_dir, split_name)
        make_clean_dir(partition_output_dir)

        # Group the examples in this partitions to write to separate TFRecords.
        example_groups = group_list(split_examples, examples_per_tfrecord)

        # Iterate over each group. Each is a list of examples.
        for group_index, example_group in enumerate(example_groups):

            print("Saving group %s w/ <= %d examples" % (str(group_index),
                                                         len(example_group)))

            # Specify the group name.
            group_tfrecords_name = tfrecords_name + '_' + split_name + '_' + str(group_index) + '.tfrecords'

            # Build the path to write the output to.
            output_path = os.path.join(partition_output_dir,
                                       group_tfrecords_name)

            # Open a writer to the provided TFRecords output location.
            with tf.io.TFRecordWriter(output_path) as writer:

                # For each example...
                for example in example_group:

                    # ...if the example isn't empty...
                    if example:

                        # print("Writing example %s" % example[0])

                        # ...instantiate a TF Example object...
                        # Ada: Your domain-specific example builder goes here.
                        tf_example = tf_example_builder_fn(example)

                        # ...and write it to the TFRecord.
                        # Ada: This is the TF serialization I mentioned.
                        writer.write(tf_example.SerializeToString())


def get_dir_content_paths(directory):
    """
    Given a directory, returns a list of complete paths to its contents.
    """
    return([os.path.join(directory, f) for f in os.listdir(directory)])


def main(flags):

    if flags.presplit:

        split_dict = {"data": 1.0}
        partition_fn = partition_examples

    else:

        # If one desires a deterministic split, pass in a splits file.
        if flags.splits_files_path:
            split_dict = flags.splits_files_path
            partition_fn = partition_examples_by_file

        # Otherwise, we'll just default to a 8/1/1 at random.
        else:
            split_dict = {"train": 0.8, "valid": 0.2}
            partition_fn = partition_examples

    # TODO: externalize this function interface.
    datapath_fn = build_speedplus_dataset
    example_builder_fn = build_speedplus_tf_example

    create_tfrecords(data_dir=flags.data_dir,
                     output_dir=flags.output_dir,
                     tfrecords_name=flags.name,
                     examples_per_tfrecord=flags.examples_per_tfrecord,
                     datapath_to_examples_fn=datapath_fn,
                     tf_example_builder_fn=example_builder_fn,
                     partition_examples_fn=partition_fn,
                     splits_dict=split_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str,
                        default="inaturalist",
                        help='Name of the dataset to build.')

    parser.add_argument('--data_dir', type=str,
                        default="./data/inaturalist",
                        help='Path to speedplus output data.')

    parser.add_argument('--output_dir', type=str,
                        default="./data/inaturalist_tfrecords",
                        help='Path to the output directory.')

    parser.add_argument("--examples_per_tfrecord",
                        type=int,
                        default=64,
                        help="Maximum number of examples to write to a file")

    parser.add_argument("--greyscale",
                        action='store_true',
                        default=False,
                        help='If true, map rgb jpgs/pngs to NTSC greyscale.')

    parser.add_argument("--presplit",
                        action='store_true',
                        default=False,
                        help='If true, assume that subdirs define splits.')

    parser.add_argument("--splits_files_path", type=str,
                        default=None,
                        help="Path to splits files, if one wants to make their splits deterministic")

    flags, unparsed = parser.parse_known_args()

    main(flags)