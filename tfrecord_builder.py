"""
This is a generic TFRecords creator, meant to replace a proliferation of
domain specific TFRecords builders in the MISS code base. The major advantage of
this particular method is the use of a input JSON so that the user can specify
arbitrarily complex mappings and partitionings between input and output
directories. Another major advantage is using a recursive search for file
discovery, so as to lessen the burden of the user in creating specific input
directory structures.

Author: Capt Ian McQuaid
Date: 29 December 2020
"""

import os
import random
import shutil
import argparse
import numpy as np
import astropy.io.fits
import tensorflow as tf
from fnmatch import fnmatch
from miss_utilities.utils import load_from_json, save_as_json


def get_spectranet_annotation_pattern(spectranet_annot_dict):
    """
    Matching annotation file to the correct data file is hard and unique to each
    problem domain. This function solves it for spectranet (or rather for the
    simulated spectral data known as specsim).

    :param spectranet_annot_dict: the dictionary corresponding to a spectranet
    annotation file
    :return: pattern which the path to the data file should match provided it
    is the correct match to this annotation.
    """
    class_name = list(
        spectranet_annot_dict["data"]["observations"].values()
    )[0]["class_name"]
    obs_name = spectranet_annot_dict["data"]["radiometry"]["object"][:-5]
    return "*" + class_name + "*/" + obs_name + "/*"


annotation_matching_functions = {
    "spectranet": get_spectranet_annotation_pattern
}


def _int64_feature(value):
    """
    Helper function for inclusion of int64 features in a TFExample.

    :param value: the integer value to be made a feature.
    :return: the TFExample feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """
    Helper function for inclusion of a bytes features in a TFExample.

    :param value: the bytes value to be made a feature.
    :return: the TFExample feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _floats_feature(value):
    """
    Helper function for inclusion of float features in a TFExample.

    :param value: the float value to be made a feature.
    :return: the TFExample feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


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


def make_spectranet_tf_example(spectra_path,
                               annotation_path,
                               object_id=0,
                               num_classes=2):
    """
    Mapping from data paths and annotation paths to a TFExample requires a
    unique function for each problem domain. This function accomplishes that
    mapping for the spectranet problem domain.

    :param spectra_path: path to the spectra (i.e. data) .fits file.
    :param annotation_path: path to the annotation .json file.
    :param object_id: ID of this object (from 0 to num_classes - 1 inclusive)
    :param num_classes: number of classes, used in 1-hot encoding class IDs
    :return: a TFExample corresponding to the provided data and annotation paths
    """
    annotation_path = os.path.abspath(annotation_path)
    _, obj_filename = os.path.split(spectra_path)

    # Read in the files for this example
    image = read_fits(spectra_path)
    annotations = load_from_json(annotation_path)["data"]
    img_height, img_width = image.shape

    # Create the features for this example
    features = {
        "images_raw": _bytes_feature([image.tostring()]),
        "class_name": _bytes_feature([annotations['observations'][obj_filename]['class_name'].encode()]),
        "class_id": _int64_feature([object_id]),
        "height": _int64_feature([img_height]),
        "width": _int64_feature([img_width]),
        "num_classes": _int64_feature([num_classes]),
        "filename": _bytes_feature([spectra_path.encode()]),
        "annotation_path": _bytes_feature([annotation_path.encode()]),
    }

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=features))

    return example


tfexample_creation_functions = {
    "spectranet": make_spectranet_tf_example
}


def search_for_files(curr_path, file_pattern):
    """
    This function realizes a recursive search, starting from a given root
    directory, and collects any files which match a provided pattern.

    :param curr_path: the root of the recursive search
    :param file_pattern: the pattern to be matched
    :return: list containing paths to every file which matched our pattern
    during the search
    """
    matched_file_list = []
    for suffix in os.listdir(curr_path):
        next_path = os.path.join(curr_path, suffix)
        if os.path.isfile(next_path) and fnmatch(next_path, file_pattern):
            # If you found a matching file, add it to our list
            matched_file_list.append(next_path)
        elif os.path.isdir(next_path):
            # If you found a directory, we need to recurse
            matched_file_list += search_for_files(next_path, file_pattern)

    return matched_file_list


def match_data_to_annotations(data_path_list,
                              annotation_path_list,
                              annotation_matching_fn):
    """
    After discovering every possible data and annotation file, we need to
    directly associate an annotation with each data file. This is done with a
    domain-specific matching function. Down the road this could be standardized
    by including a path to the data file within the annotation with a consistent
    key name.

    :param data_path_list:
    :param annotation_path_list:
    :param annotation_matching_fn:
    :return:
    """
    matched_list = []
    curr_annotation_path_list = annotation_path_list.copy()

    # We need to match each data file to an annotation
    for data_file_path in data_path_list:
        match_found = False
        for annotation_file_path in curr_annotation_path_list:
            annotation_dict = load_from_json(annotation_file_path)

            # Matching annotations to data is challenging when our annotations
            # vary wildly between problems. In the future, standardization
            # to include the relative path to the data file in each annotation
            # would make this a ton easier. In the interim...this happens.
            try:
                annot_pattern = annotation_matching_fn(annotation_dict)
            except Exception:
                # If the file doesn't parse, it must not be a valid annotation
                continue

            if fnmatch(data_file_path, annot_pattern):
                matched_list.append((data_file_path, annotation_file_path))
                curr_annotation_path_list.remove(annotation_file_path)
                match_found = True
                break

        # Did we find a match? If not, we have a problem
        if not match_found:
            raise Exception("No annotation match found for the data at "
                            + str(data_file_path))

    # Done, so return the matching we found
    return matched_list


def partition_file_list(file_list, parition_dict, random_seed=None):
    """
    Divide a list of file path tuples into partitions based on user input.
    Technically, the results need not be true partitions as they need not be
    collectively exhaustive, but technically people hate math people because
    they point out the definition of partition to people. I make no apologies
    for who I am.

    The split is also random, as file_list is shuffled before partitioning. A
    seed can be used to force deterministic behavior.

    :param file_list: list of (data, annot) tuples which will be partitioned
    :param parition_dict: dictionary with keys corresponding to partition names
    and values corresponding to partition size. Int values will result in that
    number of examples being used in that partition. Float values will take that
    proportion of the entire list (0.5 = half of file_list).
    :param random_seed: can be used to force deterministic behavior.
    :return: dictionary where keys correspond to partition name and values
    correspond to the elements of file_list assigned to that partition.
    """
    # Start by randomizing our list, subject to a random seed for repeatability
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(file_list)

    # Check if this is a true partition - meaning both mutually exclusive and
    # collectively exhaustive of our inputs
    use_entire_set = True
    current_proportion = 0.0
    for name in parition_dict.keys():
        if type(parition_dict[name]) != float:
            use_entire_set = False
            break
        else:
            current_proportion += parition_dict[name]
    if use_entire_set and current_proportion != 1.0:
        use_entire_set = False

    # Now sort our file list into partitions
    partitioned_files = {}
    start_idx = 0
    total_example_count = len(file_list)
    for name in parition_dict.keys():
        if type(parition_dict[name]) == int:
            partition_example_count = parition_dict[name]
        elif type(parition_dict[name]) == float:
            partition_example_count = round(
                parition_dict[name] * total_example_count
            )
        else:
            raise Exception("Error in partition '" + name +
                            "'. Quantity has type " +
                            str(type(parition_dict[name])) +
                            " and only int and float are valid.")

        # Take an appropriately sized portion from our shuffled list
        stop_idx = start_idx + partition_example_count
        partitioned_files[name] = file_list[start_idx:stop_idx]
        start_idx = stop_idx

    # Check if any examples are unsorted - arbitrarily add them to the last
    # partition
    if use_entire_set and stop_idx != total_example_count:
        partitioned_files[name] += file_list[stop_idx:]

    return partitioned_files


def write_tfrecords_group(file_list,
                          output_path,
                          paths_to_example_fn,
                          group_name="data",
                          examples_per_file=256):
    """
    Writes a collection of examples to TFRecords files.

    :param file_list: a list of tuples (data, annotation) where each tuple will
    be transformed into a TFExample.
    :param output_path: directory where our TFRecords should be written.
    :param paths_to_example_fn: function to map the (data, annotation) tuples to
     a TFExample
    :param group_name: name to use in the TFRecords file name to uniquely
    identify this group.
    :param examples_per_file: Maximum number of examples to include in a single
    TFRecords file.
    :return: a list of paths to each TFRecords file created, for later use in
    creating configuration JSONs.
    """
    # Make sure the path exists
    make_dir(output_path, remove_old=False)

    # Keep track of any files we make
    tfrecords_file_list = []

    # We need to write every example in this group to a tfrecords file
    file_list_idx = 0
    group_idx = 1
    while file_list_idx < len(file_list):
        # Open a new file and reset the counter
        curr_group_ex_count = 0
        tfrecords_path = os.path.join(
            output_path,
            group_name + "_" + str(group_idx) + ".tfrecords"
        )

        # Open a writer to the provided TFRecords output location
        tfrecords_file_list.append(tfrecords_path)
        with tf.io.TFRecordWriter(tfrecords_path) as writer:
            print("Writing to " + str(tfrecords_path) + "...")
            while curr_group_ex_count < examples_per_file and \
                    file_list_idx < len(file_list):
                # Retrive the paths for this example
                data_path, annot_path = file_list[file_list_idx]
                file_list_idx += 1
                curr_group_ex_count += 1

                # Transform to tfexample and write to the tfrecords
                tf_example = paths_to_example_fn(data_path, annot_path)
                writer.write(tf_example.SerializeToString())

        # Next group
        group_idx += 1

    # Done, so return the paths to every file we created
    return tfrecords_file_list


def make_dir(directory, remove_old=False):
    """
    Similar to the utilites make_clean_dir function, except has an option to
    delete the old directory contents or not. Defaults to not delete old
    contents for safety.

    This function will recursively create a directory and any non-existent
    parent directories. Also has an option to remove old contents, as described
    above.

    :param directory: path to the directory in question.
    :param remove_old: true if the user wants the old contents of the directory
    to be deleted.
    :return: nothing
    """
    if remove_old and os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory)
    elif not os.path.exists(directory):
        os.makedirs(directory)


def copy_global_setting_to_mappings(setting_dict):
    """
    Helper function to apply global level dictionary elements (user settings) to
    individual mappings, provided the individual mapping didn't have one already
    set. This has the effect of letting users apply specific settings, if
    desired, and otherwise follow global ones for brevity.

    :param setting_dict: the dict form of the user input JSON
    :return: the settings dictionary with global settings copied to the mapping
    level, where appropriate.
    """
    for mapping in setting_dict["mapping"]:
        for key in setting_dict.keys():
            if key != "mapping" and key not in mapping.keys():
                mapping[key] = setting_dict[key]
    return setting_dict


def main(flags):
    """
    Main function for the generic tfrecord creation script. Takes input from
    an input JSON and then applies each requested TFRecords mapping. Once
    complete the script generates configuration JSONs for each partition made.

    :param flags: command line arguments
    :return: nothing
    """
    input_json = load_from_json(flags.input_json)

    # If the user does not specify parameters at the mapping level, borrow those
    # from the global level.
    input_json = copy_global_setting_to_mappings(input_json)

    # Directory cleaning has to be done before anything else, otherwise we risk
    # removing files that we create in this script.
    for mapping in input_json["mapping"]:
        if "clean_destination_dir" in mapping.keys():
            make_dir(mapping["destination"], mapping["clean_destination_dir"])

    # Keep track of the things we need to build our config files
    global_config_dict = {
        "dirs": [],
        "num_examples": 0,
    }
    partition_config_dict = {}

    # Apply each mapping independently
    for mapping in input_json["mapping"]:
        # Find all data files in consideration
        data_file_list = search_for_files(
            mapping["data_source"],
            mapping["data_file_pattern"]
        )

        # Find all annotation files in consideration
        annotation_file_list = search_for_files(
            mapping["annotation_source"],
            mapping["annotation_file_pattern"]
        )

        # Get the annotation function from the ones we have registered
        annot_fn_keys = annotation_matching_functions.keys()
        if mapping["annotation_matching_fn"] not in annot_fn_keys:
            raise Exception("The current annotation matching function registry "
                            "does not contain anything for the key '" +
                            mapping["annotation_matching_fn"] + "'.")
        annotation_matching_fn = annotation_matching_functions[
            mapping["annotation_matching_fn"]
        ]

        # Match an annotation to each data file
        matched_file_list = match_data_to_annotations(
            data_file_list,
            annotation_file_list,
            annotation_matching_fn=annotation_matching_fn,
        )

        # Partition the files, if desired
        if "partitions" in mapping.keys() and mapping["partitions"] != "none":
            partitioned_file_list = partition_file_list(
                matched_file_list,
                mapping["partitions"],
                random_seed=mapping["random_seed"]
            )

        # Get the tfexample creation function from the ones we have registered
        example_fn_keys = tfexample_creation_functions.keys()
        if mapping["tfexample_creation_fn"] not in example_fn_keys:
            raise Exception(
                "The current tfexample creation function registry "
                "does not contain anything for the key '" +
                mapping["tfexample_creation_fn"] + "'.")

        # We define a local function to handle any kwargs that may be needed
        def tfexample_creation_fn(data_path, annot_path):
            return tfexample_creation_functions[
                mapping["tfexample_creation_fn"]
            ](data_path, annot_path, **mapping["tfexample_creation_kwargs"])

        # Now map to tfrecords and write to file
        group_name = mapping["dataset_name"] + "_"
        group_name += mapping["mapping_name"]
        if "partitions" in mapping.keys() and mapping["partitions"] != "none":
            for partition_name in partitioned_file_list.keys():
                new_tfrecord_paths = write_tfrecords_group(
                    partitioned_file_list[partition_name],
                    os.path.join(mapping["destination"], partition_name),
                    tfexample_creation_fn,
                    group_name=group_name + "_" + partition_name,
                    examples_per_file=mapping["examples_per_tfrecord"]
                )

                # Track what was added for our config file
                if partition_name not in partition_config_dict.keys():
                    partition_config_dict[partition_name] = {
                        "dirs": [],
                        "num_examples": 0,
                    }
                curr_config_dict = partition_config_dict[partition_name]
                curr_config_dict["dirs"] += new_tfrecord_paths
                curr_config_dict["num_examples"] += len(
                    partitioned_file_list[partition_name]
                )
        else:
            new_tfrecord_paths = write_tfrecords_group(
                matched_file_list,
                mapping["destination"],
                tfexample_creation_fn,
                group_name=group_name,
                examples_per_file=mapping["examples_per_tfrecord"]
            )

            # Track what was added for our config file
            global_config_dict["dirs"] += new_tfrecord_paths
            global_config_dict["num_examples"] += len(matched_file_list)

    # Finally, write our config jsons to disk
    if global_config_dict["num_examples"] > 0:
        save_as_json(
            os.path.join(
                input_json["config_file_dir"],
                input_json["dataset_name"] + ".json"
            ),
            global_config_dict
        )
    for partition_name in partition_config_dict.keys():
        if partition_config_dict[partition_name]["num_examples"] > 0:
            save_as_json(
                os.path.join(
                    input_json["config_file_dir"],
                    partition_name + ".json"
                ),
                partition_config_dict[partition_name]
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', type=str,
                        default="./tfrecord_maker_test.json",
                        help='Path to the input JSON which defines the data to'
                             ' tfrecords mapping.')

    parsed_flags, _ = parser.parse_known_args()
    main(parsed_flags)
