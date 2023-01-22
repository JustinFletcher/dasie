"""
Dataset generator, intended to be used with the Tensorflow Dataset API in the
form of a TFRecords file. Originally constructed to feed inputs to an
implementation of SSD, this class should be general enough to feed any model if
provided an appropriate parsing/encoding function for that model.

Author: 1st Lt Ian McQuaid
Date: 16 Nov 2018
"""

import os
import json
import fnmatch
import numpy as np
import tensorflow as tf
# from miss_utilities import get_tfrecords_list


def parse_run_config_json(filepath):
    # If the path points to a JSON, we need to parse it and read in all
    # directories listed
    tfrecords_list = []
    fp = open(filepath, "r")
    tfrecord_dirs_dict = json.load(fp)
    fp.close()

    # Tolerate a naming bug for the time being
    if "dirs" in tfrecord_dirs_dict.keys():
        dir_key = "dirs"
    elif "dir" in tfrecord_dirs_dict.keys():
        dir_key = "dir"
    else:
        raise Exception(
            "utils.parse_run_config_json: could not find either "
            "'dir' or 'dirs' in config file."
        )

    # Check every directory the JSON provided
    for tfrecord_path in tfrecord_dirs_dict[dir_key]:
        if os.path.isabs(tfrecord_path):
            # Easy if the path is absolute
            abs_tfrecord_path = tfrecord_path
        else:
            # If this path is relative
            start_dir = os.getcwd()
            os.chdir(os.path.split(filepath)[0])
            abs_tfrecord_path = os.path.abspath(tfrecord_path)
            os.chdir(start_dir)

        if os.path.isfile(abs_tfrecord_path):
            # In case individual files are listed as well
            tfrecords_list += [abs_tfrecord_path]
        else:
            tfrecords_list += get_tfrecords_in_dir(abs_tfrecord_path)

    # Done, so return
    return tfrecords_list

def get_dir_content_paths(directory):
    """
    Given a directory, returns a list of complete paths to contents.
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]

def get_tfrecords_list(filepath):
    """
    Helper function to handle parsing many tfrecords, potentially located
    in many directories, into a single list that can be treated as a single
    dataset.

    :param filepath: path to either tfrecord file, directory of tfrecords, or
    a configuration file that lists paths to tfrecords.
    :return: a list of all TFRecords files in this "dataset"
    """
    if os.path.isdir(filepath):
        # If this points to a directory, simply list everything in it
        tfrecords_list = get_dir_content_paths(filepath)
    elif fnmatch.fnmatch(filepath, "*.json"):
        tfrecords_list = parse_run_config_json(filepath)
    else:
        # In the simplest case, the path points to a single TFRecords file
        tfrecords_list = [filepath]

    return tfrecords_list

class DatasetGenerator(object):
    def __init__(self,
                 tfrecords,
                 parse_function,
                 augment=False,
                 shuffle=False,
                 crop_size=512,
                 batch_size=1,
                 num_threads=1,
                 buffer=30,
                 encoding_function=None,
                 cache_dataset_memory=False,
                 cache_dataset_file=False,
                 cache_path="",
                 max_elements=None):
        """
        Constructor for the data generator class. Takes as inputs many
        configuration choices, and returns a generator with those options set.

        :param tfrecords: the name of the TFRecord to be processed.
        :param parse_function: a custom parsing function to map input tfrecords
        examples to the format desired by the model and pipeline
        :param augment: whether or not to apply augmentation.
        :param shuffle: whether or not to shuffle the input buffer.
        :param batch_size: the number of examples in each batch produced.
        :param num_threads: the number of threads to use in processing input.
        :param buffer: the prefetch buffer size to use in processing.
        :param encoding_function: a custom encoding function to map from the
        raw image/bounding boxes to the desired format for one's specific
        network.
        :param cache_dataset_memory: whether or not to cache post-encoded
        examples to memory
        :param cache_dataset_file: whether or not to cache post-encoded
        examples to a file
        :param cache_path: path of file to use if cache_dataset_file is true
        """
        self.num_examples = None
        self.input_shape = None
        self.batch_size = batch_size
        self.encode_for_network = encoding_function
        self._parse_function = parse_function
        self.spatial_quantization = crop_size

        # Collect paths to the input tfrecords file(s)
        if type(tfrecords) != list:
            tfrecords_list = get_tfrecords_list(tfrecords)
        else:
            tfrecords_list = tfrecords

        # If this is a run configuration, see if the input size/shape are there
        num_examples = None
        input_shape = None
        if fnmatch.fnmatch(tfrecords, "*.json"):
            fp = open(tfrecords, "r")
            tfrecord_dirs_dict = json.load(fp)
            fp.close()

            if "num_examples" in tfrecord_dirs_dict.keys():
                num_examples = tfrecord_dirs_dict["num_examples"]

            if "input_shape" in tfrecord_dirs_dict.keys():
                input_shape = tfrecord_dirs_dict["input_shape"]

        self.dataset = self.build_pipeline(
            tfrecords_list,
            augment=augment,
            shuffle=shuffle,
            crop_size=crop_size,
            batch_size=batch_size,
            num_threads=num_threads,
            buffer=buffer,
            cache_dataset_memory=cache_dataset_memory,
            cache_dataset_file=cache_dataset_file,
            cache_path=cache_path,
            num_examples=num_examples,
            input_shape=input_shape,
            max_elements=max_elements,
        )

    def __len__(self):
        """
        The "length" of the generator is the number of batches expected.

        :return: the number of batches that will be produced by this generator.
        """
        return int(np.ceil(self.num_examples / self.batch_size))

    def get_dataset(self):
        return self.dataset

    def get_iterator(self):
        # return tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        self.iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        return self.iterator
        # TODO: Pick up here working towards a validation loop.

    def get_initializer(self):
        # return tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        # return tf.compat.v1.data.make_initializable_iterator(self.dataset)
        self.initializer = self.iterator.make_initializer(self.dataset)
        return self.initializer

    def _example_larger_than_crop(self, image,):
        min_size = tf.constant(self.spatial_quantization)
        img_height = tf.shape(image)[0]
        img_width = tf.shape(image)[1]
        height_enough = tf.greater(img_height, min_size)
        width_enough = tf.greater(img_width, min_size)
        isallowed = tf.math.logical_and(height_enough, width_enough)
        return isallowed

    def build_pipeline(self,
                       tfrecord_path,
                       augment,
                       shuffle,
                       batch_size,
                       num_threads,
                       buffer,
                       crop_size=None,
                       cache_dataset_memory=False,
                       cache_dataset_file=False,
                       cache_path="",
                       num_examples=None,
                       input_shape=None,
                       max_elements=None,):
        """
        Reads in data from a TFRecord file, applies augmentation chain (if
        desired), shuffles and batches the data.
        Supports prefetching and multithreading, the intent being to pipeline
        the training process to lower latency.

        :param tfrecord_path: the name of the TFRecord to be processed.
        :param augment: whether to augment data or not.
        :param shuffle: whether to shuffle data in buffer or not.
        :param batch_size: Number of examples in each batch returned.
        :param num_threads: Number of parallel subprocesses to load data.
        :param buffer: Number of images to prefetch in buffer.
        :param cache_dataset_memory: whether or not to cache post-encoded
        examples to memory
        :param cache_dataset_file: whether or not to cache post-encoded
        examples to a file
        :param cache_path: path of file to use if cache_dataset_file is true
        :return:  the next batch, to be provided when this generator is run (see
        run_generator())
        """

        # Create the TFRecord dataset
        data = tf.data.TFRecordDataset(tfrecord_path)

        # Parse the record into tensors
        data = data.map(self._parse_function, num_parallel_calls=num_threads)

        data = data.filter(self._example_larger_than_crop)


        if cache_dataset_memory:
            data = data.cache()
        elif cache_dataset_file:
            data = data.cache(cache_path)


        # Shuffle/repeat the data forever (i.e. as many epochs as we want)
        if shuffle:
            data = data.shuffle(buffer)
        # data = data.repeat()
        # If augmentation is to be applied
        if augment:
            # The only pixel-wise mutation possible on single channel imagery
            data = data.map(self._vary_contrast,
                            num_parallel_calls=num_threads)

            # Technically, we only need rotation and one flip to get all
            # possible orientations. But they are all here
            # anyways because it makes me feel better.
            data = data.map(self._flip_left_right,
                            num_parallel_calls=num_threads)
            data = data.map(self._flip_up_down,
                            num_parallel_calls=num_threads)
            data = data.map(self._rotate_random,
                            num_parallel_calls=num_threads)

        image_batch_shape = get_input_shape(data)
        print(image_batch_shape)
        if crop_size and ((crop_size < image_batch_shape[0])
                          and
                          (crop_size < image_batch_shape[1])):

            print(crop_size)
            data = data.map(self._perform_center_crop,
                            num_parallel_calls=num_threads)

        data = data.map(self._normalize,
                        num_parallel_calls=num_threads)

        # Force images to the same size
        # data = data.map(_resize_data,
        #                 num_parallel_calls=num_threads).prefetch(buffer)

        # Grab the basic non-repeating (non-cached!) data so we can do an
        # initial pass to get the set size
        if num_examples is None:
            self.num_examples = get_num_examples(data)
        else:
            self.num_examples = num_examples

        if input_shape is None:
            self.input_shape = get_input_shape(data)
        else:
            self.input_shape = input_shape

        # If the destination network requires a special encoding, do that here
        if self.encode_for_network is not None:
            data = data.map(
                lambda *args: self.encode_for_network(
                    *args,
                    image_shape=self.input_shape
                ),
                num_parallel_calls=num_threads
            )


        # Batch the data
        data = data.batch(batch_size,
                          num_parallel_calls=num_threads,
                          drop_remainder=True)

        if max_elements:

            data = data.take(max_elements)

        # Prefetch with multiple threads
        data.prefetch(buffer_size=buffer)



        image_batch_shape = get_input_shape(data)
        self.image_shape = (image_batch_shape[1], image_batch_shape[2])

        # Return a reference to this data pipeline
        return data

    def _vary_contrast(self, image, bboxs, filename=None):
        """
        Randomly varies the pixel-wise contrast of the image. This is the only
        pixel-wise augmentation that can be performed on single-channel imagery.
         The bounding boxes are not changed in any way by this function.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :return: the image and transformed bounding box tensor
        """

        cond_contrast = tf.cast(
            tf.random.uniform([], maxval=2, dtype=tf.int32),
            tf.bool)
        image = tf.cond(cond_contrast,
                        lambda: tf.image.random_contrast(image, 0.2, 1.8),
                        lambda: tf.identity(image))
        if filename is not None:
            return image, bboxs, filename
        else:
            return image, bboxs

    def _crop_random(self, image, bboxs, filename=None):
        """
        Randomly either applies the crop function or returns the unchanged input.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :return: the image and transformed bounding box tensor
        """

        cond_crop = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32),
                            tf.bool)
        image, bboxs = tf.cond(cond_crop,
                               lambda: _perform_crop(image, bboxs),
                               lambda: (
                               tf.identity(image), tf.identity(bboxs)))
        if filename is not None:
            return image, bboxs, filename
        else:
            return image, bboxs

    def _normalize(self, image):

        # Move the entire dynamic range up above zero.
        # image = image + tf.abs(tf.reduce_min(image))
        #
        # # If it was already above zero, or is now, bring it to zero.
        # image = image - tf.reduce_min(image)
        #
        # # And divide by the largest value to normalize the range to [0, 1] exactly.
        # image = image / tf.reduce_max(image)

        # image_min = tf.reduce_min(image)
        # image_max = tf.reduce_max(image)
        #
        # a = (1.0 - 0.0) / (image_max - image_min)
        # b = 1.0 - (a * image_max)
        # image = (image / (image_max - image_min)) + (1.0 - (image_max/ (image_max - image_min)))
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
        image = image + 0.0000001
        image = image / tf.reduce_max(image)
        return image

    def _standardize(self, image):

        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_mean(image))

        return image

    def _perform_center_crop(self, image, filename=None, min_crop_size=(256, 256)):
        """
        Center crops the image.


        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :param min_crop_size: the smallest dimensions that a crop can take
        :return: the image and transformed bounding box tensor
        """

        # Get the image shape tensor
        img_shape = tf.shape(image)

        # First come up with a desired crop size, from given mins to the whole image
        #  (maxval is exclusive)
        # crop_width = tf.random.uniform(
        #     [],
        #     minval=min_crop_size[1],
        #     maxval=max_crop_width,
        #     dtype=tf.int32
        # )
        # crop_height = tf.random.uniform(
        #     [],
        #     minval=min_crop_size[0],
        #     maxval=max_crop_width,
        #     dtype=tf.int32
        # )
        crop_width = self.spatial_quantization
        crop_height = self.spatial_quantization

        # The heavy lifting is done, time to make us a crop and transform our
        # bounding boxes to the new coordinates

        image = tf.image.crop_to_bounding_box(image,
                                              (img_shape[0] // 2) - (crop_height // 2),
                                              (img_shape[1] // 2) - (crop_width // 2),
                                              crop_height,
                                              crop_width)


        if filename is not None:
            return image, filename
        else:
            return image

    def _flip_left_right(self, image, bboxs, filename=None):
        """
        Randomly flips the image left or right, and transforms the bounding boxes.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :return: the image and transformed bounding box tensor
        """

        # Do the random image flip
        image_after = tf.image.random_flip_left_right(image)

        # Determine if a flip happened or not
        # Have to convert out of uint16...because tf apparently can't check
        # equality of uint16...smh...
        image_one = tf.cast(image_after, dtype=tf.float32)
        image_two = tf.cast(image, dtype=tf.float32)

        # If every pixel were equal, this would be a NOT flip, hence the outer NOT
        # to determine if this is a flip
        cond_flip = tf.logical_not(
            tf.reduce_all(tf.equal(image_one, image_two)))

        # This makes the computations a bit easier
        ymin = bboxs[:, 0]
        xmin = bboxs[:, 1]
        ymax = bboxs[:, 2]
        xmax = bboxs[:, 3]
        classes = bboxs[:, 4]

        # If we flipped, also flip the bounding boxes
        bboxs = tf.cond(cond_flip,
                        lambda: (tf.stack([ymin,
                                           1 - xmax,
                                           ymax,
                                           1 - xmin,
                                           classes], axis=1)),
                        lambda: tf.identity(bboxs))

        if filename is not None:
            return image_after, bboxs, filename
        else:
            return image_after, bboxs

    def _flip_up_down(self, image, bboxs, filename=None):
        """
        Randomly flips the image up or down, and transforms the bounding boxes.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :return: the image and transformed bounding box tensor
        """

        # Do the random image flip
        image_after = tf.image.random_flip_up_down(image)

        # Determine if a flip happened or not
        # Have to convert out of uint16...because tf apparently can't check equality
        #  of uint16...smh...
        image_one = tf.cast(image_after, dtype=tf.float32)
        image_two = tf.cast(image, dtype=tf.float32)

        # If every pixel were equal, this would be a NOT flip, hence the outer NOT
        #  to determine if this is a flip
        cond_flip = tf.logical_not(
            tf.reduce_all(tf.equal(image_one, image_two)))

        # This makes the computations a bit easier
        ymin = bboxs[:, 0]
        xmin = bboxs[:, 1]
        ymax = bboxs[:, 2]
        xmax = bboxs[:, 3]
        classes = bboxs[:, 4]

        # If we flipped, also flip the bounding boxes
        bboxs = tf.cond(cond_flip,
                        lambda: (tf.stack([1 - ymax,
                                           xmin,
                                           1 - ymin,
                                           xmax,
                                           classes], axis=1)),
                        lambda: tf.identity(bboxs))

        if filename is not None:
            return image_after, bboxs, filename
        else:
            return image_after, bboxs

    def _rotate_random(self, image, bboxs, filename=None):
        """
        Randomly applies the rotation function or returns the unchanged input.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :return: the image and transformed bounding box tensor
        """
        cond_rotate = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32),
                              tf.bool)
        image, bboxs = tf.cond(cond_rotate,
                               lambda: _perform_rotation(image, bboxs),
                               lambda: (
                               tf.identity(image), tf.identity(bboxs)))

        if filename is not None:
            return image, bboxs, filename
        else:
            return image, bboxs

    def _perform_rotation(self, image, bboxs, filename=None):
        """
        Rotates images randomly either 90 degrees left or right.
        The decision to rotate or not is decided by _rotate_random.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :return: the image and transformed bounding box tensor
        """
        # Either rotate once clockwise or counter clockwise
        cond_direction = tf.cast(
            tf.random.uniform([], maxval=2, dtype=tf.int32),
            tf.bool)
        num_rotations = tf.cond(cond_direction,
                                lambda: 1,
                                lambda: -1)

        image = tf.image.rot90(image, k=num_rotations)

        # Rotate the bounding boxes with the image
        ymin = bboxs[:, 0]
        xmin = bboxs[:, 1]
        ymax = bboxs[:, 2]
        xmax = bboxs[:, 3]

        ymin, xmin, ymax, xmax = tf.cond(cond_direction,
                                         lambda: (
                                         1 - xmax, ymin, 1 - xmin, ymax),
                                         lambda: (
                                         xmin, 1 - ymax, xmax, 1 - ymin))

        bbox_new = tf.stack([ymin, xmin, ymax, xmax, bboxs[:, 4]], axis=1)
        if filename is not None:
            return image, bbox_new, filename
        else:
            return image, bbox_new

    def _resize_data(self, image, bboxs, filename=None, image_size=(2048, 2048)):
        """
        Resizes images to specified size. Intended to be applied as an element of
        the augmentation chain via the Tensorflow Dataset API map call.

        Note: Currently this is not used, and is WRONG for use in SatNet. Resizing,
         if used, should be done via padding.

        :param image: input image tensor of Shape = Height X Width X #Channels
        :param bboxs: input bounding boxes of Shape = #Boxes X 4
        (ymin, xmin, ymax, xmax)
        :param image_size: desired/output image size
        :return: the image and transformed bounding box tensor
        """

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_images(image, image_size)
        image = tf.squeeze(image, axis=0)

        if filename is not None:
            return image, bboxs, filename
        else:
            return image, bboxs


def get_num_examples(dataset_no_repeat):
    # Do a single pass to get the size of the dataset
    sess = tf.compat.v1.keras.backend.get_session()
    single_pass_iter = tf.compat.v1.data.make_one_shot_iterator(
        dataset_no_repeat
    )
    next_elem = single_pass_iter.get_next()
    dataset_size = 0
    while True:
        try:
            sess.run(next_elem)
            dataset_size += 1
        except tf.errors.OutOfRangeError:
            break
    return dataset_size


def get_input_shape(dataset_no_repeat):
    # Do a single pass to get the size of the dataset
    sess = tf.compat.v1.keras.backend.get_session()
    single_pass_iter = tf.compat.v1.data.make_one_shot_iterator(
        dataset_no_repeat
    )
    next_elem = single_pass_iter.get_next()
    images = sess.run(next_elem)
    return images.shape

