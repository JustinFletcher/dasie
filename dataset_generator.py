"""
Dataset generator, intended to be used with the Tensorflow Dataset API in the
form of a TFRecords file. Originally constructed to feed inputs to an
implementation of SSD, this class should be general enough to feed any model if
provided an appropriate parsing/encoding function for that model.

Author: 1st Lt Ian McQuaid
Date: 16 Nov 2018
"""

import json
import fnmatch
import numpy as np
import tensorflow as tf
from miss_utilities import get_tfrecords_list


class DatasetGenerator(object):
    def __init__(self,
                 tfrecords,
                 parse_function,
                 augment=False,
                 shuffle=False,
                 batch_size=1,
                 num_threads=1,
                 buffer=30,
                 encoding_function=None,
                 cache_dataset_memory=False,
                 cache_dataset_file=False,
                 cache_path=""):
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
            batch_size=batch_size,
            num_threads=num_threads,
            buffer=buffer,
            cache_dataset_memory=cache_dataset_memory,
            cache_dataset_file=cache_dataset_file,
            cache_path=cache_path,
            num_examples=num_examples,
            input_shape=input_shape
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
        return tf.compat.v1.data.make_one_shot_iterator(self.dataset)

    def build_pipeline(self,
                       tfrecord_path,
                       augment,
                       shuffle,
                       batch_size,
                       num_threads,
                       buffer,
                       cache_dataset_memory=False,
                       cache_dataset_file=False,
                       cache_path="",
                       num_examples=None,
                       input_shape=None):
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

        # If augmentation is to be applied
        if augment:
            # The only pixel-wise mutation possible on single channel imagery
            data = data.map(_vary_contrast, num_parallel_calls=num_threads)

            # Technically, we only need rotation and one flip to get all
            # possible orientations. But they are all here
            # anyways because it makes me feel better.
            data = data.map(_flip_left_right, num_parallel_calls=num_threads)
            data = data.map(_flip_up_down, num_parallel_calls=num_threads)
            data = data.map(_rotate_random, num_parallel_calls=num_threads)

            # 50/50 chance of performing some crop, which is then randomly
            # data = data.map(_crop_random,
            #                 num_parallel_calls=num_threads).prefetch(buffer)

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

        if cache_dataset_memory:
            data = data.cache()
        elif cache_dataset_file:
            data = data.cache(cache_path)

        # Shuffle/repeat the data forever (i.e. as many epochs as we want)
        if shuffle:
            data = data.shuffle(buffer)
        data = data.repeat()

        # Batch the data
        data = data.batch(batch_size)

        # Prefetch with multiple threads
        data.prefetch(buffer_size=buffer)

        # Return a reference to this data pipeline
        return data


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
    images, _, _ = sess.run(next_elem)
    return images.shape


def _vary_contrast(image, bboxs, filename=None):
    """
    Randomly varies the pixel-wise contrast of the image. This is the only
    pixel-wise augmentation that can be performed on single-channel imagery.
     The bounding boxes are not changed in any way by this function.

    :param image: input image tensor of Shape = Height X Width X #Channels
    :param bboxs: input bounding boxes of Shape = #Boxes X 4
    (ymin, xmin, ymax, xmax)
    :return: the image and transformed bounding box tensor
    """

    cond_contrast = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32),
                            tf.bool)
    image = tf.cond(cond_contrast,
                    lambda: tf.image.random_contrast(image, 0.2, 1.8),
                    lambda: tf.identity(image))
    if filename is not None:
        return image, bboxs, filename
    else:
        return image, bboxs


def _crop_random(image, bboxs, filename=None):
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
                           lambda: (tf.identity(image), tf.identity(bboxs)))
    if filename is not None:
        return image, bboxs, filename
    else:
        return image, bboxs


def _perform_crop(image, bboxs, filename=None, min_crop_size=(400, 400)):
    """
    Randomly crops the image. Crop positions are chosen randomly, as well as
    the size (provided it is above the requested minimum size). If any bounding
    boxes are only partially included in the crop, the crop is enlarged so that
    the bounding box falls totally within the crop.

    Note: this makes batching difficult, as image sizes will no longer be
    consistent. To use this function in the augmentation chain, resizing will be
    needed after cropping but before batching.

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
    crop_width = tf.random.uniform(
        [],
        minval=min_crop_size[1],
        maxval=img_shape[1] + 1,
        dtype=tf.int32
    )
    crop_height = tf.random.uniform(
        [],
        minval=min_crop_size[0],
        maxval=img_shape[0] + 1,
        dtype=tf.int32
    )

    # Now come up with crop offsets
    offset_width = tf.random.uniform(
        [],
        minval=0,
        maxval=img_shape[1] - crop_width,
        dtype=tf.int32
    )
    offset_height = tf.random.uniform(
        [],
        minval=0,
        maxval=img_shape[0] - crop_height,
        dtype=tf.int32
    )

    # If we ever split a box with our crop, increase the crop size to include.
    #  Positives are already too scarce
    # First convert bounding boxes to pixel coordinates (rather than percents)
    bbox_convert_tensor = tf.stack([img_shape[0],
                                    img_shape[1],
                                    img_shape[0],
                                    img_shape[1],
                                    1], axis=0)
    bbox_coords = bboxs * tf.cast(bbox_convert_tensor, dtype=tf.float32)
    bbox_coords = tf.cast(bbox_coords, dtype=tf.int32)

    # Need to figure out which boxes are totally within the crop, and which ones
    #  are totally outside the crop
    bbox_x_in_crop = tf.logical_and(tf.greater(bbox_coords[:, 1], offset_width),
                                    tf.greater(offset_width + crop_width,
                                               bbox_coords[:, 3]))
    bbox_y_in_crop = tf.logical_and(tf.greater(bbox_coords[:, 0],
                                               offset_height),
                                    tf.greater(offset_height + crop_height,
                                               bbox_coords[:, 2]))
    bbox_in_crop = tf.logical_and(bbox_x_in_crop, bbox_y_in_crop)

    bbox_x_out_of_crop = tf.logical_or(tf.greater(bbox_coords[:, 1],
                                                  offset_width + crop_width),
                                       tf.greater(offset_width,
                                                  bbox_coords[:, 3]))
    bbox_y_out_of_crop = tf.logical_or(tf.greater(bbox_coords[:, 0],
                                                  offset_height + crop_height),
                                       tf.greater(offset_height,
                                                  bbox_coords[:, 2]))
    bbox_out_of_crop = tf.logical_or(bbox_x_out_of_crop, bbox_y_out_of_crop)

    # Boxes not at all in the crop should be changed to negatives
    classes = bboxs[:, 4] * tf.cast(tf.logical_not(bbox_out_of_crop),
                                    dtype=tf.float32)

    # The problematic boxes are the ones that aren't either totally in the crop
    #  or totally out of the crop
    bboxes_split_mask = tf.logical_not(tf.logical_or(bbox_in_crop,
                                                     bbox_out_of_crop))
    bboxes_split = tf.boolean_mask(bbox_coords, bboxes_split_mask)
    min_x = tf.reduce_min(bboxes_split[:, 1])
    max_x = tf.reduce_max(bboxes_split[:, 3])
    min_y = tf.reduce_min(bboxes_split[:, 0])
    max_y = tf.reduce_max(bboxes_split[:, 2])

    offset_width = tf.minimum(offset_width, min_x)
    offset_height = tf.minimum(offset_height, min_y)

    # A strange bug occurs when there were no split boxes. Have to make sure max
    # _x and max_y are defined
    max_x = tf.maximum(max_x, offset_width)
    max_y = tf.maximum(max_y, offset_height)

    crop_height = tf.maximum(crop_height, max_y - offset_height)
    crop_width = tf.maximum(crop_width, max_x - offset_width)

    # The heavy lifting is done, time to make us a crop and transform our
    # bounding boxes to the new coordinates
    image = tf.image.crop_to_bounding_box(image,
                                          offset_height,
                                          offset_width,
                                          crop_height,
                                          crop_width)

    # Precision is an issue, need to cast to float before we start doing math
    bbox_coords = tf.cast(bbox_coords, tf.float32)
    offset_width = tf.cast(offset_width, tf.float32)
    offset_height = tf.cast(offset_height, tf.float32)
    crop_width = tf.cast(crop_width, tf.float32)
    crop_height = tf.cast(crop_height, tf.float32)

    ymin = (bbox_coords[:, 0] - offset_height) / crop_height
    xmin = (bbox_coords[:, 1] - offset_width) / crop_width
    ymax = (bbox_coords[:, 2] - offset_height) / crop_height
    xmax = (bbox_coords[:, 3] - offset_width) / crop_width

    bbox_new = tf.stack([ymin, xmin, ymax, xmax, classes], axis=1)

    if filename is not None:
        return image, bbox_new, filename
    else:
        return image, bbox_new


def _flip_left_right(image, bboxs, filename=None):
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
    cond_flip = tf.logical_not(tf.reduce_all(tf.equal(image_one, image_two)))

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


def _flip_up_down(image, bboxs, filename=None):
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
    cond_flip = tf.logical_not(tf.reduce_all(tf.equal(image_one, image_two)))

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


def _rotate_random(image, bboxs, filename=None):
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
                           lambda: (tf.identity(image), tf.identity(bboxs)))

    if filename is not None:
        return image, bboxs, filename
    else:
        return image, bboxs


def _perform_rotation(image, bboxs, filename=None):
    """
    Rotates images randomly either 90 degrees left or right.
    The decision to rotate or not is decided by _rotate_random.

    :param image: input image tensor of Shape = Height X Width X #Channels
    :param bboxs: input bounding boxes of Shape = #Boxes X 4
    (ymin, xmin, ymax, xmax)
    :return: the image and transformed bounding box tensor
    """
    # Either rotate once clockwise or counter clockwise
    cond_direction = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32),
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
                                     lambda: (1 - xmax, ymin, 1 - xmin, ymax),
                                     lambda: (xmin, 1 - ymax, xmax, 1 - ymin))

    bbox_new = tf.stack([ymin, xmin, ymax, xmax, bboxs[:, 4]], axis=1)
    if filename is not None:
        return image, bbox_new, filename
    else:
        return image, bbox_new


def _resize_data(image, bboxs, filename=None, image_size=(2048, 2048)):
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
