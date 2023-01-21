import numpy as np
import tensorflow as tf
from zernike import *

class RecoveryModel(object):

    def __init__(self,
                 distributed_aperture_images_batch,
                 filter_scale,
                 num_exposures,
                 image_x_scale,
                 image_y_scale,
                 batch_size,
                 model_type="tseng2021neural",
                 optics_only_mtfs=None):



        with tf.name_scope("recovery_model"):

            recovered_image_batch = None

            if model_type == "tseng2021neural":

                recovered_image_batch = self.build_tseng2021neural(
                  distributed_aperture_images_batch,
                  filter_scale,
                  num_exposures,
                  image_x_scale,
                  image_y_scale,
                  batch_size
                )

            elif model_type == "weighted_average":

                recovered_image_batch = self.build_weighted_average(
                    distributed_aperture_images_batch,
                    num_exposures,
                    image_x_scale,
                    image_y_scale,
                    batch_size
                )

            elif model_type == "zenrike_feature_network":

                if not optics_only_mtfs:
                    "zenrike_feature_network without optics_only_mtfs!"

                recovered_image_batch = self.build_zernike_feature_stack(
                    distributed_aperture_images_batch,
                    optics_only_mtfs,
                    filter_scale,
                    num_exposures,
                    image_x_scale,
                    image_y_scale,
                    batch_size
                )

            self.recovered_image_batch = recovered_image_batch

        return


    def build_zernike_feature_stack(self,
                               distributed_aperture_images_batch,
                               optics_only_mtfs,
                               num_exposures,
                               image_x_scale,
                               image_y_scale,
                               batch_size
                               ):

        print(distributed_aperture_images_batch.shape)
        distributed_aperture_images_batch / optics_only_mtfs

        # Generate a single scalar variable for each exposure.
        weights = tf.Variable(tf.ones_like(list(range(num_exposures)),
                                           dtype=tf.float64))


        # Multiply each element by the weight vector.
        weighted_image_batch = distributed_aperture_images_batch * weights

        # Add up each element along the image stack dimension.
        weighted_sum_image_batch = tf.math.reduce_sum(
            weighted_image_batch,
            axis=[-1]
        )

        # Divide each element by the number of exposures.
        recovered_image_batch = weighted_sum_image_batch / num_exposures

        return recovered_image_batch


    def build_weighted_average(self,
                               distributed_aperture_images_batch,
                               num_exposures,
                               image_x_scale,
                               image_y_scale,
                               batch_size
                               ):


        # Generate a single scalar variable for each exposure.
        weights = tf.Variable(tf.ones_like(list(range(num_exposures)),
                                           dtype=tf.float64))


        # Multiply each element by the weight vector.
        weighted_image_batch = distributed_aperture_images_batch * weights

        # Add up each element along the image stack dimension.
        weighted_sum_image_batch = tf.math.reduce_sum(
            weighted_image_batch,
            axis=[-1]
        )

        # Divide each element by the number of exposures.
        recovered_image_batch = weighted_sum_image_batch / num_exposures

        return recovered_image_batch

    def build_tseng2021neural(self,
                              distributed_aperture_images_batch,
                              filter_scale,
                              num_exposures,
                              image_x_scale,
                              image_y_scale,
                              batch_size):

        self.image_x_scale = image_x_scale
        self.image_y_scale = image_y_scale
        self.batch_size = batch_size


        with tf.name_scope("recovery_feature_extractor"):
            input = distributed_aperture_images_batch
            # down_l0 conv-c15-k7-s1-LRelu input
            down_l0 = self.conv_block(input,
                                      input_channels=num_exposures,
                                      output_channels=filter_scale,
                                      kernel_size=7,
                                      stride=1,
                                      activation="LRelu")
            #

            # down_l0 conv-c15-k7-s1-LRelu down_l0
            down_l0 = self.conv_block(down_l0,
                                      input_channels=filter_scale,
                                      output_channels=filter_scale,
                                      kernel_size=7,
                                      stride=1,
                                      activation="LRelu")
            #

            # down_l1 conv-c30-k5-s2-LRelu down_l0
            down_l1_0 = self.conv_block(down_l0,
                                        input_channels=filter_scale,
                                        output_channels=filter_scale * 2,
                                        kernel_size=5,
                                        stride=2,
                                        activation="LRelu")
            #

            # down_l1 conv-c30-k3-s1-LRelu down_l1
            down_l1 = self.conv_block(down_l1_0,
                                      input_channels=filter_scale * 2,
                                      output_channels=filter_scale * 2,
                                      kernel_size=3,
                                      stride=1,
                                      activation="LRelu")
            #

            # down_l1 conv-c30-k3-s1-LRelu down_l1
            down_l1 = self.conv_block(down_l1,
                                      input_channels=filter_scale * 2,
                                      output_channels=filter_scale * 2,
                                      kernel_size=3,
                                      stride=1,
                                      activation="LRelu")
            #

            # down_l2 conv-c60-k5-s2-LRelu down_l1
            down_l2 = self.conv_block(down_l1_0,
                                      input_channels=filter_scale,
                                      output_channels=filter_scale * 4,
                                      kernel_size=5,
                                      stride=2,
                                      activation="LRelu")
            #

            # down_l2 conv-c60-k3-s1-LRelu down_l2
            down_l2 = self.conv_block(down_l2,
                                      input_channels=filter_scale * 4,
                                      output_channels=filter_scale * 4,
                                      kernel_size=3,
                                      stride=1,
                                      activation="LRelu")
            #

            # down_l2 conv-c60-k3-s1-LRelu down_l2
            down_l2 = self.conv_block(down_l2,
                                      input_channels=filter_scale * 4,
                                      output_channels=filter_scale * 4,
                                      kernel_size=3,
                                      stride=1,
                                      activation="LRelu")
            #
            # End of downsample and pre-skip.

            # conv_l2_k0 conv-c60-k3-s1-LRelu down_l2
            conv_l2_k0 = self.conv_block(down_l2,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l2_k1 conv-c60-k3-s1-LRelu conv_l2_k0
            conv_l2_k1 = self.conv_block(conv_l2_k0,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l2_k2 conv-c60-k3-s1-LRelu Concat([conv_l2_k0, conv_l2_k1])
            conv_l2_k2 = self.conv_block(
                tf.concat([conv_l2_k0, conv_l2_k1], axis=-1),
                input_channels=filter_scale * 8,
                output_channels=filter_scale * 4,
                kernel_size=3,
                stride=1,
                activation="LRelu")
            #

            # conv_l2_k3 conv-c60-k3-s1-LRelu conv_l2_k2
            conv_l2_k3 = self.conv_block(conv_l2_k2,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l2_k4 conv-c60-k3-s1-LRelu conv_l2_k3
            conv_l2_k4 = self.conv_block(conv_l2_k3,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l2_k5 conv-c60-k3-s1-LRelu conv_l2_k4
            conv_l2_k5 = self.conv_block(conv_l2_k4,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #
            # End of bottom pipe.

            # Mid-resolution.
            # conv_l1_k0 conv-c30-k3-s1-LRelu down_l1
            conv_l1_k0 = self.conv_block(down_l1,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l1_k1 conv-c30-k3-s1-LRelu conv_l1_k0
            conv_l1_k1 = self.conv_block(conv_l1_k0,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #
            # conv_l1_k2 conv-c30-k3-s1-LRelu Concat([conv_l1_k0, conv_l1_k1])
            conv_l1_k2 = self.conv_block(
                tf.concat([conv_l1_k0, conv_l1_k1], axis=-1),
                input_channels=filter_scale * 4,
                output_channels=filter_scale * 2,
                kernel_size=3,
                stride=1,
                activation="LRelu")
            #

            # conv_l1_k3 conv-c30-k3-s1-LRelu conv_l1_k2
            conv_l1_k3 = self.conv_block(conv_l1_k2,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l1_k4 conv-c30-k3-s1-LRelu conv_l1_k3
            conv_l1_k4 = self.conv_block(conv_l1_k3,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l1_k5 conv-c30-k3-s1-LRelu conv_l1_k4
            conv_l1_k5 = self.conv_block(conv_l1_k4,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # up_l2 convT-c30-k2-s2-LRelu conv_l2_k5
            # modded to: up_l2 convT-c60-k2-s2-LRelu conv_l2_k5
            # pull the bottom pipe up.
            up_l2 = self.convT_block(conv_l2_k5,
                                     input_downsample_factor=4,
                                     input_channels=filter_scale * 4,
                                     output_channels=filter_scale * 4,
                                     kernel_size=2,
                                     stride=2,
                                     activation="LRelu")
            #

            # conv_l1_k6 conv-c30-k3-s1-LRelu Concat([up_l2, conv_l1_k5])
            # modded to: input 60
            conv_l1_k6 = self.conv_block(tf.concat([up_l2, conv_l1_k5], axis=-1),
                                         input_channels=filter_scale * 6,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l1_k7 conv-c30-k3-s1-LRelu conv_l1_k6
            conv_l1_k7 = self.conv_block(conv_l1_k6,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #
            # End of mid-resolution pipe.

            # High Resolution.
            # conv_l0_k0 conv-c15-k3-s1-LRelu down_l0
            conv_l0_k0 = self.conv_block(down_l0,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #
            # conv_l0_k1 conv-c15-k3-s1-LRelu conv_l0_k0
            conv_l0_k1 = self.conv_block(conv_l0_k0,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l0_k2 conv-c15-k3-s1-LRelu Concat([conv_l1_k0, conv_l0_k1])c
            # halv the input size = conv_l1_k0
            # This is wrong in tseng2021neural! The sizes don't match!
            # Modded to: conv_l0_k2 conv-c15-k3-s1-LRelu Concat([conv_l0_k0, conv_l0_k1])
            # Then I moved the whole concat down to be consistent with fig 5
            conv_l0_k2 = self.conv_block(conv_l0_k1,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l0_k3 conv-c15-k3-s1-LRelu conv_l0_k2
            conv_l0_k3 = self.conv_block(conv_l0_k2,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l0_k4 conv-c15-k3-s1-LRelu conv_l0_k3
            # Move the skip connection here to be consistent with Fig 5.
            conv_l0_k4 = self.conv_block(
                tf.concat([conv_l0_k0, conv_l0_k3], axis=-1),
                input_channels=filter_scale * 2,
                output_channels=filter_scale,
                kernel_size=3,
                stride=1,
                activation="LRelu")
            #

            # conv_l0_k5 conv-c15-k3-s1-LRelu conv_l0_k4
            conv_l0_k5 = self.conv_block(conv_l0_k4,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # up_l1 convT-c15-k2-s2-LRelu conv_l1_k5
            # modded to: up_l1 convT-c30-k2-s2-LRelu conv_l1_k5
            up_l1 = self.convT_block(conv_l1_k5,
                                     input_downsample_factor=2,
                                     input_channels=filter_scale * 2,
                                     output_channels=filter_scale * 2,
                                     kernel_size=2,
                                     stride=2,
                                     activation="LRelu")
            #

            # conv_l0_k6 conv-c15-k3-s1-LRelu Concat([up_l1, conv_l0_k5])
            conv_l0_k6 = self.conv_block(tf.concat([up_l1, conv_l0_k5], axis=-1),
                                         input_channels=filter_scale * 3,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l0_k7 conv-c15-k3-s1-LRelu conv_l0_k6
            conv_l0_k7 = self.conv_block(conv_l0_k6,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")

        with tf.name_scope("recovery_feature_propagator"):
            # fp_l0 Feature Propagator (PSF_1x, conv_l0_k7)
            # TODO: Implement feature propagator.
            fp_l0 = conv_l0_k7
            # fp_l1 Feature Propagator (PSF_2x, conv_l1_k7)
            # TODO: Implement feature propagator.
            fp_l1 = conv_l1_k7
            # fp_l2 Feature Propagator (PSF_4x, conv_l2_k5)
            # TODO: Implement feature propagator.
            fp_l2 = conv_l2_k5

        with tf.name_scope("recovery_decoder"):
            # conv_l0_k0 conv-c30-k5-s1-LRelu fp_l0
            conv_l0_k0 = self.conv_block(fp_l0,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=5,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l0_k1 conv-c30-k5-s1-LRelu conv_l0_k0
            conv_l0_k1 = self.conv_block(conv_l0_k0,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=5,
                                         stride=1,
                                         activation="LRelu")
            #
            # down_l0 conv-c30-k5-s2-LRelu conv_l0_k1
            down_l0 = self.conv_block(conv_l0_k1,
                                      input_channels=filter_scale,
                                      output_channels=filter_scale * 2,
                                      kernel_size=5,
                                      stride=2,
                                      activation="LRelu")
            #

            # conv_l1_k0 conv-c60-k3-s1-LRelu Concat([fp_l1, down_l0])
            conv_l1_k0 = self.conv_block(tf.concat([fp_l1, down_l0], axis=-1),
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #
            # conv_l1_k1 conv-c60-k3-s1-LRelu conv_l1_k0
            conv_l1_k1 = self.conv_block(conv_l1_k0,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # down_l1 conv-c60-k3-s2-LRelu conv_l1_k1
            down_l1 = self.conv_block(conv_l1_k1,
                                      input_channels=filter_scale * 4,
                                      output_channels=filter_scale * 4,
                                      kernel_size=3,
                                      stride=2,
                                      activation="LRelu")
            #

            # conv_l2_k0 conv-c120-k3-s1-LRelu Concat([fp_l2, down_l1])
            # Modded to 60.
            conv_l2_k0 = self.conv_block(tf.concat([fp_l2, down_l1], axis=-1),
                                         input_channels=filter_scale * 8,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l2_k1 conv-c120-k3-s1-LRelu conv_l2_k0
            # Report error - this is never used, even in the paper.
            conv_l2_k1 = self.conv_block(conv_l2_k0,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l2_k2 conv-c120-k3-s1-LRelu Concat([conv_l2_k0, fp_l2, down_l1])
            # NOTE: This was wrong in the tseng2021neural! conv_l2_k0 -> conv_l2_k1
            conv_l2_k2 = self.conv_block(
                tf.concat([conv_l2_k1, fp_l2, down_l1], axis=-1),
                # change to 10 if breaks
                input_channels=filter_scale * 12,
                output_channels=filter_scale * 4,
                kernel_size=3,
                stride=1,
                activation="LRelu")
            #

            # conv_l2_k3 conv-c120-k3-s1-LRelu conv_l2_k2
            # modded to: conv_l2_k3 conv-c60-k3-s1-LRelu conv_l2_k2
            conv_l2_k3 = self.conv_block(conv_l2_k2,
                                         input_channels=filter_scale * 4,
                                         output_channels=filter_scale * 4,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # up_l2 convT-c60-k2-s2-LRelu conv_l2_k3
            # modded to 60 input
            up_l2 = self.convT_block(conv_l2_k3,
                                     input_downsample_factor=4,
                                     input_channels=filter_scale * 4,
                                     output_channels=filter_scale * 4,
                                     kernel_size=2,
                                     stride=2,
                                     activation="LRelu")
            #

            # conv_l1_k2 conv-c60-k3-s1-LRelu Concat([conv_l1_k1, up_l2])
            # modded to 30.
            conv_l1_k2 = self.conv_block(tf.concat([conv_l1_k1, up_l2], axis=-1),
                                         input_channels=filter_scale * 8,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # conv_l1_k3 conv-c60-k3-s1-LRelu conv_l1_k2
            conv_l1_k3 = self.conv_block(conv_l1_k2,
                                         input_channels=filter_scale * 2,
                                         output_channels=filter_scale * 2,
                                         kernel_size=3,
                                         stride=1,
                                         activation="LRelu")
            #

            # up_l1 convT-c60-k2-s2-LRelu conv_l2_k3
            # NOTE: This was wrong in the tseng2021neural! conv_l2_k3 -> conv_l1_k3
            up_l1 = self.convT_block(conv_l1_k3,
                                     input_downsample_factor=2,
                                     input_channels=filter_scale * 2,
                                     output_channels=filter_scale * 2,
                                     kernel_size=2,
                                     stride=2,
                                     activation="LRelu")
            #

            # conv_l0_k2 conv-c30-k5-s1-LRelu Concat([conv_l0_k1, up_l1])
            conv_l0_k2 = self.conv_block(tf.concat([conv_l0_k1, up_l1], axis=-1),
                                         input_channels=filter_scale * 3,
                                         output_channels=filter_scale,
                                         kernel_size=5,
                                         stride=1,
                                         activation="LRelu")
            #

            # Output RGB conv_l0_k2
            # NOTE: This was underspecified in tseng2021neural!
            conv_l0_k3 = self.conv_block(conv_l0_k2,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=2,
                                         stride=1,
                                         activation="LRelu")
            #
            conv_l0_k4 = self.conv_block(conv_l0_k3,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=2,
                                         stride=1,
                                         activation="LRelu")
            conv_l0_k5 = self.conv_block(conv_l0_k4,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=2,
                                         stride=1,
                                         activation="LRelu")
            conv_l0_k6 = self.conv_block(conv_l0_k5,
                                         input_channels=filter_scale,
                                         output_channels=filter_scale,
                                         kernel_size=2,
                                         stride=1,
                                         activation="LRelu")
            conv_l0_k7 = self.conv_block(conv_l0_k6,
                                         input_channels=filter_scale,
                                         output_channels=1,
                                         kernel_size=2,
                                         stride=1,
                                         activation="LRelu")
            conv_l0_k0 = self.conv_block(conv_l0_k7,
                                         input_channels=1,
                                         output_channels=1,
                                         kernel_size=2,
                                         stride=1,
                                         activation="LRelu")

        # Remove the now irrelevant channel dim.
        recovered_image_batch = tf.squeeze(conv_l0_k0)

        return recovered_image_batch


    def convT_block(self,
                   input_feature_map,
                   input_downsample_factor,
                   input_channels,
                   output_channels=1,
                   kernel_size=2,
                   stride=1,
                   activation="LRelu",
                   name=None):

        if not name:

            name = "convT-c" + str(output_channels) + "-k" + str(kernel_size) + "-s" + str(stride) + "-" + activation

        with tf.name_scope(name):

            # Initialize the filter variables as he2015delving.
            he_relu_init_std = np.sqrt(2 / (input_channels * (kernel_size**2)))
            filters = tf.Variable(tf.random.normal((kernel_size,
                                                    kernel_size,
                                                    input_channels,
                                                    output_channels),
                                                   stddev=he_relu_init_std,
                                                   dtype=tf.float64))

            # Encode the strides for TensorFlow, and build the conv graph.
            strides = [1, stride, stride, 1]

            # Given the base quantization, div by downsample, mul by stride.
            # print(self.image_x_scale)
            output_x_scale = (self.image_x_scale // input_downsample_factor) * stride
            output_y_scale = (self.image_y_scale // input_downsample_factor) * stride
            output_shape = (self.batch_size,
                            output_x_scale,
                            output_y_scale,
                            output_channels)

            conv_output = tf.nn.conv2d_transpose(input_feature_map,
                                                 filters,
                                                 output_shape,
                                                 strides,
                                                 padding="SAME",
                                                 data_format='NHWC',
                                                 dilations=None,
                                                 name=name)

            # Apply an activation function.
            output_feature_map = tf.nn.leaky_relu(conv_output, alpha=0.02)

        return output_feature_map

    def conv_block(self,
                   input_feature_map,
                   input_channels,
                   output_channels=1,
                   kernel_size=2,
                   stride=1,
                   activation="LRelu",
                   name=None):

        if not name:

            name = "conv-c" + str(output_channels) + "-k" + str(kernel_size) + "-s" + str(stride) + "-" + activation

        with tf.name_scope(name):

            # Initialize the filter variables as he2015delving.
            he_relu_init_std = np.sqrt(2 / (input_channels * (kernel_size**2)))
            filters = tf.Variable(tf.random.normal((kernel_size,
                                                    kernel_size,
                                                    input_channels,
                                                    output_channels),
                                                   stddev=he_relu_init_std,
                                                   dtype=tf.float64))

            # Encode the strides for TensorFlow, and build the conv graph.
            strides = [1, stride, stride, 1]
            conv_output = tf.nn.conv2d(input_feature_map,
                                       filters,
                                       strides,
                                       padding="SAME",
                                       data_format='NHWC',
                                       dilations=None,
                                       name=None)

            # Apply an activation function.
            output_feature_map = tf.nn.leaky_relu(conv_output, alpha=0.02)

        return output_feature_map

    def zernike_block(self,
                      input_feature_map,
                      input_downsample_factor,
                      input_channels,
                      output_channels=1,
                      kernel_size=2,
                      stride=1,
                      activation="LRelu",
                      name=None):

        input_spectrum = tf.signal.fft2d(
            tf.cast(
                input_feature_map,
                dtype=tf.complex128
            )
        )

        # TODO: always read the zernikes from the forward model, becuase these
        #       will be known at inference time. Learn masking thresholds and
        #       biases to the coefficients! Then, you can learn several zernike
        #       filters per exposure, each representing hypotheses about the
        #       spatial frequency features most relevant to a particular
        #       articulation choice. Finally, we can choose between two heads
        #       One simply applies the thresholds, averages, and then weighted
        #       averages across exposures, which regularizes strongly. The
        #       other is a typical Unet like thing that maps the raw feature
        #       maps from all exposures together.

        # TODO: Externalize.
        num_zernike_indices = 6
        num_apertures = 15
        spatial_quantization = 512
        pupil_extent = 3.67 * 1.1
        radius_meters = 3.67
        subaperture_radius_meters = 1.0

        u = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        v = np.linspace(-pupil_extent/2, pupil_extent/2, spatial_quantization)
        pupil_grid_u, pupil_grid_v = np.meshgrid(u, v)

        pupil_size = (spatial_quantization,
                      spatial_quantization)

        # Build the pupil plane quantization grid for this exposure.
        optics_only_pupil_plane = tf.zeros(pupil_size,
                                           dtype=tf.complex128)

        for aperture_num in range(num_apertures):
            print("Building aperture number %d." % aperture_num)

            # Compute the subap centroid cartesian coordinates.
            rotation = (aperture_num + 1) / num_apertures
            edge_radius = radius_meters - subaperture_radius_meters
            mu_u = edge_radius * np.cos((2 * np.pi) * rotation)
            mu_v = edge_radius * np.sin((2 * np.pi) * rotation)

            # Initialize the coefficients for this subaperture.
            subap_zernike_coeffs = init_zernike_coefficients(
                num_zernike_indices=num_zernike_indices,
                zernike_init_type="constant",
            )

            # Build TF Variables around the coefficients.
            subap_zernike_coefficients_vars = self._build_zernike_coefficient_variables(
                subap_zernike_coeffs,
            )

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

        # Compute the PSF of the pupiil
        pupil_spectrum = tf.signal.fft2d(optics_only_pupil_plane)
        shifted_pupil_spectrum = tf.signal.fftshift(pupil_spectrum)
        psf = tf.abs(shifted_pupil_spectrum) ** 2

        # Compute the OTF, which is the Fourier transform of the PSF.
        with tf.name_scope("filter_otf"):

            otf = tf.signal.fft2d(tf.cast(psf, tf.complex128))

        # Compute the mtf, which is the real component of the OTF.
        with tf.name_scope("filter_mtf"):

            filter_mtf = tf.math.abs(otf)

        feature_spectrum = input_spectrum * tf.cast(filter_mtf, dtype=tf.complex128)
        output_feature_map = tf.abs(tf.signal.fft2d(feature_spectrum))


        return output_feature_map


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



