import argparse

def main(flags):


    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:
        print("Restoring Recovery Model." % i)

        restore_dict = json.load(open(dasie_model_save_file, 'r'))

        dasie_model = DASIEModel(sess, restore_dict["kwargs"])

        dasie_model.restore(dasie_model_save_file)
        print("Recovery Model Restored." % i)
        die

    return "Hello"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='provide arguments for training.')

    parser.add_argument('--gpu_list', type=str,
                        default="0",
                        help='GPUs to use with this model.')

    parser.add_argument('--dasie_model_save_file',
                        type=str,
                        help='The save file for the model to load.')

    parser.add_argument('--image_ensemble_path',
                        type=str,
                        help='The path to a directory of images, from which \
                              the recovered model will infer to produce a \
                              reconstructed image.')

    main(flags)