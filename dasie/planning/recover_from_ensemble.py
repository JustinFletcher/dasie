import os
import json
import argparse

import tensorflow as tf
from matplotlib import image

from differentiable_dasie import DASIEModel


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


    # Begin by creating a new session.
    with tf.compat.v1.Session() as sess:
        print("Restoring Recovery Model.")

        restore_dict = json.load(open(flags.dasie_model_save_file, 'r'))

        dasie_model = DASIEModel(sess, **restore_dict["kwargs"])

        dasie_model.restore(flags.dasie_model_save_file)
        print("Recovery Model Restored.")

        images = list()

        # Iterate over each file in the provided directory, loading images.
        for f in os.listdir(flags.image_ensemble_path):
            full_image_path = os.path.join(flags.image_ensemble_path, f)
            if os.path.isfile(full_image_path):

                # TODO: check extension.
                filename, extension = os.path.splitext(f)
                if extension == ".jpg":
                    image = read_jpg(full_image_path)
                elif extension == ".png":
                    image = read_png(full_image_path)
                elif extension == ".fits":
                    image = read_fits(full_image_path)
                else:
                    raise NotImplementedError("The supplied file type %s\
                                               is not yet supported. Extend\
                                               recover_from_ensemble.py to \
                                               add new types." % extension)
                images.append(image)

        if len(images) != restore_dict["kwargs"]["num_exposures"]:
            raise Exception("The image_ensemble_path contains %d images, but \
                              the model restored from dasie_model_save_file \
                              has a num_exposures of %d. These must be equal.\
                              " % (len(images),
                                   restore_dict["kwargs"]["num_exposures"]))

        recovered_image = dasie_model.recover(images)


        full_output_file_path =  os.path.join(flags.output_file_path,
                                              flags.output_file_name)
        if extension == ".jpg":
            write_jpg(recovered_image, full_output_file_path)
        elif extension == ".png":
            write_png(recovered_image, full_output_file_path)
        elif extension == ".fits":
            write_fits(recovered_image, full_output_file_path)
        else:
            raise NotImplementedError("The supplied file type %s\
                                       is not yet supported. Extend\
                                       recover_from_ensemble.py to \
                                       add new types." % extension)

    return


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
                        default=os.path.join(".",
                                             "dasie",
                                             "resources",
                                             "example_ensemble"),
                        help='The path to a directory of images, from which \
                              the recovered model will infer to produce a \
                              reconstructed image.')

    parser.add_argument('--output_file_path',
                        type=str,
                        default=".",
                        help='The name of the output image file.')

    parser.add_argument('--output_file_name',
                        type=str,
                        default="recovered_image",
                        help='The path at at which the recovered image will\
                              be saved.')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)