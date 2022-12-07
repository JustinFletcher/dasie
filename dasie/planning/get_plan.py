import os
import json
import argparse


import numpy as np



def dasie_save_to_plan_arry(dasie_model_save_dict):

    kwargs = dasie_model_save_dict["kwargs"]
    num_exposures = kwargs["num_exposures"]
    num_apertures = kwargs["num_apertures"]
    num_zernike_indices = kwargs["num_zernike_indices"]
    plan_array = np.zeros((num_exposures, num_apertures, num_zernike_indices))

    for (var_name, var_value) in dasie_model_save_dict["variables"]:
        "dasie_model/exposure_0/pupil_plane_model/subaperture_0/zernike_coefficient_0:0"
        # Split variable name by /.
        var_scope_list = var_name.split('/')
        if var_scope_list[0] == "dasie_model":

            # Parse the exposure number.
            exposure_name = var_scope_list[1]
            exposure_num = exposure_name.split('_')[-1]

            # Parse the subaperture number.
            subaperture_name = var_scope_list[3]
            subaperture_num = subaperture_name.split('_')[-1]

            # Parse the zernike term number; extra step is due to a tf quirk.
            zernike_coefficient_name = var_scope_list[4]
            zernike_coefficient_name = zernike_coefficient_name.split(':')[0]
            zernike_coefficient_num = zernike_coefficient_name.split('_')[-1]

            plan_array[exposure_num,
                       subaperture_num,
                       zernike_coefficient_num] = var_value

    return plan_array

def main(flags):

    dasie_model_save_dict = json.load(flags.dasie_model_save_file)

    plan_array = dasie_save_to_plan_array(dasie_model_save_dict)

    print(plan_array, indent=4)

    return plan_array


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
