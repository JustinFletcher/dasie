import os
import glob
import json
import argparse
import collections
import numpy as np

# import mutedcolors
from matplotlib import pyplot as plt

import matplotlib


def save_and_close_current_plot(logdir, plot_name="default", dpi=600):
    fig_path = os.path.join(logdir, str(plot_name) + '.png')
    plt.gcf().set_dpi(dpi)
    plt.savefig(fig_path)
    plt.close()


def plot_dasie_performance(logdir, step=None):
    results_dict = get_latest_results_dict(logdir)
    # Plotting Palette
    # results_dict["results"]["train_loss_list"] = list()
    # results_dict["results"]["train_dist_mse_list"] = list()
    # results_dict["results"]["train_mono_mse_list"] = list()
    # results_dict["results"]["train_mse_ratio_list"] = list()
    # results_dict["results"]["train_dist_ssim_list"] = list()
    # results_dict["results"]["train_mono_ssim_list"] = list()
    # results_dict["results"]["train_ssim_ratio_list"] = list()
    # results_dict["results"]["train_dist_psnr_list"] = list()
    # results_dict["results"]["train_mono_psnr_list"] = list()
    # results_dict["results"]["train_psnr_ratio_list"] = list()
    # results_dict["results"]["valid_loss_list"] = list()
    # results_dict["results"]["valid_dist_mse_list"] = list()
    # results_dict["results"]["valid_mono_mse_list"] = list()
    # results_dict["results"]["valid_mse_ratio_list"] = list()
    # results_dict["results"]["valid_dist_ssim_list"] = list()
    # results_dict["results"]["valid_mono_ssim_list"] = list()
    # results_dict["results"]["valid_ssim_ratio_list"] = list()
    # results_dict["results"]["valid_dist_psnr_list"] = list()
    # results_dict["results"]["valid_mono_psnr_list"] = list()
    # results_dict["results"]["valid_psnr_ratio_list"] = list()
    # results_dict["results"]["train_epoch_time_list"] = list()

    if step:

        plot_logdir = os.path.join(logdir, "step_" + str(step) + "_plots")

    else:
        plot_logdir = logdir

    fig, ax = plt.subplots()

    ax.plot(results_dict["results"]["train_loss_list"],
            label="Training Loss")

    ax.plot(results_dict["results"]["valid_loss_list"],
            label="Validation Loss")

    ax.set(xlabel='Epoch',
           ylabel='Loss Value (%s)' % results_dict["loss_name"],
           title='DASIE Training Learning Curve')

    ax.legend()

    ax.grid()

    save_and_close_current_plot(plot_logdir, plot_name="learning_curve")

    return

def get_latest_results_dict(logdir):


    # Get the latest json file from the logdir.
    jsons = glob.glob(os.path.join(logdir, "results_*"))

    jsons.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    latest_json = jsons[-1]
    print("Reading: " + str(latest_json))
    results_dict = json.load(open(latest_json))

    return(results_dict)

def main(flags):

    results_dict = get_latest_results_dict(flags.logdir)

    plot_dasie_performance(results_dict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='provide arguments for visualizations.')

    parser.add_argument('--logdir',
                        type=str,
                        default=sorted(glob.glob(os.path.join(".", "logs", "20*")))[-1],
                        help='The directory to which summaries are written.')


    parser.add_argument('--plot_type',
                        type=str,
                        default="ensemble",
                        help='The type of plot to display.')


    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)