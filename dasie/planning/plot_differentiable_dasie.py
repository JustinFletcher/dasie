import os
import glob
import json
import argparse
import collections
import numpy as np

import mutedcolors
from matplotlib import pyplot as plt

import matplotlib

def plot_differentiability_ablation(results_dicts):

    differentiable_dicts = list()
    nondifferentiable_dicts = list()

    for results_dict in results_dicts:
        if results_json["lock_ttp_values"]:
            nondifferentiable_dicts.append(results_dict)
        else:
            differentiable_dicts.append(results_dict)

def plot_ensemble_ablation(results_dicts):

    for results_dict in results_dicts:
        if results_json["lock_ttp_values"]:
            nondifferentiable_dicts.append(results_dict)
        else:
            differentiable_dicts.append(results_dict)

def trim_lists_to_shortest(list_of_lists):

    min_len = np.min([len(r) for r in list_of_lists])
    # Plot the mean and variance of the colleciton of lists.
    return [r[0:min_len] for r in list_of_lists]

def get_latest_results_dict(logdir):


    # Get the latest json file from the logdir.
    jsons = glob.glob(os.path.join(logdir, "results_*"))
    jsons.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    if not jsons:
        print("No results_* matches in %s! Skipping." % logdir)
        results_dict = None
    else:
        latest_json = jsons[-1]
        print("Reading: " + str(latest_json))
        results_dict = json.load(open(latest_json))

    return(results_dict)


def smooth_list(input_list, window_size=2):
    window = np.ones(window_size) / window_size
    smoothed_list = np.convolve(input_list,
                                window,
                                mode='valid')
    return smoothed_list
def main(flags):

    mutedcolors.new_cmaps()

    # results_dict = get_latest_results_dict(flags.logdir)
    # # print(results_dict)
    #
    # for (result_name, result_list) in results_dict["results"].items():
    #     print(result_name)
    #     print(result_list)

        # fig_path = os.path.join(logdir, str(run_id) + '.png')
        # plt.gcf().set_dpi(1200)
        # plt.savefig(fig_path)
        # plt.close()

    # plot_differentiability_ablation(results_jsons)

    if flags.plot_type == "num_exposures_running_time":


        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

        experiment_result_dict = dict()
        print(flags.logdir)
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            print(subdirectories)
            for subdirectory in subdirectories:
                print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                if results_dict:
                    print(results_dict["run_name"])
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    valid_ssim_ratio = np.array(results_dict["results"]["valid_ssim_ratio_list"])
                    valid_psnr_ratio = np.array(results_dict["results"]["valid_psnr_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])
                    train_mono_da_mse_ratio = 1 / train_mse_ratio
                    valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                    plot_metric = train_epoch_time

                    plot_result_key_value = results_dict["num_exposures"]
                    plot_filter_dict = {"plan_diversity_regularization": False}

                    # Apply the filter list to throw out these results.
                    include_results = True
                    for key, value in plot_filter_dict.items():
                        # TODO: Generalize to any evaluation.
                        if results_dict[key] == value:
                            include_results = True

                    if include_results:
                        if plot_result_key_value in experiment_result_dict:
                            experiment_result_dict[plot_result_key_value].append(plot_metric)
                        else:
                            experiment_result_dict[plot_result_key_value] = list()
                            experiment_result_dict[plot_result_key_value].append(plot_metric)

        od = collections.OrderedDict(sorted(experiment_result_dict.items()))

        for n, color, (num_exposures,
                       metric_lists) in mutedcolors.eczip(od.items(),
                                                         cmap='grormute',
                                                         start=1,
                                                         step=1):

            # print("Start outputs for tables")
            # print(label)
            # print(metric_lists)
            # best_epoch = np.argmax(metric_lists)
            # print(metric_list[best_epoch])
            # print(np.max(metric_list))
            # print("End outputs for tables")

            for metric_list in metric_lists:

                plt.scatter(num_exposures, np.median(metric_list), color=color, label=num_exposures)

                one_line = np.ones_like(metric_list)

        # plt.plot(one_line,
        #          color="black",
        #          label="Equal MSE",
        #          linewidth=1.5,
        #          linestyle="--",
        #          zorder=-1,
        #          alpha=0.5)
        # from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Times']})
        # rc('text', usetex=True)
        # plt.title("Differentiability Improves Recovery Performance")
        plt.legend()
        plt.tight_layout()
        # plt.xlim(0, 128)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.show()
        plt.close()


        # For each, plot the train/val curves of the last results.

        # results_dict = get_latest_results_dict(flags.logdir)
        #
        # results_dicts = None
        #
        # plot_ensemble_ablation(results_dicts)

    if flags.plot_type == "diversity_regularization_num_exposures":


        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

        experiment_result_dict = dict()
        print(flags.logdir)
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            print(subdirectories)
            for subdirectory in subdirectories:
                print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                if results_dict:
                    print(results_dict["run_name"])
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    valid_ssim_ratio = np.array(results_dict["results"]["valid_ssim_ratio_list"])
                    valid_psnr_ratio = np.array(results_dict["results"]["valid_psnr_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])
                    train_mono_da_mse_ratio = 1 / train_mse_ratio
                    valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                    plot_metric = valid_psnr_ratio

                    plot_result_key_value = results_dict["num_exposures"]
                    plot_filter_dict = {"plan_diversity_regularization": False}

                    # Apply the filter list to throw out these results.
                    include_results = False
                    for key, value in plot_filter_dict.items():
                        # TODO: Generalize to any evaluation.
                        if results_dict[key] == value:
                            include_results = True

                    if include_results:
                        if plot_result_key_value in experiment_result_dict:
                            experiment_result_dict[plot_result_key_value].append(plot_metric)
                        else:
                            experiment_result_dict[plot_result_key_value] = list()
                            experiment_result_dict[plot_result_key_value].append(plot_metric)

        od = collections.OrderedDict(sorted(experiment_result_dict.items()))

        for n, color, (num_exposures,
                       metric_lists) in mutedcolors.eczip(od.items(),
                                                         cmap='grormute',
                                                         start=1,
                                                         step=1):

            # print("Start outputs for tables")
            # print(label)
            # print(metric_lists)
            # best_epoch = np.argmax(metric_lists)
            # print(metric_list[best_epoch])
            # print(np.max(metric_list))
            # print("End outputs for tables")

            for metric_list in metric_lists:


                plt.plot(smooth_list(metric_list, 16),
                         color=color,
                         label=num_exposures)

                one_line = np.ones_like(metric_list)

        # plt.plot(one_line,
        #          color="black",
        #          label="Equal MSE",
        #          linewidth=1.5,
        #          linestyle="--",
        #          zorder=-1,
        #          alpha=0.5)
        # from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Times']})
        # rc('text', usetex=True)
        # plt.title("Differentiability Improves Recovery Performance")
        plt.legend()
        plt.tight_layout()
        plt.xlim(0, 1000)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{PSNR}_{\mathbf{d} / \mathbf{m}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.show()
        plt.close()


        # For each, plot the train/val curves of the last results.

        # results_dict = get_latest_results_dict(flags.logdir)
        #
        # results_dicts = None
        #
        # plot_ensemble_ablation(results_dicts)

    if flags.plot_type == "diversity_regularization":


        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

        experiment_result_dict = dict()
        print(flags.logdir)
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            print(subdirectories)
            for subdirectory in subdirectories:
                print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                if results_dict:
                    print(results_dict["run_name"])
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])
                    train_mono_da_mse_ratio = 1 / train_mse_ratio
                    valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                    plot_metric = valid_mono_da_mse_ratio
                    plot_metric = train_mono_da_mse_ratio

                    plot_result_key_value = results_dict["plan_diversity_regularization"]
                    if plot_result_key_value in experiment_result_dict:
                        experiment_result_dict[plot_result_key_value].append(plot_metric)
                    else:
                        experiment_result_dict[plot_result_key_value] = list()
                        experiment_result_dict[plot_result_key_value].append(plot_metric)

        od = collections.OrderedDict(sorted(experiment_result_dict.items()))

        for n, color, (plan_diversity_regularization,
                       metric_lists) in mutedcolors.eczip(od.items(),
                                                         cmap='grormute',
                                                         start=1,
                                                         step=1):

            if plan_diversity_regularization:
                label = "Diversity Regularization ($\mu \pm \sigma^2$)"
            else:
                label = "No Regularization ($\mu \pm \sigma^2$)"

            # print("Start outputs for tables")
            # print(label)
            # print(metric_lists)
            # best_epoch = np.argmax(metric_lists)
            # print(metric_list[best_epoch])
            # print(np.max(metric_list))
            # print("End outputs for tables")

            for metric_list in metric_lists:

                plt.plot(metric_list, color=color, label=label)

                one_line = np.ones_like(metric_list)

        # plt.plot(one_line,
        #          color="black",
        #          label="Equal MSE",
        #          linewidth=1.5,
        #          linestyle="--",
        #          zorder=-1,
        #          alpha=0.5)
        # from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Times']})
        # rc('text', usetex=True)
        # plt.title("Differentiability Improves Recovery Performance")
        plt.legend()
        plt.tight_layout()
        plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.show()
        plt.close()


        # For each, plot the train/val curves of the last results.

        # results_dict = get_latest_results_dict(flags.logdir)
        #
        # results_dicts = None
        #
        # plot_ensemble_ablation(results_dicts)

    if flags.plot_type == "diversity_regularization_mean":

        experiment_result_dict = dict()
        print(flags.logdir)
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            print(subdirectories)
            for subdirectory in subdirectories:
                print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                if results_dict:
                    print(results_dict["run_name"])
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])
                    train_mono_da_mse_ratio = 1 / train_mse_ratio
                    valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                    plot_metric = valid_mono_da_mse_ratio
                    plot_metric = train_mono_da_mse_ratio

                    plot_result_key_value = results_dict["plan_diversity_regularization"]

                    if plot_result_key_value in experiment_result_dict:
                        experiment_result_dict[plot_result_key_value].append(plot_metric)
                    else:
                        experiment_result_dict[plot_result_key_value] = list()
                        experiment_result_dict[plot_result_key_value].append(plot_metric)

        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

        od = collections.OrderedDict(sorted(experiment_result_dict.items()))

        for n, color, (plan_diversity_regularization,
                       metric_list) in mutedcolors.eczip(od.items(),
                                                                           cmap='grormute',
                                                                           start=1,
                                                                           step=1):

            trimmed_results = trim_lists_to_shortest(metric_list)
            results_array = np.array(trimmed_results)
            t = list(range(len(trimmed_results[0])))

            metric_mean = np.mean(results_array, axis=0)
            metric_std = np.std(results_array, axis=0)

            plt.fill_between(t,
                             metric_mean + metric_std / 2,
                             metric_mean - metric_std / 2,
                             facecolor=color,
                             alpha=0.5)

            if plan_diversity_regularization:
                label = "Diversity Regularization ($\mu \pm \sigma^2$)"
            else:
                label = "No Regularization ($\mu \pm \sigma^2$)"


            print("Start outputs for tables")
            print(label)
            best_epoch = np.argmax(metric_mean)
            print(metric_mean[best_epoch])
            print(metric_std[best_epoch])
            print(np.max(results_array))
            print("End outputs for tables")

            plt.plot(metric_mean,
                 color=color,
                 label=label)
            # plt.plot(valid_mono_da_mse_ratios_mean,
            #          color=color,
            #          label="Immutable TTP = " + str(lock_ttp_values_var))
            one_line = np.ones_like(metric_mean)


            # # Plot each line in the same color
            # for valid_mono_da_mse_ratios in valid_mono_da_mse_ratios_list:
            #     plt.plot(np.convolve(valid_mono_da_mse_ratios, np.ones(16) / 16, mode="same"),
            #              color=color,
            #              label="Immutable TTP = " + str(lock_ttp_values_var))
            #     plt.plot(valid_mono_da_mse_ratios,
            #              color=color,
            #              linewidth=1,
            #              alpha=0.5)
            #
            #     one_line = np.ones_like(valid_mono_da_mse_ratios)

        plt.plot(one_line,
                 color="black",
                 label="Equal MSE",
                 linewidth=1.5,
                 linestyle="--",
                 zorder=-1,
                 alpha=0.5)
        # from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Times']})
        # rc('text', usetex=True)
        # plt.title("Differentiability Improves Recovery Performance")
        plt.legend()
        plt.tight_layout()
        plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.show()
        plt.close()


        # For each, plot the train/val curves of the last results.

        # results_dict = get_latest_results_dict(flags.logdir)
        #
        # results_dicts = None
        #
        # plot_ensemble_ablation(results_dicts)

    if flags.plot_type == "ensemble":

        ensemble_experiment_dict = dict()
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            for subdirectory in subdirectories:
                # print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))


                print(results_dict["run_name"])
                train_loss = np.array(results_dict["results"]["train_loss_list"])
                train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])

                print(np.min(train_mse_ratio))
                train_mono_da_mse_ratio = 1 / train_mse_ratio
                valid_mono_da_mse_ratio = 1 / valid_mse_ratio

                ensemble_size = results_dict["num_exposures"]

                if ensemble_size in ensemble_experiment_dict:
                    ensemble_experiment_dict[ensemble_size].append(valid_mono_da_mse_ratio)
                else:
                    ensemble_experiment_dict[ensemble_size] = list()
                    ensemble_experiment_dict[ensemble_size].append(valid_mono_da_mse_ratio)


        plt.style.use('.\\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

        od = collections.OrderedDict(sorted(ensemble_experiment_dict.items()))

        for n, color, (num_exposures,
                       valid_mono_da_mse_ratios_list) in mutedcolors.eczip(od.items(),
                                                                           cmap='grormute',
                                                                           start=1,
                                                                           step=1):
        # for (num_exposures, valid_mono_da_mse_ratios_list) in ensemble_experiment_dict.items():
            label = "Ensemble Size = " + str(num_exposures)
            # print(valid_mono_da_mse_ratios_list)
            for valid_mono_da_mse_ratios in valid_mono_da_mse_ratios_list:
                plt.plot(np.convolve(valid_mono_da_mse_ratios, np.ones(16) / 16, mode="same"),
                         color=color,
                         label=label)
                plt.plot(valid_mono_da_mse_ratios,
                         color=color,
                         linewidth=1,
                         alpha=0.5)

                one_line = np.ones_like(valid_mono_da_mse_ratios)

                print("Start outputs for tables")
                print(label)
                best_epoch = np.argmax(valid_mono_da_mse_ratios[0:2048])
                print(valid_mono_da_mse_ratios[best_epoch])
                # print(np.max(valid_mono_da_mse_ratios))
                print("End outputs for tables")

        plt.plot(one_line,
                 color="black",
                 label="Equal MSE",
                 linewidth=1.5,
                 linestyle="--",
                 alpha=0.5)

        plt.legend()
        plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.show()
        plt.close()

    if flags.plot_type == "diff":

        experiment_result_dict = dict()
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            for subdirectory in subdirectories:
                # print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))


                print(results_dict["run_name"])
                train_loss = np.array(results_dict["results"]["train_loss_list"])
                train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])
                train_mono_da_mse_ratio = 1 / train_mse_ratio
                valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                plot_metric = valid_mono_da_mse_ratio
                plot_metric = train_mono_da_mse_ratio

                plot_result_key_value = results_dict["lock_ttp_values"]

                if plot_result_key_value in experiment_result_dict:
                    experiment_result_dict[plot_result_key_value].append(plot_metric)
                else:
                    experiment_result_dict[plot_result_key_value] = list()
                    experiment_result_dict[plot_result_key_value].append(plot_metric)

        plt.style.use('.\\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

        od = collections.OrderedDict(sorted(experiment_result_dict.items()))

        for n, color, (lock_ttp_values_var,
                       valid_mono_da_mse_ratios_list) in mutedcolors.eczip(od.items(),
                                                                           cmap='grormute',
                                                                           start=1,
                                                                           step=1):
        # for (num_exposures, valid_mono_da_mse_ratios_list) in ensemble_experiment_dict.items():

            # print(valid_mono_da_mse_ratios_list)
            min_len = np.min([len(r) for r in valid_mono_da_mse_ratios_list])
            # Plot the mean and variance of the colleciton of lists.
            short_lists = [r[0:min_len] for r in valid_mono_da_mse_ratios_list]
            results_array = np.array(short_lists)
            t = list(range(min_len))

            valid_mono_da_mse_ratios_mean = np.mean(results_array, axis=0)
            valid_mono_da_mse_ratios_std = np.std(results_array, axis=0)

            plt.fill_between(t,
                             valid_mono_da_mse_ratios_mean+valid_mono_da_mse_ratios_std/2,
                             valid_mono_da_mse_ratios_mean-valid_mono_da_mse_ratios_std/2,
                             facecolor=color,
                             alpha=0.5)

            if lock_ttp_values_var:
                label = "Immutable Articulations ($\mu \pm \sigma^2$)"
            else:
                label = "Mutable Articulations ($\mu \pm \sigma^2$)"


            print("Start outputs for tables")
            print(label)
            best_epoch = np.argmax(valid_mono_da_mse_ratios_mean[0:2048])
            print(valid_mono_da_mse_ratios_mean[best_epoch])
            print(valid_mono_da_mse_ratios_std[best_epoch])
            print(np.max(results_array[:, 2048]))
            print("End outputs for tables")

            plt.plot(valid_mono_da_mse_ratios_mean,
                 color=color,
                 label=label)
            # plt.plot(valid_mono_da_mse_ratios_mean,
            #          color=color,
            #          label="Immutable TTP = " + str(lock_ttp_values_var))
            one_line = np.ones_like(valid_mono_da_mse_ratios_mean)


            # # Plot each line in the same color
            # for valid_mono_da_mse_ratios in valid_mono_da_mse_ratios_list:
            #     plt.plot(np.convolve(valid_mono_da_mse_ratios, np.ones(16) / 16, mode="same"),
            #              color=color,
            #              label="Immutable TTP = " + str(lock_ttp_values_var))
            #     plt.plot(valid_mono_da_mse_ratios,
            #              color=color,
            #              linewidth=1,
            #              alpha=0.5)
            #
            #     one_line = np.ones_like(valid_mono_da_mse_ratios)

        plt.plot(one_line,
                 color="black",
                 label="Equal MSE",
                 linewidth=1.5,
                 linestyle="--",
                 zorder=-1,
                 alpha=0.5)
        # from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Times']})
        # rc('text', usetex=True)
        # plt.title("Differentiability Improves Recovery Performance")
        plt.legend()
        plt.tight_layout()
        plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.show()
        plt.close()


        # For each, plot the train/val curves of the last results.

        # results_dict = get_latest_results_dict(flags.logdir)
        #
        # results_dicts = None
        #
        # plot_ensemble_ablation(results_dicts)

    if flags.plot_type == "diameter":

        experiment_result_dict = dict()
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):

            for subdirectory in subdirectories:
                # print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                train_loss = np.array(results_dict["results"]["train_loss_list"])
                train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])

                train_mono_da_mse_ratio = 1 / train_mse_ratio
                valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                plot_metric = valid_mono_da_mse_ratio
                # plot_metric = valid_mono_mse

                plot_result_key_value = str(results_dict["aperture_diameter_meters"])
                if plot_result_key_value in experiment_result_dict:
                    print("this value was present.")
                    experiment_result_dict[plot_result_key_value].append(plot_metric)
                else:
                    experiment_result_dict[plot_result_key_value] = list()
                    experiment_result_dict[plot_result_key_value].append(plot_metric)

                print(plot_result_key_value)
                for (k, v) in experiment_result_dict.items():
                    print(len(v))
                # print(experiment_result_dict)


        plt.style.use('.\\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')


        # First, plot metric vs training time.
        od = collections.OrderedDict(sorted(experiment_result_dict.items()))
        for n, color, (lock_ttp_values_var,
                       valid_mono_da_mse_ratios_list) in mutedcolors.eczip(od.items(),
                                                                           cmap='grormute',
                                                                           start=1,
                                                                           step=1):
        # for (num_exposures, valid_mono_da_mse_ratios_list) in ensemble_experiment_dict.items():

            # print(valid_mono_da_mse_ratios_list)
            label = "Maximum Optical Baseline = " + str(lock_ttp_values_var)
            for valid_mono_da_mse_ratios in valid_mono_da_mse_ratios_list:
                plt.plot(np.convolve(valid_mono_da_mse_ratios, np.ones(16) / 16, mode="valid"),
                         color=color,
                         label=label)
                plt.plot(valid_mono_da_mse_ratios,
                         color=color,
                         linewidth=1,
                         alpha=0.5)

                print("Start outputs for tables")
                print(label)
                best_epoch = np.argmax(valid_mono_da_mse_ratios[0:2048])
                print(valid_mono_da_mse_ratios[best_epoch])
                # print(np.max(valid_mono_da_mse_ratios))
                print("End outputs for tables")

                one_line = np.ones_like(valid_mono_da_mse_ratios)

        plt.plot(one_line,
                 color="black",
                 label="Equal MSE",
                 linewidth=1.5,
                 linestyle="--",
                 alpha=0.5)

        plt.legend()
        # plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.show()
        plt.close()

        # # Next, plot wrt the variable of interest.
        # for experiment_value, experiment_array_list in experiment_result_dict.items():
        #
        #     experiment_value_maxima = list()
        #     for experiment_array in experiment_array_list:
        #         experiment_value_maxima.append(np.max(experiment_array))
        #         print(experiment_value_maxima)
        #
        #     experiment_value_mean = np.mean(experiment_value_maxima)
        #     experiment_value_var = np.var(experiment_value_maxima)
        #     print(experiment_value)
        #     plt.plot(experiment_value,
        #              experiment_value_mean,
        #              'rp')
        #     plt.errorbar(experiment_value,
        #                  experiment_value_mean,
        #                  yerr=experiment_value_var)
        #
        #
        # plt.title("Maximum Validation MSE Ration")
        # # plt.xlim(0, 2048)
        # plt.xlabel("Optical Baseline (m)")
        # plt.ylabel("MSE Ratio")
        # plt.show()
        # plt.close()

    if flags.plot_type == "overfitting":

        experiment_result_dict = dict()
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):
            train_losses = list()
            valid_losses = list()
            valid_mono_da_mse_ratios = list()
            train_mono_da_mse_ratios = list()

            for subdirectory in subdirectories:
                # print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))


                if results_dict:
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])

                    train_mono_da_mse_ratio = 1 / train_mse_ratio
                    valid_mono_da_mse_ratio = 1 / valid_mse_ratio
                    plot_metric = train_mono_da_mse_ratio

                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    train_mono_da_mse_ratios.append(train_mono_da_mse_ratio)
                    valid_mono_da_mse_ratios.append(valid_mono_da_mse_ratio)

                    plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
                    plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')

                    plt.subplot(121)
                    # First, plot metric vs training time.

            for (train_loss,
                 valid_loss,
                 train_mono_da_mse_ratio,
                 valid_mono_da_mse_ratio) in zip(train_losses,
                                                 valid_losses,
                                                 valid_mono_da_mse_ratios,
                                                 train_mono_da_mse_ratios,):

                plt.subplot(121)
                plt.plot(train_mono_da_mse_ratio,
                         label="Train MSE Metric",
                         color='g')
                # plt.plot(np.convolve(train_mono_da_mse_ratio, np.ones(16) / 16, mode="valid"),
                #          label="Train MSE Metric",
                #          color='g')
                plt.plot(valid_mono_da_mse_ratio,
                         label="Valid MSE Metric",
                         color='b')
                # plt.plot(np.convolve(valid_mono_da_mse_ratio, np.ones(16) / 16, mode="valid"),
                #          label="Train MSE Metric",
                #          color='r')

                plt.legend()
                plt.subplot(122)
                plt.plot(train_loss,
                         label="Train Loss",
                         color='g')
                # plt.plot(np.convolve(train_loss, np.ones(16) / 16, mode="valid"),
                #          label="Train Loss",
                #          color='g')
                plt.plot(valid_loss,
                         label="Valid Loss",
                         color='b')
                # plt.plot(np.convolve(valid_loss, np.ones(16) / 16, mode="valid"),
                #          label="Valid Loss",
                #          color='r')
                # plt.plot(valid_mono_da_mse_ratio)

                plt.legend()
                one_line = np.ones_like(train_mono_da_mse_ratio)

        # plt.plot(one_line,
        #          color="black",
        #          label="Equal MSE",
        #          linewidth=1.5,
        #          linestyle="--",
        #          alpha=0.5)

        # plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.ylabel(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$')
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        # plt.xlim(0, 2048)
        plt.xlabel("Training Epoch")
        plt.show()
        plt.close()

        # # Next, plot wrt the variable of interest.
        # for experiment_value, experiment_array_list in experiment_result_dict.items():
        #
        #     experiment_value_maxima = list()
        #     for experiment_array in experiment_array_list:
        #         experiment_value_maxima.append(np.max(experiment_array))
        #         print(experiment_value_maxima)
        #
        #     experiment_value_mean = np.mean(experiment_value_maxima)
        #     experiment_value_var = np.var(experiment_value_maxima)
        #     print(experiment_value)
        #     plt.plot(experiment_value,
        #              experiment_value_mean,
        #              'rp')
        #     plt.errorbar(experiment_value,
        #                  experiment_value_mean,
        #                  yerr=experiment_value_var)
        #
        #
        # plt.title("Maximum Validation MSE Ration")
        # # plt.xlim(0, 2048)
        # plt.xlabel("Optical Baseline (m)")
        # plt.ylabel("MSE Ratio")
        # plt.show()
        # plt.close()

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