import os
import glob
import json
import argparse
import collections
import numpy as np

import matplotlib.ticker as ticker

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
            #

            min_mean = 1e16
            display_metric_list = None
            for metric_list in metric_lists:
                if np.mean(metric_list) < min_mean:

                    min_mean = np.mean(metric_list)

                    display_metric_list = metric_list

            plt.scatter(num_exposures, np.mean(display_metric_list), color=color, label=num_exposures)
            plt.errorbar(num_exposures,
                         np.mean(display_metric_list),
                         yerr=np.std(display_metric_list),
                         color=color,
                         marker=".")


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
        # plt.legend(loc='lower center', ncol=len(list(od.keys())))
        plt.legend(loc='upper left')
        plt.tight_layout()
        # plt.xlim(0, 128)
        plt.xlabel("Ensemble Size")
        plt.xticks(list(od.keys()))
        plt.xscale('log', base=2)
        plt.ylabel(r'Step Running Time [s]')
        plt.gcf().set_size_inches(1.0, 2.0, forward=True)
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.3)
        plt.subplots_adjust(left=0.3)
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
        num_x_ticks = 1
        num_y_ticks = 3
        ymax = 1.0
        xmax = 1450
        plt.rcParams["font.family"] = "Times New Roman"

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
                    # plot_filter_dict = {"r0_mean": lambda x: x<1.0}

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

            # for metric_list in metric_lists:


            plt.plot(smooth_list(metric_lists[-1], 16),
                     color=color,
                     label=num_exposures)


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
        plt.ylabel(r'$\mathbf{PSNR}_{\mathbf{d} / \mathbf{m}}$', labelpad=0)
        plt.tight_layout()
        plt.legend(loc='upper left', prop={'size': 5}, shadow=True, handlelength=2.0, labelspacing=0.1, title="Actuations", title_fontsize=6)
        # plt.xlim(0, 1000)
        plt.xlabel(r'Training Epoch', fontsize=6, labelpad=0)
        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")

        # plt.ylim(0.0, ymax)
        ax = plt.gca()

        ax.get_yaxis().set_major_formatter('{x:.2f}')
        ax.get_xaxis().set_major_locator(
            ticker.MultipleLocator(int(len(metric_lists[0]) / num_x_ticks)))
        # ax.get_yaxis().set_major_formatter('{x:.1f}')
        # ax.get_yaxis().set_major_locator(
        #     ticker.MultipleLocator(ymax / num_y_ticks))
        ax.tick_params(axis='both', which='major', labelsize=6, pad=2.0)
        plt.xlim(0.0, xmax)
        plt.grid(axis='both')
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(right=0.975)
        plt.subplots_adjust(left=0.2)
        plt.subplots_adjust(top=0.925)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(2.5, 1.54508)
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

    if flags.plot_type == "metrics_vs_epoch_model_gen":

        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')
        num_x_ticks = 4
        num_y_ticks = 7
        ymax = 1.8
        xmax = 132
        plt.rcParams["font.family"] = "Times New Roman"
        experiment_result_dict = dict()
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):
            train_losses = list()
            valid_losses = list()
            train_metrics = list()
            valid_metrics = list()

            for subdirectory in subdirectories:
                # print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                # TODO: make this more flexible.
                if results_dict:
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    train_ssim_ratio = np.array(results_dict["results"]["train_ssim_ratio_list"])
                    train_psnr_ratio = np.array(results_dict["results"]["train_psnr_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    valid_ssim_ratio = np.array(results_dict["results"]["valid_ssim_ratio_list"])
                    valid_psnr_ratio = np.array(results_dict["results"]["valid_psnr_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])

                    train_mono_da_mse_ratio = 1. / train_mse_ratio
                    valid_mono_da_mse_ratio = 1. / valid_mse_ratio
                    train_plot_metric = train_mono_da_mse_ratio
                    valid_plot_metric = valid_mono_da_mse_ratio

                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    train_metrics.append([train_mono_da_mse_ratio,
                                          train_ssim_ratio,
                                          train_psnr_ratio])
                    valid_metrics.append([valid_mono_da_mse_ratio,
                                          valid_ssim_ratio,
                                          valid_psnr_ratio])




            for (train_loss,
                 valid_loss,
                 train_metric,
                 valid_metric) in zip(train_losses,
                                                 valid_losses,
                                                 train_metrics,
                                                 valid_metrics,):

                (train_mse, train_ssim, train_psnr) =  train_metric
                (valid_mse, valid_ssim, valid_psnr) =  valid_metric

                ax = plt.subplot(131)
                plt.plot(train_mse,
                         label="Training",
                         color='g',
                         alpha=0.5)
                # plt.plot(np.convolve(train_mono_da_mse_ratio, np.ones(16) / 16, mode="valid"),
                #          label="Train MSE Metric",
                #          color='g')
                plt.plot(valid_mse,
                         label="Validation",
                         color='b',
                         alpha=0.5)
                plt.xlabel(r'Training Epoch', fontsize=6, labelpad=0)

                one_line = np.ones_like(train_mse)
                plt.plot(one_line,
                         color="black",
                         label="Parity",
                         linewidth=1.0,
                         linestyle="dashed",
                         alpha=0.5)
                plt.title(r'$\mathbf{MSE}_{\mathbf{m} / \mathbf{d}}$', fontsize=6, pad=0)
                # plt.xlabel("Training Epoch")
                plt.ylim(0.0, ymax)
                plt.xlim(0.0, xmax)
                ax.get_xaxis().set_major_locator(ticker.MultipleLocator(int(len(train_mse) / num_x_ticks)))
                ax.get_yaxis().set_major_formatter('{x:.1f}')
                ax.get_yaxis().set_major_locator(ticker.MultipleLocator(ymax / num_y_ticks))
                ax.tick_params(axis='both', which='major', labelsize=6, pad=2.0)
                plt.grid(axis='both')

                # ax = axs.flat[3]
                # ax.plot([x1, x2], [y1, y2], ".")
                # el = mpatches.Ellipse((x1, y1), 0.3, 0.4, angle=30, alpha=0.2)
                # ax.add_artist(el)
                ax.annotate("Recovery Parity",
                            xy=(28, 1.0), xycoords='data',
                            xytext=(8, 0.41), textcoords='data',
                            fontsize=6,
                            arrowprops=dict(arrowstyle="fancy",
                                            color="0.5",
                                            # patchB=el,
                                            connectionstyle="arc3,rad=0.6",
                                            ),
                            )
                # ax.text(20, 0.5, "Recovery Parity", transform=ax.transAxes, ha="left",
                #         va="top")
                # ax.annotate('local max', xy=(32, 1.0), xytext=(32, 1.0))

                ax = plt.subplot(132)
                plt.plot(train_ssim,
                         label="Training",
                         color='g',
                         alpha=0.5)
                # plt.plot(np.convolve(train_mono_da_mse_ratio, np.ones(16) / 16, mode="valid"),
                #          label="Train MSE Metric",
                #          color='g')
                plt.plot(valid_ssim,
                         label="Validation",
                         color='b',
                         alpha=0.5)

                one_line = np.ones_like(train_mse)
                plt.plot(one_line,
                         color="black",
                         label="Parity",
                         linewidth=1.0,
                         linestyle="dashed",
                         alpha=0.5)
                plt.title(r'$\mathbf{SSIM}_{\mathbf{d} / \mathbf{m}}$', fontsize=6, pad=0)
                plt.xlabel(r'Training Epoch', fontsize=6, labelpad=0)
                plt.grid(axis='both')
                ax.get_yaxis().set_ticklabels([])
                ax.get_xaxis().set_major_locator(ticker.MultipleLocator(int(len(train_mse) / num_x_ticks)))
                ax.get_yaxis().set_major_locator(ticker.MultipleLocator(ymax / num_y_ticks))
                ax.tick_params(axis='both', which='major', labelsize=6, pad=2.0)
                plt.ylim(0, ymax)
                plt.xlim(0.0, xmax)

                ax = plt.subplot(133)
                plt.plot(train_psnr,
                         label="Training",
                         color='g',
                         alpha=0.5)
                # plt.plot(np.convolve(train_mono_da_mse_ratio, np.ones(16) / 16, mode="valid"),
                #          label="Train MSE Metric",
                #          color='g')
                plt.plot(valid_psnr,
                         label="Validation",
                         color='b',
                         alpha=0.5)

                one_line = np.ones_like(train_mse)
                plt.plot(one_line,
                         color="black",
                         label="Parity",
                         linewidth=1.0,
                         linestyle="dashed",
                         alpha=0.5)
                plt.title(r'$\mathbf{PSNR}_{\mathbf{d} / \mathbf{m}}$', fontsize=6, pad=0)
                plt.xlabel(r'Training Epoch', fontsize=6, labelpad=0)
                ax.get_yaxis().set_ticklabels([])
                ax.get_xaxis().set_major_locator(ticker.MultipleLocator(int(len(train_mse) / num_x_ticks)))
                ax.get_yaxis().set_major_locator(ticker.MultipleLocator(ymax / num_y_ticks))
                ax.tick_params(axis='both', which='major', labelsize=6, pad=2.0)
                # plt.xlabel("Training Epoch")
                plt.grid(axis='both')
                plt.ylim(0.0, ymax)
                plt.xlim(0.0, xmax)
                #
                # plt.legend()
                # ax = plt.subplot(144)
                # plt.plot(train_loss,
                #          label="Training",
                #          color='g',
                #          alpha=0.5)
                # # plt.plot(np.convolve(train_loss, np.ones(16) / 16, mode="valid"),
                # #          label="Train Loss",
                # #          color='g')
                # plt.plot(valid_loss,
                #          label="Validation",
                #          color='b',
                #          alpha=0.5)
                #
                # plt.xlabel("Training Epoch")
                # ax.yaxis.set_label_position("right")
                # ax.yaxis.tick_right()
                # plt.title(r'$\mathcal{L}$')
                # # plt.plot(np.convolve(valid_loss, np.ones(16) / 16, mode="valid"),
                # #          label="Valid Loss",
                # #          color='r')
                # # plt.plot(valid_mono_da_mse_ratio)

        # plt.ylabel("$\mathbf{MSE}_{\mathbf{DA}} / \mathbf{MSE}_\mathbf{MONO}$")
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(right=0.975)
        plt.subplots_adjust(left=0.075)
        plt.subplots_adjust(top=0.925)

        # plt.xlim(0, 2048)
        # plt.xlabel("Training Epoch")
        plt.legend(loc='lower center', prop={'size': 5}, shadow=True, handlelength=2.0, labelspacing=0.1)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(2.5, 1.54508)
        # fig.set_size_inches(2.5, 4.04508)
        plt.show()
        # fig.savefig('.png', dpi=100)

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

    if flags.plot_type == "loss":

        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\muted.mplstyle')
        plt.style.use('.\\dasie\\utils\\plottools\\mutedplots-master\\stylelib\\small.mplstyle')
        num_x_ticks = 8
        num_y_ticks = 10
        ymax = 0.2
        xmax = 117
        plt.rcParams["font.family"] = "Times New Roman"
        experiment_result_dict = dict()
        # iterate over each experiment instance in the logdir.
        for root, subdirectories, files in os.walk(flags.logdir):
            train_losses = list()
            valid_losses = list()
            train_metrics = list()
            valid_metrics = list()

            for subdirectory in subdirectories:
                # print(subdirectory)
                results_dict = get_latest_results_dict(os.path.join(root, subdirectory))

                # TODO: make this more flexible.
                if results_dict:
                    train_loss = np.array(results_dict["results"]["train_loss_list"])
                    train_dist_mse = np.array(results_dict["results"]["train_dist_mse_list"])
                    train_mono_mse = np.array(results_dict["results"]["train_mono_mse_list"])
                    train_mse_ratio = np.array(results_dict["results"]["train_mse_ratio_list"])
                    train_ssim_ratio = np.array(results_dict["results"]["train_ssim_ratio_list"])
                    train_psnr_ratio = np.array(results_dict["results"]["train_psnr_ratio_list"])
                    valid_loss = np.array(results_dict["results"]["valid_loss_list"])
                    valid_dist_mse = np.array(results_dict["results"]["valid_dist_mse_list"])
                    valid_mono_mse = np.array(results_dict["results"]["valid_mono_mse_list"])
                    valid_mse_ratio = np.array(results_dict["results"]["valid_mse_ratio_list"])
                    valid_ssim_ratio = np.array(results_dict["results"]["valid_ssim_ratio_list"])
                    valid_psnr_ratio = np.array(results_dict["results"]["valid_psnr_ratio_list"])
                    train_epoch_time = np.array(results_dict["results"]["train_epoch_time_list"])

                    train_mono_da_mse_ratio = 1. / train_mse_ratio
                    valid_mono_da_mse_ratio = 1. / valid_mse_ratio
                    train_plot_metric = train_mono_da_mse_ratio
                    valid_plot_metric = valid_mono_da_mse_ratio

                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    train_metrics.append([train_mono_da_mse_ratio,
                                          train_ssim_ratio,
                                          train_psnr_ratio])
                    valid_metrics.append([valid_mono_da_mse_ratio,
                                          valid_ssim_ratio,
                                          valid_psnr_ratio])


            for (train_loss, valid_loss) in zip(train_losses, valid_losses,):

                ax = plt.subplot(111)
                plt.plot(train_loss,
                         label="Training",
                         color='g',
                         alpha=0.5)
                plt.plot(valid_loss,
                         label="Validation",
                         color='b',
                         alpha=0.5)

                plt.xlabel(r'Training Epoch', fontsize=6, labelpad=0)
                ax.get_xaxis().set_major_locator(
                    ticker.MultipleLocator(int(len(train_loss) / num_x_ticks)))
                ax.get_yaxis().set_major_locator(
                    ticker.MultipleLocator(ymax / num_y_ticks))
                ax.tick_params(axis='both', which='major', labelsize=6,
                               pad=2.0)
                plt.grid(axis='both')
                plt.xlim(0.0, xmax)
                plt.ylabel(r'$\mathcal{L}$', labelpad=0.0)
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(right=0.975)
        plt.subplots_adjust(left=0.15)

        # plt.xlim(0, 2048)
        # plt.xlabel("Training Epoch")
        plt.legend(loc='upper center', prop={'size': 5}, shadow=True, handlelength=2.0, labelspacing=0.1)
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(2.5, 1.54508)
        # fig.set_size_inches(2.5, 4.04508)
        plt.show()
        # fig.savefig('.png', dpi=100)

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