"""
A library for the computation of DASIE MTF ensembles.

Author: Justin Fletcher
"""

import os
import time
import copy
import joblib
import argparse

import json
import codecs
import hcipy
import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatch
from decimal import Decimal

import codecs

from multi_aperture_psf import MultiAperturePSFSampler

def compute_dasie_configuration_space_size(flags):
    mft_count =  flags.num_subapertures * (flags.num_actuation_states**3)
    print("MFT count = %.2E" % Decimal(mft_count))

    bytes_per_mft = (32 / 8) *  flags.mft_scale**2
    print("GB/mft = %f GB" % (bytes_per_mft/(1000*1000*1000)))

    bytes_for_all_mft = bytes_per_mft * mft_count
    print("bytes total = %f GB" % (bytes_for_all_mft/(1000*1000*1000)))
    print("bytes total = %f TB" % (bytes_for_all_mft/(1000*1000*1000*1000)))
    print("bytes total = %f PB" % (bytes_for_all_mft/(1000*1000*1000*1000*1000)))
    print("bytes total = %f EB" % (bytes_for_all_mft/(1000*1000*1000*1000*1000*1000)))



class Annealer(object):

    def __init__(self,
                 state,
                 state_perturber,
                 state_cost_evaluator,
                 temperature_function,
                 acceptance_function,
                 initial_temperature,
                 state_save_dir=None):

        # Initialize annealing governing functions.
        self.state_perturber = state_perturber
        self.state_cost_evaluator = state_cost_evaluator
        self.state = state
        self.temperature_function = temperature_function
        self.acceptance_function = acceptance_function
        self.initial_temperature = initial_temperature
        self.state_save_dir = state_save_dir

        # Intialize the annealing state variables.
        self.epoch = 0
        self.temperature = initial_temperature
        self.current_cost = []
        self.best_cost = 10000000000000.0

    def update_temperature(self):

        self.temperature = self.temperature_function(self.epoch,
                                                     self.initial_temperature)

    def reanneal(self):

        self.epoch = 0
        self.temperature = self.initial_temperature

    def __call__(self, perturb_params=[], input_data=[]):

        print("====== Start of Epoch ======")

        if self.epoch == 0:

            self.current_cost = self.state_cost_evaluator.evaluate(self.state,
                                                                   input_data)

        # Incement the epoch count.
        self.epoch = self.epoch + 1

        print("self.epoch = %.d" % self.epoch)

        print("self.current_cost = %f" % self.current_cost)

        # Update the temperature.
        self.update_temperature()

        print("self.temperature = %f" % self.temperature)
        # Store the current state.
        self.state.store()

        # Create a new state using the perturbation function.
        self.state_perturber.perturb(self.state, perturb_params)

        # Compute the cost of this state using the provided grader.
        candidate_cost = self.state_cost_evaluator.evaluate(self.state,
                                                            input_data)

        print("candidate_cost = %f" % candidate_cost)

        # Compute the cost function delta induced by this state.
        cost_delta = candidate_cost - self.current_cost

        print("cost_delta = %f" % cost_delta)

        # If the candidate state reduces the cost...
        if(cost_delta <= 0.0):

            print("Accept 1")
            # ...accept the candidate state implicitly by updating the cost.
            self.current_cost = candidate_cost

        # If the cost is increased...
        else:

            print('np.exp(-d / t) = %.9f' % np.exp(-cost_delta / self.temperature))
            # ...and if the acceptance function returns True...
            if(self.acceptance_function(self.temperature, cost_delta)):

                print("Accept 2")
                # ...accept the state implicitly by updating the cost.
                self.current_cost = candidate_cost

            # If not, we reject the state by restoring the prior state.
            else:

                print("Reject")
                self.state.restore()

        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best_state = self.state

            if self.state_save_dir:
                save_file_name = "best_state_" + str(self.epoch) + ".json"
                file_path = os.path.join(self.state_save_dir, save_file_name)
                json.dump(self.best_state.value.tolist(),
                          codecs.open(file_path, 'w', encoding='utf-8'),
                          separators=(',', ':'),
                          sort_keys=True,
                          indent=4)

            else:

                save_file_name = "best_state_" + str(self.epoch) + ".json"
                file_path = os.path.join(".", save_file_name)
                json.dump(self.best_state.value.tolist(),
                          codecs.open(file_path, 'w', encoding='utf-8'),
                          separators=(',', ':'),
                          sort_keys=True,
                          indent=4)

        print("best_cost = %f" % self.best_cost)

        print("====== End of Epoch ======")
        return()

def psf_from_dasie_actuation(dasie_psf_sampler, ptt):

    # Ian's code.
    actuated_dasie_psf, _ = dasie_psf_sampler.sample(ptt_actuate=ptt)
    actuated_dasie_psf = actuated_dasie_psf[..., 0]
    return(actuated_dasie_psf)

def mtf_from_dasie_actuation(dasie_psf_sampler, ptt):

    # Ian's code.
    actuated_dasie_psf = psf_from_dasie_actuation(dasie_psf_sampler, ptt)
    actuated_dasie_mtf = np.fft.fft2(actuated_dasie_psf, norm="ortho")
    real_actuated_dasie_mtf = np.abs(actuated_dasie_mtf)
    return(real_actuated_dasie_mtf)


class DASIEState(object):

    def __init__(self, ensemble_size, num_apertures):

        self.ensemble_size = ensemble_size
        self.value = np.zeros((ensemble_size, num_apertures, 3))
        self.value = np.random.randn(ensemble_size, num_apertures, 3)

    def __len__(self):

        return self.ensemble_size

    # State set function.
    def store(self):

        self.prior_value = copy.deepcopy(self.value)

    # State set function.
    def restore(self):

        self.value = copy.deepcopy(self.prior_value)

    # Plot the state
    def plot(self,
             mtf_scale,
             dasie_psf_sampler,
             plot_list=None,
             show_plot=True,
             save_plot=True,
             run_id=None,
             save_dir=None,
             epoch=None):

        perfect_image = plt.imread('sample_image.png')
        perfect_image = perfect_image / np.max(perfect_image)
        perfect_image_spectrum = np.fft.fft2(perfect_image, norm="ortho")

        mtf_list = list()
        psf_list = list()
        perturbed_image_list = list()

        # Iterate over the slices of state representing ptt arrays...
        for dasie_actuation in self.value:

            # ...compute the MTF...
            mtf = mtf_from_dasie_actuation(dasie_psf_sampler, dasie_actuation)
            mtf_list.append(mtf)

            psf = psf_from_dasie_actuation(dasie_psf_sampler, dasie_actuation)
            psf_list.append(psf)

            perturbed_image_spectrum = perfect_image_spectrum * mtf

            perturbed_image = abs(np.fft.fft2(perturbed_image_spectrum, norm="ortho"))

            perturbed_image_list.append(perturbed_image)

        # Compute the restored image assuming mean restoration.
        restored_image = np.mean(perturbed_image_list, axis=0)

        restored_image_residual = perfect_image - np.fliplr(np.flipud(restored_image))

        monolithic_mtf = get_monolithic_mft()

        monolithic_image_spectrum = perfect_image_spectrum * monolithic_mtf

        monolithic_image = abs(np.fft.fft2(monolithic_image_spectrum, norm="ortho"))

        monolithic_image_residual = perfect_image - np.fliplr(np.flipud(monolithic_image))
        # First, set up the gridspec into which we will draw plots
        nrows = 4
        ncols = len(mtf_list)

        fig = plt.figure(constrained_layout=True, figsize=(21, 9), dpi=160)
        gs0 = fig.add_gridspec(1, 3)
        gs00 = gs0[0].subgridspec(nrows, ncols)

        # Now, draw the left subgridspec.
        for index, (ptt,
                    psf,
                    mtf,
                    perturbed_image) in enumerate(zip(self.value,
                                                      psf_list,
                                                      mtf_list,
                                                      perturbed_image_list)):
            ppt_ax = fig.add_subplot(gs00[0, index])
            ppt_ax.imshow(ptt)
            ppt_ax.axis("off")


            chip_center_x = 32
            chip_center_y = 32
            chip_radius = 16
            psf_center = int(np.round(psf.shape[0] / 2))
            dark_spot_chip_x_start = psf_center + chip_center_x - chip_radius
            dark_spot_chip_x_stop = psf_center + chip_center_x + chip_radius
            dark_spot_chip_y_start = psf_center + chip_center_y - chip_radius
            dark_spot_chip_y_stop = psf_center + chip_center_y + chip_radius

            # plot_psf = psf
            # plot_psf[dark_spot_chip_x_start:dark_spot_chip_x_stop,
            #          dark_spot_chip_y_start:dark_spot_chip_y_stop] = 0.0

            mtf_ax = fig.add_subplot(gs00[1, index])
            mtf_ax.imshow(np.log10(abs(psf)))
            mtf_ax.axis("off")

            rectangles = {'spot': mpatch.Rectangle((dark_spot_chip_x_start, dark_spot_chip_y_start),
                                                   2 * chip_radius,
                                                   2 * chip_radius,
                                                   fill=False,
                                                   edgecolor='white')}

            for r in rectangles:
                mtf_ax.add_artist(rectangles[r])
                rx, ry = rectangles[r].get_xy()
                cx = rx + rectangles[r].get_width() / 2.0
                cy = ry + rectangles[r].get_height() / 2.0

                # mtf_ax.annotate(r, (cx, cy), color='w', weight='bold',
                #                 fontsize=6, ha='center', va='center')

            psf_chip = psf[dark_spot_chip_x_start:dark_spot_chip_x_stop,
                           dark_spot_chip_y_start:dark_spot_chip_y_stop]

            mtf_ax = fig.add_subplot(gs00[2, index])
            mtf_ax.imshow(np.log10(psf_chip))
            mtf_ax.axis("off")

            mtf_ax = fig.add_subplot(gs00[3, index])
            mtf_ax.imshow(perturbed_image)
            mtf_ax.axis("off")

        # aggregate_mtf = np.zeros((mtf_scale, mtf_scale))
        # for mtf in mtf_list:
        #     aggregate_mtf += mtf
        #
        # # Take the grand sum, and divide it by the ensemble length.
        # cost = (np.sum(np.sum(mtf_list)) / len(self.value))

        # Now, draw the right subgridspec.
        gs01 = gs0[1].subgridspec(4, 2)

        restored_image_ax = fig.add_subplot(gs01[0, 0])
        title = "Perfect Image"
        restored_image_ax.set_title(title)
        restored_image_ax.imshow(perfect_image)

        restored_image_ax = fig.add_subplot(gs01[1, 0])
        title = "Mean Restored MTF"
        restored_image_ax.set_title(title)
        restored_image_ax.imshow(np.log10(np.fft.fftshift(np.mean(mtf_list, axis=0))))



        restored_image_ax = fig.add_subplot(gs01[2, 0])
        title = "Restored Image"
        restored_image_ax.set_title(title)
        restored_image_ax.imshow(restored_image)

        restored_image_residual_ax = fig.add_subplot(gs01[3, 0])
        title = "Restored Image Residual = %f" % rmse(restored_image_residual)
        restored_image_residual_ax.set_title(title)
        restored_image_residual_ax.imshow(restored_image_residual)


        monolithic_image_ax = fig.add_subplot(gs01[0, 1])
        title = "Perfect Image"
        monolithic_image_ax.set_title(title)
        monolithic_image_ax.imshow(perfect_image)

        monolithic_mtf_ax = fig.add_subplot(gs01[1, 1])
        title = "Monolithic MTF"
        monolithic_mtf_ax.set_title(title)
        monolithic_mtf_ax.imshow(np.log10(np.fft.fftshift(monolithic_mtf)))

        monolithic_image_ax = fig.add_subplot(gs01[2, 1])
        title = "Monolithic Image"
        monolithic_image_ax.set_title(title)
        monolithic_image_ax.imshow(monolithic_image)

        monolithic_image_residual_ax = fig.add_subplot(gs01[3, 1])
        title = "Monolithic Image Residual = %f" % rmse(monolithic_image_residual)
        monolithic_image_residual_ax.set_title(title)
        monolithic_image_residual_ax.imshow(monolithic_image_residual)


        gs02 = gs0[2].subgridspec(1, 1)

        if plot_list:
            plot_ax = fig.add_subplot(gs02[0, 0])

            plot_ax.plot(plot_list)

            plt.yscale('log')

        if save_plot:

            if run_id:

                fig_path = os.path.join(save_dir, run_id + '.png')
                plt.savefig(fig_path)

            else:

                plt.savefig('axis_flipped_30_5_experiment.png')

        if show_plot:

            plt.show()


def get_monolithic_mft():
    # Ian's code to make a monolithic aperture MTF.
    mono_setup = joblib.load('21-01-25_256_mas_setup_2.5as_1um_.05frac.pkl')
    mir_centers = hcipy.CartesianGrid(np.array([[0], [0]]))
    mir_diamater, pup_diamater = 2.5, 3.6
    aper_config = ['circular_central_obstruction', mir_diamater, .3]
    mono_setup['mirror_config']['positions'] = mir_centers
    mono_setup['mirror_config']['aperture_config'] = aper_config
    mono_sampler = MultiAperturePSFSampler(**mono_setup)
    x, _ = mono_sampler.sample(ptt_actuate=None)
    mono_psf = x[..., 0]
    mono_mtf = np.abs(np.fft.fft2(mono_psf, norm="ortho"))

    return(mono_mtf)

def rmse(predictions, targets=None):

    if targets:
        return np.sqrt(np.mean((predictions - targets) ** 2))

    else:

        return np.sqrt(np.mean((predictions) ** 2))


class DASIECostEvaluator(object):

    def __init__(self, mtf_scale, dasie_psf_sampler):

        self.mtf_scale = mtf_scale
        self.dasie_psf_sampler = dasie_psf_sampler
        self.one_matrix = np.ones((mtf_scale, mtf_scale))
        self.aggregate_mtf = np.zeros((mtf_scale, mtf_scale))

        mono_mtf = get_monolithic_mft()
        # inverted_mono_mtf = np.divide(self.one_matrix, mono_mtf + 1e-16)

        # Compute the grand sum of the monolithic MTF for scaling.
        self.monolithic_mtf_grand_sum = rmse(mono_mtf)

        self.cost = None



    def reset_aggregates(self):

        self.aggregate_mtf = np.zeros((self.mtf_scale, self.mtf_scale))

    def evaluate(self, dasie_state, input_data=None):

        # Iterate over the slices of state representing ptt arrays...
        for dasie_actuation in dasie_state.value:

            # ...compute the MTF...
            mtf = mtf_from_dasie_actuation(self.dasie_psf_sampler,
                                           dasie_actuation)

            # ... recipricate and aggregate it, with stability...
            # self.aggregate_mtf += np.divide(self.one_matrix, mtf + 1e-16)
            self.aggregate_mtf += mtf
            # print(self.aggregate_mtf)

        # Take the grand sum, and divide it by the ensemble length.
        distributed_mtf_grand_sum = rmse((self.aggregate_mtf) / len(dasie_state))

        # print(self.monolithic_mtf_grand_sum)

        # Scale by the monolithic aperture sum; <1 means gt of dasie>gt of mono
        cost = self.monolithic_mtf_grand_sum / distributed_mtf_grand_sum

        self.cost = cost

        # Clear the aggregate MTF.
        self.reset_aggregates()

        # Return the cost.
        return cost

class DarkSpotCostEvaluator(object):

    def __init__(self, mtf_scale, dasie_psf_sampler):

        self.mtf_scale = mtf_scale
        self.dasie_psf_sampler = dasie_psf_sampler
        self.one_matrix = np.ones((mtf_scale, mtf_scale))
        self.aggregate_mtf = np.zeros((mtf_scale, mtf_scale))

        mono_mtf = get_monolithic_mft()
        # inverted_mono_mtf = np.divide(self.one_matrix, mono_mtf + 1e-16)

        # Compute the grand sum of the monolithic MTF for scaling.
        self.monolithic_mtf_grand_sum = rmse(mono_mtf)

        self.cost = None


    def reset_aggregates(self):

        self.aggregate_mtf = np.zeros((self.mtf_scale, self.mtf_scale))

    def evaluate(self, dasie_state, input_data=None):

        dasie_actuation = None
        # Iterate over the slices of state representing ptt arrays...
        for dasie_actuation in dasie_state.value:

            # ...compute the MTF...
            psf = psf_from_dasie_actuation(self.dasie_psf_sampler,
                                           dasie_actuation)
            dasie_actuation = dasie_actuation

        chip_center_x = 64
        chip_center_y = 64
        chip_radius = 16
        psf_center = int(np.round(psf.shape[0] / 2))
        dark_spot_chip_x_start = psf_center + chip_center_x - chip_radius
        dark_spot_chip_x_stop = psf_center + chip_center_x + chip_radius
        dark_spot_chip_y_start = psf_center + chip_center_y - chip_radius
        dark_spot_chip_y_stop = psf_center + chip_center_y + chip_radius

        psf_chip = psf[dark_spot_chip_x_start:dark_spot_chip_x_stop,
                       dark_spot_chip_y_start:dark_spot_chip_y_stop]

        alpha = 1.0
        beta = 1.0

        # cost = alpha * np.sum(psf_chip) + beta * np.sqrt(np.sum(np.power(dasie_actuation, 2)))

        cost = alpha * np.sqrt(np.sum(np.power(dasie_actuation, 2))) + beta * np.sqrt(np.sum(np.power(dasie_actuation, 2)))
        # cost = np.sqrt(np.sum(np.power(dasie_actuation, 2)))

        self.cost = cost

        # Return the cost.
        return cost

class DASIEPerturber(object):

    def __init__(self, scale=1.0, type="csa"):

        self.scale = scale

        if type == "csa":
            self.perturb = self.perturb_classical

        if type == "fsa":
            self.perturb = self.perturb_quantum

    def perturb_classical(self, state, perturb_params=""):

        state.value += self.scale * np.random.randn(*state.value.shape)

    def perturb_quantum(self, state, perturb_params=""):

        p = np.tan(np.pi * (np.random.randn(*state.value.shape) - 0.5))

        state.value += self.scale *  p

def csa_acceptance_function(temperature, cost_delta):
    accept = np.exp(-cost_delta / temperature) > np.random.rand()

    return accept

def csa_temperature_function(epoch, initial_temperature):

    return initial_temperature / np.log(float(epoch) + 1.0)

def fsa_temperature_function(epoch, initial_temperature):

    return initial_temperature / (float(epoch) + 1.0)

def load_best_state():
    obj_text = codecs.open("./best_state.json", 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    return np.array(b_new)


def main(flags):

    # Instantiate DASIE model.
    mas_setup = joblib.load('21-01-25_256_mas_setup_2.5as_1um_.05frac.pkl')
    dasie_psf_sampler = MultiAperturePSFSampler(**mas_setup)
    num_apertures = dasie_psf_sampler.nMir

    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(".", "logs", timestamp)
    os.makedirs(save_dir, exist_ok=True)


    state = DASIEState(flags.ensemble_size, num_apertures)
    state_cost_evaluator = DarkSpotCostEvaluator(flags.mtf_scale,
                                                  dasie_psf_sampler)

    type = "csa"
    # state_perturber = DASIEPerturber(scale=flags.actuation_scale)
    state_perturber = DASIEPerturber(scale=flags.actuation_scale,
                                     type=type)

    if type == "csa":
        temperature_function = csa_temperature_function
        acceptance_function = csa_acceptance_function

    elif type == "fsa":
        temperature_function = fsa_temperature_function
        acceptance_function = csa_acceptance_function


    initial_temperature = flags.initial_temperature

    annealer = Annealer(state,
                        state_perturber,
                        state_cost_evaluator,
                        temperature_function,
                        acceptance_function,
                        initial_temperature,
                        state_save_dir=save_dir)

    costs = list()
    for epoch in range(flags.num_epochs):

        annealer()
        costs.append(state_cost_evaluator.cost)
        if epoch % flags.plot_periodicity == 0:
            costs.append(state_cost_evaluator.cost)
            state.plot(flags.mtf_scale,
                       dasie_psf_sampler,
                       plot_list=costs,
                       show_plot=flags.show_plot,
                       save_plot=flags.save_plot,
                       save_dir=save_dir,
                       run_id=str(epoch),
                       epoch=epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for training.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1234,
                        help='A random seed for repeatability.')

    parser.add_argument('--plot_periodicity',
                        type=int,
                        default=100,
                        help='Number of epochs to wait before plotting.')

    parser.add_argument('--num_subapertures',
                        type=int,
                        default=15,
                        help='Number of DASIE subapertures.')

    parser.add_argument('--num_actuation_states',
                        type=int,
                        default=2**14,
                        help='Number of possible actuation states.')

    parser.add_argument('--actuation_scale',
                        type=float,
                        default=0.01,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--initial_temperature',
                        type=float,
                        default= 0.1,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--mtf_scale',
                        type=int,
                        default=256,
                        help='Square array size for MFT.')

    parser.add_argument('--ensemble_size',
                        type=int,
                        default=1,
                        help='Number of samples to take.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=100000000,
                        help='Number annealing steps to run.')

    parser.add_argument("--show_plot", action='store_true',
                        default=False,
                        help="Show the plot?")

    parser.add_argument("--save_plot", action='store_true',
                        default=False,
                        help='Save the plot?')


    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()


    main(parsed_flags)