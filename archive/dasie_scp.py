

import os
import time
import random
import argparse
import itertools
from datetime import datetime
from collections import deque
from decimal import Decimal

import joblib
import numpy as np
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
                 initial_temperature):

        # Initialize annealing governing functions.
        self.state_perturber = state_perturber
        self.state_cost_evaluator = state_cost_evaluator
        self.state = state
        self.temperature_function = temperature_function
        self.acceptance_function = acceptance_function
        self.initial_temperature = initial_temperature

        # Intialize the annealing state variables.
        self.epoch = 0
        self.temperature = initial_temperature
        self.current_cost = []

    def update_temperature(self):

        self.temperature = self.temperature_function(self.epoch,
                                                     self.initial_temperature)

    def reanneal(self):

        self.epoch = 0
        self.temperature = self.initial_temperature

    def __call__(self, perturb_params=[], input_data=[]):

        if self.epoch == 0:

            self.current_cost = self.state_cost_evaluator.evaluate(input_data)

        # Incement the epoch count.
        self.epoch = self.epoch + 1

        # print("self.epoch = %.d" % self.epoch)

        # Update the temperature.
        self.update_temperature()

        # print("self.temperature = %.6f" % self.temperature)

        # Store the current state.
        self.state.store()

        # Create a new state using the perturbation function.
        self.state_perturber.perturb(perturb_params)

        # Compute the cost of this state using the provided grader.
        candidate_cost = self.state_cost_evaluator.evaluate(input_data)

        # Compute the cost function delta induced by this state.
        cost_delta = candidate_cost - self.current_cost

        # print("cost_delta = %.6f" % cost_delta)

        # If the candidate state reduces the cost...
        if(cost_delta <= 0.0):

            # print("Accept 1")
            # ...accept the candidate state implicitly by updating the cost.
            self.current_cost = candidate_cost

        # If the cost is increased...
        else:

            # print('np.exp(-d / t) = %.9f' % np.exp(-cost_delta / self.temperature))
            # ...and if the acceptance function returns True...
            if(self.acceptance_function(self.temperature, cost_delta)):

                # print("Accept 2")
                # ...accept the state implicitly by updating the cost.
                self.current_cost = candidate_cost

            # If not, we reject the state by restoring the prior state.
            else:

                # print("Reject")
                self.state.restore()

        return()



class DASIECostEvaluator(object):

    def __init__(self, mtf_scale, dasie_psf_sampler):

        self.mtf_scale = mtf_scale
        self.dasie_psf_sampler = dasie_psf_sampler
        self.one_matrix = np.ones((mtf_scale, mtf_scale))
        self.aggregate_mtf = np.zeros((mtf_scale, mtf_scale))

    def mtf_from_dasie_actuation(self, ptt):

        actuated_dasie_psf = self.dasie_psf_sampler.sample(ptt_actuate=ptt)
        actuated_dasie_mtf = np.fft.fft2(actuated_dasie_psf, norm="ortho")
        return(actuated_dasie_mtf)


    def reset_aggregates(self):

        self.aggregate_mtf = np.zeros((self.mtf_scale, self.mtf_scale))

    def evaluate(self, dasie_actuation_ensemble):

        # TODO: refactor to 3D matrix for faster evaluation.

        # Compute and store the MTFs of the actuation state from the ensemble.
        mtfs = list()
        for dasie_actuation in dasie_actuation_ensemble:
            mtfs.append(self.mtf_from_dasie_actuation(dasie_actuation))

        # Average the MTFs; that average is the cost for this ensemble.
        for mft in mtfs:
            self.aggregate_mtf += np.divide(self.one_matrix, mft)
        cost = self.aggregate_mtf / len(dasie_actuation_ensemble)

        # Clear the aggregate MTF.
        self.reset_aggregates()

        # Return the cost.
        return cost


def csa_acceptance_function(temperature, cost_delta):
    accept = np.exp(-cost_delta / temperature) > np.random.rand()

    return accept

def csa_temperature_function(temperature, cost_delta):

    accept = np.exp(-cost_delta / temperature) > np.random.rand()

    return accept

def main(flags):

    # Instantiate DASIE model.
    mas_setup = joblib.load('21-01-25_256_mas_setup_2.5as_1um_.05frac.pkl')
    dasie_psf_sampler = MultiAperturePSFSampler(**mas_setup)

    # list of PTTs
    dasie_psf_sampler.sample(ptt_actuate=ptt)

    state = np.random.rand()
    state_perturber = 0
    state_cost_evaluator = DASIECostEvaluator(flags.mtf_scale,
                                              dasie_psf_sampler)
    temperature_function = csa_temperature_function
    acceptance_function = csa_acceptance_function
    initial_temperature = 100.0
    annealer = Annealer(state,
                        state_perturber,
                        state_cost_evaluator,
                        temperature_function,
                        acceptance_function,
                        initial_temperature)

    # for subaperture_id in range(flags.num_subapertures):
    #
    #     actuation_states = list(range(flags.num_actuation_states))
    #     print(actuation_states)
    #     joint_actuation_tuples = itertools.product(actuation_states,
    #                                                actuation_states,
    #                                                actuation_states)
    #
    #     for (tip_state, tilt_state, piston_state) in joint_actuation_tuples:
    #
    #         print(subaperture_id, tip_state, tilt_state, piston_state)

        # compute_mft


    # for () in zip(itertools.product())
    #
    #     ttp =

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for training.')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1234,
                        help='A random seed for repeatability.')

    parser.add_argument('--num_subapertures',
                        type=int,
                        default=16,
                        help='Number of DASIE subapertures.')

    parser.add_argument('--num_actuation_states',
                        type=int,
                        default=2**14,
                        help='Number of possible actuation states.')

    parser.add_argument('--mtf_scale',
                        type=int,
                        default=256,
                        help='Square array size for MFT.')

    parser.add_argument('--ensemble_size',
                        type=int,
                        default=8,
                        help='Number of samples to take.')

    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()


    main(parsed_flags)