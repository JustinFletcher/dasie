"""
A library for the computation of DASIE MTF ensembles.

Author: Justin Fletcher
"""

import os
import math
import time
import copy
import json
import hcipy
import codecs
import joblib
import datetime
import argparse
import itertools

import pandas as pd
import numpy as np

from decimal import Decimal

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulate_multi_aperture import SimulateMultiApertureTelescope

from multi_aperture_psf import MultiAperturePSFSampler

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

def mtf_from_psf(psf):

    return  np.abs(np.fft.fft2(psf))

def get_monolithic_mft(pup_diamater=3.6):
    # Ian's code to make a monolithic aperture MTF.
    mono_setup = joblib.load('21-01-25_256_mas_setup_2.5as_1um_.05frac.pkl')
    mir_centers = hcipy.CartesianGrid(np.array([[0], [0]]))
    mir_diamater = 2.5
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

def load_best_state():
    obj_text = codecs.open("./best_state.json", 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    return np.array(b_new)

def cosine_similarity(u, v):
    """

    :param u: Any np.array matching u in shape (and semantics probably)
    :param v: Any np.array matching u in shape (and semantics probably)
    :return: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    """
    u = u.flatten()
    v = v.flatten()

    return (np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v)))

def compute_cosine_similarity_ratio(distributed_aperture_diameter,
                                    monolithic_aperture_diameter,
                                    perfect_image_spectrum,
                                    perfect_image_flipped,
                                    filter_psf_extent=1.0,
                                    pupil_plane_resolution=2**8,
                                    num_apertures=15,
                                    subaperture_radius=None):

    # Simulate a distributed aperture telescope.
    distributed_aperture_radius = distributed_aperture_diameter / 2
    distributed_telescope_sim = SimulateMultiApertureTelescope(mirror_layout='elf',
                                                               telescope_radius=distributed_aperture_radius,
                                                               pupil_plane_resolution=pupil_plane_resolution,
                                                               num_apertures=num_apertures,
                                                               filter_psf_extent=filter_psf_extent,
                                                               subaperture_radius=subaperture_radius)
    distributed_ptt_shape = (distributed_telescope_sim.num_apertures, 3)
    distributed_ptt_actuation = np.zeros(distributed_ptt_shape)
    distributed_telescope_psf, Y, strehls = distributed_telescope_sim.get_observation(distributed_ptt_actuation)
    distributed_telescope_psf = distributed_telescope_psf[..., 0]
    distributed_telescope_mtf = mtf_from_psf(distributed_telescope_psf)
    distributed_image_spectrum = perfect_image_spectrum * distributed_telescope_mtf

    distributed_image = np.abs(np.fft.fft2(distributed_image_spectrum))
    distributed_image = distributed_image / np.max(distributed_image)
    distributed_cosine_similarity = cosine_similarity(distributed_image, perfect_image_flipped)
    print("Dist cosine similarity: " + str(distributed_cosine_similarity))

    # Simulate a monolithic aperture telescope.
    monolithic_aperture_radius = monolithic_aperture_diameter / 2
    monolithic_telescope_sim = SimulateMultiApertureTelescope(mirror_layout='monolithic',
                                                              telescope_radius=monolithic_aperture_radius,
                                                              filter_psf_extent=filter_psf_extent,
                                                              pupil_plane_resolution=pupil_plane_resolution)
    monolithic_ptt_shape = (monolithic_telescope_sim.num_apertures, 3)
    monolithic_ptt_actuation = np.zeros(monolithic_ptt_shape)
    monolithic_telescope_psf, Y, strehls = monolithic_telescope_sim.get_observation(monolithic_ptt_actuation)
    monolithic_telescope_psf = monolithic_telescope_psf[..., 0]
    monolithic_telescope_mtf = mtf_from_psf(monolithic_telescope_psf)
    monolithic_image_spectrum = perfect_image_spectrum * monolithic_telescope_mtf

    monolithic_image = np.abs(np.fft.fft2(monolithic_image_spectrum))
    monolithic_image = monolithic_image / np.max(monolithic_image)
    monolithic_cosine_similarity = cosine_similarity(monolithic_image, perfect_image_flipped)
    print("Mono cosine similarity: " + str(monolithic_cosine_similarity))

    dist_mono_cos_sim_ratio = distributed_cosine_similarity / monolithic_cosine_similarity
    print("dist/mono cosine similarity: " + str(dist_mono_cos_sim_ratio))

    return(dist_mono_cos_sim_ratio)


def compute_cosine_similarity_ratios(distributed_aperture_diameter_start,
                                     distributed_aperture_diameter_stop,
                                     monolithic_aperture_diameter_start,
                                     monolithic_aperture_diameter_stop,
                                     aperture_diameter_num,
                                     perfect_image,
                                     filter_psf_extent=1.0,
                                     pupil_plane_resolution=2**9,
                                     sample_mode='linspace',
                                     subaperture_mode='fixed',
                                     num_apertures=15):

    # Simulate a perfect image only once.
    perfect_image = perfect_image / np.max(perfect_image)
    perfect_image_spectrum = np.fft.fft2(perfect_image)
    perfect_image_flipped = np.fliplr(np.flipud(perfect_image))

    # Create a list to hold the output tuples for this group.
    similarity_ratio_mono_dist_tuples = list()

    if sample_mode == 'linspace':

        distributed_aperture_diameters = np.linspace(distributed_aperture_diameter_start,
                                                     distributed_aperture_diameter_stop,
                                                     aperture_diameter_num)
        monolithic_aperture_diameters = np.linspace(monolithic_aperture_diameter_start,
                                                    monolithic_aperture_diameter_stop,
                                                    aperture_diameter_num)

        sample_parameter_tuples = itertools.product(distributed_aperture_diameters,
                                                    monolithic_aperture_diameters)
        num_samples = len(distributed_aperture_diameters) * len(monolithic_aperture_diameters)

    elif sample_mode == 'uniform':

        distributed_aperture_diameters = np.random.uniform(low=distributed_aperture_diameter_start,
                                                           high=distributed_aperture_diameter_stop,
                                                           size=aperture_diameter_num)
        monolithic_aperture_diameters = np.random.uniform(low=monolithic_aperture_diameter_start,
                                                          high=monolithic_aperture_diameter_stop,
                                                          size=aperture_diameter_num)


        sample_parameter_tuples = itertools.product(distributed_aperture_diameters,
                                                    monolithic_aperture_diameters)
        num_samples = len(distributed_aperture_diameters) * len(monolithic_aperture_diameters)

    else:

        sample_parameter_tuples = None
        num_samples = 0


    for index, sample_parameter_tuple in enumerate(sample_parameter_tuples):

        (distributed_aperture_diameter, monolithic_aperture_diameter) = sample_parameter_tuple
        print("#---- Sample %d / %d ----#" % (index, num_samples))
        print("#---- d_D = %s, d_M = %s ----#" % (distributed_aperture_diameter, monolithic_aperture_diameter))

        if subaperture_mode == 'fixed':

            num_apertures = num_apertures
            subaperture_radius = None

        elif subaperture_mode == 'monolithic_area_match':

            # Compute the number of apertures which most closely matches this monolithic aperture, rounding down.
            # TODO: this is wrong.
            distributed_aperture_radius = None
            num_apertures = int(((monolithic_aperture_diameter / 2) ** 2) / ((distributed_aperture_radius) ** 2))

            # Set the subaperture radius to match the number of apertures choosen.
            subaperture_radius = np.sqrt(((monolithic_aperture_diameter / 2) ** 2) / float(num_apertures))

        dist_mono_cos_sim_ratio = compute_cosine_similarity_ratio(distributed_aperture_diameter,
                                                                  monolithic_aperture_diameter,
                                                                  perfect_image_spectrum,
                                                                  perfect_image_flipped,
                                                                  filter_psf_extent=filter_psf_extent,
                                                                  pupil_plane_resolution=pupil_plane_resolution,
                                                                  num_apertures=num_apertures,
                                                                  subaperture_radius=subaperture_radius)

        similarity_ratio_mono_dist_tuples.append((dist_mono_cos_sim_ratio,
                                                  monolithic_aperture_diameter,
                                                  distributed_aperture_diameter))

    return similarity_ratio_mono_dist_tuples


def main(flags):

    # Set up some log directories.
    timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(".", "logs", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Read the target image only once.
    perfect_image = plt.imread('sample_image.png')

    # (dist_mono_cos_sim_ratios,
    #  distributed_aperture_diameters,
    #  monolithic_aperture_diameters) = compute_cosine_similarity_ratios(distributed_aperture_diameter_start=flags.distributed_aperture_diameter_start,
    #                                                                    distributed_aperture_diameter_stop=flags.distributed_aperture_diameter_stop,
    #                                                                    monolithic_aperture_diameter_start=flags.monolithic_aperture_diameter_start,
    #                                                                    monolithic_aperture_diameter_stop=flags.monolithic_aperture_diameter_stop,
    #                                                                    aperture_diameter_num=flags.aperture_diameter_num,
    #                                                                    perfect_image=perfect_image,
    #                                                                    pupil_plane_resolution=2**8)

    similarity_ratio_mono_dist_tuples = compute_cosine_similarity_ratios(distributed_aperture_diameter_start=flags.distributed_aperture_diameter_start,
                                                                         distributed_aperture_diameter_stop=flags.distributed_aperture_diameter_stop,
                                                                         monolithic_aperture_diameter_start=flags.monolithic_aperture_diameter_start,
                                                                         monolithic_aperture_diameter_stop=flags.monolithic_aperture_diameter_stop,
                                                                         aperture_diameter_num=flags.aperture_diameter_num,
                                                                         filter_psf_extent=flags.filter_psf_extent,
                                                                         sample_mode='uniform',
                                                                         perfect_image=perfect_image,
                                                                         pupil_plane_resolution=2**7)


    # TODO: Big decouple here.

    cols = ['similarity_ratio', 'mono', 'dist']

    df = pd.DataFrame(similarity_ratio_mono_dist_tuples, columns=cols)
    distributed_aperture_diameters = np.sort(df['dist'].unique())
    monolithic_aperture_diameters = np.sort(df['mono'].unique())
    pivot_df = df.pivot(index='dist', columns='mono', values='similarity_ratio')
    dist_mono_cos_sim_ratios = df.pivot(index='dist', columns='mono', values='similarity_ratio').to_numpy()

    xy_pairs = list(itertools.product(list(pivot_df.columns.values), list(pivot_df.index.values)))
    x = [p[0] for p in xy_pairs]
    y = [p[1] for p in xy_pairs]
    print(monolithic_aperture_diameters)
    print(dist_mono_cos_sim_ratios)

    if flags.save_plot:
        # Plot.
        nrows = 1
        ncols = 3

        fig = plt.figure(constrained_layout=True, figsize=(16, 4), dpi=160)
        gs0 = fig.add_gridspec(1, 1)
        gs00 = gs0[0].subgridspec(nrows, ncols)

        # Plot distributed aperture images.
        dist_mono_cos_sim_ratios_ax = fig.add_subplot(gs00[0, 0])
        # ratio_plot = dist_mono_cos_sim_ratios_ax.matshow(dist_mono_cos_sim_ratios, origin='lower')

        ratio_plot = dist_mono_cos_sim_ratios_ax.pcolormesh(monolithic_aperture_diameters,
                                                            distributed_aperture_diameters,
                                                            dist_mono_cos_sim_ratios)
        fig.colorbar(ratio_plot, ax=dist_mono_cos_sim_ratios_ax)


        dist_mono_cos_sim_ratios_ax = fig.add_subplot(gs00[0, 1])
        ratio_plot = dist_mono_cos_sim_ratios_ax.contour(monolithic_aperture_diameters,
                                                         distributed_aperture_diameters,
                                                         dist_mono_cos_sim_ratios)
        fig.colorbar(ratio_plot, ax=dist_mono_cos_sim_ratios_ax)

        dist_mono_cos_sim_ratios_ax = fig.add_subplot(gs00[0, 2])
        ratio_plot = dist_mono_cos_sim_ratios_ax.scatter(x,
                                                         y)

        print(list(distributed_aperture_diameters))
        # dist_mono_cos_sim_ratios_ax.set_xticklabels(distributed_aperture_diameters)
        # dist_mono_cos_sim_ratios_ax.set_yticklabels(monolithic_aperture_diameters)

        # Plot monolithic aperture images.
        # monolithic_psf_ax = fig.add_subplot(gs00[0, 1])
        # monolithic_psf_ax.imshow(np.log10(np.fft.fftshift(monolithic_telescope_mtf)))
        # monolithic_psf_ax.imshow((np.fft.fftshift(monolithic_telescope_mtf)))
        # monolithic_psf_ax.imshow(monolithic_image)
        # monolithic_psf_ax.axis("off")

        run_id = None
        if run_id:

            fig_path = os.path.join(save_dir, run_id + '.png')
            plt.savefig(fig_path)

        else:

            fig_path = os.path.join('./', 'tmp.png')
            plt.savefig(fig_path)


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

    parser.add_argument('--distributed_aperture_diameter_start',
                        type=float,
                        default=1.0,
                        help='Diameter of the distributed aperture system in meters.')

    parser.add_argument('--filter_psf_extent',
                        type=float,
                        default=2.0,
                        help='Angular extent of simulated PSF (arcsec)')

    parser.add_argument('--monolithic_aperture_diameter_start',
                        type=float,
                        default=1.0,
                        help='Diameter of the monolithic aperture system in meters.')

    parser.add_argument('--distributed_aperture_diameter_stop',
                        type=float,
                        default=30.0,
                        help='Diameter of the distributed aperture system in meters.')

    parser.add_argument('--monolithic_aperture_diameter_stop',
                        type=float,
                        default=30.0,
                        help='Diameter of the monolithic aperture system in meters.')

    parser.add_argument('--aperture_diameter_num',
                        type=int,
                        default=64,
                        help='Number of linspaced aperture values to simulate')

    parser.add_argument('--num_actuation_states',
                        type=int,
                        default=2**14,
                        help='Number of possible actuation states.')

    parser.add_argument('--actuation_scale',
                        type=float,
                        default=0.001,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--initial_temperature',
                        type=float,
                        default= 1.0,
                        help='Scale of actuations, as a real.')

    parser.add_argument('--mtf_scale',
                        type=int,
                        default=256,
                        help='Square array size for MFT.')

    parser.add_argument('--ensemble_size',
                        type=int,
                        default=2,
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