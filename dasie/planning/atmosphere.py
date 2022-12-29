import argparse

import numpy as np

import tensorflow as tf
from matplotlib import  pyplot as plt


# def ft_phase_screen(r0, N, delta, L0, l0, FFT=None, seed=None):
#     """
#     Creates a random phase screen with Von Karmen statistics.
#     (Schmidt 2010)
#
#     Parameters:
#         r0 (float): r0 parameter of scrn in metres
#         N (int): Size of phase scrn in pxls
#         delta (float): size in Metres of each pxl
#         L0 (float): Size of outer-scale in metres
#         l0 (float): inner scale in metres
#
#     Returns:
#         ndarray: numpy array representing phase screen
#     """
#     delta = float(delta)
#     r0 = float(r0)
#     L0 = float(L0)
#     l0 = float(l0)
#
#     R = random.SystemRandom(time.time())
#     if seed is None:
#         seed = int(R.random() * 100000)
#     numpy.random.seed(seed)
#
#     del_f = 1. / (N * delta)
#
#     fx = numpy.arange(-N / 2., N / 2.) * del_f
#
#     (fx, fy) = numpy.meshgrid(fx, fx)
#     f = numpy.sqrt(fx ** 2. + fy ** 2.)
#
#     fm = 5.92 / l0 / (2 * numpy.pi)
#     f0 = 1. / L0
#
#     PSD_phi = (0.023 * r0 ** (-5. / 3.) * numpy.exp(-1 * ((f / fm) ** 2)) / (
#                 ((f ** 2) + (f0 ** 2)) ** (11. / 6)))
#
#     PSD_phi[int(N / 2), int(N / 2)] = 0
#
#     cn = ((numpy.random.normal(size=(N, N)) + 1j * numpy.random.normal(
#         size=(N, N))) * numpy.sqrt(PSD_phi) * del_f)

#
#     phs = ift2(cn, 1, FFT).real
#
#     return phs


def make_von_karman_phase_grid(r0,
                           spatial_quantization,
                           pupil_plane_extent,
                           outer_scale,
                           inner_scale):

    """
    Credit to: https://aotools.readthedocs.io/en/v1.0.6/_modules/aotools/turbulence/phasescreen.html
    Creates a random phase screen with Von Karman statistics.
    (Schmidt 2010)
    The phase screen is returned as a 2d array, with each element representing
    the phase change in radians. This means that to obtain the physical phase
    distortion in nanometres, it must be multiplied by (wavelength / (2*pi)),
    (where wavelength here is the same wavelength in which r0 is given in the
    function arguments)

    Parameters:
        r0 (float): r0 parameter of screen in meters .
        spatial_quantization (int): Size of phase screen in pixels.
        pupil_plane_extent (float): size in meters of the pupil plane.
        outer_scale (float): Size of outer-scale in metres.
        inner_scale (float): inner scale in metres.

    Returns:
        ndarray: np array representing phase screen
    """

    # Build a distance grid.
    fx = np.arange(-spatial_quantization / 2.0,
                   spatial_quantization / 2.0) / (pupil_plane_extent)


    (fx, fy) = np.meshgrid(fx, fx)
    f = np.sqrt(fx ** 2. + fy ** 2.)

    # Compute a static phase grid base
    fm = 5.92 / inner_scale / (2 * np.pi)
    f0 = 1. / outer_scale
    # PSD_phi
    static_phase_grid = (0.023 *
                         r0 ** (-5. / 3.) *
                         np.exp(-1 * ((f / fm) ** 2)) /
                         (((f ** 2) + (f0 ** 2)) ** (11. / 6)))

    # Set the center point to zero.
    static_phase_grid[int(spatial_quantization / 2),
                      int(spatial_quantization / 2)] = 0

    return (static_phase_grid)


def make_phase_screen_radians(pupil_plane_extent,
                              spatial_quantization,
                              static_phase_grid,
                              real_sample,
                              img_sample):
    """
    Credit to: https://aotools.readthedocs.io/en/v1.0.6/_modules/aotools/turbulence/phasescreen.html
    Creates a random phase screen with Von Karman statistics.
    (Schmidt 2010)
    The phase screen is returned as a 2d array, with each element representing
    the phase change in radians. This means that to obtain the physical phase
    distortion in nanometres, it must be multiplied by (wavelength / (2*pi)),
    (where wavelength here is the same wavelength in which r0 is given in the
    function arguments)

    Parameters:
        pupil_plane_extent (float): size in meters of the pupil plane.
        static_phase_grid: ndarray representing Von Karman statistics.
        spatial_quantization (int): Size of phase screen in pixels
    Returns:
        Tensor: a array representing phase screen
    """

    cn = tf.complex(real_sample, img_sample) * np.sqrt(static_phase_grid) / pupil_plane_extent

    # TODO: Validate. Delta f is set to 1.0 in the source, but might be a bug.
    delta_f = 1
    # delta_f = 1.0 / (pupil_plane_extent)

    phase_radians = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(cn))) * (spatial_quantization * delta_f) ** 2

    return tf.abs(phase_radians)

def main(flags):

    sample_size = (flags.spatial_quantization, flags.spatial_quantization)

    static_phase_grid = make_static_phase_grid(flags.r0,
                                               flags.spatial_quantization,
                                               flags.pupil_plane_extent,
                                               flags.outer_scale,
                                               flags.inner_scale)

    num_exposures = 10
    sample_scale = 1.0

    # Initialize base sample matrices.
    real_sample = np.random.normal(size=sample_size)
    img_sample = np.random.normal(size=sample_size)

    phase_screens_radians = list()

    # Evolve the atmospheric phase screen.
    for t in range(num_exposures):

        # Add a new sample and re-normalize to prevent convergence.
        real_sample += sample_scale * np.random.normal(size=sample_size)
        img_sample += sample_scale * np.random.normal(size=sample_size)

        # real_sample = real_sample / np.max(real_sample)
        # img_sample = img_sample / np.max(img_sample)


        phase_screen_radians = make_phase_screen_radians(flags.pupil_plane_extent,
                                                         flags.spatial_quantization,
                                                         static_phase_grid,
                                                         real_sample,
                                                         img_sample)
        phase_screens_radians.append(phase_screen_radians)

    # Plot the screens
    for phase_screen_radians in phase_screens_radians:

        plt.imshow(phase_screen_radians)
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='provide arguments.')

    parser.add_argument('--r0', type=float,
                        default=0.20,
                        help='r0 parameter of scrn in metres.')

    parser.add_argument('--spatial_quantization', type=int,
                        default=512,
                        help='Size of phase screen in pixels.')

    parser.add_argument('--pupil_plane_extent', type=float,
                        default=3.6,
                        help='Size in meters of each pixel.')

    parser.add_argument('--outer_scale', type=float,
                        default=200.0,
                        help='Size of outer-scale in meters.')

    parser.add_argument('--inner_scale', type=float,
                        default=0.1,
                        help='inner scale in meters.')



    # Parse known arguments.
    parsed_flags, _ = parser.parse_known_args()

    main(parsed_flags)