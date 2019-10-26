import numpy as np
import time

from matplotlib import pyplot as plt

# Disertation
# Map problem to an MDP - Paper
# Simulate - Characterize and optimize simulation - Paper
# Integrate sim w/ openAI Gym - pull request
# Build a control model - paper
# Build a real system and transfer - paper
class DistributedapertureSystem(object):
    """docstring for DistributedapertureSystem"""

    def __init__(self,
                 num_apertures=7,
                 phase_simulation_resolution=2**10,
                 apertrue_tip_phase_error_scale=0.01,
                 apertrue_tilt_phase_error_scale=0.01,
                 apertrue_piston_phase_error_scale=0.01):
        # 2**10 == 1024

        super(DistributedapertureSystem, self).__init__()

        self.num_apertures = num_apertures
        self.phase_simulation_resolution = phase_simulation_resolution

        self.apertures = list()

        for aperture_index in range(num_apertures):

            start_time = time.time()

            angular_position = 2 * np.pi * aperture_index / num_apertures

            phase_map_midpoint = np.floor(phase_simulation_resolution // 2)

            aperture_scale = (np.cos(np.pi / num_apertures) / np.sin(np.pi / num_apertures)) + 1

            aperture_radius = (phase_map_midpoint / aperture_scale)
            aperture_annulus_radius = (( aperture_radius) * 1 / np.sin(np.pi / num_apertures))

            # aperture_radius = aperture_scale * phase_map_midpoint
            phase_map_x_centroid = phase_map_midpoint + (aperture_annulus_radius * np.sin(angular_position))
            phase_map_y_centroid = phase_map_midpoint + (aperture_annulus_radius * np.cos(angular_position))

            # For each apeture, compute all of it's pixel coordinates.
            xx, yy = np.mgrid[:self.phase_simulation_resolution,
                              :self.phase_simulation_resolution]

            circle = (xx - phase_map_x_centroid) ** 2 + \
                     (yy - phase_map_y_centroid) ** 2

            # Model the aperture as a plane.
            # Compute a mask for the plane in the composite aperture.
            # Update the composite mask using matrix addition.
            circle = np.sqrt(circle) <= aperture_radius

            r_min = phase_simulation_resolution
            r_max = 0
            c_min = phase_simulation_resolution
            c_max = 0
            aperture_pixels = list()
            for r, row in enumerate(circle):
                for c, value in enumerate(row):
                    if circle[r, c]:
                        aperture_pixels.append((r, c))
                        if r < r_min:
                            r_min = r
                        if r > r_max:
                            r_max = r
                        if c < c_min:
                            c_min = c
                        if c > c_max:
                            c_max = c

            # As the number of apertures increases, pixel-wise is better.
            # As the number of apertures decreases, matrix addition is better.
            # Both scale with the square of the resolution, but pixel-wise scales worse.

            aperture = dict()

            # Store a reference index for this aperture.
            aperture['index'] = aperture_index

            # Initialize the tip, tilt, and piston for this aperture.
            tip = 0.0 + apertrue_tip_phase_error_scale * np.random.randn(1)
            tilt = 0.0 + apertrue_tilt_phase_error_scale * np.random.randn(1)
            piston = 0.5 + apertrue_piston_phase_error_scale * np.random.randn(1)
            aperture['tip_phase'] = tip
            aperture['tilt_phase'] = tilt
            aperture['piston_phase'] = piston

            # Compute a grid of phase for this aperture, and record it's value.
            xx, yy = np.mgrid[:(r_max - r_min),
                              :(c_max - c_min)]
            patch = (tip * xx) + (tilt * yy) + piston
            aperture['phase_map_patch'] = patch

            # Store the bounds of this aperture, relative to the global phase.
            aperture['phase_map_patch_bounds'] = [r_min, r_max, c_min, c_max]
            # Store this apertures ciruclar mask. Eases later computations.
            aperture['phase_map_circle_patch'] = circle[r_min:r_max, c_min:c_max]



            # aperture controid
            aperture['phase_map_radius'] = aperture_radius
            aperture['phase_map_x_centroid'] = phase_map_x_centroid
            aperture['phase_map_y_centroid'] = phase_map_y_centroid

            aperture['pixel_list'] = aperture_pixels

            self.apertures.append(aperture)

            print("--- Setup time = %s seconds ---" % (time.time() - start_time))

        # Create an empty phase matrix to modify.
        self.system_phase_matrix = 0.0 * np.ones((phase_simulation_resolution,
                                                  phase_simulation_resolution))

        print(self.system_phase_matrix)

        num_updates = 100
        mode = "use_aperture_patch"

        for update_number in range(num_updates):
            start_time = time.time()
            self.randomly_perterb_ttp(scale=0.001)
            self.update_system_phase_matrix(mode=mode)
            self.update_system_psf_matrix()
            print("--- Update time = %s seconds ---" % (time.time() - start_time))

    def randomly_perterb_ttp(self, scale=1):

        for aperture in self.apertures:

            aperture['piston_phase'] += scale * np.random.randn(1)
            aperture['tip_phase'] += scale * np.random.randn(1)
            aperture['tilt_phase'] += scale * np.random.randn(1)

    def update_system_phase_matrix(self, mode="use_matrix_addition"):

        for aperture in self.apertures:

            if mode=="use_matrix_addition":

                # This meshgrid approach is extremely inefficient.
                xx, yy = np.mgrid[:self.phase_simulation_resolution,
                                  :self.phase_simulation_resolution]

                circle = (xx - aperture['phase_map_x_centroid']) ** 2 + \
                         (yy - aperture['phase_map_y_centroid']) ** 2

                circle = np.sqrt(circle) <= aperture['phase_map_radius']

                self.system_phase_matrix += np.random.randn(1) * circle
                self.system_phase_matrix += 1 * circle

            elif mode=="use_pixel_list":
                value = np.random.randn(1)

                for (r, c) in aperture['pixel_list']:
                    self.system_phase_matrix[r, c] += value

            elif mode == "use_aperture_patch":

                piston = aperture['piston_phase']
                tip = aperture['tip_phase']
                tilt = aperture['tilt_phase']

                [r_min, r_max, c_min, c_max] = aperture['phase_map_patch_bounds']

                # Create current patch based on the tip, tilt, piston.
                xx, yy = np.mgrid[:(r_max - r_min),
                                  :(c_max - c_min)]
                new_patch = (tip * xx) + (tilt * yy) + piston

                # Get the last patch, and use it to compute a phase map delta.
                old_patch = aperture['phase_map_patch']
                patch_delta = new_patch - old_patch

                # Update the local phase patch.
                aperture['phase_map_patch'] = new_patch

                patch_delta = patch_delta * aperture['phase_map_circle_patch']
                self.system_phase_matrix[r_min:r_max, c_min:c_max] += patch_delta


    def update_system_psf_matrix(self):


        self.optical_psf_matrix = np.fft.fft2(self.system_phase_matrix,
                                              norm="ortho")

    def show_phase_matrix(self):

        plt.matshow(self.system_phase_matrix)
        plt.colorbar()
        plt.show()

    def show_optical_psf_matrix(self):

        plt.matshow(np.log(np.abs(np.fft.fftshift(self.optical_psf_matrix))))
        plt.colorbar()
        plt.show()


if __name__ == '__main__':

    da_system = DistributedapertureSystem()

    da_system.show_phase_matrix()

    da_system.show_optical_psf_matrix()
