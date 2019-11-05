"""

Distributed Aperture System for Interferometric Exploitation control system
simulation environment.

Author: Justin Fletcher
"""

import gym
import time
import numpy as np

from gym import spaces, logger
from gym.utils import seeding

class DasieEnv(gym.Env):
    """
    Description:
        A distributed aperture telescope is tasked to observe an extended space
        object. This object and the intervening atmosphere cause an at-aperture
        illuminance for each aperture. The joint phase state of the apertures
        create an optical point spread function with which the at-aperture
        illuminance is convolved creating a speckle image.

    Source:
        This environment corresponds to the Markov decision process described
        in the forthcoming work by Fletcher et. al.

    Observation: 
        Type: Tuple(observation_window_size )
        Each observation is a tuple of the last N_w (window size)
        
    Actions:
        Type: Tuple(3 * num_apertures)
        Each tuple corresponds to one aperture's tip, tilt, and piston command.
        These commands will be attempted, but may be interrupted, and take time
        steps to execute. The truth tip, tilt, and piston state are hidden. The
        observers of the system not access them directly.

    Reward:
        Currently undefined. Eventually computed from SNIIRS gains.

    Starting State:
        Currently undefined.

    Episode Termination:
        Currently undefined.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs):

        print(kwargs)

        # First, parse the keyword arguments.
        self.num_apertures = kwargs['num_apertures']
        self.phase_simulation_resolution = kwargs['phase_simulation_resolution']
        self.aperture_tip_phase_error_scale = kwargs['aperture_tip_phase_error_scale']
        self.aperture_tilt_phase_error_scale = kwargs['aperture_tilt_phase_error_scale']
        self.aperture_piston_phase_error_scale = kwargs['aperture_piston_phase_error_scale']

        # Create an empty phase matrix to modify.
        self.system_phase_matrix = 0.0 * np.ones((self.phase_simulation_resolution,
                                                  self.phase_simulation_resolution))

        # TODO: Compute the number of tau needed for one model inference.

        # Define the action space for DASIE, by setting the lower limits...
        tip_lower_limit = 0.0
        tilt_lower_limits = 0.0
        piston_lower_limits = 0.0
        lower_limits = np.array([tip_lower_limit,
                                 tilt_lower_limits,
                                 piston_lower_limits])

        # ...and upper limits for each aperture.
        tip_upper_limit = 1.0
        tilt_upper_limits = 1.0
        piston_upper_limits = 1.0
        upper_limits = np.array([tip_upper_limit,
                                 tilt_upper_limits,
                                 piston_upper_limits])

        # We then create a list to hold the composite action spaces, and...
        aperture_action_spaces = list()

        # ...iterate over each aperture...
        for aperture_num in range(self.num_apertures):

            # ...instantiating a box space using the established limits...
            aperture_space = spaces.Box(lower_limits,
                                        upper_limits, dtype=np.float32)

            # ...and adding that space to the list of spaces.
            aperture_action_spaces.append(aperture_space)

        # Finally, we complete the space by building it from the template list.
        self.action_space = spaces.Tuple(aperture_action_spaces)

        # TODO: Build a correct observation space.
        self.observation_space = spaces.Tuple(aperture_action_spaces)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        # Set the initial state. This is the first thing called in an episode.

        self.apertures = list()
        # TODO: Once refactored, parallelize this.
        for aperture_index in range(self.num_apertures):

            # TODO: Refactor to a build_aperture function.

            start_time = time.time()

            # Compute the angular position for this aperture.
            angular_position = 2 * np.pi * aperture_index / self.num_apertures

            # Compute the global midpoint for all apertures.
            phase_map_midpoint = np.floor(self.phase_simulation_resolution // 2)

            # Compute a scale factor; shrink the subapertures by just enough.
            # TODO: Determine why this works. I made it. I'm just not sure how.
            aperture_scale = (np.cos(np.pi / self.num_apertures) / np.sin(np.pi / self.num_apertures)) + np.pi

            # Scale the midpoint to get the radius.
            aperture_radius = (phase_map_midpoint / aperture_scale)
            aperture_annulus_radius = (aperture_radius / np.sin(np.pi / self.num_apertures))

            # Compute the global-reference midpoint using position and radius.
            phase_map_x_centroid = phase_map_midpoint + (aperture_annulus_radius * np.sin(angular_position))
            phase_map_y_centroid = phase_map_midpoint + (aperture_annulus_radius * np.cos(angular_position))

            # For each aperture, compute all of it's pixel coordinates.
            xx, yy = np.mgrid[:self.phase_simulation_resolution,
                              :self.phase_simulation_resolution]

            # Compute the meshgrid map for the radius of this apertures circle.
            circle = (xx - phase_map_x_centroid) ** 2 + \
                     (yy - phase_map_y_centroid) ** 2

            # Compute the inclusive region of a circle given by this radius.
            circle = np.sqrt(circle) <= aperture_radius

            # Next, we bookkeep the aperture, by computing its extent...
            r_min = self.phase_simulation_resolution
            r_max = 0
            c_min = self.phase_simulation_resolution
            c_max = 0

            # ...and its pixels. Both methods are retained for comparisons.
            aperture_pixels = list()

            # Iterate over each pixle in the whole phase map. This is slow.
            for r, row in enumerate(circle):
                for c, value in enumerate(row):

                    # If the pixel is in the circle of this aperture...
                    if circle[r, c]:

                        # ...store the pixel...
                        aperture_pixels.append((r, c))

                        # ...and use it to assess aperture pixel bounds.
                        if r < r_min:
                            r_min = r
                        if r > r_max:
                            r_max = r
                        if c < c_min:
                            c_min = c
                        if c > c_max:
                            c_max = c

            # Now, create a new dict to track this aperture through simulation.
            aperture = dict()

            # Store a reference index for this aperture.
            aperture['index'] = aperture_index

            # Initialize the tip, tilt, and piston for this aperture.
            tip = 0.0 + self.aperture_tip_phase_error_scale * np.random.randn(1)
            tilt = 0.0 + self.aperture_tilt_phase_error_scale * np.random.randn(1)
            piston = 1.0 + self.aperture_piston_phase_error_scale * np.random.randn(1)
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

            # Store this apertures circular mask. Eases later computations.
            aperture['phase_map_circle_patch'] = circle[r_min:r_max, c_min:c_max]
            aperture['phase_map_radius'] = aperture_radius
            aperture['phase_map_x_centroid'] = phase_map_x_centroid
            aperture['phase_map_y_centroid'] = phase_map_y_centroid

            aperture['pixel_list'] = aperture_pixels

            self.apertures.append(aperture)

            print("--- Setup time = %s seconds ---" % (time.time() - start_time))

        self._update_phase_matrix()
        self._update_psf_matrix()

        # TODO: Initialize state.
        self.state = None
        self.steps_beyond_done = None
        return np.array(self.state)

    def step(self, action):

        # First, ensure the step action is valid.
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # The action is valid. Use it to update the ttp in each aperture.
        for (aperture, aperture_command) in zip(self.apertures, action):

            aperture['tip_phase_command'] = aperture_command[0]
            aperture['tilt_phase_command'] = aperture_command[1]
            aperture['piston_phase_command'] = aperture_command[2]

            # TODO: Move this and do incremental updates.
            aperture['tip_phase'] = aperture['tip_phase_command']
            aperture['tilt_phase'] = aperture['tilt_phase_command']
            aperture['piston_phase'] =aperture['piston_phase_command']

        # # TODO: Write main simulation loop here.
        # # num_timesteps is probably the inference latency of the system.
        # for t in range(num_timesteps):
        #     # TODO: Iterate the target model state.
        #     # TODO: Iterate the at-aperture irradiance.
        #     # TODO: Iterate toward commanded articulations.

        # # Parse the state.
        # state = self.state
        # x, x_dot, theta, theta_dot = state
        # self.state = (x,x_dot,theta,theta_dot)

        self._update_phase_matrix()
        self._update_psf_matrix()

        # TODO: Figure out what this means for our problem, if anything.
        done = False

        # TODO: Build a reward function.
        # In this implementation, reward is a function of done.
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Just finished
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        # compute_reward()
        # compute_done()

        self.state = self.optical_psf_matrix

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):

        if self.state is None: return None

        def positive_shift_norm_scale_img(img, scale=2**8):

            shifted_img = img + np.abs(np.min(img))

            normalized_img = shifted_img / np.max(shifted_img)

            scaled_img = (scale) * normalized_img

            return(scaled_img)

        def to_rgb(im, dtype=np.uint8):
            """
            This function maps a matrix to a greyscale RGB image quickly.
            """
            w, h = im.shape
            ret = np.empty((w, h, 3), dtype=dtype)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
            return ret

        log_real_optical_psf = np.log(np.abs(np.fft.fftshift(self.optical_psf_matrix)))
        log_real_optical_psf_img = to_rgb(positive_shift_norm_scale_img(log_real_optical_psf))
        system_phase_matrix_img = to_rgb(positive_shift_norm_scale_img(self.system_phase_matrix))

        render_image = np.hstack([system_phase_matrix_img,
                                  log_real_optical_psf_img])

        if mode == 'rgb_array':

            return render_image

        elif mode == 'human':

            from gym.envs.classic_control import rendering

            if self.viewer is None:

                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(render_image)

            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _update_phase_matrix(self):
        mode = "use_aperture_patch"

        # TODO: Multithread this.
        for aperture in self.apertures:

            piston = aperture['piston_phase']
            tip = aperture['tip_phase']
            tilt = aperture['tilt_phase']

            [r_min, r_max, c_min, c_max] = aperture['phase_map_patch_bounds']

            # Get the last patch, and use it to compute a phase map delta.
            old_patch = aperture['phase_map_patch']

            # Create current patch based on the tip, tilt, piston.
            xx, yy = np.mgrid[:(r_max - r_min),
                     :(c_max - c_min)]
            new_patch = (tip * xx) + (tilt * yy) + piston

            # As the number of apertures increases, pixel-wise is better.
            # As the number of apertures decreases, matrix addition is better.
            # Both scale with the square of the resolution,  but pixel-wise scales worse.
            aperture['phase_map_patch'] = new_patch

            if mode == "use_matrix_addition":

                # This meshgrid approach is extremely inefficient.
                xx, yy = np.mgrid[:self.phase_simulation_resolution,
                         :self.phase_simulation_resolution]

                circle = (xx - aperture['phase_map_x_centroid']) ** 2 + \
                         (yy - aperture['phase_map_y_centroid']) ** 2

                circle = np.sqrt(circle) <= aperture['phase_map_radius']

                patch_delta = new_patch - old_patch
                patch_delta = patch_delta * aperture[
                    'phase_map_circle_patch']

                self.system_phase_matrix[r_min:r_max,
                c_min:c_max] += patch_delta * circle[r_min:r_max,
                                              c_min:c_max]

            elif mode == "use_pixel_list":

                for (r, c) in aperture['pixel_list']:
                    self.system_phase_matrix[r, c] = new_patch[
                        r - r_min - 1, c - c_min - 1]

            elif mode == "use_aperture_patch":

                patch_delta = new_patch - old_patch
                patch_delta = patch_delta * aperture['phase_map_circle_patch']
                self.system_phase_matrix[r_min:r_max, c_min:c_max] += patch_delta

    def _update_psf_matrix(self):

        # Finally, compute the psy of this state.
        self.optical_psf_matrix = np.fft.fft2(self.system_phase_matrix,
                                              norm="ortho")