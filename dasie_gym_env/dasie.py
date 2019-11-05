"""

Distributed Aperture System for Interferometric Exploitation control system
simulation environment.

Author: Justin Fletcher
"""

import math
import time
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from distributed_apeture_simulation import DistributedApertureSystem

class DasieEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed
              but depends on the angle the pole is pointing. This is because
              the center of gravity of the pole increases the amount of energy
              needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to
        195.0 over 100 consecutive trials.
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

        print(self.system_phase_matrix)

        # Actions may include increasing or decreasing tip, tilt, and piston.
        # TODO: Confirm this.
        action_space_size = self.num_apertures * 3
        self.action_space = spaces.Discrete(action_space_size)

        # TODO: Figure this out.
        # TODO: Edit this space to be variable in time length.

        # self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


        # --------------------------------------------

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            # Out of bounds
            self.x_threshold * 2,
            # max speed is numerically limited.
            np.finfo(np.float32).max,
            # Fell too far
            self.theta_threshold_radians * 2,
            # Max angular speed is numerically limited.
            np.finfo(np.float32).max])

        tip_lower_limit = 0.0
        tilt_lower_limits = 0.0
        piston_lower_limits = 0.0

        tip_upper_limit = 1.0
        tilt_upper_limits = 1.0
        piston_upper_limits = 1.0


        lower_limits = np.array([tip_lower_limit,
                                 tilt_lower_limits,
                                 piston_lower_limits])

        upper_limits = np.array([tip_upper_limit,
                                 tilt_upper_limits,
                                 piston_upper_limits])

        aperture_action_spaces_tuple = list()

        for aperture_num in range(self.num_apertures):

            aperture_space = spaces.Box(lower_limits,
                                        upper_limits, dtype=np.float32)

            aperture_action_spaces_tuple.append(aperture_space)

        self.action_space = spaces.Tuple(aperture_action_spaces_tuple)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # First, ensure the step action is valid.
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # The action is valid. Use it to update the ttp in each aperture.
        for (aperture, aperture_command) in zip(self.apertures,
                                                     action):

            aperture['tip_phase'] = aperture_command[0]
            aperture['tilt_phase'] = aperture_command[1]
            aperture['piston_phase'] = aperture_command[2]

        # TODO: Update phase update mechanism to be itertive in time.

        self._update_phase_matrix()
        self._update_psf_matrix()

        # TODO: Figure out what this means for our problem, if anything

        # Parse the state.
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        done = False

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

        # The first return of this function is the observation.

        return np.array(self.state), reward, done, {}

    def reset(self):
        # Set the inital state. This is the first thing called in an episode.

        self.apertures = list()
        # TODO: Once refactored, parallelize this.
        for aperture_index in range(self.num_apertures):

            # TODO: Refactor to a build_aperture function.

            start_time = time.time()

            angular_position = 2 * np.pi * aperture_index / self.num_apertures

            phase_map_midpoint = np.floor(self.phase_simulation_resolution // 2)

            aperture_scale = (np.cos(np.pi / self.num_apertures) / np.sin(np.pi / self.num_apertures)) + 1

            aperture_radius = (phase_map_midpoint / aperture_scale)
            aperture_annulus_radius = (( aperture_radius) * 1 / np.sin(np.pi / self.num_apertures))

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

            r_min = self.phase_simulation_resolution
            r_max = 0
            c_min = self.phase_simulation_resolution
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
            # Store this apertures circuclar mask. Eases later computations.
            aperture['phase_map_circle_patch'] = circle[r_min:r_max, c_min:c_max]



            # aperture controid
            aperture['phase_map_radius'] = aperture_radius
            aperture['phase_map_x_centroid'] = phase_map_x_centroid
            aperture['phase_map_y_centroid'] = phase_map_y_centroid

            aperture['pixel_list'] = aperture_pixels

            self.apertures.append(aperture)

            print("--- Setup time = %s seconds ---" % (time.time() - start_time))

        self._update_phase_matrix()
        self._update_psf_matrix()

        # OLD BELOW -----------------------------------------------------------

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

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

            [r_min, r_max, c_min, c_max] = aperture[
                'phase_map_patch_bounds']

            # Get the last patch, and use it to compute a phase map delta.
            old_patch = aperture['phase_map_patch']

            # Create current patch based on the tip, tilt, piston.
            xx, yy = np.mgrid[:(r_max - r_min),
                     :(c_max - c_min)]
            new_patch = (tip * xx) + (tilt * yy) + piston

            # As the number of apertures increases, pixel-wise is better.
            # As the number of apertures decreases, matrix addition is
            # better.
            # Both scale with the square of the resolution,
            # but pixel-wise scales worse.
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