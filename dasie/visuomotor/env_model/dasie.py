"""

Distributed Aperture System for Interferometric Exploitation control system
simulation environment.

Author: Justin Fletcher, Ian Cunnyngham
"""

import os

import gym
import glob
import time
import datetime
import numpy as np

from collections import deque

from PIL import Image, ImageSequence

from gym import spaces, logger
from gym.utils import seeding

from matplotlib import pyplot as plt

# Simulation of multi aperture telescope built on top of MutiPSFSampler built on top of HCIPy
from .simulate_multi_aperture import SimulateMultiApertureTelescope


def cosine_similarity(u, v):
    """

    :param u: Any np.array matching u in shape (and semantics probably)
    :param v: Any np.array matching v in shape (and semantics probably)
    :return: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    """
    u = u.flatten()
    v = v.flatten()

    return (np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v)))


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
        Type: Box(observation_window_size, height, width)
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
        
        ### Inialize multi-aperture telescope simulator
        # Pass all parsed kwargs in, will only use the ones it needs and grab defaults for anything missing
        self.telescope_sim = SimulateMultiApertureTelescope(**kwargs)
        
        # Copy this locally
        self.num_apertures = self.telescope_sim.num_apertures

        # If an extended object image is provided, load it only once.
        # TODO: generalize to accomodate changing images.
        self.extended_object_image_file = kwargs['extended_object_image_file']
        if self.extended_object_image_file:
            perfect_image = plt.imread(self.extended_object_image_file)
            self.perfect_image = perfect_image / np.max(perfect_image)
        
        
        # Keep track of simulation time for atmosphere evolution at least (probably more later)
        self.simulation_time = 0
        
        # Parse the rest of the keyword arguments
        self.step_time_granularity = kwargs['step_time_granularity']
        self.tip_phase_error_scale = kwargs['tip_phase_error_scale']
        self.tilt_phase_error_scale = kwargs['tilt_phase_error_scale']
        self.piston_phase_error_scale = kwargs['piston_phase_error_scale']

        # TODO: Compute the number of tau needed for one model inference.

        # Define the action space for DASIE, by setting the lower limits...
        # (Ian) Switched this to be symetric about zero
        tip_lower_limit = -1.0
        tilt_lower_limits = -1.0
        piston_lower_limits = -1.0
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

        # # We then create a list to hold the composite action spaces, and...
        # aperture_action_spaces = list()
        #
        # # ...iterate over each aperture...
        # for aperture_num in range(self.num_apertures):
        #
        #     # ...instantiating a box space using the established limits...
        #     aperture_space = spaces.Box(lower_limits,
        #                                 upper_limits, dtype=np.float32)
        #     # ...and adding that space to the list of spaces.
        #     aperture_action_spaces.append(aperture_space)
        #
        # # Finally, we complete the space by building it from the template list.
        # self.action_space = spaces.Tuple(aperture_action_spaces)

        # Define a symmetric action space.
        self.action_shape = (self.num_apertures, 3)
        action_bound_low = -1.0 * np.ones(self.action_shape)
        action_bound_high = np.ones(self.action_shape)
        self.action_space = spaces.Box(low=action_bound_low,
                                       high=action_bound_high,
                                       dtype=np.float32)

        # TODO: Refactor action space to allow Box-based asymmetric spaces.

        # TODO: Build a correct observation space.
        # Get image shape.
        self.image_shape = kwargs['filter_psf_resolution']

        # Get window depth.
        self.observation_window_size = kwargs['observation_window_size']

        # Define a 3D observation space
        self.observation_shape = (self.image_shape,
                                  self.image_shape,
                                  self.observation_window_size)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        self.render_frequency = kwargs['render_frequency']
        

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        # Set the initial state. This is the first thing called in an episode.

        print("Resetting Environment")
        
        self.apertures = list()
        
        # Iterate over each aperture.
        for aperture_index in range(self.num_apertures):
            # Now, create a new dict to track this aperture through simulation.
            aperture = dict()
            
            # Initialize the tip, tilt, and piston for this aperture.
            tip = 0.0 + self.tip_phase_error_scale * np.random.randn(1)
            tilt = 0.0 + self.tilt_phase_error_scale * np.random.randn(1)
            piston = 1.0 + self.piston_phase_error_scale * np.random.randn(1)
            aperture['tip_phase'] = tip[0]
            aperture['tilt_phase'] = tilt[0]
            aperture['piston_phase'] = piston[0]
            
            self.apertures.append(aperture)
            
        # Reset simulation time
        self.simulation_time = 0
        
        # Reset telescope simulator (atmosphere if any)
        self.telescope_sim.reset()

        # Update observables 
        # ( Since state is set to None, is this important? )
        self._update_sampler()
        
        # Initialize state.

        # Construct the observation queue (FIFO) using blank frames.
        state_shape = np.stack(self.observation_space.sample()).shape
        blank_frame = np.zeros(state_shape[:-1] + (1,))
        self.observation_stack = deque()

        for _ in range(self.observation_window_size):

            self.observation_stack.append(blank_frame)

        # Let the system run uncontrolled to populate the frame buffer.
        initial_state = self.observation_stack

        print("Resetting Environment: Populating Initial State")

        for _ in range(self.observation_window_size):

            no_action = np.zeros_like(self.action_space.sample())

            (initial_state, _, _, _) = self.step(action=no_action)

        self.state = initial_state
        self.steps_beyond_done = None


        print("Resetting Environment: Complete")
        return np.array(self.state)

    def step(self, action):

        # First, ensure the step action is valid.
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # The action is valid. Use it to update the ttp in each aperture.
        for (aperture, aperture_command) in zip(self.apertures, action):

            aperture['tip_phase_command'] = aperture_command[0]
            aperture['tilt_phase_command'] = aperture_command[1]
            aperture['piston_phase_command'] = aperture_command[2]

            # print("Start Aperture Commands")
            # print(aperture['tip_phase_command'])
            # print(aperture['tilt_phase_command'])
            # print(aperture['piston_phase_command'])
            # print("Stop Aperture Commands")

            # TODO: Move this and do incremental updates.
            aperture['tip_phase'] = aperture['tip_phase_command']
            aperture['tilt_phase'] = aperture['tilt_phase_command']
            aperture['piston_phase'] = aperture['piston_phase_command']

        # # TODO: Write actuation simulation loop here.
        # # num_timesteps is probably the inference latency of the system.
        # for t in range(num_timesteps):
        #     # TODO: Iterate the target model state.
        #     # TODO: Iterate the at-aperture irradiance.
        #     # TODO: Iterate toward commanded articulations.

        # Move time forward by one time-step
        self.simulation_time += self.step_time_granularity

        # Evolve telescope simulation (atmosphere)
        self.telescope_sim.evolve_to(self.simulation_time)

        # Update observables.
        self._update_sampler()

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

        # TODO: compute_reward()
        # TODO: compute_done()

        # Build the state by popping the oldest frame and appending the newest.
        self.observation_stack.popleft()
        self.observation_stack.append(self.focal_plane_obs)

        # Set the state to the updated queue.
        self.state = self.observation_stack

        self.reward_function = "strehl"
        self.reward_function = "truth_cosine_similarity"

        if self.reward_function == "strehl":

            reward = self.strehl_scalar

        if self.reward_function == "truth_cosine_similarity":

            truth_image = self.telescope_sim.get_extended_object_image()

            if truth_image is None:

                print("No extended_object_image_file was provided.")

            recovered_image = np.mean(np.stack(self.observation_stack), axis=(0, -1))

            reward = cosine_similarity(truth_image, recovered_image)

        print(reward)

        return np.array(self.state), reward, done, {}

    def render(self,
               render_mode,
               logdir=None,
               run_name=None,
               episode=None,
               step=None):

        if not logdir:
            logdir = "."

        if not run_name:
            # run_name = str(datetime.timestamp(datetime.now()))
            run_name = "temp"


        if self.state is None: return None

        def positive_shift_norm_scale_img(img, scale=2**8):

            shifted_img = img + np.abs(np.min(img))

            normalized_img = shifted_img / np.max(shifted_img)

            scaled_img = scale * normalized_img

            return(scaled_img)

        def to_rgb(im, dtype=np.uint8):
            """
            This function maps a matrix to a greyscale RGB image quickly.
            """
            w, h = im.shape
            ret = np.empty((w, h, 3), dtype=dtype)
            ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
            return ret

        perfect_image = to_rgb(positive_shift_norm_scale_img(self.perfect_image))
        system_phase_matrix_img = to_rgb(positive_shift_norm_scale_img(self.pupil_plane_phase_screen))
        # (Ian) The output of sampler is (res, res, num_wavelengths), so need to select just one
        log_real_focal_plane = np.abs(self.focal_plane_obs[..., 0])
        log_real_focal_plane = to_rgb(positive_shift_norm_scale_img(log_real_focal_plane))
        restored_image = np.mean(np.stack(self.observation_stack), axis=(0, -1))
        restored_image = to_rgb(positive_shift_norm_scale_img(restored_image))

        # Compose the component images into a single view.
        if self.extended_object_image_file:

            render_image = np.hstack([perfect_image,
                                      system_phase_matrix_img,
                                      log_real_focal_plane,
                                      restored_image])

        else:

            render_image = np.hstack([system_phase_matrix_img,
                                      log_real_focal_plane])

        # TODO: Make render gifs work here.

        if render_mode == 'gif':

            if step % self.render_frequency == 0:

                render_output_path = os.path.join(logdir,
                                                  run_name,
                                                  str(episode))
                if not os.path.exists(render_output_path):
                    os.makedirs(render_output_path)

                im = Image.fromarray(render_image)
                im.save(os.path.join(render_output_path, str(step) + ".jpeg"))

            # self.render_images.append(render_image)
            #
            # print(self.steps_until_render)

            # if self.steps_until_render == 0:
                #
                # print("Rendering")
                #
                # # First, load prior frame from disk.
                # prior_images = list()
                #
                # # if os.path.isfile("render.gif"):
                # #
                # #     for im in ImageSequence.Iterator(Image.open("render.gif")):
                # #
                # #         prior_images.append(im)
                # #     # prior_images = list(ImageSequence.Iterator(Image.open("render.gif")))
                # #
                # #     print(len(prior_images))
                #
                # render_images = prior_images + self.render_images
                #
                # print(len(render_images))
                # render_images[0].save("render.gif",
                #                       save_all=True,
                #                       append_images=render_images[1:],
                #                       duration=50,
                #                       loop=0)
                #
                # self.steps_until_render = self.render_frequency
                # self.render_images = list()


            # render_stack.append(render_image)
            # imgs = [Image.fromarray(img) for img in render_stack]
            # # duration is the number of milliseconds between frames; this is 40 frames per second
            # imgs[0].save("render.gif",
            #              save_all=True,
            #              append_images=imgs[1:],
            #              duration=50,
            #              loop=0)
            #
            # render_stack.append(render_image)

            # Count down the steps until render.
            return render_image

        elif render_mode == 'human':

            from gym.envs.classic_control import rendering

            if self.viewer is None:

                self.viewer = rendering.SimpleImageViewer()

            self.viewer.imshow(render_image)

            return self.viewer.isopen

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _update_sampler(self):

        # Retrieve piston, tip, and tilt phases
        sampler_ptt_phases = []
        for aperture in self.apertures:
            piston = aperture['piston_phase']
            tip = aperture['tip_phase']
            tilt = aperture['tilt_phase']
            
            sampler_ptt_phases += [[piston, tip, tilt]]
        
        # Sampler takes a (nMir, 3) numpy array with piston, tip, tilt for each sub-aperture
        sampler_ptt_phases = np.array(sampler_ptt_phases)
        
        # X: Stack of focal plane observations
        #    PSF by default, extended image convolved with PSF if provided
        # Y: Returns the optimal P/T/T (n_aper, 3) phases to get optimal strehl (measured vs atmosphere)
        # strehls: If meas_strehls set, returns strehl vs perfectly phase mirror
        X, Y, strehls  = self.telescope_sim.get_observation(
            piston_tip_tilt = sampler_ptt_phases,  # (n_aper, 3) piston, tip, tilts to set telescope to
        )

        self.strehl_scalar = strehls[0]
        self.focal_plane_obs = X

        # self.truth_image = self.telescope_sim.get_extended_object_image()
        
        # Maybe only fetch the phase screen if render is set to True?
        # Get a numpy matrix instead of an HCIPy "Field"
        self.pupil_plane_phase_screen = self.telescope_sim.pupil_plane_phase_screen(np_array=True)
