"""

Distributed Aperture System for Interferometric Exploitation control system
simulation environment.

Author: Justin Fletcher, Ian Cunnyngham
"""

import gym
import glob
import time
import numpy as np

from gym import spaces, logger
from gym.utils import seeding

from matplotlib import pyplot as plt

# Load HCIPy and Ian's sample generator built on top of it
import hcipy 
from multi_aperture_psf import MultiAperturePSFSampler
import joblib

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
        
        ### Load extended object if specified
        # Notes: 
        # - This is of course highly static, assume will be upgraded later
        # - Plate scale (angular pixel size) fixed in simulator
        #   by focal-plane extent and resolution, scale the images?
        # - In current code, if render is enabled, pupil-plane res and image
        #   resolution must match.  This is of course extremely arbitrary
        #   as they absolutely don't need to be related. 
        self.extended_object_image_file = kwargs['extended_object_image_file']
        if self.extended_object_image_file == "none":
            self.ext_im = None
        else:
            self.ext_im = plt.imread( self.extended_object_image_file )
        
        ### Genereate ELF-like telescope geometry
        ###  annulus of sub-apertures with centers at telescope_radius
        self.num_apertures = kwargs['num_apertures']
        self.telescope_radius = kwargs['telescope_radius'] # meters

        # Linear space of angular coordinates for mirror centers
        thetas = np.linspace(0, 2*np.pi, self.num_apertures+1)[:-1]
        
        # Use HCIPy coordinate generation to quickly generate mirror centers
        aper_coords = hcipy.SeparatedCoords((np.array([self.telescope_radius]), thetas))
        # Create an HCIPy "CartesianGrid" by creating PolarGrid and converting
        m_cens = self.aperture_centers = hcipy.PolarGrid(aper_coords).as_('cartesian')
        
        # Calculate sub-aperture diamater from the distance between centers 
        # (Assuming dense packing of apertures for now, could simulate gaps later)
        self.aperture_diamater = np.sqrt((m_cens.x[1]-m_cens.x[0])**2 + (m_cens.y[1]-m_cens.y[0])**2)
        
        # Calculate extent of pupil-plane simulation (meters)
        self.pupil_plane_diamater = max(m_cens.x.max() - m_cens.x.min(), m_cens.y.max() - m_cens.y.min()) + self.aperture_diamater
        # Add a little extra for edges, not convinced not cutting them off
        self.pupil_plane_diamater *= 1.05  
        
        print(f'Sub-aperture diamater {self.aperture_diamater :.02f}m')
        print(f'Pupil-plane extent {self.pupil_plane_diamater :.02f}m')
        
        
        ### Build up rest of sampler configuration dictionary from kwargs
        self.sampler_setup = {
            'mirror_config': {
                'positions': self.aperture_centers,
                'aperture_config': ['circular', self.aperture_diamater],
                'pupil_extent': self.pupil_plane_diamater ,
                'pupil_res': kwargs['pupil_plane_resolution'],
                'piston_scale': kwargs['piston_actuate_scale'],   # meters
                'tip_tilt_scale': kwargs['tip_tilt_actuate_scale']  # meters
            },
            # Single filter for now, sampler designed for with multiple in mind
            'filter_configs': [ 
            {
                'central_lam': kwargs['filter_central_wavelength'],    # meters
                'focal_extent': kwargs['filter_psf_extent'],      # arcsec
                'focal_res': kwargs['filter_psf_resolution'],
                'frac_bandwidth': kwargs['filter_fractional_bandwidth'],
                'num_samples': kwargs['filter_bandwidth_samples']
            } ] 
        }
        # Initialize sampler with above setup
        self.mas_psf_sampler = MultiAperturePSFSampler(**self.sampler_setup)

        ### If enabled, setup atmosphere
        self.atmosphere_type = kwargs['atmosphere_type']
        if self.atmosphere_type != "single" and self.atmosphere_type != "multi":
            self.atmos = None
        else:
            self.atmosphere_fried_paramater = kwargs['atmosphere_fried_paramater']
            self.atmosphere_outer_scale = kwargs['atmosphere_outer_scale']
            self.atmosphere_velocity = kwargs['atmosphere_velocity']
            
            # Calculate C_n^2 from given Fried param, r0 @ 550nm 
            self.cn2 = hcipy.Cn_squared_from_fried_parameter(self.atmosphere_fried_paramater, 550e-9)
            
            if self.atmosphere_type == "single":
                # Single layer atmosphere
                self.atmos = hcipy.InfiniteAtmosphericLayer(
                    self.mas_psf_sampler.pupil_grid, 
                    self.cn2, 
                    self.atmosphere_outer_scale, 
                    self.atmosphere_velocity, 
                    100  # Height of single layer in meters, but may not be important for now
                )
            if self.atmosphere_type == "multi":
                layers = hcipy.make_standard_atmospheric_layers(
                    self.mas_psf_sampler.pupil_grid, 
                    self.atmosphere_outer_scale
                )
                self.atmos = hcipy.MultiLayerAtmosphere(
                    layers, 
                    scintilation=kwargs['enable_atmosphere_scintilation']
                )
                self.atmos.Cn_squared = self.cn2
                # Need to set individual layer velocities, for now stays at 10 m/s
                #self.atmos.velocity = self.atmosphere_velocity
        
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
        
        # Iterate over each aperture.
        for aperture_index in range(self.num_apertures):
            # Now, create a new dict to track this aperture through simulation.
            aperture = dict()
            
            # Initialize the tip, tilt, and piston for this aperture.
            tip = 0.0 + self.tip_phase_error_scale * np.random.randn(1)
            tilt = 0.0 + self.tilt_phase_error_scale * np.random.randn(1)
            piston = 1.0 +self.piston_phase_error_scale * np.random.randn(1)
            aperture['tip_phase'] = tip[0]
            aperture['tilt_phase'] = tilt[0]
            aperture['piston_phase'] = piston[0]
            
            self.apertures.append(aperture)
            
        # Reset simulation time
        self.simulation_time = 0
        
        # Reset atmosphere (if any)
        if self.atmos is not None:
            self.atmos.reset()

        # Update observables 
        # ( Since state is set to None, is this important? )
        self._update_sampler()
        
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
            aperture['piston_phase'] = aperture['piston_phase_command']

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
        
        # Move time forward by one time-step
        self.simulation_time += self.step_time_granularity
        
        # Evolve atmosphere (if any)
        if self.atmos is not None:
            self.atmos.evolve_until( self.simulation_time )

        # Update observaables
        self._update_sampler()
        
        done = False

        # TODO: Build a reward function
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

        self.state = self.focal_plane_obs

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):

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

        # (Ian) The output of sampler is (res, res, num_wavelengths), so need to select just one
        log_real_focal_plane = np.log(np.abs(self.focal_plane_obs[..., 0]))

        log_real_focal_plane = to_rgb(positive_shift_norm_scale_img(log_real_focal_plane))
        
        system_phase_matrix_img = to_rgb(positive_shift_norm_scale_img(self.pupil_plane_phase_screen))
        render_image = np.hstack([system_phase_matrix_img,
                                  log_real_focal_plane])

        # Uncomment to zoom
        # psf_shape = log_real_optical_psf.shape
        # x_length = psf_shape[0]
        # y_length = psf_shape[1]
        # x_center = int(x_length / 2)
        # y_center = int(y_length / 2)
        #
        # patch_fraction = 0.1
        # patch_x_half_width = int(patch_fraction * x_center)
        # patch_y_half_width = int(patch_fraction * x_center)
        #
        # patch = log_real_optical_psf[x_center-patch_x_half_width:x_center+patch_x_half_width,
        #                              y_center-patch_y_half_width:y_center+patch_y_half_width]
        #
        # from skimage.transform import resize
        # patch_resized = resize(patch, (x_length, y_length))

        # log_real_optical_psf = patch_resize

        # render_image = np.hstack([system_phase_matrix_img,
        #                           patch_resized])
        # file_id = str(len(glob.glob('.\\temp\\*.png')))
        # plt.imsave('.\\temp\\' + file_id + '.png', render_image)

        # Save render image

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
        X, Y, strehls  = self.mas_psf_sampler.sample(
            sampler_ptt_phases,  # (n_aper, 3) piston, tip, tilts to set telescope to
            atmos=self.atmos,    # Pass in HCIPy atmosphere, applied to each pupil-plane (or None is fine)
            convolve_im=self.ext_im, # Image to convolve PSF with 
                                 # (Note: assuemd matches sampler filters angular extent/pixel scale)
            meas_strehl=True     # If True, returns third output which is the measured strehl for each filter 
        )
        
        self.focal_plane_obs = X
        
        # Maybe only fetch the phase screen if render is set to True?
        self.pupil_plane_phase_screen = self.mas_psf_sampler.getPhaseScreen(
            atmos=self.atmos,   # atmos not stored in sampler, but last PTT is
            np_array=True       # Get a numpy matrix instead of an HCIPy "Field"
        )
