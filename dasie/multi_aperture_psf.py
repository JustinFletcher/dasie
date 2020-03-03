"""
Simulate multi-aperture telescope optics with piston, tip, and tilt control using HCIPy

Author: Ian Cunnyngham (Institute for Astronomy, University of Hawai'i) 2019-2020

Features:
 - Sets up multi-aperture pupil plane with tip-tilt piston control
 - Sets up multiple broadband filters consisting of a number of discrete 
   monochromatic simulations around a central wavelength
 - Generates PSFs for all filters/focal-planes for a given piston, tip, tilt, and HCIPy atmosphere
   and optionally...
   - Scale PSFs (by max possible intensity, and/or by exponentiation)
   - Convove an image with PSFs (returned instead of PSFs)
   - Bundle FFTs of PSFs (or convolved images) (real and imaginary) for machine learning
   - Add noise to output samples
 - Measure linear fit of pupil-plane piston, tip, tilt (if atmosphere provided)
 - Return strehl measures
 
 .sample() returns data structured for ML: 
     X: (res,res,samples) tensor of PSFs (or convolved images) + (optionally) FFTs
     Y: (nMir, 3) matrix of piston, tip, tilt (measured if atmosphere passed in)
 
"""

import numpy as np
import hcipy
from scipy.signal import fftconvolve

class MultiAperturePSFSampler:
    """
    Multi-aperture telescope PSF sampler class
    
    Parameters
    - - - - - -
    mirror_config : (dict) mirror configuration details
        {
            'positions': ...,      # HCIPy cartesian grid of positions (meters)
            'aperture': ...,       # HCIPy aperture fuction
                                   #   e.g.: hcipy.circular_aperture(diameter)  (meters)
            'pupil_extent':, ...,  # (float) Spatial extent of pupil plane (meters)
            'pupil_res': ...,      # (int) Resolution of pupil plane
            'piston_scale', ...,   # (float) Scale piston actuation (meters)
            'tip_tilt_scale', ...  # (float) Scale of tip-tilt actuation (meters)
                                   #   Note: meausred pupil-plane errors will be 2x larger than segment actuation
        }
    filter_configs : (list of dicts) specifying filter configuations
        [ {
            'central_lam': ...,    # (float) Central wavelgnth of filter (meters)
            'focal_extent': ...,   # (float) Angular extent of focal plane (arcsec)
            'focal_res': ...,      # (int) resolution of focal-plane PSF generated
            'frac_bandwidth': ..., # (float) fractional bandwidth of filter ( (1 +/- this/2 ) * central_wavelength )
            'num_samples': ...     # (int) Number of monocromatic PSFs across the bandwidth range
        }, ...]
    extra_processing: dict of extra steps the sampler can do
       {
           'include_fft': ...,     # (bool) Bundle FFT (real, imag) with PSF samples for machine learning
           'max_inten_norm': ...,  # (bool) Normalize PSFs by max acheivable intensity
           'pow_scale': ...,       # (float or False) Scale output by this power 
           'gauss_noise': ...,     # (float or False) sigma of gaussian noise (added after int norm, before power scaling)
       }
    
    """
    def __init__(self, 
                 mirror_config, 
                 filter_configs, 
                 extra_processing=None):
        self.mirror_config = mirror_config
        self.filter_configs = filter_configs
        
        # defaults for extra_processing (could do this with other configs...)
        ep_default = {
           'include_fft': False,
           'max_inten_norm': True,
           'pow_scale': False,
           'gauss_noise': False   
        }
        if extra_processing is None:
            extra_processing = ep_default
        else:
            # Add in any missing keys
            for key, value in ep_default.items():
                if key not in extra_processing:
                    extra_processing[key] = value
        self.extra_processing = extra_processing
        
        ### Setup HCI
        
        self.pupil_grid = hcipy.make_pupil_grid(mirror_config['pupil_res'], mirror_config['pupil_extent'])
        
        mPos = mirror_config['positions']
        self.nMir = mPos.x.shape[0]
        
        aper, segments = hcipy.make_segmented_aperture( 
                            mirror_config['aperture'],  
                            mPos, 
                            return_segments=True)
        self.aper = hcipy.evaluate_supersampled(aper, self.pupil_grid, 1)
        self.segments = hcipy.evaluate_supersampled(segments, self.pupil_grid, 1)
        self.sm = hcipy.SegmentedDeformableMirror(self.segments)

        self.lam_setups = []
        for f_config in filter_configs:
            lam = f_config['central_lam']
            fov = f_config['focal_extent']
            f_res = f_config['focal_res']
            frac_bw = f_config['frac_bandwidth']
            num_samples  = f_config['num_samples']
            
            f_grid = hcipy.make_uniform_grid([f_res]*2, fov*np.pi/(180*3600))
            prop = hcipy.FraunhoferPropagator(self.pupil_grid, f_grid)
            
            if num_samples > 1:
                filter_lams = lam * np.linspace(1 - frac_bw / 2., 1 + frac_bw / 2., num_samples)
            else:
                filter_lams = [ lam ]
            
            wfs = [ hcipy.Wavefront(self.aper, fil_lam ) for fil_lam in filter_lams ]
            
            lam_setup = {
                'f_grid': f_grid,
                'prop': prop,
                'fil_lams': filter_lams,
                'wfs': wfs,
            }
            
            # Generate PSF with no errors
            ref_psf = self._psf(lam_setup)
            lam_setup['peak_int'] = ref_psf.max()
            lam_setup['peak_ind'] = ref_psf.argmax()
            
            self.lam_setups += [ lam_setup ]
        
        # Setup pupil-plane segment coordinates for measuring PTT errors
        self.seg_coords = []
        for s in self.segments:
            inds = s.nonzero()
            xs = self.pupil_grid.x[inds]
            ys = self.pupil_grid.y[inds]
            self.seg_coords += [{
                'inds': inds,
                'xs': xs-xs.mean(),
                'ys': ys-ys.mean(),
                'offset': np.zeros(len(inds[0]))+1
            }]
    
    def _psf(self, lam_setup, atmos=None):
        prop = lam_setup['prop']
        wfs = lam_setup['wfs']
        
        focal_total = 0
        for wf in wfs:
            # Apply SM to pupil plane wf
            wf_sm = self.sm(wf)
            
            if atmos is not None:
                wf_sm = atmos(wf_sm)

            # Propagate from SM to image plane
            focal_total += prop(wf_sm).intensity
            
        return focal_total.shaped
        
    def _fft_sample(self, psf):
        """Take PSF's FFT, separate into real and imag components, stack into tensor"""
        psf_fft = np.fft.fft2(psf, norm="ortho")
        return np.stack((psf_fft.real, psf_fft.imag), axis=2)
    
    def _measure_atmos_ptt(self, atmos):
        """Given an HCIPy atmosphere, measure best-fit PTT scaled to mirror config"""
        
        # Sample 1 micron atmosphere
        ref_atmos = atmos.phase_for(1e-6)/(2*np.pi)
        
        fits = np.zeros((self.nMir, 3))
        for i_s, sc in enumerate(self.seg_coords):
            inds, off, xs, ys = sc['inds'], sc['offset'], sc['xs'], sc['ys']
            #print(inds[0].shape, off.shape, xs.shape, ys.shape, ref_atmos[inds].shape)
            x, _, _, _ = np.linalg.lstsq(np.vstack([off, xs, ys]).T, ref_atmos[inds], rcond=None)
            fits[i_s] = x
        fits[:, 0] -= fits[:, 0].mean()
        fits[:, 0] *= (1e-6)/self.mirror_config['piston_scale']/2
        fits[:, 1:] *= (1e-6)/self.mirror_config['tip_tilt_scale']/2
        return fits
    
    def sample(self, 
               ptt_actuate=None, 
               atmos=None,
               convolve_im=None,
               meas_strehl=False
              ):
        """
        Generate a PSF based on PTT and/or atmosphere, return PSFs or convolved images (X) and best PTT (Y)
        
        Parameters
        - - - - - - 
        ptt_actuate: (ndarray) (optional) (nMir x 3), piston, tip, tilt actuation to impose scaled by 
                     factors set up in mirror_config.  Note: Pupil plane errors are 2x these!
                     If none given, set to zero errors (could be set to differential instead...)
        atmos:       (HCIPy atmosphere) (optional) Applies atmosphere, output TTP measured
        convolve_im: (ndarray) (optional) the image you want to convolve with the PSF or False return PSF
                     Note: Assumes the pixel scale of the filter PSFs is set for image's angular extent 
                     Convolves for each filter, extra_processing steps will be appleid to conv images (including FFTs)
        meas_strehl: (bool) If true, returns the strehls for each feature
                     
        Output
        - - - - - - 
        X: (res,res,samples) tensor of PSFs (or convolved images) + (optionally) FFTs
        Y: (nMir, 3) matrix of piston, tip, tilt (measured if atmosphere passed in)
        
        """
        
        # Build up stack of samples to stack into tensor for prediction
        Xs = []
        
        if ptt_actuate is None:
            ptt_actuate = np.zeros((self.nMir, 3))
        
        pist = ptt_actuate[:, 0] * self.mirror_config['piston_scale']
        tts = ptt_actuate[:, 1:] * self.mirror_config['tip_tilt_scale']
        self.sm.set_segment_actuators(np.arange(self.nMir), pist, tts[:, 0], tts[:, 1])

        strehls = []
        for lam_setup in self.lam_setups:
            psf = self._psf(lam_setup, atmos=atmos)
            
            if meas_strehl:
                strehls += [ psf.flat[ lam_setup['peak_ind'] ] / lam_setup['peak_int'] ]
            if self.extra_processing['max_inten_norm']:
                psf /= lam_setup['peak_int']
                
            if convolve_im is not None:
                out_samp = fftconvolve(convolve_im, psf, mode='same')
            else:
                out_samp = psf
            
            if self.extra_processing['gauss_noise'] is not False:
                out_samp += np.abs(np.random.normal(0, self.extra_processing['gauss_noise'], psf.shape))
            
            if self.extra_processing['pow_scale'] is not False:
                out_samp = np.power(out_samp[..., None], self.extra_processing['pow_scale']) 
      
            Xs += [ out_samp[..., None] ]
            if self.extra_processing['include_fft']:
                Xs += [ self._fft_sample(out_samp) ]
        
        # If atmosphere is provided, return the best fit piston, tip, and tilts to correct
        if atmos is not None:
            out_ptt = ptt_actuate+self._measure_atmos_ptt(atmos)
        else:
            out_ptt = ptt_actuate
        
        # Combine list of X samples into tensor
        Xs = np.concatenate(Xs, axis=2)
        if meas_strehl:
            return Xs, out_ptt, strehls
        else:
            return Xs, out_ptt
