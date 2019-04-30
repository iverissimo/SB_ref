import ctypes
import numpy as np
import scipy as sp
import platform
from math import *
import os
import glob
import json

# MRI analysis imports
import nibabel as nb
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus
import popeye.css as css
import popeye.og as og
import cifti
from joblib import Parallel, delayed
from scipy import signal

from scipy.signal import fftconvolve, savgol_filter
from scipy.stats import linregress
from popeye.base import PopulationModel
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask

from tqdm import tqdm


class CompressiveSpatialSummationModel(PopulationModel):
    
    r"""
    A Compressive Spatial Summation population receptive field model class    
    """
    
    def __init__(self, stimulus, hrf_model, cached_model_path=None, nuisance=None):
                
        PopulationModel.__init__(self, stimulus, hrf_model, nuisance)

    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, n, beta, baseline):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # units
        model /= np.max(model)
        
        # offset
        model += baseline
        
        # scale it by beta
        model *= beta

        return model

class CompressiveSpatialSummationModelFiltered(PopulationModel):
    
    r"""
    A Compressive Spatial Summation population receptive field model class
    Adapted to include a savgol_filter
    
    """
    
    def __init__(self, stimulus, hrf_model, cached_model_path=None, nuisance=None, sg_filter_window_length=120, sg_filter_polyorder=3, sg_filter_deriv = 0, tr=1.5):
                
        PopulationModel.__init__(self, stimulus, hrf_model, nuisance)

        # sg filter
        self.sg_filter_window = np.int(sg_filter_window_length / tr)
        if self.sg_filter_window % 2 == 0:
            self.sg_filter_window += 1
        self.sg_filter_polyorder = sg_filter_polyorder
        self.sg_filter_deriv = sg_filter_deriv

    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, n, beta, baseline):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # units
        model /= np.max(model)
        
        # at this point, add filtering with a savitzky-golay filter
        model_drift = savgol_filter(model, 
                                    window_length = self.sg_filter_window, 
                                    polyorder = self.sg_filter_polyorder,
                                    deriv = self.sg_filter_deriv, 
                                    mode = 'nearest')
        
        # demain model_drift, so baseline parameter is still interpretable
        model_drift_demeaned = model_drift-np.mean(model_drift)
        
        # and apply to data
        model -= model_drift_demeaned    
        
        # offset
        model += baseline
        
        # scale it by beta
        model *= beta

        return model
    
class GaussianModel(PopulationModel):
    
    r"""
    A Gaussian Spatial Summation population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model, cached_model_path=None, nuisance=None):
                
        PopulationModel.__init__(self, stimulus, hrf_model, nuisance)

    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta, baseline):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
                
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # units
        model /= np.max(model)
        
        # offset
        model += baseline
        
        # scale it by beta
        model *= beta

        return model
    
class GaussianModelFiltered(PopulationModel):
    
    r"""
    A Gaussian Spatial Summation population receptive field model class
    Adapted to include a savgol_filter
    
    """
    
    def __init__(self, stimulus, hrf_model, cached_model_path=None, nuisance=None, sg_filter_window_length=120, sg_filter_polyorder=3, sg_filter_deriv = 0, tr=1.5):
                
        PopulationModel.__init__(self, stimulus, hrf_model, nuisance)

        # sg filter
        self.sg_filter_window = np.int(sg_filter_window_length / tr)
        if self.sg_filter_window % 2 == 0:
            self.sg_filter_window += 1
        self.sg_filter_polyorder = sg_filter_polyorder
        self.sg_filter_deriv = sg_filter_deriv

    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta, baseline):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
                
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # units
        model /= np.max(model)
        
        # at this point, add filtering with a savitzky-golay filter
        model_drift = savgol_filter(model, 
                                    window_length = self.sg_filter_window, 
                                    polyorder = self.sg_filter_polyorder,
                                    deriv = self.sg_filter_deriv, 
                                    mode = 'nearest')
        
        # demain model_drift, so baseline parameter is still interpretable
        model_drift_demeaned = model_drift-np.mean(model_drift)
        
        # and apply to data
        model -= model_drift_demeaned    
        
        # offset
        model += baseline
        
        # scale it by beta
        model *= beta

        return model


def fit_gradient_descent(model, data, ballpark, bounds, verbose=0):
    return utils.gradient_descent_search(data,
                                         utils.error_function,
                                         model.generate_prediction,
                                         ballpark,
                                         bounds,
                                         verbose)[0]

class PRF_fit(object):
    
    def __init__(self, data, fit_model, visual_design, screen_distance, screen_width, 
                 scale_factor, tr, bound_grids, grid_steps, bound_fits, n_jobs,
                 sg_filter_window_length=210, sg_filter_polyorder=3, sg_filter_deriv=0):

        # immediately convert nans to nums
        self.data = np.nan_to_num(data)
        self.data_var = self.data.var(axis=-1)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[1]

        self.stimulus = VisualStimulus( stim_arr = visual_design,
                                        viewing_distance = screen_distance, 
                                        screen_width = screen_width,
                                        scale_factor = scale_factor,
                                        tr_length = tr,
                                        dtype = np.short)

        assert self.n_timepoints == self.stimulus.run_length, \
            "Data and design matrix do not have the same nr of timepoints, %i vs %i!"%(self.n_timepoints, self.stimulus.run_length)

        if fit_model == 'gauss':
            self.model_func = GaussianModel(stimulus = self.stimulus, 
                                                    hrf_model = utils.spm_hrf)    
        elif fit_model == 'gauss_sg':
            self.model_func = GaussianModelFiltered(stimulus = self.stimulus,
                                            hrf_model = utils.spm_hrf,
                                            sg_filter_window_length = sg_filter_window_length, 
                                            sg_filter_polyorder = sg_filter_polyorder,
                                            sg_filter_deriv = sg_filter_deriv,
                                            tr = tr)

        elif fit_model == 'css':
            self.model_func = CompressiveSpatialSummationModel( stimulus = self.stimulus,
                                                                    hrf_model = utils.spm_hrf)
        elif fit_model == 'css_sg':
            self.model_func = CompressiveSpatialSummationModelFiltered( stimulus = self.stimulus,
                                                                    hrf_model = utils.spm_hrf,
                                                                    sg_filter_window_length = sg_filter_window_length, 
                                                                    sg_filter_polyorder = sg_filter_polyorder,
                                                                    sg_filter_deriv = sg_filter_deriv,
                                                                    tr = tr)
        self.model_func.hrf_delay = 0
        self.predictions = None      
        self.fit_model =  fit_model
        self.bound_grids = bound_grids
        self.grid_steps = grid_steps
        self.bound_fits = bound_fits
        self.n_jobs = n_jobs
        
    def make_grid(self):
        prf_xs = np.linspace(self.bound_grids[0][0],self.bound_grids[0][1],self.grid_steps)
        prf_ys = np.linspace(self.bound_grids[1][0],self.bound_grids[1][1],self.grid_steps)
        prf_sigma = np.linspace(self.bound_grids[2][0],self.bound_grids[2][1],self.grid_steps)
        
        if self.fit_model == 'gauss' or self.fit_model == 'gauss_sg':
            self.prf_xs, self.prf_ys, self.prf_sigma = np.meshgrid(prf_xs, prf_ys, prf_sigma)
        elif self.fit_model == 'css' or self.fit_model == 'css_sg':
            prf_n = np.linspace(self.bound_grids[3][0],self.bound_grids[3][1],self.grid_steps)
            self.prf_xs, self.prf_ys, self.prf_sigma, self.prf_n = np.meshgrid(prf_xs, prf_ys, prf_sigma, prf_n)
    
    def make_predictions(self, out_file=None):
        if not hasattr(self, 'prf_xs'):
            self.make_grid()
        self.predictions = np.zeros(list(self.prf_xs.shape) + [self.stimulus.run_length])
        self.predictions = self.predictions.reshape(-1, self.predictions.shape[-1]).T
        
        if self.fit_model == 'gauss' or self.fit_model == 'gauss_sg':
            for i, (x, y, s) in tqdm(enumerate(zip(self.prf_xs.ravel(), self.prf_ys.ravel(), self.prf_sigma.ravel()))):
                self.predictions[:, i] = self.model_func.generate_prediction(x, y, s, 1, 0)
        elif self.fit_model == 'css' or self.fit_model == 'css_sg':
            for i, (x, y, s, n) in tqdm(enumerate(zip(self.prf_xs.ravel(), self.prf_ys.ravel(), self.prf_sigma.ravel(), self.prf_n.ravel()))):
                self.predictions[:, i] = self.model_func.generate_prediction(x, y, s, n, 1, 0)
        self.predictions = np.nan_to_num(self.predictions)
        if out_file != None:
            np.save(out_file, self.predictions)

    def load_grid_predictions(self, prediction_file):
        self.make_grid()
        self.predictions = np.load(prediction_file)

    def grid_fit(self):
        
        if self.fit_model == 'gauss' or self.fit_model == 'gauss_sg':
            prediction_params = np.ones((self.n_units, 6))*np.nan
        elif self.fit_model == 'css' or self.fit_model == 'css_sg':
            prediction_params = np.ones((self.n_units, 7))*np.nan

        # set up book-keeping to minimize memory usage.
        self.gridsearch_r2 = np.zeros(self.n_units)
        self.best_fitting_prediction_thus_far = np.zeros(self.n_units, dtype=int)
        self.best_fitting_beta_thus_far = np.zeros(self.n_units, dtype=float)
        self.best_fitting_baseline_thus_far = np.zeros(self.n_units, dtype=float)

        for prediction_num in tqdm(range(self.predictions.shape[1])):
            # scipy implementation?
            # slope, intercept, rs, p_values, std_errs = linregress(self.predictions[:,prediction_num], self.data)
            # rsqs = rs**2
            # numpy implementation is slower?
            dm = np.vstack([np.ones_like(self.predictions[:,prediction_num]),self.predictions[:,prediction_num]]).T
            (intercept, slope), residual, _, _ = sp.linalg.lstsq(dm, self.data.T, check_finite=False) #  , lapack_driver='gelsy')
            rsqs = ((1 - residual / (self.n_timepoints * self.data_var)))

            improved_fits = rsqs > self.gridsearch_r2
            # fill in the improvements
            self.best_fitting_prediction_thus_far[improved_fits] = prediction_num
            self.gridsearch_r2[improved_fits] = rsqs[improved_fits]
            self.best_fitting_baseline_thus_far[improved_fits] = intercept[improved_fits]
            self.best_fitting_beta_thus_far[improved_fits] = slope[improved_fits]

        if self.fit_model == 'gauss' or self.fit_model == 'gauss_sg':
            self.gridsearch_params = np.array([ self.prf_xs.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_ys.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_sigma.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.best_fitting_beta_thus_far,
                                                    self.best_fitting_baseline_thus_far
                                                ])
        elif self.fit_model == 'css' or self.fit_model == 'css_sg':
            self.gridsearch_params = np.array([ self.prf_xs.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_ys.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_sigma.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.prf_n.ravel()[self.best_fitting_prediction_thus_far],
                                                    self.best_fitting_beta_thus_far,
                                                    self.best_fitting_baseline_thus_far
                                                ])
                
    def iterative_fit(self):
        if self.gridsearch_params is None:
            raise Exception('First use self.fit_grid!')
        
        prf_params = Parallel(self.n_jobs,verbose = 10)(delayed(fit_gradient_descent)(self.model_func, data, ballpark, self.bound_fits)
                                       for (data,ballpark) in zip(self.data, self.gridsearch_params))
        prf_params = np.vstack(prf_params)
        if self.fit_model == 'gauss' or self.fit_model == 'gauss_sg':
            output = np.ones((self.n_units,6))*nan
        elif self.fit_model == 'css' or self.fit_model == 'css_sg':
            output = np.ones((self.n_units,7))*nan
            
        for vox in range(0,self.n_units):
            data_tc = self.data[:,vox]
            if self.fit_model == 'gauss' or self.fit_model == 'gauss_sg':
                model_tc = self.model_func.generate_prediction(prf_params[vox,0],prf_params[vox,1],prf_params[vox,2],prf_params[vox,3],prf_params[vox,4])
            elif self.fit_model == 'css' or self.fit_model == 'css_sg':
                model_tc = self.model_func.generate_prediction(prf_params[vox,0],prf_params[vox,1],prf_params[vox,2],prf_params[vox,3],prf_params[vox,4],prf_params[vox,5])
                
            output[vox,:] = np.hstack([prf_params[vox,:], utils.coeff_of_determination(data_tc,model_tc)/100.0])
        
        self.fit_output = output
        return output
