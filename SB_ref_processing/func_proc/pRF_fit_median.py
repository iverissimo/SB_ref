#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:13:11 2019

@author: inesverissimo

Do pRF fit on median run and save outputs

"""

'/home/shared/2018/visual/SB-prep/SB-ref/derivatives/post_fmriprep/sub-04/ses-01/func'

import os, json
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import scipy as sp
import scipy.stats as stats
import nibabel as nb


# define participant number and open json parameter file
if len(sys.argv)<2:	
    raise NameError('Please add subject number (ex:01) '	
                    'as 1st argument in the command line!')	

elif len(sys.argv)<3:	
    raise NameError('Please add if running in ex:aeneas or mac ' 	
                    'as 2nd argument in the command line!')    	 
else:	
    sub_num = str(sys.argv[1]).zfill(2) #fill subject number with 0 in case user forgets	

    with open('SBref_analysis_params.json','r') as json_file:	
            params = json.load(json_file)	
    
    if sys.argv[2]=='mac':	
        
        base_dir = params['local_basedir']
        output_dir = params['local_outputdir']
        
    elif sys.argv[2]=='aeneas':	

        base_dir = params['aeneas_basedir']
        output_dir = params['aeneas_outputdir']
        
    else:	
        raise NameError('Machine not defined, no parametes will be given') 
        

## define paths and list of files
        
if sub_num=='11':
    ses_num = '02'
else:
    ses_num = '01'
    
filepath = base_dir+'fmriprep/sub-'+sub_num+'/ses-'+ses_num+'/func/'

filename = [run for run in os.listdir(filepath) if 'prf' in run and run.endswith('T1w_desc-preproc_bold.nii.gz')]
filename.sort() #sorted filenames for preprocessed functionals

mask_files = [run for run in os.listdir(filepath) if 'prf' in run and run.endswith('T1w_desc-brain_mask.nii.gz')]
mask_files.sort() #sorted filenames for brain masks
     

# load design matrix   
prf_dm = np.load('prf_dm.npy')
prf_dm = np.moveaxis(prf_dm, 0, -1) #swap axis for popeye (x,y,time)

# load median file or create it
median_dir = output_dir+'sub-'+sub_num+'/ses-'+ses_num+'/func/' if sys.argv[2]=='mac' else base_dir+'post_fmriprep/sub-'+sub_num+'/ses-'+ses_num+'/func/median/'

median_filename = filename[0].replace('run-01','run-median')
median_maskfiles = mask_files[0].replace('run-01','run-median')

if not os.path.exists(median_dir+median_filename): #if file doesn't exist
    
    if not os.path.exists(median_dir): # check if path to save median run exist
        os.makedirs(median_dir)       # if not create it
    
    # Load data for all runs then compute median run
    data = []
    mask = []
    for i in range(len(filename)):
        dataload = np.array(nb.load(filepath+filename[i]).get_fdata())
        mask_load = np.array(nb.load(filepath+mask_files[i]).get_fdata().astype(bool))

        if i ==0:
            data = dataload[:,:,:,:,np.newaxis]
            mask = mask_load
        else: 
            data = np.concatenate((data,dataload[:,:,:,:,np.newaxis]),axis=4)
            mask = np.bitwise_or(mask,mask_load)

    data_avg = np.median(data,axis=4)
    data_input = data_avg.reshape((-1,data_avg.shape[-1])) #has to be 2D 
    mask_input = mask
    
    #load a run for header info
    run1 = nb.load(filepath+filename[0])
    mask1 = nb.load(filepath+mask_files[0])

    # save median run and median mask
    data_input_nii = nb.Nifti1Image(data_avg, affine=run1.affine, header=run1.header)
    data_input_nii.to_filename(median_dir+median_filename)

    mask_input_nii = nb.Nifti1Image(mask_input, affine=mask1.affine, header=mask1.header)
    mask_input_nii.to_filename(median_dir+median_maskfiles)

else:
    #load a run for header info
    run1 = nb.load(filepath+filename[0])
    mask1 = nb.load(filepath+mask_files[0])
    # load precomputed input
    data_input = np.array(nb.load(median_dir+median_filename).get_fdata())
    data_input = data_input.reshape((-1,data_input.shape[-1])) #has to be 2D 
    mask_input = np.array(nb.load(median_dir+median_maskfiles).get_fdata().astype(bool))
    

# define model params

fit_model = params["fit_model"]
TR = 1.5 if sub_num == '01' or sub_num == '03' and ses_num == '01' else params["TR"]

# Fit: define search grids
x_grid_bound = (-params["max_eccen"], params["max_eccen"])
y_grid_bound = (-params["max_eccen"], params["max_eccen"])
sigma_grid_bound = (params["min_size"], params["max_size"])
n_grid_bound = (params["min_n"], params["max_n"])
grid_steps = params["grid_steps"]

# Fit: define search bounds
x_fit_bound = (-params["max_eccen"]*2, params["max_eccen"]*2)
y_fit_bound = (-params["max_eccen"]*2, params["max_eccen"]*2)
sigma_fit_bound = (1e-6, 1e2)
n_fit_bound = (1e-6, 2)
beta_fit_bound = (-1e6, 1e6)
baseline_fit_bound = (-1e6, 1e6)

if fit_model == 'gauss' or fit_model == 'gauss_sg':
    bound_grids  = (x_grid_bound, y_grid_bound, sigma_grid_bound)
    bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound, beta_fit_bound, baseline_fit_bound)
elif fit_model == 'css' or fit_model == 'css_sg':
    bound_grids  = (x_grid_bound, y_grid_bound, sigma_grid_bound, n_grid_bound)
    bound_fits = (x_fit_bound, y_fit_bound, sigma_fit_bound, n_fit_bound, beta_fit_bound, baseline_fit_bound)        
        
N_PROCS = 8

from prf_fit_lyon import * #import lyon script to use relevante functions

# intitialize prf analysis
prf = PRF_fit(data = data_input[mask_input.ravel()],
            fit_model = fit_model, 
            visual_design = prf_dm, 
            screen_distance = params["screen_distance"],
            screen_width = params["screen_width"],
            scale_factor = 1/2.0, 
            tr =  TR,
            bound_grids = bound_grids,
            grid_steps = grid_steps,
            bound_fits = bound_fits,
            n_jobs = N_PROCS,
            sg_filter_window_length = params["sg_filt_window_length"],
            sg_filter_polyorder = params["sg_filt_polyorder"],
            sg_filter_deriv = params["sg_filt_deriv"], 
            )
 
prediction_pth = output_dir+'sub-'+sub_num+'/run-median/'     
prediction_file = prediction_pth+'predictions.npy'

if not os.path.exists(prediction_file): #if file doesn't exist
    
    if not os.path.exists(prediction_pth): # check if path to save predictions exists
        os.makedirs(prediction_pth)       # if not create it
         
    prf.make_predictions(out_file=prediction_file)      
    
else: # load predicitons
    prf.load_grid_predictions(prediction_file=prediction_file)
    
    
prf.grid_fit() # do grid fit

# save outputs
#rsq
rsq_output = np.zeros(mask_input.shape)
rsq_output[mask_input] = prf.gridsearch_r2
rsq_out_nii = nb.Nifti1Image(rsq_output, affine=mask1.affine, header=mask1.header)
rsq_out_nii.to_filename(prediction_pth+'rsq.nii.gz')

#parameters
params_output = np.zeros(list(mask_input.shape) + [prf.gridsearch_params.shape[0]])
params_output[mask_input] = prf.gridsearch_params.T
params_out_nii = nb.Nifti1Image(params_output, affine=mask1.affine, header=mask1.header)
params_out_nii.to_filename(prediction_pth+'params.nii.gz')  
    
    
