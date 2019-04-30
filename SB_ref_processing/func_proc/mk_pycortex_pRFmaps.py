#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:51:14 2019

@author: inesverissimo
"""

import os, json
import sys
import nibabel as nb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import seaborn as sns

import hrf_estimation
import scipy.signal as signal
import scipy.stats as stats

from joblib import Parallel, delayed
from nipype.interfaces.freesurfer import BBRegister
import cortex


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
        
        base_dir = '/Users/inesverissimo/Documents/SB_project/mp2rage/func_nosubmm/derivatives/'
        output_dir = '/Users/inesverissimo/Documents/SB_project/pRF_fit/outputs/sub-'+sub_num+'/run-median/'
        
    elif sys.argv[2]=='aeneas':	

        base_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/'
        output_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/pRF_fit/sub-'+sub_num+'/run-median/'
        
    else:	
        raise NameError('Machine not defined, no parametes will be given') 
        

## define paths and list of files
        
if sub_num=='11':
    ses_num = '02'
else:
    ses_num = '01'
      
# paths to derivatives data (get freesurfer volumes), output and mean functionals
fmriprep_func_dir = base_dir+'fmriprep/sub-'+sub_num+'/ses-'+ses_num+'/func/'
fmriprep_anat_dir = base_dir+'fmriprep/sub-'+sub_num+'/anat/'
fs_dir = base_dir+'freesurfer/'
median_dir = base_dir+'post_fmriprep/sub-'+sub_num+'/ses-'+ses_num+'/func/median/'

    
# compute mean boldref epi and save it ###
epi_files = [run for run in os.listdir(fmriprep_func_dir) if 'prf' in run and run.endswith('_space-T1w_boldref.nii.gz')]
epi_files.sort()

median_epi_file = epi_files[0].replace('run-01','run-median')

if not os.path.exists(median_dir+median_epi_file): #if file doesn't exist
    
    if not os.path.exists(median_dir): # check if path to save median run exist
        os.makedirs(median_dir)       # if not create it
    
    # Load data for all runs then compute median run
    data = []
    for i in range(len(epi_files)):
        dataload = np.array(nb.load(fmriprep_func_dir+epi_files[i]).get_fdata())

        if i ==0:
            data = dataload[:,:,:,np.newaxis]
        else: 
            data = np.concatenate((data,dataload[:,:,:,np.newaxis]),axis=3)

    data_avg = np.median(data,axis=3)
    
    #load a run for header info
    run1 = nb.load(fmriprep_func_dir+epi_files[0])

    # save median run and median mask
    data_input_nii = nb.Nifti1Image(data_avg, affine=run1.affine, header=run1.header)
    data_input_nii.to_filename(median_dir+median_epi_file)


####

# params
rsq_threshold = 0.2
     
# for registration into pycortex

T1_file = os.path.join(fmriprep_anat_dir, 'sub-'+sub_num+'_desc-preproc_T1w.nii.gz')
fs_T1_file = os.path.join(fs_dir, 'sub-'+sub_num, 'mri', 'T1.nii.gz')  

if not os.path.exists(fs_T1_file): # convert the FS T1 .mgz file to .nii.gz if it wasn't already done
    os.system('mri_convert {mgz} {nii}'.format(
            mgz=fs_T1_file.replace('.nii.gz', '.mgz'),
            nii=fs_T1_file))
    
# prf results
prf_par_file = os.path.join(output_dir, 'params.nii.gz')
prf_rsq_file = os.path.join(output_dir, 'rsq.nii.gz')

# save fsl registration file in the subject folder
fsl_reg_file = os.path.join(output_dir, 'flirt.mat')

# website
web_path = os.path.join(output_dir, 'webgl')

if not os.path.exists(fsl_reg_file): # if no flirt.mat file already in dir
    bbreg = BBRegister(subject_id='sub-'+sub_num, 
                       source_file=fmriprep_func_dir+epi_files[0], 
                       init='fsl', 
                       contrast_type='t2',
                       out_fsl_file=fsl_reg_file,
                       subjects_dir=fs_dir)
    bbreg.run()

# only need to run this once for each subject
# will load into build folder in this dir
cortex.freesurfer.import_subj(subject='sub-'+sub_num, sname='sub-'+sub_num, freesurfer_subject_dir=fs_dir)

# create and save pycortex 'coord' transform from registration flirt file
xfm = cortex.xfm.Transform.from_fsl(xfm=fsl_reg_file, 
                                    func_nii=fmriprep_func_dir+epi_files[0], 
                                    anat_nii=fs_T1_file)
xfm.save('sub-'+sub_num, 'fmriprep_T1', 'coord')

# define colormaps
# load params and rsq to build maps
prf_params = nb.load(prf_par_file)
p_data = prf_params.get_data()
rsq = nb.load(prf_rsq_file).get_data()

# now construct polar angle and eccentricity values
# grid fit function gives out [x,y,size,baseline,betas]
complex_location = p_data[...,0] + p_data[...,1] * 1j
polar_angle = np.angle(complex_location)
eccentricity = np.abs(complex_location)
size = p_data[...,2]
baseline = p_data[...,3]
beta = p_data[...,4]

# normalize polar angles to have values in circle between 0 and 1 
polar_ang_norm = (polar_angle + np.pi) / (np.pi * 2.0)

# use "resto da divisão" so that 1 == 0 (because they overlapp in circle)
# why have an offset?
angle_offset = 0.1
polar_ang_norm = np.fmod(polar_ang_norm+angle_offset, 1.0)

# convert angles to colors, using correlations as weights
hsv = np.zeros(list(polar_ang_norm.shape) + [3])
hsv[..., 0] = polar_ang_norm # different hue value for each angle
hsv[..., 1] = np.ones_like(rsq) # saturation weighted by rsq
hsv[..., 2] = np.ones_like(rsq) # value weighted by rsq

# convert hsv values of np array to rgb values (values assumed to be in range [0, 1])
rgb = colors.hsv_to_rgb(hsv)

# define alpha channel - which specifies the opacity for a color
# 0 = transparent = values with rsq below thresh and 1 = opaque = values above thresh
alpha_mask = (rsq <= rsq_threshold).T #why transpose? because of orientation of pycortex volume?
alpha = np.ones(alpha_mask.shape)
alpha[alpha_mask] = 0

#create volumes

#contains RGBA colors for each voxel in a volumetric dataset
# volume for polar angles
vrgba = cortex.VolumeRGB(
    red=rgb[..., 0].T,
    green=rgb[..., 1].T,
    blue=rgb[..., 2].T,
    subject='sub-'+sub_num,
    alpha=alpha,
    xfmname='fmriprep_T1')

# volume for ecc
vecc = cortex.Volume2D(eccentricity.T, rsq.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=0, vmax=10,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='BROYG_2D')

# volume for size
vsize = cortex.Volume2D(size.T, rsq.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=0, vmax=10,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='BROYG_2D')

# volume for betas (amplitude?)
vbeta = cortex.Volume2D(beta.T, rsq.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=-2.5, vmax=2.5,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='RdBu_r_alpha')

# volume for baseline
vbaseline = cortex.Volume2D(baseline.T, rsq.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=-1, vmax=1,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='RdBu_r_alpha')

# volume for rsq
vrsq = cortex.Volume2D(rsq.T, rsq.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=0, vmax=0.8,
                           vmin2=rsq_threshold, vmax2=1.0, cmap='fire_alpha')

# volume for mean(median) and normalized epi for veins etc.
mean_epid = nb.load(median_dir+median_epi_file).get_data().T
mean_epid /= mean_epid.max()
mean_epi = cortex.Volume(mean_epid, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=mean_epid.min(), vmax=mean_epid.max(),
                           cmap='cubehelix')
#convert into a `Dataset`
DS = cortex.Dataset(polar=vrgba, ecc=vecc, size=vsize, 
                    amplitude=vbeta, baseline=vbaseline, rsq=vrsq, mean_epi=mean_epi)
# save in prf params dir
#DS.save(os.path.join(output_dir, 'pycortex_ds.h5'))

# Creates a static webGL MRI viewer in your filesystem
cortex.webgl.make_static(web_path, DS)#,template='SBref_sub-'+sub_num+'.html')


