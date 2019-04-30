#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:51:34 2019

@author: inesverissimo

make pycortex volumes with different tasks plotted

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
        pRF_dir = '/Users/inesverissimo/Documents/SB_project/pRF_fit/outputs/sub-'+sub_num+'/run-median/'
        soma_dir = '/Users/inesverissimo/Documents/SB_project/SOMA_fit/outputs/sub-'+sub_num+'/'
        output_dir = '/Users/inesverissimo/Documents/SB_project/cortex/sub-'+sub_num+'/'
        
    elif sys.argv[2]=='aeneas':	

        base_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/'
        pRF_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/pRF_fit/sub-'+sub_num+'/run-median/'
        soma_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/soma_fit/sub-'+sub_num+'/'
        output_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/cortex/sub-'+sub_num+'/'
        
    else:	
        raise NameError('Machine not defined, no parametes will be given') 
        

## define paths and list of files
        
if sub_num=='11':
    raise NameError('Not possible for sub 11, tasks in different sessions, see how to solve this') 
      
# paths to derivatives data (get freesurfer volumes), output and mean functionals
fmriprep_func_dir = base_dir+'fmriprep/sub-'+sub_num+'/ses-01/func/'
fmriprep_anat_dir = base_dir+'fmriprep/sub-'+sub_num+'/anat/'
fs_dir = base_dir+'freesurfer/'
median_dir = base_dir+'post_fmriprep/sub-'+sub_num+'/ses-01/func/median/'

    
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
z_threshold = 3.1 #2
     
# for registration into pycortex

T1_file = os.path.join(fmriprep_anat_dir, 'sub-'+sub_num+'_desc-preproc_T1w.nii.gz')
fs_T1_file = os.path.join(fs_dir, 'sub-'+sub_num, 'mri', 'T1.nii.gz')  

if not os.path.exists(fs_T1_file): # convert the FS T1 .mgz file to .nii.gz if it wasn't already done
    os.system('mri_convert {mgz} {nii}'.format(
            mgz=fs_T1_file.replace('.nii.gz', '.mgz'),
            nii=fs_T1_file))
    
# prf results
prf_par_file = os.path.join(pRF_dir, 'params.nii.gz')
prf_rsq_file = os.path.join(pRF_dir, 'rsq.nii.gz')

# soma results
face_file = os.path.join(soma_dir, 'z_face_contrast.nii.gz')
upper_file = os.path.join(soma_dir, 'z_upper_limb_contrast.nii.gz')
lower_file = os.path.join(soma_dir, 'z_lower_limb_contrast.nii.gz')

RLupper_file = os.path.join(soma_dir, 'z_right-left_hand_contrast.nii.gz')
RLlower_file = os.path.join(soma_dir, 'z_right-left_leg_contrast.nii.gz')

# save fsl registration file in the subject folder
fsl_reg_file = os.path.join(pRF_dir, 'flirt.mat')

# website
if not os.path.exists(output_dir): # check if path to save median run exist
        os.makedirs(output_dir)       # if not create it
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
vrsq = cortex.Volume(rsq.T, 'sub-'+sub_num, 'fmriprep_T1',
                     vmin=rsq_threshold, vmax=1.0, cmap='Reds')

# volume for mean(median) and normalized epi for veins etc.
mean_epid = nb.load(median_dir+median_epi_file).get_data().T
norm_mean_epid =(mean_epid - mean_epid.min())/(mean_epid.max()- mean_epid.min())
mean_epi = cortex.Volume(norm_mean_epid, 'sub-'+sub_num, 'fmriprep_T1',
                         vmin=norm_mean_epid.min(), vmax=norm_mean_epid.max(),
                         cmap='cubehelix')
                           
# somatotopy

face_zscore = nb.load(face_file).get_data()
upper_zscore = nb.load(upper_file).get_data()
lower_zscore = nb.load(lower_file).get_data()

RLupper_zscore = nb.load(RLupper_file).get_data()
RLlower_zscore = nb.load(RLlower_file).get_data()

# threshold left vs right, to only show relevant voxel
z_RLupper_data1D = RLupper_zscore.ravel()
z_RLlower_data1D = RLlower_zscore.ravel()

data_threshed_up=np.zeros(z_RLupper_data1D.shape) # set at 0 whatever is outside thresh
data_threshed_down=np.zeros(z_RLlower_data1D.shape)

for i in range(len(z_RLupper_data1D)):
    if z_RLupper_data1D[i] < -z_threshold or z_RLupper_data1D[i] > z_threshold:
        data_threshed_up[i]=z_RLupper_data1D[i]
    
    if z_RLlower_data1D[i] < -z_threshold or z_RLlower_data1D[i] > z_threshold:
        data_threshed_down[i]=z_RLlower_data1D[i]

z_RLupper_thresh = np.reshape(data_threshed_up,RLupper_zscore.shape) #back to original shape
z_RLlower_thresh = np.reshape(data_threshed_down,RLlower_zscore.shape)

#


# volume for face, upper and lower limb zscore
v_face =  cortex.Volume(face_zscore.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=z_threshold, vmax=face_zscore.max(),
                           cmap='BuGn')
v_upper =  cortex.Volume(upper_zscore.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=z_threshold, vmax=upper_zscore.max(),
                           cmap='Blues')
v_lower =  cortex.Volume(lower_zscore.T, 'sub-'+sub_num, 'fmriprep_T1',
                           vmin=z_threshold, vmax=lower_zscore.max(),
                           cmap='OrRd')
# left vs right
rl_upper = cortex.Volume(z_RLupper_thresh.T, 'sub-'+sub_num, 'fmriprep_T1',
                         vmin=-z_RLupper_thresh.max(), vmax=z_RLupper_thresh.max(),
                         cmap='bwr')
rl_lower = cortex.Volume(z_RLlower_thresh.T, 'sub-'+sub_num, 'fmriprep_T1',
                         vmin=-z_RLlower_thresh.max(), vmax=z_RLlower_thresh.max(),
                         cmap='bwr')

#convert into a `Dataset`
DS = cortex.Dataset(polar=vrgba, ecc=vecc, size=vsize, 
                    amplitude=vbeta, baseline=vbaseline, rsq=vrsq, mean_epi=mean_epi,
                    face=v_face,upper_limb=v_upper,lower_limb=v_lower,
                    RvsL_upper=rl_upper,RvsL_lower=rl_lower)
# save in prf params dir
#DS.save(os.path.join(output_dir, 'pycortex_ds.h5'))

# Creates a static webGL MRI viewer in your filesystem
cortex.webgl.make_static(web_path, DS)#,template='SBref_sub-'+sub_num+'.html')


