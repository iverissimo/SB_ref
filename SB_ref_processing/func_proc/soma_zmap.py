#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Apr 29 11:13:11 2019

@author: inesverissimo

Do SOMA contrasts and save outputs

"""

# import packages
from nilearn import image, plotting
import matplotlib.pyplot as plt

import os, sys
import glob
import json
import numpy as np
import nibabel as nb
from spynoza.filtering.nodes import savgol_filter, savgol_filter_confounds
from spynoza.conversion.nodes import Percent_signal_change
from nipype import Node, Function
import nipype.pipeline as pe
import nipype.interfaces.io as nio
#from bids.grabbids import BIDSLayout
import pandas as pd

from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show
from nistats.reporting import plot_design_matrix, plot_contrast_matrix

from nistats.first_level_model import FirstLevelModel
from sklearn.decomposition import PCA

from nipype.interfaces import fsl
from nipype.interfaces import freesurfer
from nipype.interfaces.utility import Function, IdentityInterface


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
        
        base_dir = '/Users/inesverissimo/Documents/SB_project/mp2rage/func_nosubmm/derivatives/fmriprep/sub-'+sub_num+'/ses-01/func/'
        #median_dir = '/Users/inesverissimo/Documents/SB_project/SOMA_fit/outputs/sub-'+sub_num+'/run-median/' 
        output_dir = '/Users/inesverissimo/Documents/SB_project/SOMA_fit/outputs/'
        event_dir = '/Users/inesverissimo/Documents/SB_project/mp2rage/func_nosubmm/sourcedata/sub-'+sub_num+'/ses-01/func/'

        
    elif sys.argv[2]=='aeneas':	

        base_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/fmriprep/sub-'+sub_num+'/ses-01/func/'
        #median_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/post_fmriprep/sub-02/ses-01/func/median/'
        output_dir = '/home/shared/2018/visual/SB-prep/SB-ref/derivatives/soma_fit/sub-'+sub_num+'/'
        event_dir = '/home/shared/2018/visual/SB-prep/SB-ref/sourcedata/sub-'+sub_num+'/ses-01/func/'
        
    else:	
        raise NameError('Machine not defined, no parametes will be given') 
        

if (sub_num=='01' or sub_num=='03'): # and ses_num=='01': # exception for some initial subjects' sessions
    TR = 1.5
else:
    TR = params['TR']
    
if not os.path.exists(output_dir): # check if path for outputs exist
        os.makedirs(output_dir)       # if not create it

# list of functional files
filenames = [run for run in os.listdir(base_dir) if 'soma' in run and run.endswith('T1w_desc-preproc_bold.nii.gz')]
filenames.sort()
# list of confounds
confounds = [run for run in os.listdir(base_dir) if 'soma' in run and run.endswith('_desc-confounds_regressors.tsv')]
confounds.sort()
# list of stimulus onsets
events = [run for run in os.listdir(event_dir) if 'soma' in run and run.endswith('events.tsv')]
events.sort()
#list of boldref files
#epi_files = [run for run in os.listdir(base_dir) if 'soma' in run and run.endswith('T1w_boldref.nii.gz')]
#epi_files.sort()

# high pass filter all runs and save 

print('High pass filtering runs')

filenames_sg = []
for j in range(len(filenames)):
    
    file_sg = savgol_filter(base_dir+filenames[j], polyorder=params['sg_filt_polyorder'], deriv=params['sg_filt_deriv'], window_length=params['sg_filt_window_length'], tr=TR)
    filenames_sg.append(os.path.basename(file_sg))
    # move file to median directory
    os.rename(file_sg, output_dir+os.path.basename(file_sg))

# do same for confounds

print('High pass filtering confounds')

all_confs = []

for j in range(len(confounds)):
    # high pass confounds
    confounds_SG = savgol_filter_confounds(base_dir+confounds[j], polyorder=params['sg_filt_polyorder'], deriv=params['sg_filt_deriv'], window_length=params['sg_filt_window_length'], tr=TR)

    confs = pd.read_csv(confounds_SG, sep='\t', na_values='n/a')
    confs = confs[params['nuisance_columns']]
    
    #choose the minimum number of principal components such that at least 95% of the variance is retained.
    #pca = PCA(0.95,whiten=True) 
    pca = PCA(n_components=2,whiten=True) #had to chose 2 because above formula messes up len of regressors 
    pca_confs = pca.fit_transform(np.nan_to_num(confs))
    print('%d components selected for run' %pca.n_components_)

    # make list of dataframes 
    all_confs.append(pd.DataFrame(pca_confs, columns=['comp_{n}'.format(n=n) for n in range(pca.n_components_)]))
    
    # move file to median directory
    os.rename(confounds_SG, output_dir+os.path.basename(confounds_SG))
    

# Append all events in same dataframe

print('Loading events')

all_events = []
for e in range(len(events)):
    
    events_pd = pd.read_csv(event_dir+events[e],sep = '\t')

    new_events = []
    
    for ev in events_pd.iterrows():
        row = ev[1]   
        if row['trial_type'][0] == 'b': # if both hand/leg then add right and left events with same timings
            new_events.append([row['onset'],row['duration'],'l'+row['trial_type'][1:]])
            new_events.append([row['onset'],row['duration'],'r'+row['trial_type'][1:]])
        else:
            new_events.append([row['onset'],row['duration'],row['trial_type']])
   
    df = pd.DataFrame(new_events, columns=['onset','duration','trial_type'])  #make sure only relevant columns present
    all_events.append(df)

# Do GLM
    
fmri_glm = FirstLevelModel(t_r=TR,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='glover',
                           n_jobs=4)

all_sg_soma = [output_dir+filenames_sg[s] for s in range(len(filenames_sg))]

print('Fitting GLM...')

fmri_glm = fmri_glm.fit(all_sg_soma, events=all_events, confounds=all_confs)

design_matrix = fmri_glm.design_matrices_[0]

# Compute z-score of contrasts

print('Computing contrasts')

all_contrasts = {'upper_limb':['lhand_fing1','lhand_fing2','lhand_fing3','lhand_fing4','lhand_fing5',
             'rhand_fing1','rhand_fing2','rhand_fing3','rhand_fing4','rhand_fing5'],
             'lower_limb':['lleg','rleg'],
             'face':['eyes','eyebrows','tongue','mouth']}

for index, (contrast_id, contrast_val) in enumerate(all_contrasts.items()):
    contrast = np.zeros(len(design_matrix.columns)) # array of zeros with len = num predictors
    for i in range(len(contrast)):
        if design_matrix.columns[i] in contrast_val:
            contrast[i] = 1
    
    z_map = fmri_glm.compute_contrast(contrast, output_type='z_score')
    z_map.to_filename(output_dir+'z_%s_contrast.nii.gz' % contrast_id)


# compare each finger with the others of same hand
bhand_label = ['lhand','rhand']
for j,lbl in enumerate(bhand_label):
    
    hand_label = [s for s in all_contrasts['upper_limb'] if lbl in s]
    
    for index, label in enumerate(hand_label):

        contrast = np.zeros(len(design_matrix.columns)) # array of zeros with len = num predictors
    
        for i in range(len(contrast)):
            if design_matrix.columns[i]==label: 
                contrast[i] = 1
            elif lbl in design_matrix.columns[i]: # -1 to other fingers of same hand
                contrast[i] = -1
                
        z_map = fmri_glm.compute_contrast(contrast, output_type='z_score')
        z_map.to_filename(output_dir+'z_%s-all_%s_contrast.nii.gz' % (label,lbl))
    

#compare left vs right
rl_limb = ['hand','leg']

for j in range(len(rl_limb)):
    contrast = np.zeros(len(design_matrix.columns)) # array of zeros with len = num predictors
    for i in range(len(contrast)):
        if 'r'+rl_limb[j] in design_matrix.columns[i]:
            contrast[i] = 1
        elif 'l'+rl_limb[j] in design_matrix.columns[i]:
            contrast[i] = -1
    
    plot_contrast_matrix(contrast, design_matrix=design_matrix)        

    z_map = fmri_glm.compute_contrast(contrast,
                                  output_type='z_score')
    z_map.to_filename(output_dir+'z_right-left_'+rl_limb[j]+'_contrast.nii.gz')

print('Success!')

