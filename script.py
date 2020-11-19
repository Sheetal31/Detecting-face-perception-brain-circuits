


import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import os 
import nibabel as nib
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from nilearn import input_data
import time
from utils import shift_timing

sns.set(style = 'white', context='poster', rc={"lines.linewidth": 2.5})
sns.set(palette="colorblind")

# Get the nifti object
nii = nib.load("sub-1_task-objectviewing_run-01_bold.nii.gz")
hdr=nii.get_header()
 
 #convert tsv to csv
tsv_file='sub-1_task-objectviewing_run-01_events.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('sub-1_task-objectviewing_run-01_events.csv',index=False)


def seed_correlation(wbBold, seedBold):
    
    num_voxels = wbBold.shape[1]
    seed_corr = np.zeros((num_voxels, 1))
    for v in range(num_voxels):    
        seed_corr[v, 0] = np.corrcoef(seedBold.flatten(), wbBold[:, v])[0, 1]
    # Transfrom the correlation values to Fisher z-scores    
    seed_corr_fishZ = np.arctanh(seed_corr)
    return seed_corr, seed_corr_fishZ



#covert names to labels by using label encoding 
timing = pd.read_csv('sub-1_task-objectviewing_run-01_events.csv')
timing.drop(['duration'],axis=1,inplace=True)
timing.columns = ['onset','type']
from sklearn import preprocessing  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
timing.type= label_encoder.fit_transform(timing.type) 
timing.type.unique()

time_secs = timing.onset
labels = timing.type
plt.figure()
# Plot the stimulus data
plt.plot(time_secs, labels)
plt.title('stimulus presentation')
plt.xlabel('time in secs')
plt.show()



atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = atlas.maps

# This is where the atlas is saved.
print("Atlas path: " + atlas_filename + "\n\n")
# Plot the ROIs
plotting.plot_roi(atlas_filename);
print('Harvard-Oxford cortical atlas')
atlas_pd = pd.DataFrame(atlas)
print(atlas_pd['labels'])


# Create a masker object that we can use to select ROIs
masker_ho = NiftiLabelsMasker(labels_img=atlas_filename)
print(masker_ho.get_params())

# Apply our atlas to the Nifti object so we can pull out data from single parcels/ROIs
bold_ho = masker_ho.fit_transform(nii)
print('shape: parcellated bold time courses: ', np.shape(bold_ho))

# Get data 
var = timing.type 
var = np.concatenate([var, np.zeros(25)])
bold_ho_r = (bold_ho[(var != 3),:])

# What does our data structure look like?
print("Parcellated data shape (time points x num ROIs)")
print("All time points  ", bold_ho.shape)
print("Rightward attention trials: ", bold_ho_r.shape)

# Pull out a single ROI corresponding to the posterior parahippocampal cortex
# Note that Label #35 is the Parahippocampal Gyrus, posterior division. 
roi_id = 36
bold_ho_pPHG_r = np.array(bold_ho_r[:, roi_id])
bold_ho_pPHG_r = bold_ho_pPHG_r.reshape(bold_ho_pPHG_r.shape[0],-1)
print("Posterior PPC (region 35) rightward attention trials: ", bold_ho_pPHG_r.shape)

plt.figure(figsize=(14,4))
plt.plot(bold_ho_pPHG_r)
plt.ylabel('Evoked activity');
plt.xlabel('Timepoints');
plt.xticks(range(1, 50))
sns.despine()
plt.show()

# Like before we want to correlate the whole brain time course with the seed we have pulled out

corr_pPHG_r, corr_fz_pPHG_r = seed_correlation(bold_ho_r, bold_ho_pPHG_r) 

# Print the range of correlations.
print("PHG correlation Fisher-z transformed: min = %.3f; max = %.3f" % (
    corr_fz_pPHG_r.min(), corr_fz_pPHG_r.max())
)

# Plot a histogram
plt.hist(corr_fz_pPHG_r)
plt.ylabel('Frequency');
plt.xlabel('Fisher-z score');
plt.show()

# Map back to the whole brain image
img_corr_pPHG_r = masker_ho.inverse_transform(
    corr_fz_pPHG_r.T
)

threshold = .8

# Find the cut coordinates of this ROI, using parcellation.
# This function takes the atlas path and the hemisphere and outputs all centers of the ROIs
roi_coords = plotting.find_parcellation_cut_coords(atlas_filename,label_hemisphere='left')

# Pull out the coordinate for this ROI
roi_coord = roi_coords[roi_id,:]

# Plot the correlation as a map on a standard brain. 
# For comparison, we also plot the position of the sphere we created ealier.
h2 = plotting.plot_stat_map(
    img_corr_pPHG_r, 
    threshold=threshold,
    cut_coords=roi_coord,
)

# Create a glass brain
plotting.plot_glass_brain(
    img_corr_pPHG_r, 
    threshold=threshold,
    colorbar=True, 
    display_mode='lyrz', 
    plot_abs=False
)

plt.show()


# Set up the connectivity object
correlation_measure = ConnectivityMeasure(kind='correlation')
# Calculate the correlation of each parcel with every other parcel
corr_mat_ho_r = correlation_measure.fit_transform([bold_ho_r])[0]
# Remove the diagonal for visualization (guaranteed to be 1.0)
np.fill_diagonal(corr_mat_ho_r, np.nan)
# Plot the correlation matrix
# The labels of the Harvard-Oxford Cortical Atlas that we are using 
# start with the background (0), hence we skip the first label
fig = plt.figure(figsize=(11,10))
plt.imshow(corr_mat_ho_r, interpolation='None', cmap='RdYlBu_r')
plt.yticks(range(len(atlas.labels)), atlas.labels[1:]);
plt.xticks(range(len(atlas.labels)), atlas.labels[1:], rotation=90);
plt.title('Parcellation correlation matrix')
plt.colorbar();
plt.show()

# Load the atlas
atlas_nii = nib.load(atlas_filename)
atlas_data = atlas_nii.get_data()
labels = np.unique(atlas_data)

# Iterate through all of the ROIs
coords = []
for label_id in labels:
    
    # Skip the background
    if label_id == 0:
        continue
        
    # Pull out the ROI of within the mask    
    roi_mask = (atlas_data == label_id)
    
    # Create as a nifti object so it can be read by the cut coords algorithm
    nii = nib.Nifti1Image(roi_mask.astype('int16'), atlas_nii.affine)
    
    # Find the centre of mass of the connectome
    coords.append(plotting.find_xyz_cut_coords(nii))
    

# Plot the connectome
from nilearn import plotting
plotting.plot_connectome(corr_mat_ho_r, coords, edge_threshold='95%')
plt.show()
