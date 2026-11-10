# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:42:54 2026

@author: cre10128
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import re
from natsort import natsorted   
from scipy.stats import  lognorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import skimage as ski
from scipy import ndimage as ndi
from skimage import io, util, data, exposure, color
from skimage.filters import difference_of_gaussians, threshold_li
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.color import label2rgb

########################### prepare figure ##################################
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 6,
    'axes.labelsize': 8,
    'axes.linewidth' : 0.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.top': True,
    'ytick.minor.right': True,
    'xtick.minor.ndivs' : 5,
    'ytick.minor.ndivs' : 5,
    'ytick.right': True,
    'xtick.top': True,
    'xtick.major.pad' : 3,
    'ytick.major.pad' : 3,
    'xtick.major.width' : 0.5,
    'ytick.major.width' : 0.5,
    'xtick.minor.width' : 0.5,
    'ytick.minor.width' : 0.5
    
    
})

# ############# create figure 
fig = plt.figure(figsize=(9, 4), dpi=300)#, sharex=True, sharey=True)

gs = gridspec.GridSpec(1, 2, wspace=0.1) #, hspace=0.1)   ## main gridspec


gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.2)  

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.1, hspace=0.2, width_ratios=[0.05, 0.95])  

ax_dist = fig.add_subplot(gs2[0, 1])

axins = ax_dist.inset_axes([0.65, 0.65, 0.3, 0.3])
axins.set_xlabel(r'$c $ [wt\%]')
axins.set_ylabel(r'Diameter [µm]')


########### load images and read 

# fp = r"E:\Semester 5\Experiments\confocal\microgel_concentration_variation_FITC\concentrations\same_size"
fp = r'E:\Semester 5\Experiments\confocal\microgel_concentration_variation_FITC\concentrations\raw\preprocessed'
dat_files = sorted([f for f in os.listdir(fp) if f.endswith('.tif')])
dat_files = natsorted(dat_files) #sorting files in natural order :)

#### colormap
min_val, max_val = 0.1, 1

cmap = plt.colormaps['viridis_r']
colors = cmap(np.linspace(min_val, max_val, len(dat_files)))

###colorbar
norm_color = Normalize(vmin=0.05, vmax=1.25)
sm_color = ScalarMappable(cmap=cmap, norm=norm_color)


axs = []
wtpcs = []
diams = []
x = np.linspace(0, 15, 200)

max_radius = [13.392/2, 10.4888/2, 8.10/2, 5.597/2] ### these are obtained by measuring the biggest particle on imageJ
eq_microns = [4.9, 8.3079, 16.6320, 16.6320] ### these are the scale bar obtained from the microscope pixel to micron equivalence pixels/micron

for i, (filename) in enumerate(dat_files):
    filepath = os.path.join(fp, filename)
    
    ### create axis
    axs.append(i)
    if i <= len(dat_files)/2 - 1:
        axs[i] =  fig.add_subplot(gs1[0, i]) 
    else : 
        axs[i] =  fig.add_subplot(gs1[1, i-2]) 
    
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)

    for spine in axs[i].spines.values():
        spine.set_edgecolor(colors[i])
        spine.set_linewidth(3)
        
    ### get concentration of sample
    match = re.search(r'Mg_(\d+)_', filename)  
    if match:
        num_str = match.group(1)
        wt_percent = int(num_str) / 100  # Convert to float wt%
    print(f'\n  # # # # Processing image for: {wt_percent:.2f} wt % # # # # \n')
    
    ### load image
    sample = ski.io.imread(filepath)

    im = util.img_as_float(sample) #### use image as float...
    # axs[i].imshow(im, cmap=plt.cm.gray)
    axs[i].set_title(fr'$c =$ {wt_percent:.2f} wt \%')
    
    ### Difference of Gaussians 
    filtered_image = difference_of_gaussians(im, low_sigma=1, high_sigma=20)
    # axs[i].imshow(filtered_image, cmap=plt.cm.gray)
    
    ### Enhanced edges and threshold
    enhanced_edges = exposure.equalize_hist(filtered_image)
    thresh = threshold_li(enhanced_edges)
    binary_objects = enhanced_edges > thresh
    axs[i].imshow(binary_objects, cmap=plt.cm.gray)
    
    ### watershed segmentation
    ### apply watershed segmentation bounded
    distance = ndi.distance_transform_edt(binary_objects)
    minimal_lengthscale_detectable = int(0.5*eq_microns[i])
    coords = peak_local_max(distance, min_distance=minimal_lengthscale_detectable, labels=binary_objects) ### min distance is the minimum lengthscale that can be detected
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_objects)
        
    valid_centers_x = []
    valid_centers_y = []
    valid_radii = []

    for region in regionprops(labels):
        # Calculate equivalent radius: area = pi * r^2  =>  r = sqrt(area / pi)
        radius = np.sqrt(region.area / np.pi)
        
        # Filter out regions that don't match the target radius size (8 to 52 pixels)
        if 0.5*eq_microns[i] <= radius <=  max_radius[i] * eq_microns[i]: 
            cy, cx = region.centroid
            valid_centers_x.append(cx)
            valid_centers_y.append(cy)
            valid_radii.append(radius)
        
    labels_overlay = label2rgb(labels, image=binary_objects, bg_label=0)

    
    # axs[i].imshow(labels_overlay)

    # axs[i].imshow(filtered_image, cmap=plt.cm.gray)

    # for cx, cy, r in zip(valid_centers_x, valid_centers_y, valid_radii):
    #     circle = plt.Circle((cx, cy), r, color='red', fill=False, linewidth=0.5, alpha=0.5)
    #     axs[i].add_patch(circle)
    
    
    ############# plotting the size distribution
    ### convert radius from pixels to microns!!!
    valid_radii_microns = np.array(valid_radii) / eq_microns[i]
    diameters = 2*valid_radii_microns
    
    ######### lognorm distribution fit 
    shape, loc, scale = lognorm.fit(diameters, floc=0)
    pdf = lognorm.pdf(x, shape, loc, scale)
    mu = np.log(scale)
    std = np.sqrt((np.exp(shape**2)-1)*np.exp(2*mu + shape**2))
    ax_dist.plot(x, pdf, color=colors[i], linewidth=1)
    
    
    counts, bins, patches = ax_dist.hist(
        diameters,
        bins= 10,# len(radius/10) round up fill=False, histtype="step", stacked=True,
        color=colors[i],
        # stacked=True,
        density=True,
        alpha=0.7,
        edgecolor='white',
        # label= rf'$c =$ {wt_percent:.2f} wt \%, $\bar{{D}}$ ={mu:.2f} µm, $\sigma$ ={std:.2f} µm' 
    )
    axins.scatter(wt_percent, mu*2, color=colors[i], marker='s')
    wtpcs.append(wt_percent)
    diams.append(mu*2)    
    
#########  plot settings ############
ax_dist.set_xlabel(r'Diameter [µm]')
ax_dist.set_ylabel(r'Density of microgel particles')
ax_dist.legend(framealpha=0)

# axins.set_ylim(1, 5)
    
fig.tight_layout()
plt.show()    

# ### apply difference of gaussians filter 
# sigma_low = 1
# sigma_high = 20
# filtered_image = difference_of_gaussians(im, low_sigma=sigma_low, high_sigma=sigma_high)
# ax0.set_title('Gaussian blur', fontsize = 6)
# ax0.imshow(filtered_image, cmap=plt.cm.gray)
    
# ### Enhance edges and threshold
# enhanced_edges = exposure.equalize_hist(filtered_image)
# thresh = threshold_li(enhanced_edges)
# binary_objects = enhanced_edges > thresh

# ax1.set_title('Binary mask', fontsize = 6)

# ax1.imshow(binary_objects, cmap=plt.cm.gray)

# ### apply watershed segmentation bounded
# distance = ndi.distance_transform_edt(binary_objects)
# coords = peak_local_max(distance, min_distance=16, labels=binary_objects) ### 16 pixels is the minimum diameter that can be detected
# mask = np.zeros(distance.shape, dtype=bool)
# mask[tuple(coords.T)] = True
# markers, _ = ndi.label(mask)
# labels = watershed(-distance, markers, mask=binary_objects)
    
# valid_centers_x = []
# valid_centers_y = []
# valid_radii = []

# for region in regionprops(labels):
#     # Calculate equivalent radius: area = pi * r^2  =>  r = sqrt(area / pi)
#     radius = np.sqrt(region.area / np.pi)
    
#     # Filter out regions that don't match the target radius size (8 to 52 pixels)
#     if 8 <= radius <= 63.34782: 
#         cy, cx = region.centroid
#         valid_centers_x.append(cx)
#         valid_centers_y.append(cy)
#         valid_radii.append(radius)
    
# labels_overlay = label2rgb(labels, image=binary_objects, bg_label=0)

# ax2.set_title('Watershed segmentation', fontsize = 6)
# ax2.imshow(labels_overlay)

# ax2.imshow(filtered_image, cmap=plt.cm.gray)

# for cx, cy, r in zip(valid_centers_x, valid_centers_y, valid_radii):
#     circle = plt.Circle((cx, cy), r, color='red', fill=False, linewidth=0.5) ##, alpha=0.3)
#     ax2.add_patch(circle)

# plt.tight_layout()
# plt.show()

