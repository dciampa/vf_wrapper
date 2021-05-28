#!/usr/bin/env python3

# To run this python script on the python command line,
# do the following:
# import runpy
# runpy.run_module(mod_name='uv_line_fits')
# OR
# exec(open('uv_line_fits.py').read())

# To run from the regular command line:
# python uv_line_fits.py
# OR add a "-i" flag to be dumped in the python interpreter upon completion
# python -i uv_line_fits.py

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, traceback
import os
import numpy as np
import re # So we can string split with multiple diliminators
np.set_printoptions(threshold=sys.maxsize)
home_dir=os.environ['HOME']+'/'

import VoigtFit
import linecache # Read a specific line in a file
#home_dir is defined in the startup file
from numpy.lib.recfunctions import append_fields # So that fields can be added to structured "ndarray" arrays

#### Plotting stuff here ########################################################################################

fig=plt.plot()
plt.title(target,size=10,loc='center')
plt.plot(ion["vel"],ion["norm"],c='k',alpha=0.8)

# "Get Current Axis" information from the plot using "gca"
axes = plt.gca() # plt.gca().get_ylim() also works
plt.xlim(-velocity_span,velocity_span)

### Shade the masked regions
# It seems that if this limit isn't explicitly set somewhere, then it will end up changing as more stuff is plotted.
plt.ylim(axes.get_ylim()[0],axes.get_ylim()[1])
yfill=[axes.get_ylim()[0],axes.get_ylim()[0],axes.get_ylim()[1],axes.get_ylim()[1],axes.get_ylim()[0]]
if mask_index.size != 0:
    for i in range(len(mask_index)):
        xfill=[mask_regions['vmin'][mask_index[i]], \
        mask_regions['vmax'][mask_index[i]], \
        mask_regions['vmax'][mask_index[i]], \
        mask_regions['vmin'][mask_index[i]], \
        mask_regions['vmin'][mask_index[i]]]
        axes.fill(xfill,yfill, alpha=1, edgecolor='none', facecolor='thistle',zorder=0)

#plt.errorbar(ion["vel"],ion["norm"],yerr=ion["norm_err"],c='k',alpha=0.4,zorder=1)
plt.fill_between(ion["vel"], ion['norm']-ion["norm_err"], ion['norm']+ion["norm_err"], \
    facecolor="k",edgecolor='none',alpha=0.3,zorder=1)

plt.plot(fit_array['vel'],fit_array['fit'],linewidth=5,c='lightskyblue',zorder=2)
# Plot all of the individually fitted components
component_strings=list(profiles.keys())
num_components=len(component_strings) # The first column is either a velocity or wavelength array
for i in range(num_components-1):
    # Only plot the fitted component at the absorption feature and not across the entire spectral region
    # The round command is rounding the normalized profile data points to the 2nd decimal position.
    line_index=np.where(profiles[component_strings[i+1]] <= 0.999)
    plt.plot(profiles[component_strings[0]][line_index],profiles[component_strings[i+1]][line_index], \
        c='royalblue',alpha=1,linewidth=1.5, zorder=i+3) #linestyle='--'

plt.plot(ion["vel"],ion["norm"],c='k',alpha=0.6,zorder=i+4)
plt.ylabel('Normalized Flux',size=10)
plt.ylabel('Normalized Flux',size=10)
# This adds tick marks at the top of the plotting panel pointed inward with no labels
# Also adding ticks to the right side without labels
plt.tick_params(labeltop=False,labelbottom=False,top=True,labelright=False,right=True,direction='in')

# Plot a dotted horizontal line at y=1
plt.plot(axes.get_xlim(),[1,1],':',color='black')

### Plot residuals ###

divider = make_axes_locatable(axes)
# Neat that you can set a percentage size for the subplot.
cax = divider.append_axes('bottom', size='30%', pad=0., sharex=axes, zorder=1)
cax.set_axisbelow(False) # This makes sure that the ticks are plotted on top layer of the plot
                         # so that they are not covered by the masked polynomial
cax.tick_params(labeltop=False,labelbottom=True,top=True,labelright=False,right=True,direction='in')
cax.set_ylabel('Residuals',size=10)
plt.xlabel('LSR velocity [km s$^{-1}$]',size=10)
# Obnoxiously, the data and the fit might not have the exact same array length
# Extrapolating fixes that issue.
norm_spectrum = np.interp(fit_array['vel'], ion['vel'], ion['norm'])
resid_index=np.where(fit_array['mask'] == 1)

cax.fill_between(fit_array['vel'][resid_index], \
    (norm_spectrum[resid_index]-fit_array['fit'][resid_index])-fit_array['norm_err'][resid_index], \
    (norm_spectrum[resid_index]-fit_array['fit'][resid_index])+fit_array['norm_err'][resid_index], \
    facecolor="k",edgecolor='none',alpha=0.3,zorder=1)
# Plot a two pink lines in the residuals that represent 3 times the error from y=0 (not residuals ± 3 times the error).
# If the residuals go past this pink line, then the signal there might be significant and the quality of the fit
# should be scrutinized.
cax.plot(fit_array['vel'][resid_index], 3*fit_array['norm_err'][resid_index], color='PaleVioletRed', lw=1., zorder=0)
cax.plot(fit_array['vel'][resid_index], -3*fit_array['norm_err'][resid_index], color='PaleVioletRed', lw=1., zorder=0)
cax.axhline(0., ls=':', color='black') # Plot a dotted horizontal line at y=0

plt.plot(fit_array['vel'][resid_index],norm_spectrum[resid_index]-fit_array['fit'][resid_index],c='k',alpha=0.8,zorder=1)
res_min = 1.25*np.nanmin(((norm_spectrum[resid_index]-fit_array['fit'][resid_index]))-fit_array['norm_err'][resid_index])
res_max = 1.25*np.nanmax(((norm_spectrum[resid_index]-fit_array['fit'][resid_index]))+fit_array['norm_err'][resid_index])
cax.set_ylim(res_min, res_max)
# Fill the masked regions with a red polygon
if mask_index.size != 0:
    for i in range(len(mask_index)):
        xfill=[mask_regions['vmin'][mask_index[i]], \
        mask_regions['vmax'][mask_index[i]], \
        mask_regions['vmax'][mask_index[i]], \
        mask_regions['vmin'][mask_index[i]], \
        mask_regions['vmin'][mask_index[i]]]
        yfill=[res_min,res_min,res_max,res_max,res_min]
        cax.fill(xfill,yfill, alpha=1, edgecolor='none', facecolor='thistle',zorder=2,clip_on=True)

### Add titles of ion wave and the reduced chi squared

plt.title("  "+ion_string+' $\lambda$'+wave_string,size=10,loc='left')
plt.title("Reduced $\chi^2$: % 5.2f   " % (popt.redchi),size=10,loc='right')

plt.show()
