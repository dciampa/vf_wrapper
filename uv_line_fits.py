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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, traceback
import os
import numpy as np
import re # So we can string split with multiple deliminators
np.set_printoptions(threshold=sys.maxsize)
from os.path import expanduser
home_dir = os.path.join(expanduser('~'),'') # This is to ensure that it works on all platforms
# the os.path.join( ,"") will add a forward (Mac/Linux) or backslash (Windows) to the home_dir variable

import VoigtFit
import linecache # Read a specific line in a file
#home_dir is defined in the startup file
from numpy.lib.recfunctions import append_fields # So that fields can be added to structured "ndarray" arrays

#### User input ################################################################################################

# Common options: 'OI_1302', 'CII_1334', 'CIIx_1335', 'NI_1199', 'NI_1200', 'SiII_1260',
#                 'SiII_1193', 'SiII_1190', 'SiII_1304', 'SII_1250 'SII_1253', 'SII_1259',
#                 'FeII_1144', 'PII_1152', 'SiIII_1206', 'SiIV_1393', 'SiIV_1402', 'NV_1238', 'NV_1242'
###   Note: Added a line CIIa_1335 in VoigtFit/VoigtFit/static/linelist.dat
###   to read CIIx_1335 so that it is compatible with our line naming convention
ion_wave_string='SiIV_1402' # Must be a string formatted ion_wave
velocity_span = 1500.0
rebin_factor=3

#### Load the spectral data ####################################################################################

# Name of the QSO background target as given in the file naming convention
target='UVQSJ203335.89-032038.5'
# target='UVQSJ204402.02-075810.0'

# This _ISM.dat file likely does not contain the night only observations....
# This is the file that contains the 'stacked' and already normalized spectrum that Bart Wakker created for us.
file=target+'_ISM.dat' # UVQSJ203335.89-032038.5_ISM.dat

# This is just the location of where you have your Smith Cloud project saved.
base_directory=os.path.join(home_dir,'vf','')    # /Users/kat/HST/Smith_Cloud

# This is the location in which you have your '_ISM.dat' file saved.
data_directory=os.path.join(base_directory,'reduction','')  # /Users/kat/HST/Smith_Cloud/reduction/UVQSJ203335.89-032038.5

# This is the directory that you have your uv_line_fits.py, fit_masks.dat, and fit_guesses.dat file saved.
fit_directory=os.path.join(base_directory,'analysis','UV_Fits','') # /Users/kat/HST/Smith_Cloud/Analysis/UV_Fits

# Make sure that all of the directories above have a '/' at the end of those strings.

###############################################################################################################

### Read in and parse the stacked columns of data that are split by row headers
### that begin with the target name.

# Find all line numbers in the file that have the target name,
# which act like mini headers
lnum=[] # Create a blank list for the line numbers of the header rows.
transition_list=[] # Blank list for the ion and wavelength from header rows in the spectral file
with open(data_directory+file) as myFile:
    # Read all of each line in the file and store the information as line.
    # For each line read in, num will grow by 1.
    for num, line in enumerate(myFile, 1):
        # Search for the target name in the read in line. If it matches, save that line number.
        if target in line:
            lnum=lnum+[num] # Concatinate the lnum list with the line number with target name in it.
            # Split the text where ever there is a dash, plus, or space
            tmp=re.split("[-+ ]", line)
            # The split on tmp[3] is to remove any periods from the line transition strings
            transition=[tmp[2]+'_'+tmp[3].split(".")[0]] # square brackets are needed to make this a list
            transition_list=transition_list+transition # ion+line list of strings
    lnum=lnum+[num] # To grab the last line in the data file

# After this loop runs, num = number of lines in the file
total_num_lines=num

for i in range(len(lnum)-1):
    # This will create structured arrays with names ion+wave (e.g., CII1334)
    # that contains the spectral segment for each line transition.
    wave,vel,flux,err,cont,flux_binned=np.genfromtxt(data_directory+file, \
        skip_header=lnum[i],max_rows=lnum[i+1]-lnum[i]-1,unpack=True)

    # Using the VoigtFit.output.rebin_spectrum method to rebin the dataset by a factor of rebin_factor
    # Note that the method is expecting just wave, flux, err, but it seems that it really just needs
    # x, y, yerr to rebin. Therefore, we should also be able to use it to rebin the velocity and continuum
    wave_rebinned,flux_rebinned,err_rebinned=VoigtFit.output.rebin_spectrum( \
        wave,flux,err,int(rebin_factor),method='median')
    vel_rebinned,cont_rebinned,err_rebinned=VoigtFit.output.rebin_spectrum( \
        vel,cont,err,int(rebin_factor),method='median')

    exec(transition_list[i]+"={'ion_wave':transition_list[i],'wave':wave_rebinned,'vel':vel_rebinned, \
        'flux':flux_rebinned,'err':err_rebinned, \
        'norm':flux_rebinned/cont_rebinned,'norm_err':err_rebinned/cont_rebinned, \
        'cont':cont_rebinned}")
    #print(transition_list[i], i)

# This is behaving like a where or string match function that provides the
# index of the list that matches the string.
ion_string=ion_wave_string.split('_')[0]
wave_string=ion_wave_string.split('_')[1]
exec('ion ='+ ion_wave_string)

#### VoigtFit Stuff ##############################################################################################################

# Add systemic redshift
z_sys = 0.0

# This creates the object "dataset" using the function "DataSet" in the VoigtFit package.
# Notice that "DataSet" is a function and "dataset" is an object; CASE SENSITIVE!!!!
# dataset and DataSet are two different things!!!
dataset = VoigtFit.DataSet(z_sys)
# Stores the redshift in the dataset.redshift variable

# Set the width of velocity fit region
dataset.velspan = velocity_span

root_file_name=target.split('.')[0]+'_'+ion_wave_string
# Note that a "method" is a function of an "object".
# Here we are using the object's method "set_name" to define the file.
# So the syntax is "object.method(inputs)"
dataset.set_name(root_file_name) # dataset.name
# object = class()

# Supply verbose print statements to the screen
dataset.verbose = True

# The resolving power of HST/COS/G130M
res_g130m = 16000.
c = 299792.458 # speed of light in km/s

dataset.add_data(ion['wave'], ion['norm'], c/res_g130m, \
	err=ion['norm_err'], normalized=True, mask=None)

# Specify which line transitions will be fit.
dataset.add_line(ion_wave_string)

# Read in the mask ranges for each line transition spectral segment.
column_names=['ion_wave','vmin','vmax']
mask_file='fit_masks.dat'
mask_regions=np.genfromtxt(fit_directory+mask_file, \
	skip_header=1,names=column_names,dtype="U10,f8,f8")
mask_index=(np.where(mask_regions['ion_wave']==ion_wave_string)[0])
if mask_index.size != 0:
    for i in range(len(mask_index)):
        # Ensure that the min and max mask ranges are passed such that min < max
        # just incase the are ordered incorrectly in the fit_masks.dat file
        min_mask=np.min([mask_regions['vmin'][mask_index[i]],mask_regions['vmax'][mask_index[i]]])
        max_mask=np.max([mask_regions['vmin'][mask_index[i]],mask_regions['vmax'][mask_index[i]]])
        dataset.mask_range(ion_wave_string, min_mask, max_mask)

# Read in the initial guesses for the vcen, b, logN for each component in the line transition spectral segment.
# This includes flags for fixing or varying these parameters.
column_names=['ion','vcen','vcen_var','b','b_var','logN','logN_var']
guesses_file='fit_guesses.dat'
component_constraints=np.genfromtxt(fit_directory+guesses_file, \
	skip_header=1,names=column_names,dtype="U5,f8,i2,f8,i2,f8,i2")
component_index=(np.where(component_constraints['ion']==ion_string)[0])

if len(component_index) == 0:
    print('\nNo inital parameter guesses have been passed for '+ion['ion_wave'])
    print('At least one component guess must be set to run a fit...')
    print('These are set in this file: '+fit_directory+guesses_file+'\n\n')
    raise SystemExit(1)
else:
    for i in range(len(component_index)):
        dataset.add_component_velocity(ion_string, \
        component_constraints['vcen'][component_index[i]], \
        component_constraints['b'][component_index[i]], \
        component_constraints['logN'][component_index[i]], \
        # Note that there is not a var_v for velocity, but var_z has the same result.
        var_z=component_constraints['vcen_var'][component_index[i]], \
        var_b=component_constraints['b_var'][component_index[i]], \
        var_N=component_constraints['logN_var'][component_index[i]])

# Set the nomalized flag to false, meaning do not normalzied.
# Set the mask flag to false, meaning do not ask us to specify masked regions.
# Setting velocity=True enables normalization and masking plots to be done in velocity space.
# ... although these plots will not ever appear anyways with the mask and norm flags set to False...
dataset.prepare_dataset(norm=False, mask=False, velocity=True)

# Fit the dataset with the spectrum rebined by rebin factor:
popt, not_reduced_chi2 = dataset.fit(verbose=True, rebin=True, sampling=rebin_factor)
# Obnoxiously, the message attribute is only not set if a chi2 is not set as an output,
# so we need to add a junk variable to quite it.

# If the fit was not successful enough to estimate uncertainties for the fitted parameters,
# kill the program and do not save the fit.
if popt.message == 'Fit succeeded. Could not estimate error-bars.':
    print('Bad fit.')
    print('Set different fit guesses for '+ion['ion_wave'])
    print('These are set in this file: '+fit_directory+guesses_file+'\n\n')
    raise SystemExit('1')

# Print the reduced chi-squared to the terminal screen
print("\nreduced chi-squared: % 5.2f\n" % (popt.redchi))
# Remember that values near 1 indicate that the fit is really good.
# Values significantly below 1, e.g., 0.5, are not indicators of a "better fit";
# instead, these values could indicate that the uncertainties have been over estimated
# such that your error bars might be larger than they should be for some reason.
# If this occurs, do not artificially reduce the size of your error bars, but instead try
# to understand how and why this occured and try to correct that issue.

#### Save results ############################################################################

# Create save directory named ion_wave if it does exist
# Save all fit products to that directory.
if not os.path.exists(fit_directory+ion_wave_string):
    os.makedirs(fit_directory+ion_wave_string)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

#########################################################
# Switch to save to file instead of to the prompt
#orig_stdout = sys.stdout
#f = open(os.path.join(ion_wave_string,root_file_name+'.fit_params'), 'w')
#sys.stdout = f

# Prints the best fit parameters in velocity.
#dataset.print_results(velocity=True)
#print("\nreduced chi-squared: % 5.2f\n" % (popt.redchi))
#print("\nContinuum Polynomial Order: % 2d\n" % (dataset.cheb_order))
#sys.stdout = orig_stdout
#f.close()
# Close the file and set printing back to prompt
#########################################################

# Create PDF figure of the fit and residuals.
# Unfortunately, this does not show the individual components and there does not seem to be a flag to turn that on.
# This is the code causing the '_asarray.py:85: UserWarning: Warning: converting a masked element to nan.' error message
VoigtFit.output.plot_all_lines(dataset,plot_fit=True,\
    filename=os.path.join(fit_directory+ion_wave_string,root_file_name+".pdf"),show=False)

# Save the dataset to a Python save file?
dataset.save(filename=os.path.join(fit_directory+ion_wave_string,root_file_name+".hdf5"))

# Save the best fit parameters and also saves a python script for adding those best fit components
dataset.save_parameters(filename=os.path.join(fit_directory+ion_wave_string,root_file_name+".fit"))
# Unfortunately, the reduced chi-squared is not saved. Append the file to include this.
file=open(os.path.join(fit_directory+ion_wave_string,root_file_name+".fit"),"a+")
# Set file.write() equal to tmp so that it doesn't print the number of characters to the screen...
tmp=file.write("\nchi2: % 5.2f\n" % (popt.redchi))
# Add this line if you also want to add the polynomial order for a continuum fit.
# Skipping this as we are using already normalized data.
# tmp=file.write("\nContinuum Polynomial Order: % 2d\n" % (dataset.cheb_order))
#f.close()

# Save the continuum fit. Skipping this step as the dataset is already normalized.
# dataset.save_cont_parameters_to_file(root_file_name+".cont")

# Save the wavelength, normalized flux, normalized error, best fit profile, and mask flags.
# Added velocity as the 2nd column, but not sure that this will work propertly when multiple lines are fit.
fit_array=dataset.save_fit_regions(os.path.join(fit_directory+ion_wave_string,root_file_name+".reg"))

# The individual component fits with wavelength in the first column
# and normalized flux of each component in the subsequent columns.
# The velocity is stored in the ion["vel"] variable.
profiles=VoigtFit.output.save_individual_components(dataset=dataset, \
    filename=os.path.join(fit_directory+ion_wave_string,root_file_name+'.ind'),get_profiles=True, velocity=True)

####### plot all the individual line components #####################################

# Currently only plots in a python window, not yet saved to a PDF
# Does not include the residuals.
# Does not display masked regions.

fit_array=np.genfromtxt(fit_directory+ion_wave_string+'/'+root_file_name+".reg", \
    skip_header=6,names=['wave','vel','norm','norm_err','fit','mask'])
## Central wavelength
#l0=dataset.lines[ion_wave_string].l0
#l_center=l0*(dataset.redshift + 1.)
#fit_vel=(fit_array['wave'] - l_center)/l_center*c

## Read in the individual components.
## Note: I cannot find which variable this information is saved in.
## It only seems to be calculated and used when needed and not saved in a variable?
#components=np.genfromtxt(fit_directory+ion_wave_string+'/'+root_file_name+'.ind',skip_header=1)
## first column: components[:,0] # This is wavelength, can use ion['vel'] instead, which is the same length
## second column: components[:,1]
#num_components=len(components[0,:])-1

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
        axes.fill(xfill,yfill, alpha=0.75, edgecolor='none', facecolor='thistle',zorder=0)
print(mask_index)
print(mask_regions['vmax'])
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
# Plot a two pink lines in the residuals that represent 3 times the error from y=0 (not residuals ?? 3 times the error).
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
        cax.fill(xfill,yfill, alpha=0.75, edgecolor='none', facecolor='thistle',zorder=2,clip_on=True)

### Add titles of ion wave and the reduced chi squared

plt.title("  "+ion_string+' $\lambda$'+wave_string,size=10,loc='left')
plt.title("Reduced $\chi^2$: % 5.2f   " % (popt.redchi),size=10,loc='right')

plt.show()
