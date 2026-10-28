# ---------------------------------------------------------------------------
# LEGACY NOTEBOOK EXPORT (code + markdown only; outputs/figures stripped)
# Source:   ADF_SSN_Plots.ipynb
# Status:   tracked, clean
# SHA-256 of original .ipynb: 791d1462d9b92846fe235d311719eb66f14fb477f77c49e2e6436467a967ffcf
# Cells:    11 (code 8, markdown 3, raw 0)
# Exported: 2026-10-28 by legacy/tools/export_notebook_text.ps1
# Format:   py:percent ('# %%' = code cell, '# %% [markdown]' = markdown cell)
# Data paths inside refer to the ORIGINAL repo layout; see legacy/README.md.
# ---------------------------------------------------------------------------

# %% [markdown]
# # Analysing SSN fits with differing ADF flags
# Sam Hollenbach, Andrés Muñoz-Jaramillo

# %% [markdown]
# # Modules, colors, and functions

# %%
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
  return false;
}

# %%
# Modules
import scipy as sp
from scipy import sparse, signal, stats
from scipy.stats import gaussian_kde
from astropy import convolution as conv
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import pandas as pd 
import time
import datetime
import calendar
import csv
from copy import deepcopy

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

%matplotlib inline 

import sys
sys.path.insert(1, r'functions')  # add to pythonpath
from detect_peaks import detect_peaks


# Color definitions
ClrS = (0.74, 0.00, 0.00)
ClrN = (0.20, 0.56, 1.00)

Clr = [(0.00, 0.00, 0.00),
      (0.31, 0.24, 0.00),
      (0.43, 0.16, 0.49),
      (0.32, 0.70, 0.30),
      (0.45, 0.70, 0.90),
      (1.00, 0.82, 0.67)]

# Font Size

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size'   : 21}

plt.rc('font', **font)

# Histogram and scatter plot
def plotHistSqrtSSN(ax, calRefT, calObsT, maxN):

    # Number of bins to use
    Nbins = np.sqrt(calRefT.shape[0])/5
    Nbins = 15
    
    
    calRefT = np.sqrt(calRefT+1)
    calObsT = np.sqrt(calObsT+1)
    
    #Edges and Centers
    edges = np.arange(1,np.ceil(maxN), (np.ceil(maxN))/Nbins)
    centers = (edges[1:edges.shape[0]] + edges[0:edges.shape[0]-1])/2
    
    # Average group number 
    ax.hist2d(calObsT, calRefT, bins=edges, cmap=plt.cm.magma_r,cmin=1)
    
    # Calculating Quantities for plot
    hist, xedges, yedges = np.histogram2d(calObsT, calRefT, bins=edges)
    xx, yy = np.meshgrid(centers, centers)

    meanY = np.divide(np.sum(np.multiply(hist,xx),axis=1),np.sum(hist,axis=1))
    alphaY = np.sum(hist,axis=1)/np.max(np.sum(hist,axis=1))*0.95+0.05
    
    meanX = np.divide(np.sum(np.multiply(hist,yy),axis=0),np.sum(hist,axis=0))
    alphaX = np.sum(hist,axis=0)/np.max(np.sum(hist,axis=0))*0.95+0.05     
    
    mx, my = np.meshgrid(meanY, meanY)
    
    sDY = np.divide(np.sum(np.multiply(hist,np.power(xx-my,2)),axis=1),np.sum(hist,axis=1))
    
    # Calculating quantities for assessment
    y = meanY
    x = centers
    
    x = x[np.isfinite(y)]
    y = y[np.isfinite(y)]
    

    
    # R squared
    yMean = np.mean(y)    
    SStot = np.sum(np.power(y-yMean,2)) 
    SSreg = np.sum(np.power(y-x,2))
    rSq   = (1 - SSreg/SStot)*sp.stats.pearsonr(x,y)[0]
    
    # Mean Residual
    mRes = np.mean(y-x)
    
    print(meanY)
    print(sDY)
    
    # Plot centers in Y
    for i in range(0, meanY.shape[0]):
        ax.errorbar(centers[i], meanY[i], sDY[i], color='k', zorder=11, alpha=alphaY[i])
        ax.scatter(centers[i], meanY[i], color='w', edgecolor='k', s=100, linewidths = 3, zorder=11, alpha=alphaY[i])

        
    # for i in range(0, meanY.shape[0]):
    #     ax.scatter(meanX[i], centers[i],  color='b', edgecolor='b', s=50, linewidths = 3, zorder=11, alpha=alphaY[i])

    ax.plot(edges, edges, '--'
             , color = Clr[4], linewidth = 3)

    ax.text(0.5, 0.95,'$PCCR$=' + str(np.round(rSq,decimals=2)) + '  MR=' + str(np.round(mRes,decimals=2)), horizontalalignment='center', verticalalignment='center',transform=ax1.transAxes)


# Histogram Outline
def histOutline(dataIn, *args, **kwargs):
    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return (bins, data)

# %% [markdown]
# # Reading CSV Data
# Read CSV file for each flag

# %%
### READ IN MAIN OBSERVER DATA ###
dat_AM = pd.read_csv('out/A_M_Observer_ADF.csv', quotechar = '"', encoding = 'utf-8',header = 0)
dat_AM = dat_AM[np.isfinite(dat_AM['R2'])]
dat_QM = pd.read_csv('out/Q_M_Observer_ADF.csv', quotechar = '"', encoding = 'utf-8',header = 0)
dat_QM = dat_QM[np.isfinite(dat_QM['R2'])]
dat_AO = pd.read_csv('out/A_O_Observer_ADF.csv', quotechar = '"', encoding = 'utf-8',header = 0)
dat_AO = dat_AO[np.isfinite(dat_AO['R2'])]
dat_QO = pd.read_csv('out/Q_O_Observer_ADF.csv', quotechar = '"', encoding = 'utf-8',header = 0)
dat_QO = dat_QO[np.isfinite(dat_QO['R2'])]

# Show columns
print(dat_AM.columns.values)

### READ IN BEHAVIOR DATA ###
read_AM = csv.reader(open('input_data/A_M_behaviors.txt','r'), delimiter=' ')
behaviors_AM = {}
for l in read_AM:
    if len(l) == 2:
        behaviors_AM[l[0]] = l[1]
        
read_AO = csv.reader(open('input_data/A_O_behaviors.txt','r'), delimiter=' ')
behaviors_AO = {}
for l in read_AO:
    if len(l) == 2:
        behaviors_AO[l[0]] = l[1]
        
read_QO = csv.reader(open('input_data/Q_O_behaviors.txt','r'), delimiter=' ')
behaviors_QO = {}
for l in read_QO:
    if len(l) == 2:
        behaviors_QO[l[0]] = l[1]
        
read_QM = csv.reader(open('input_data/Q_M_behaviors.txt','r'), delimiter=' ')
behaviors_QM = {}
for l in read_QM:
    if len(l) == 2:
        behaviors_QM[l[0]] = l[1]

#print(dat_AM)

# %%
# Distribution Plots for Observer

# Size definitions
dpi = 300
pxx = 3000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 0.8    # Fraction of the panel devoted to histograms

nph = 1      # Number of horizontal panels
npv = 1      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 50 #Horizontal padding in pixels at the edge of the figure
padh2 = 0 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in inches
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in inches

# Conversion to relative unites
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszv     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszv    #Horizontal padding between panels in relative units



#Normalize vals on 0 to 1 scale
def normalize_vals(x, log = False):
    return ((x - min(x)) / (max(x) - min(x)))

xvar = 'Avg.Res'
yvar = 'R2'

# Create a dictionary with necessary data for each flag
data = {'AM' : (dat_AM, Clr[4], behaviors_AM), 'AO': (dat_AO, Clr[3], behaviors_AO),
        'QM' : (dat_QM, Clr[2], behaviors_QM), 'QO' : (dat_QO, Clr[5], behaviors_QO)}


# Create a horizonal histogram
def do_hist_horiz(ax, x, y, col, bins):
    ax.hist(y, bins = bins,  color=col, rwidth = 0.5, alpha=.6, orientation = 'horizontal',
            normed = True);

# Create a histogram
def do_hist(ax, x, y, col, bins):
    ax.hist(y, bins = bins, color=col, rwidth = 0.5, alpha=.6, normed = True);

def do_scatter(ax, xy_dict, behaviors, col, label='R2'):
    # Create list of x,y vals for each specified type
    xy_b = []
    xy_bb = []
    xy_s = []
    xy_bs = []
    xy_norm = []
    for i, xy in xy_dict.items():
        if i in behaviors['B']:
            xy_b.append(xy)
        elif i in behaviors['BB']:
            xy_bb.append(xy)
        elif i in behaviors['S']:
            xy_s.append(xy)
        elif i in behaviors['BS']:
            xy_bs.append(xy)
        else:
            xy_norm.append(xy)
            
    # Plot each specified type with a different marker. Interesting points are more opaque.
    ax.scatter([x[0] for x in xy_b], [x[1] for x in xy_b], marker='s', color=col, edgecolor="k",
               alpha = 0.9, s=110, linewidths = 2, label = '{} : Some Bifurcation'.format(label))
    
    ax.scatter([x[0] for x in xy_bb], [x[1] for x in xy_bb], marker='p', color=col, edgecolor="k",
               alpha = 0.9, s=150, linewidths = 2, label = '{} : Significant Bifurcation'.format(label))
    
    ax.scatter([x[0] for x in xy_s], [x[1] for x in xy_s], marker='v', color=col, edgecolor="k",
               alpha = 0.9, s=110, linewidths = 2, label = '{} : Split'.format(label))
    
    ax.scatter([x[0] for x in xy_bs], [x[1] for x in xy_bs], marker='^', color=col, edgecolor="k",
               alpha = 0.9, s=120, linewidths = 2, label = '{} : Bifurcation and split'.format(label))
    
    # Plot remaining normal points, more transparent
    ax.scatter([x[0] for x in xy_norm], [x[1] for x in xy_norm], color=col, edgecolor="k", alpha = 0.5, s=100, linewidths = 2, label = label)

    
   

    
# General function to make plot depending on specified x and y variables from main dataframe
def make_plot(xvar, yvar = 'R2', include_data = data.keys(), flag_plot = False, norm = False, full_x = False, full_y = False):
    
    if flag_plot:
        if xvar is not yvar:
            raise ValueError("X and Y variables must be the same in a flag plot")
        if len(include_data) is not 2:
            raise ValueError("Please specify exactly 2 flag combinations in \'include_data\' to use for a flag plot")
    
    ## Start Figure
    fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))


    # Top Distribution
    axd = fig.add_axes([ppadh, ppadv + pxy/fszv*frc, pxx/fszh*frc, pxy/fszv*(1-frc)])

    # Define standard limits
    lims = {'left': -0.05, 'right': 1.05, 'bottom' : -0.05,'top': 1.05}
    
    # Find actual lower and upper limits from all data
    max_x = 0
    min_x = 0
    max_y = 0
    min_y = 0
    for key, val in data.items():
        if key not in include_data:
            continue
        if full_x:
            mi_x = min(val[0][xvar])
            ma_x = max(val[0][xvar])
            if ma_x > max_x:
                max_x = ma_x
            if mi_x < min_x:
                min_x = mi_x
        if full_y:
            mi_y = min(val[0][yvar])
            ma_y = max(val[0][yvar])
            if ma_y > max_y:
                max_y = ma_y
            if mi_y < min_y:
                min_y = mi_y        
    if full_x:
        lims['left'] = min_x - .05
        lims['right'] = max_x + .05
    if full_y:
        lims['bottom'] = min_y - .05
        lims['top'] = max_y + .05
        
    
    # Axes properties
    axd.set_xlim(left = lims['left'], right = lims['right']);
    axd.set_axis_off()

    # Right Distribution
    ax2 = fig.add_axes([ppadh + pxx/fszh*frc, ppadv, pxx/fszh*frc*(1-frc), pxy/fszv*frc])

    # Axes properties
    ax2.set_ylim(bottom = lims['bottom'], top = lims['top']);
    ax2.set_axis_off()

    # Scatter Plot
    ax1 = fig.add_axes([ppadh, ppadv, pxx/fszh*frc, pxy/fszv*frc], sharex=axd)
    
    
    b_ = []
    bb_ = []
    s_ = []
    bs_ = [] 
    
    x = []
    y = []
    
    # Iterate over different datasets (flags)
    for key, val in data.items():
        if key not in include_data:
            continue
            
        # Find observers with matching tags from {}_{}_behaviors file
        if not flag_plot:
            b_ = []
            bb_ = []
            s_ = []
            bs_ = []
        for observer_id in val[0]['Observer']:
            for i, b in val[2].items():
                if int(i) == observer_id:
                    if b == 'B':
                        b_.append(observer_id)
                    elif b == 'BB':
                        bb_.append(observer_id)
                    elif b == 'S':
                        s_.append(observer_id)
                    elif b == 'BS':
                        bs_.append(observer_id)
                
        # Create dictionary of x,y vals with observer id as key
        if flag_plot:
            if include_data.index(key) == 0:
                ids_x = val[0]['Observer']
                x = list(val[0][xvar])
                if not y:
                    continue
            else:
                ids_y = val[0]['Observer']
                y = list(val[0][yvar])
                if not x:
                    continue
        else:
            x = list(val[0][xvar])
            y = list(val[0][yvar])
        ids = val[0]['Observer']
        xy_dict = {}
        
        if flag_plot:
            for index_x, obs_id_x in enumerate(ids_x):
                for index_y, obs_id_y in enumerate(ids_y):
                    if obs_id_x == obs_id_y:
                        xy_dict[obs_id_x] = (x[index_x], y[index_y])
        else:
            for index, obs_id in enumerate(ids):
                xy_dict[obs_id] = (x[index], y[index])

        # Normalize x val to 0->1 if norm flag is set
        if norm:
            x = normalize_vals(x)
             
        # Compute estimated PDF and CDF for x and y distributions
        y_sm = np.linspace(min(x), max(x), 200)
        density_x = gaussian_kde(y)
        density_x.covariance_factor = lambda : .08
        density_x._compute_covariance()
        
        x_sm = np.linspace(min(y), max(y), 200)
        density_y = gaussian_kde(x)
        density_y.covariance_factor = lambda : .08
        density_y._compute_covariance()
        
        pdf_x = density_x(x_sm) / max(density_x(x_sm))
        cdf_x = np.cumsum(pdf_x / pdf_x.sum())
        
        pdf_y = density_y(y_sm) / max(density_y(y_sm))
        cdf_y = np.cumsum(pdf_y / pdf_y.sum())
        
        # Plot PDF on axes
        #ax2.plot(pdf_x, x_sm, ':', alpha = 0.5, color = val[1])
        #axd.plot(y_sm, pdf_y, ':', alpha = 0.5, color = val[1])
        
        # Plot CDF on axes with opposing axis CDF superimposed at lower alpha
        ax2.plot(cdf_x, x_sm, color = val[1])
        ax2.plot(cdf_y, y_sm, ':', alpha = 0.5, color = val[1])
        axd.plot(y_sm, cdf_y, color = val[1])
        axd.plot(x_sm, cdf_x, ':', alpha = 0.5, color = val[1])
        
        # Create scatter points for each flag
        label = key
        if flag_plot:
            label = '{} in {} vs {}'.format(xvar, include_data[0], include_data[1])
        
        do_scatter(ax1, xy_dict, {'B': b_, 'BB': bb_, 'S': s_, 'BS': bs_}, val[1], label = label)

    # Plot x=y line
    ax1.plot([lims['left'],lims['right']], [lims['left'],lims['right']], 'k--', alpha=0.75, zorder=0)
    
    # Create legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)


    # Set axis labels
    xlab = xvar
    ylab = yvar
    
    if flag_plot:
        xlab = '{} ({})'.format(xlab, include_data[0])
        ylab = '{} ({})'.format(ylab, include_data[1])
    
    if norm:
        xlab = '{} {}'.format("Normalized", xlab)
    ax1.set_xlabel(xlab)
    
    ax1.set_ylabel(ylab)
    #ax1.text(1.02, 0.52,'Observation day fraction', horizontalalignment='center', transform = ax1.transAxes, 
             #rotation = 'vertical', verticalalignment = 'center')
    
    # Axes properties
    #ax1.text(0.5, 1.01,'Active day fraction', horizontalalignment='center', transform = ax1.transAxes)
    ax1.set_xlim(left = lims['left'], right = lims['right'])
    ax1.set_ylim(bottom = lims['bottom'], top = lims['top']);

    
### MAKE PLOTS ###
make_plot('R2', 'R2', flag_plot = True, include_data = ('AM', 'QM'))
make_plot('R2', 'R2', flag_plot = True, include_data = ('QO', 'QM'))
#make_plot('R2', 'R2', flag_plot = True, include_data = ('AO', 'QM'))
#make_plot('R2OO', 'R2OO', flag_plot = True, include_data = ('QM', 'AO'))
#make_plot('R2OO', 'R2OO', flag_plot = True, include_data = ('QM', 'AO'))
#make_plot('R2DT', 'R2OO', full_x = True, full_y = True)
make_plot('R2', 'R2OO')
#make_plot('R2OO', 'R2')
#make_plot('Avg.ResDT', 'R2DT', full_x = True)
#make_plot('SDThreshold', 'R2', full_x = True)
#make_plot('AvThreshold', 'SDThreshold', full_x = True, full_y = True)
#make_plot('AvThreshold', include_data = ['QM', 'QO','AO', 'AM'], norm = True)

# %%

#FLAGS and VARIABLE(S)
flags = ('AM', 'AO', 'QM', 'QO')
var = 'R2'

# Size definitions
dpi = 300
pxx = 1500   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = len(flags)     # Number of horizontal panels
npv = len(flags)     # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 100  #Vertical padding in pixels between panels
padh  = 50 #Horizontal padding in pixels at the edge of the figure
padh2 = 100 #Horizontal padding in pixels between panels

padv  = 0
padv2 = 0
padh  = 0
padh2 = 0

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in inches
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in inches

# Conversion to relative unites
#ppadv  = padv/fszv     #Vertical padding in relative units
#ppadv2 = padv2/fszv    #Vertical padding in relative units
#ppadh  = padh/fszv     #Horizontal padding the edge of the figure in relative units
#ppadh2 = padh2/fszv    #Horizontal padding between panels in relative units
ppadv  = 0
ppadv2 = 0
ppadh  = 0
ppadh2 = 0


## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi, frameon=False)
#plt.suptitle("X-flag - Y-flag distribution")

# Reference axis object to share with others
x_ax_ref = None

# Store reults and find optimal limits
results = {}
xlim = 0
for i in range(0, nph):
    for j in range(0,npv):
        if i == j:
            continue 
        dat_1 = data[flags[i]][0]
        dat_2 = data[flags[j]][0]
        df = pd.merge(dat_1, dat_2, on='Observer')
        res = df['{}_x'.format(var)] - df['{}_y'.format(var)]
        
        results[i*nph + j] = res
        if max(res) > xlim:
            xlim = max(res)
        if abs(min(res)) > xlim:
            xlim = abs(min(res))
xlim += 0.05

# Define bins array
bins = np.linspace(-xlim, xlim, 18)
bin_size = (bins[1]-bins[0])
bins += bin_size/2

for i in range(0, nph):
    for j in range(0,npv):
        
        # Axis object
        ax1 = fig.add_axes([ppadh + i*(pxx/fszh+ppadh2), ppadv + j*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b')
        
        # Ticks and labels
        if i == nph-1:
            ax1.set_yticks([]) 
            ax1.set_ylabel(flags[j], fontsize=24)
            ax1.yaxis.set_label_position("right")
        elif i > 0:
            ax1.get_yaxis().set_visible(False)
        if j == npv-1:
            ax1.set_xticks([]) 
            ax1.set_xlabel(flags[i], fontsize=24)
            ax1.xaxis.set_label_position("top")
        elif j > 0:
            ax1.get_xaxis().set_visible(False)
        
        if i == 0:
            ax1.set_ylabel('Normalized Histogram Counts', fontsize=18)
            ax1.yaxis.set_label_position("left")
        if j == 0:
            ax1.set_xlabel('Difference in {}'.format(var), fontsize=18)
            ax1.xaxis.set_label_position("bottom")
        
        if x_ax_ref is None:
            ax1.set_xlim([-xlim, xlim])
            x_ax_ref = ax1
        else:
            ax1.get_shared_x_axes().join(ax1, x_ax_ref)
        
        # Draw vertical dashed line at 0
        ax1.axvline(0, color='k', linestyle='dashed', alpha = 0.8, linewidth=1, zorder=0)
        
        # Skip if using the same two falgs
        if i == j:
            continue 
        
        # Get results
        res = results[i*nph + j] 
        res = [x for x in res if x != 0.0]
        
        # Main histgram
        counts, bins = np.histogram(res, bins=bins)
        counts = list(counts)
        s = float(max(counts))
        counts = [x/s for x in counts]
        ax1.bar(bins[:-1], counts, width = bin_size, align='edge',  color = data[flags[i]][1], alpha = 0.6)
         
        # Red/green fraction lines
        neg = [x for x in res if x < 0]
        pos = [x for x in res if x > 0]
        pos_frac = float(len(pos))/(len(pos)+len(neg))
        neg_frac = float(len(neg))/(len(pos)+len(neg))
        ax1.plot([-xlim,0], [neg_frac,neg_frac], 'r-', alpha = 0.8, linewidth=3, zorder=0)
        ax1.plot([0,xlim], [pos_frac,pos_frac], 'g-', alpha = 0.8, linewidth=3, zorder=0)
        
        # CDF function
        cumulative = np.cumsum(counts)
        cumulative *= max(counts)/max(cumulative)
        ax1.plot(np.linspace(-xlim+bin_size, xlim+bin_size, len(cumulative)), cumulative, '--', color = data[flags[i]][1])
        
       

# %%


# %%


# %%


