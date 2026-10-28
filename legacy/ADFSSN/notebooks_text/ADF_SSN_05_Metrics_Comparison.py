# ---------------------------------------------------------------------------
# LEGACY NOTEBOOK EXPORT (code + markdown only; outputs/figures stripped)
# Source:   ADF_SSN_05_Metrics_Comparison.ipynb
# Status:   tracked, clean
# SHA-256 of original .ipynb: 990534d0b50357b94a36624df2e7e9da5d8c2f6238f44a3dba13dd442fa07446
# Cells:    29 (code 27, markdown 2, raw 0)
# Exported: 2026-10-28 by legacy/tools/export_notebook_text.ps1
# Format:   py:percent ('# %%' = code cell, '# %% [markdown]' = markdown cell)
# Data paths inside refer to the ORIGINAL repo layout; see legacy/README.md.
# ---------------------------------------------------------------------------

# %% [markdown]
# # Modules

# %%
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}

# %%
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# %%
# Modules
import scipy as sp
from scipy import sparse, signal, stats
from astropy import convolution as conv
import numpy as np
import numpy.ma as ma
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import pandas as pd 
import time
import datetime
import calendar
import seaborn as sns
from copy import deepcopy
import json

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

# %% [markdown]
# # Reading data

# %%
path = 'D:/Mis Documentos/AAResearch/Codes/ADFSSN/test/MulL2Ov'

# EMD
dfEMD1 = pd.read_csv(path + '/A_O_NB50_Observer1_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
dfEMD2 = pd.read_csv(path + '/A_O_NB50_Observer2_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
dfEMD3 = pd.read_csv(path + '/A_O_NB50_Observer3_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')

dfEMD = pd.concat([dfEMD1, dfEMD2, dfEMD3])
dfEMD=dfEMD.reset_index(drop=True)

dfEMD.head()

# %%
dfEMD.shape[0]

# %%

y = np.array(list(map(lambda x: np.percentile(np.array(ast.literal_eval(dfL2['BestThSF'][x].replace('.',',').replace(',]',']'))),10), np.arange(0,dfL2.shape[0]))))
y

# %%

dfL2['TestTh'] = dfL2['RealThreshold']*np.nan

for i in range(0,dfL2.shape[0]):

    tmp = np.array(ast.literal_eval(dfL2['BestThSF'][i].replace('.',',').replace(',]',']')))

    ub = 120
    lb = 0

#     print(dfL2['Observer'][i])    

    for n in range(0,5):
        his, edges = np.histogram(tmp,bins=np.array([lb, (ub+lb)/2, ub]))
        inx = (his>1).nonzero()
#         print(his, his>1, edges, inx[0], len(inx[0]))
        
        if len(inx[0])>0:
        
            if inx[0][0]==0:
                ub = (ub+lb)/2
            else:
                lb = (ub+lb)/2
            
    dfL2.loc[i,'TestTh'] = (ub+lb)/2 

# %%
# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 1      # Number of horizontal panels
npv = 1      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                                
enges = np.arange(15,120,5)-2.5 
vmax = 20
    
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)   

ax1 = fig.add_axes([ppadh, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax1.hist2d(dfEMD['RealThreshold'],dfEMD['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax) 
ax1.plot([0,120],[0,120], linestyle='dotted')


ax2 = fig.add_axes([ppadh+pxx/fszh*frc, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax2.hist2d(dfEMDm1['RealThreshold'],dfEMDm1['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax) 
ax2.plot([0,120],[0,120], linestyle='dotted')


y = np.array(list(map(lambda x: np.percentile(np.array(ast.literal_eval(dfL2['BestThSF'][x].replace('.',',').replace(',]',']'))),10), np.arange(0,dfL2.shape[0]))))
y = dfL2['TestTh']

ax3 = fig.add_axes([ppadh, ppadv+pxy/fszv*frc, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax3.hist2d(dfL2['RealThreshold'],y, bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax) 
ax3.plot([0,120],[0,120], linestyle='dotted')

ax4 = fig.add_axes([ppadh+pxx/fszh*frc, ppadv+pxy/fszv*frc, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax4.hist2d(dfL2m1['RealThreshold'],dfL2m1['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax)
ax4.plot([0,120],[0,120], linestyle='dotted')

# %%
path = 'D:/Mis Documentos/AAResearch/Codes/ADFSSN/scripts/output/output/'

# EMD
dfEMD = pd.read_csv(path + 'TestL2/A_O_NB1_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
dfEMD.head()

# EMDmm1
dfEMDm1 = pd.read_csv(path + 'TestL220/A_O_NB1_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
dfEMDm1.head()

# L2
dfL2 = pd.read_csv(path + 'TestL225/A_O_NB1_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
dfL2.head()

# L2mm1
dfL2m1 = pd.read_csv(path + 'TestL230/A_O_NB1_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
dfL2m1.head()

# %%
# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 1      # Number of horizontal panels
npv = 1      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                                
enges = np.arange(20,120,10)-5 
vmax = 20
    
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)   

ax1 = fig.add_axes([ppadh, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax1.hist2d(dfEMD['RealThreshold'],dfEMD['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax) 
ax1.plot([0,120],[0,120], linestyle='dotted')


ax2 = fig.add_axes([ppadh+pxx/fszh*frc, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax2.hist2d(dfEMDm1['RealThreshold'],dfEMDm1['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax) 
ax2.plot([0,120],[0,120], linestyle='dotted')


ax3 = fig.add_axes([ppadh, ppadv+pxy/fszv*frc, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax3.hist2d(dfL2['RealThreshold'],dfL2['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax) 
ax3.plot([0,120],[0,120], linestyle='dotted')

ax4 = fig.add_axes([ppadh+pxx/fszh*frc, ppadv+pxy/fszv*frc, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')
ax4.hist2d(dfL2m1['RealThreshold'],dfL2m1['AvThreshold'], bins=[enges,enges], cmap=plt.cm.magma_r,cmin=1, vmin=1, vmax=vmax)
ax4.plot([0,120],[0,120], linestyle='dotted')

# %%
path ='D:/Mis Documentos/AAResearch/Codes/ADFSSN/runcsvs/' # use your path

PLH = [100]
PHH = [60, 75, 90]
QDH = [0]
ADH = [0.6, 0.75, 0.9]


PLL = [90]
PHL = [100]
QDL = [0.1, 0.25, 0.4]
ADL = [1.0]

# Reading reference
df = pd.read_csv(path + 'A_O_NB1_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
comObs = df['Observer'].values.copy() 
badObs = df['Observer'].values.copy()
badObs = badObs[np.logical_not(np.isfinite(df['mreSth']))]

# First pass to identify common observers
for ipl, pl in enumerate(PLL):
    for iph, ph in enumerate(PHL):
        for iqd, qd in enumerate(QDL):
            for iad, ad in enumerate(ADL):
                
                df = pd.read_csv(path + 'A_D_NB1_PL' + str(pl) + '_PH' + str(ph) + '_QD' + str(qd) + '_AD' + str(ad) + '_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
                comObs = np.intersect1d(comObs, df['Observer'].values, assume_unique=False, return_indices=False)
                
                tbadObs = df['Observer'].values.copy()
                tbadObs = tbadObs[np.logical_not(np.isfinite(df['mreSth']))]                
                badObs = np.concatenate((badObs, tbadObs))
                
                
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
                df = pd.read_csv(path + 'A_D_NB1_PL' + str(pl) + '_PH' + str(ph) + '_QD' + str(qd) + '_AD' + str(ad) + '_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last')
                comObs = np.intersect1d(comObs, df['Observer'].values, assume_unique=False, return_indices=False)

                tbadObs = df['Observer'].values.copy()
                tbadObs = tbadObs[np.logical_not(np.isfinite(df['mreSth']))]                
                badObs = np.concatenate((badObs, tbadObs))                
                
badObs = np.unique(badObs)
comObs = np.setdiff1d(comObs,badObs)
                
                
## Creating multi-index dataframe from reference

# Reading reference and keeping only common observers
df = pd.read_csv(path + 'A_O_NB1_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last').set_index('Observer')
df = df.loc[comObs,:]

# Create Multi-index
MxInx = pd.MultiIndex.from_product([[0],[0],[0],[0],comObs], names=['PL', 'PH', 'QD', 'AD','Observer'])
MxDf = pd.DataFrame(df.values, index=MxInx, columns = df.columns)

# Append Runs
for ipl, pl in enumerate(PLL):
    for iph, ph in enumerate(PHL):
        for iqd, qd in enumerate(QDL):
            for iad, ad in enumerate(ADL):
                
                df = pd.read_csv(path + 'A_D_NB1_PL' + str(pl) + '_PH' + str(ph) + '_QD' + str(qd) + '_AD' + str(ad) + '_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last').set_index('Observer')
                df = df.loc[comObs,:]
                
                MxInx = pd.MultiIndex.from_product([[pl],[ph],[qd],[ad],comObs], names=['PL', 'PH', 'QD', 'AD','Observer'])
                tmpDf = pd.DataFrame(df.values, index=MxInx, columns = df.columns)
                                
                MxDf = MxDf.append(tmpDf)
                
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
                df = pd.read_csv(path + 'A_D_NB1_PL' + str(pl) + '_PH' + str(ph) + '_QD' + str(qd) + '_AD' + str(ad) + '_Observer_ADF.csv', header=0).drop_duplicates(subset='Observer', keep='last').set_index('Observer')
                df = df.loc[comObs,:]
                
                MxInx = pd.MultiIndex.from_product([[pl],[ph],[qd],[ad],comObs], names=['PL', 'PH', 'QD', 'AD','Observer'])
                tmpDf = pd.DataFrame(df.values, index=MxInx, columns = df.columns)
                                
                MxDf = MxDf.append(tmpDf)                
                
                

# %%
MxDf

# %%
bins = np.arange(-1.3,1.3,.05)+0.025
top = 5
left = -1
right = 1

# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 3      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)                    
                
    
pl = 100
ph = 60
qd = 0
ad = 0.6

n = (nph*(iad)+iph)
ax1 = fig.add_axes([ppadh, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')

ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values), density = True, bins = bins)

print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad) + ' All Observers')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values)),decimals=3))


pl = 0
ph = 0
qd = 0
ad = 0    
    
ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values), density = True, bins = bins, color = 'r', alpha = 0.4)

print('Act/Obs All Observers')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values)),decimals=3))

ax1.set_title('All Observers')                
ax1.set_ylim(top = top, bottom = 0)
ax1.set_xlim(right = right, left = left)

# ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values), density = True, bins = bins, color = 'r', alpha = 0.4)
    

pl = 100
ph = 60
qd = 0
ad = 0.6
    
ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values))
ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  

n = (nph*(iad)+iph)
ax1 = fig.add_axes([ppadh + (pxx/fszh+ppadh2), ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b2')

ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask]), density = True, bins = bins)
ax1.set_title('Only observers with Th>30')                
ax1.set_ylim(top = top, bottom = 0)
ax1.set_xlim(right = right, left = left)
ax1.set_xticks(np.arange(-0.5,1,0.5))  
ax1.legend(['ADF = 1 for high AL', 'ADF = Act/Obs'], loc = 'upper left', ncol = 1, frameon = True, edgecolor = 'none')

print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad) + ' Th>30')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask])),decimals=3))




pl = 0
ph = 0
qd = 0
ad = 0   

ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values))
ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  


ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask]), density = True, bins = bins, color = 'r', alpha = 0.4)
ax1.yaxis.set_label_position("right")    
ax1.yaxis.tick_right()    
ax1.set_xticks(np.arange(-0.5,1,0.5))  

print('Ac/Obs Th>30')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask])),decimals=3))
    
    

# %%
bins = np.arange(0,2,.05)+0.025
top = 5
left = 0
right = 2

# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 3      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)                    
                
    
pl = 100
ph = 60
qd = 0
ad = 0.6

n = (nph*(iad)+iph)
ax1 = fig.add_axes([ppadh, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b1')

ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values), density = True, bins = bins)

print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad) + ' All Observers')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values)),decimals=3))


pl = 0
ph = 0
qd = 0
ad = 0    
    
ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values), density = True, bins = bins, color = 'r', alpha = 0.4)

print('Act/Obs All Observers')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values)),decimals=3))

ax1.set_title('All Observers')                
ax1.set_ylim(top = top, bottom = 0)
ax1.set_xlim(right = right, left = left)
ax1.set_xticks(np.arange(0.5,2,0.5))

# ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values), density = True, bins = bins, color = 'r', alpha = 0.4)
    

pl = 100
ph = 60
qd = 0
ad = 0.6
    
ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values))
ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  

n = (nph*(iad)+iph)
ax1 = fig.add_axes([ppadh + (pxx/fszh+ppadh2), ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b2')

ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values[ValMask]), density = True, bins = bins)
ax1.set_title('Only observers with Th>30')                
ax1.set_ylim(top = top, bottom = 0)
ax1.set_xlim(right = right, left = left)
# ax1.set_xticks(np.arange(-0.5,1,0.5))  
ax1.legend(['ADF = 1 for high AL', 'ADF = Act/Obs'], loc = 'upper left', ncol = 1, frameon = True, edgecolor = 'none')

print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad) + ' Th>30')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values[ValMask])),decimals=3))




pl = 0
ph = 0
qd = 0
ad = 0   

ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values))
ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  


ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values[ValMask]), density = True, bins = bins, color = 'r', alpha = 0.4)
ax1.yaxis.set_label_position("right")    
ax1.yaxis.tick_right()    
ax1.set_xticks(np.arange(0.5,2.5,0.5))  

print('Ac/Obs Th>30')
print(np.round(np.mean((MxDf.loc[([pl],[ph],[qd],[ad]),'KSth'].values[ValMask])),decimals=3))
    
    

# %%
bins = np.arange(-1.3,1.3,.05)+0.025
top = 5
left = -1
right = 1

# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 3      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)                    
                
                
# High
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
#                 ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'mneMth'].values))
#                 ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  

                ValMask = (MxDf.loc[([pl],[ph],[qd],[ad]),'RiseCount']+MxDf.loc[([pl],[ph],[qd],[ad]),'DecCount'])>5
                ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad) + ' Th>30')
                print(np.round(np.mean(np.abs(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask])),decimals=3))        
                print(np.round(np.mean(np.abs(MxDf.loc[([pl],[ph],[qd],[ad]),'mneMth'].values[ValMask])),decimals=3))        
                
                n = (nph*(iad)+iph)
                ax1 = fig.add_axes([ppadh + iph*(pxx/fszh+ppadh2), ppadv + iad*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
#                 ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask]), density = True, bins = bins)
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask]), density = True, bins = bins)
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneMth'].values[ValMask]), density = True, bins = bins, color = 'r', alpha = 0.4)

                ax1.set_title('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))                
                ax1.set_ylim(top = top, bottom = 0)
                ax1.set_xlim(right = right, left = left)
                

               
                

# %%
np.sum((MxDf.loc[([pl],[ph],[qd],[ad]),'RiseCount']+MxDf.loc[([pl],[ph],[qd],[ad]),'DecCount'])>2)

# %%
bins = np.arange(-1.3,1.3,.05)+0.025
top = 5
left = -1
right = 1

# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 3      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)                    
                
                
# High
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values))
                ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)  
                
                n = (nph*(iad)+iph)
                ax1 = fig.add_axes([ppadh + iph*(pxx/fszh+ppadh2), ppadv + iad*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values[ValMask]), density = True, bins = bins)
                ax1.set_title('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))                
                ax1.set_ylim(top = top, bottom = 0)
                ax1.set_xlim(right = right, left = left)
                

                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'mneSth'].values), density = True, bins = bins, color = 'r', alpha = 0.4)
               
                

# %%
MxDf['AbsmreSth'] = np.abs(MxDf['mreSth'])

minAbsmreSth = MxDf['AbsmreSth'].min(level='Observer')

# Referencia
MxDf.loc[([0],[0],[0],[0]),'AbsmreSthR'] = MxDf.loc[([0],[0],[0],[0]),'AbsmreSth'] - minAbsmreSth

# Low
print('Low Runs')
print('--------')
for ipl, pl in enumerate(PLL):
    for iph, ph in enumerate(PHL):
        for iqd, qd in enumerate(QDL):
            for iad, ad in enumerate(ADL):
                
                MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSthR'] = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSth'] - minAbsmreSth
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSthR'].values))
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                print(np.round(np.mean(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSthR'].values[ValMask]),decimals=2))

# High
print('')
print('High Runs')
print('--------')
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
                MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSthR'] = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSth'] - minAbsmreSth
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSthR'].values))
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                print(np.round(np.mean(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmreSthR'].values[ValMask]),decimals=2))

print(' ')                
print('Reference')
print('--------')
ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmreSthR'].values))
print(np.round(np.mean(MxDf.loc[([0],[0],[0],[0]),'AbsmreSthR'].values[ValMask]),decimals=2))

# %%
MxDf['AbsmneSth'] = np.abs(MxDf['mneSth'])
MxDf['Sign'] = np.sign(MxDf['mneSth'])

minAbsmneSth = MxDf['AbsmneSth'].min(level='Observer')

# Referencia
#MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'] = MxDf.loc[([0],[0],[0],[0]),'AbsmneSth'] - minAbsmneSth
MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'] = (MxDf.loc[([0],[0],[0],[0]),'AbsmneSth'] - minAbsmneSth)*MxDf.loc[([0],[0],[0],[0]),'Sign']

bins = np.arange(-1.3,1.3,.05)+0.025
top = 10
left = -0.7
right = 0.7

       




# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 3      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)  

# Low
print('')
print('Low Runs')
print('--------')
for ipl, pl in enumerate(PLL):
    for iph, ph in enumerate(PHL):
        for iqd, qd in enumerate(QDL):
            for iad, ad in enumerate(ADL):
                
#                 MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'] = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSth'] - minAbsmneSth
                MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'] = (MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSth'] - minAbsmneSth)*MxDf.loc[([pl],[ph],[qd],[ad]),'Sign']
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'].values))
                ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)        
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                print(np.round(np.mean(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'].values[ValMask]), decimals=2))
                
                n = (nph*(iqd)+ipl)
                ax1 = fig.add_axes([ppadh + iqd*(pxx/fszh+ppadh2), ppadv + ipl*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'].values[ValMask]), density = True, bins = bins)
                ax1.set_title('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))                
                ax1.set_ylim(top = top, bottom = 0)
                ax1.set_xlim(right = right, left = left)
                
                ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values))
                ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values))
                ValMask = np.logical_and(ValMask, MxDf.loc[([0],[0],[0],[0]),'AvThreshold']<30)    
                ax1.hist(np.vstack(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values[ValMask]), density = True, bins = bins, color = 'r', alpha = 0.4)
                



                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)                    
                
                
# High
print('')
print('High Runs')
print('--------')
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
#                 MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'] = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSth'] - minAbsmneSth
                MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'] = (MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSth'] - minAbsmneSth)*MxDf.loc[([pl],[ph],[qd],[ad]),'Sign']
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'].values))
                ValMask = np.logical_and(ValMask, MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']<30)
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                print(np.round(np.mean(np.abs(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'].values[ValMask])),decimals=3))
                
                n = (nph*(iad)+iph)
                ax1 = fig.add_axes([ppadh + iph*(pxx/fszh+ppadh2), ppadv + iad*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsmneSthR'].values[ValMask]), density = True, bins = bins)
                ax1.set_title('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))                
                ax1.set_ylim(top = top, bottom = 0)
                ax1.set_xlim(right = right, left = left)
                
                ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values))
                ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values))
                ValMask = np.logical_and(ValMask, MxDf.loc[([0],[0],[0],[0]),'AvThreshold']<30)    
                ax1.hist(np.vstack(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values[ValMask]), density = True, bins = bins, color = 'r', alpha = 0.4)
               
                
print('')
print('Reference')
print('--------')
ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values))
ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values))
ValMask = np.logical_and(ValMask, MxDf.loc[([0],[0],[0],[0]),'AvThreshold']<30)    
print(np.round(np.mean(np.abs(MxDf.loc[([0],[0],[0],[0]),'AbsmneSthR'].values[ValMask])),decimals=3))

# %%
np.sum(ValMask)

# %%
MxDf['AbsKSth'] = np.abs(MxDf['KSth']-1)
minAbsKSth = MxDf['AbsKSth'].min(level='Observer')


# Referencia
MxDf.loc[([0],[0],[0],[0]),'AbsKSthR'] = MxDf.loc[([0],[0],[0],[0]),'AbsKSth'] - minAbsKSth

# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 0.8    # Fraction of the panel devoted to histograms

nph = 1      # Number of horizontal panels
npv = 1      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)  


print('')                
print('Reference')
print('--------')
ValMask = np.isfinite(np.double(MxDf.loc[([0],[0],[0],[0]),'AbsKSthR'].values))
print(np.round(np.mean(MxDf.loc[([0],[0],[0],[0]),'AbsKSthR'].values[ValMask]),decimals=2))

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi) 
ax1 = fig.add_axes([ppadh, ppadv, pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))

ax1.hist(np.vstack(MxDf.loc[([0],[0],[0],[0]),'AbsKSthR'].values[ValMask]), density = True, bins = np.arange(0,1.3,.1))
ax1.set_title('Reference')                
ax1.set_ylim(top = 6, bottom = 0)
ax1.set_xlim(right = 1.3, left = 0)


# Size definitions
dpi = 300
pxx = 1000   # Horizontal size of each panel
pxy = pxx    # Vertical size of each panel
frc = 1    # Fraction of the panel devoted to histograms

nph = 3      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 50 #Vertical padding in pixels
padv2 = 200  #Vertical padding in pixels between panels
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
                
## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)  


# Low
print(' ')
print('Low Runs')
print('--------')
for ipl, pl in enumerate(PLL):
    for iph, ph in enumerate(PHL):
        for iqd, qd in enumerate(QDL):
            for iad, ad in enumerate(ADL):
                
                MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'] = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSth'] - minAbsKSth
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values))
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                print(np.round(np.mean(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values[ValMask]),decimals=2))

                n = (nph*(iqd)+ipl)
                ax1 = fig.add_axes([ppadh + iqd*(pxx/fszh+ppadh2), ppadv + ipl*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values[ValMask]), density = True, bins = np.arange(0,1.3,.1))
                ax1.set_title('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                ax1.set_xlim(right = 1.3, left = 0)
                ax1.set_ylim(top = 6, bottom = 0)
                

## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)                
                

    
# High
print('')
print('High Runs')
print('--------')
for ipl, pl in enumerate(PLH):
    for iph, ph in enumerate(PHH):
        for iqd, qd in enumerate(QDH):
            for iad, ad in enumerate(ADH):
                
                print(iph)
                
                MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'] = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSth'] - minAbsKSth
                ValMask = np.isfinite(np.double(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values))
                
                print('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                print(np.round(np.mean(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values[ValMask]),decimals=2))                

                n = (nph*(iad)+iph)
                ax1 = fig.add_axes([ppadh + iph*(pxx/fszh+ppadh2), ppadv + iad*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
                ax1.hist(np.vstack(MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values[ValMask]), density = True, bins = np.arange(0,1.3,.1))
                ax1.set_title('PL: ' + str(pl) + ' PH: ' + str(ph) + 'QD: ' + str(qd) + 'AD: ' + str(ad))
                ax1.set_xlim(right = 1.3, left = 0)
                ax1.set_ylim(top = 6, bottom = 0)
                

# %%
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi), dpi = dpi)   
ax1 = fig.add_axes([ppadh + iph*(pxx/fszh+ppadh2), ppadv + iad*(pxy/fszv+ppadv2), pxx/fszh*frc, pxy/fszv*frc], label= 'b'+str(n))
                
tmp = MxDf.loc[([pl],[ph],[qd],[ad]),'AvThreshold']   
    
ax1.hist(np.vstack(tmp))

# %%
tmp = MxDf.loc[([pl],[ph],[qd],[ad]),'AbsKSthR'].values[ValMask]

# %%
np.array(tmp)

# %%
np.arange(0,1.3,.1)

# %%
np.max(MxDf['AvThreshold'])

# %%


