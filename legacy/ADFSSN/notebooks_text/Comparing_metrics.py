# ---------------------------------------------------------------------------
# LEGACY NOTEBOOK EXPORT (code + markdown only; outputs/figures stripped)
# Source:   Comparing metrics.ipynb
# Status:   tracked, clean
# SHA-256 of original .ipynb: a67573161c25a37b662537155136e73f718039feed8988daf8713a482031a3aa
# Cells:    5 (code 2, markdown 3, raw 0)
# Exported: 2026-10-28 by legacy/tools/export_notebook_text.ps1
# Format:   py:percent ('# %%' = code cell, '# %% [markdown]' = markdown cell)
# Data paths inside refer to the ORIGINAL repo layout; see legacy/README.md.
# ---------------------------------------------------------------------------

# %% [markdown]
# # Modules

# %%
# Modules
import scipy as sp
from scipy import sparse, signal, stats
from astropy import convolution as conv
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
import pandas as pd 
import time
import datetime
import calendar
import seaborn as sns
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

# %% [markdown]
# # Reading data

# %%
import glob
import pandas as pd
path ='C:/Users/usuario/Documents/Fisica/Investigacion/27-Boulder/ADFSSNGit/scripts/output/Run-2018-12-18-SSN3-2' # use your path
allFiles = glob.glob(path + "/*.csv")
pieces = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, usecols = [0,1,6,7,8,9,10,11], header=0)
    pieces.append(df)
    
data = pd.concat(pieces, axis=1)

fdifmreSth = []
fdifmneSth = []
fdifkSth = []
fdifmreMth = []
fdifmneMth = []
fdifkMth = []

# for i in range(0,208)
for i in range(0,160):
    alldata = data.loc[i]
    print(alldata)
    mreSth = alldata.iloc[[2,10,18,26,34,42,50,58,66]]
    mneSth = alldata.iloc[[3,11,19,27,35,43,51,59,67]]
    kSth = alldata.iloc[[4,12,20,28,36,44,52,60,68]]
    mreMth = alldata.iloc[[5,13,21,29,37,45,53,61,69]]
    mneMth = alldata.iloc[[6,14,22,30,38,46,54,62,70]]
    kMth = alldata.iloc[[7,15,23,31,39,47,55,63,71]]
    
    minmreSth = np.min(abs(mreSth))
    difmreSth = abs(mreSth)-minmreSth
    minmneSth = np.min(abs(mneSth))
    difmneSth = abs(mneSth)-minmneSth
    minkSth = np.min(abs(kSth))
    difkSth = abs(kSth)-minkSth
    
    minmreMth = np.min(abs(mreMth))
    difmreMth = abs(mreMth)-minmreMth
    minmneMth = np.min(abs(mneMth))
    difmneMth = abs(mneMth)-minmneMth
    minkMth = np.min(abs(kMth))
    difkMth = abs(kMth)-minkMth
    
    fdifmreSth.append(difmreSth)
    fdifmneSth.append(difmneSth)
    fdifkSth.append(difkSth)
    fdifmreMth.append(difmreMth)
    fdifmneMth.append(difmneMth)
    fdifkMth.append(difkMth)

finaldifmreSth = pd.concat(fdifmreSth, axis=1)
finaldifmneSth = pd.concat(fdifmneSth, axis=1)
finaldifkSth = pd.concat(fdifkSth, axis=1)
finaldifmreMth = pd.concat(fdifmreMth, axis=1)
finaldifmneMth = pd.concat(fdifmneMth, axis=1)
finaldifkMth = pd.concat(fdifkMth, axis=1)

finaldifmreSth['sumdif'] = finaldifmreSth.sum(axis=1)
finaldifmneSth['sumdif'] = finaldifmneSth.sum(axis=1)
finaldifkSth['sumdif'] = finaldifkSth.sum(axis=1)
finaldifmreMth['sumdif'] = finaldifmreMth.sum(axis=1)
finaldifmneMth['sumdif'] = finaldifmneMth.sum(axis=1)
finaldifkMth['sumdif'] = finaldifkMth.sum(axis=1)
print(finaldifmreSth)
print(finaldifmneSth)
print(finaldifkSth)
print(finaldifmreMth)
print(finaldifmneMth)
print(finaldifkMth)

# %% [markdown]
# 0 - 60/0.6
# 1 - 60/0.75
# 2- 60/0.9
# 3- 75
# 4-
# 5-
# 6- 90
# 7-
# 8-

