# ---------------------------------------------------------------------------
# LEGACY NOTEBOOK EXPORT (code + markdown only; outputs/figures stripped)
# Source:   AMJ_JMV_MM_Perspective.ipynb
# Status:   untracked (exists only in the original working copy)
# SHA-256 of original .ipynb: 43f463524584b478f3663727b9889a2cb5c725fc6beaff179d5fc6e9a40756da
# Cells:    69 (code 51, markdown 18, raw 0)
# Exported: 2026-10-28 by legacy/tools/export_notebook_text.ps1
# Format:   py:percent ('# %%' = code cell, '# %% [markdown]' = markdown cell)
# Data paths inside refer to the ORIGINAL repo layout; see legacy/README.md.
# ---------------------------------------------------------------------------

# %%
%%javascript
IPython.OutputArea.prototype._should_scroll = function(lines) {
  return false;
}

# %% [markdown]
# ## Setup

# %%
# Modules
import scipy as sp
from scipy import sparse, signal, stats, interpolate
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
import networkx as nx

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

%matplotlib inline 

import sys
sys.path.insert(1, r'functions')  # add to pythonpath

figDPI = 600

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
        'size'   : 30}

plt.rc('font', **font)

# %% [markdown]
# # Reading observations and observers
#
# The files used were downloaded from [SILSO's Group Number page](http://www.sidc.be/silso/groupnumberv3).  The current version used is JV_V1-12, but turned into a .csv file.  The original file contains leap years that should not be there.  February 29th for those years were removed by hand (1700, 1800, 1900).
#
# ## Reading observations
#
# We recast the number of groups as a float so that we can use NaNs for days with missing observations.

# %%
GN_Dat = pd.read_csv('input_data/GNObservations_JV_V1.22.csv', quotechar = '"', encoding = 'ansi',header = 15)

GN_Dat['GROUPS'] = GN_Dat['GROUPS'].astype(float)
GN_Dat

# %% [markdown]
# ## Reading observers
#
# We create a variable to store each unique observer.  **NOTE THAT CURRENTLY EACH STATION HAS _ONLY_ ONE OBSERVER SO WE DO ALL OUR LOGICAL OPERATIONS USING THE 'STATION' FIELD**.
#
# We also remove the '0' station which indicates a day without observations.
#
# The **_print_** statement on this cell can be use for reference as it shows the station number associated with each observer.

# %%
GN_Obs = pd.read_csv('input_data/GNObservers_JV_V1.22.csv', quotechar = '"', encoding = 'ansi')
print(GN_Obs[['INITIAL','FINAL','STATION','OBSERVER','TOT.OBS']].to_string())

UnObs = sp.unique(GN_Dat.STATION)
UnObs = UnObs[UnObs>0]

# %% [markdown]
# ## Reading Sunspot Group Number Data

# %%
JV_Dat = pd.read_csv('input_data/Vaquero_Uncertainty.csv', quotechar = '"', encoding = 'ansi')
# Adding 0.5 years to Jose's data to match the other people's standards
JV_Dat['Year'] = JV_Dat['Year']+0.5
# Combining limits to create a single S
JV_Dat['S'] = (JV_Dat['S2'] + JV_Dat['S1'])/2
JV_Dat = JV_Dat.loc[:,['Year','G','S','S1','S2']]

# Interpolating Nadya's data to half years
# High
NZH_Dat = pd.read_csv('input_data/Zolotova_high.csv', quotechar = '"', encoding = 'ansi')
NZH_Dat = NZH_Dat.sort_values('Date')

NZ_int = interpolate.splrep(NZH_Dat['Date'].values, NZH_Dat['G'].values, s=1)
yr = np.arange(np.floor(np.min(NZH_Dat['Date'])),np.ceil(np.max(NZH_Dat['Date'])))+0.5
G = interpolate.splev(yr, NZ_int)
# Create new Zolotova database
NZH_Dat = pd.DataFrame(data={'Date': yr, 'G': G})

# Low
NZL_Dat = pd.read_csv('input_data/Zolotova_low.csv', quotechar = '"', encoding = 'ansi')
NZL_Dat = NZL_Dat.sort_values('Date')

NZ_int = interpolate.splrep(NZL_Dat['Date'].values, NZL_Dat['G'].values, s=7)
yr = np.arange(np.floor(np.min(NZL_Dat['Date'])),np.ceil(np.max(NZL_Dat['Date'])))+0.5
G = interpolate.splev(yr, NZ_int)
# Create new Zolotova database
NZL_Dat = pd.DataFrame(data={'Date': yr, 'G': G})

# Mixed
yr = np.intersect1d(NZH_Dat['Date'],NZL_Dat['Date'])
G = (NZH_Dat.loc[np.logical_and(NZH_Dat['Date']>=np.min(yr),NZH_Dat['Date']<=np.max(yr)),'G'] +
     NZL_Dat.loc[np.logical_and(NZL_Dat['Date']>=np.min(yr),NZL_Dat['Date']<=np.max(yr)),'G'] )/2

S = (NZH_Dat.loc[np.logical_and(NZH_Dat['Date']>=np.min(yr),NZH_Dat['Date']<=np.max(yr)),'G'] -
     NZL_Dat.loc[np.logical_and(NZL_Dat['Date']>=np.min(yr),NZL_Dat['Date']<=np.max(yr)),'G'] )/2

MS = np.mean(np.abs((NZH_Dat.loc[np.logical_and(NZH_Dat['Date']>=np.min(yr),NZH_Dat['Date']<=1670),'G'] -
     NZL_Dat.loc[np.logical_and(NZL_Dat['Date']>=np.min(yr),NZL_Dat['Date']<=1670),'G'] )/2))

S[S<MS] = MS

NZM_Dat = pd.DataFrame(data={'Date': yr, 'G': G, 'S':G*0+S})



SS_Dat = pd.read_csv('input_data/GNbb2_y.csv', quotechar = '"', encoding = 'ansi')
# Removing zero uncertainties
SS_Dat.loc[SS_Dat['S']==0,'S'] = np.min(SS_Dat.loc[SS_Dat['S']!=0,'S'])

U_Dat = pd.read_csv('input_data/GNiu_y2.csv', quotechar = '"', encoding = 'ansi')
U_Dat = U_Dat.loc[:,['Year','G','S']]
# Removing zero uncertainties
U_Dat.loc[U_Dat['S']==0,'S'] = np.min(U_Dat.loc[U_Dat['S']!=0,'S'])



HS_Dat = pd.read_csv('input_data/GNhs_y.csv', quotechar = '"', encoding = 'ansi')
# Changing the uncertainty of HS for the one in SS
HS_Dat['S'] = SS_Dat.loc[0:HS_Dat.shape[0],'S']

LO_Dat = pd.read_csv('input_data/Lockwood_etal_2014.csv', quotechar = '"', encoding = 'ansi')
# Changing the uncertainty of HS for the one in SS
LO_Dat['S'] = SS_Dat.loc[0:LO_Dat.shape[0],'S']
# Re-scaling to group number
LO_Dat['G'] = LO_Dat['G']/12.08


WSN_Dat = pd.read_csv('input_data/SN_y_tot_V2.0.csv', quotechar = '"', encoding = 'ansi')
WSN_Dat = WSN_Dat.loc[:,['Year','G','S']]
# Removing zero uncertainties
WSN_Dat.loc[WSN_Dat['S']<=0,'S'] = np.min(WSN_Dat.loc[WSN_Dat['S']!=0,'S'])
# Re-scaling to group number
WSN_Dat['G'] = WSN_Dat['G']/20
WSN_Dat['S'] = WSN_Dat['S']/20

Ch_Dat = pd.read_csv('input_data/Chatzistergos.csv', quotechar = '"', encoding = 'ansi')
Ch_Dat['S'] = (Ch_Dat['S2'] + Ch_Dat['S1'])/2

IM_Dat = pd.read_csv('input_data/Ivanov_Miletsky.csv', quotechar = '"', encoding = 'ansi')
IM_Dat['Year'] = IM_Dat['Year']+0.5


CL_Dat = pd.read_csv('input_data/Cliver_ling_2016.csv', quotechar = '"', encoding = 'ansi')
CL_Dat['Year'] = CL_Dat['Year']+0.5
CL_Dat['G'] = CL_Dat['RGC']/12.08

# Changing the uncertainty of CL for the one in SS
CL_Dat['S'] = SS_Dat.loc[231:370,'S'].values
U_Dat

# %% [markdown]
# ## Reading Butterfly Diagram Data

# %%
BflyMM = pd.read_csv('input_data/Sunspot_Latitudes_MM.csv', quotechar = '"', encoding = 'ansi',header=14)
BflyMod = pd.read_csv('input_data/composite_bfly_all.csv', quotechar = '"', encoding = 'ansi')
BflyMod['ORDINAL'] = BflyMod.apply(lambda x: datetime.date(int(x['YEAR']),int(x['MONTH']),int(x['DAY'])).toordinal(),axis=1)

BflyMM1 = pd.read_csv('input_data/Bfly_Gissendi_Malapert_Macgraf.csv', quotechar = '"', encoding = 'ansi')

BflyHrt = pd.read_csv('input_data/Bfly_harriot.csv', quotechar = '"', encoding = 'ansi')
BflyHrt['ORDINAL'] = BflyHrt.apply(lambda x: datetime.date(int(x['Year']),int(x['Month']),int(x['Day'])).toordinal(),axis=1)
BflyHrt['FRACYEAR'] = BflyHrt.apply(lambda x: x['Year'].astype(int)
                                            + (  datetime.date(x['Year'].astype(int),x['Month'].astype(int),x['Day'].astype(int)).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                            / (  datetime.date(x['Year'].astype(int)+1,1,1).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                  ,axis=1)


# BflyGal = pd.read_csv('input_data/Bfly_Galileo.csv', quotechar = '"', encoding = 'ansi')
BflyGal = pd.read_csv('input_data/Bfly_Galileo_Zolotova.csv', quotechar = '"', encoding = 'ansi')

BflyGal['ORDINAL'] = BflyGal.apply(lambda x: datetime.date(int(x['Year']),int(x['Month']),int(x['Day'])).toordinal(),axis=1)
BflyGal['FRACYEAR'] = BflyGal.apply(lambda x: x['Year'].astype(int)
                                            + (  datetime.date(x['Year'].astype(int),x['Month'].astype(int),x['Day'].astype(int)).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                            / (  datetime.date(x['Year'].astype(int)+1,1,1).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                  ,axis=1)

BflyHvl = pd.read_csv('input_data/Bfly_hevelius_JV.csv', quotechar = '"', encoding = 'ansi')

BflySchn = pd.read_csv('input_data/Bfly_scheiner.csv', quotechar = '"', encoding = 'ansi')
BflySchn['ORDINAL'] = BflySchn.apply(lambda x: datetime.date(int(x['Year']),int(x['Month']),int(x['Day'])).toordinal(),axis=1)
BflySchn['FRACYEAR'] = BflySchn.apply(lambda x: x['Year'].astype(int)
                                            + (  datetime.date(x['Year'].astype(int),x['Month'].astype(int),x['Day'].astype(int)).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                            / (  datetime.date(x['Year'].astype(int)+1,1,1).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                  ,axis=1)


BflyStdc = pd.read_csv('input_data/Bfly_staudacher.csv', quotechar = '"', encoding = 'ansi')
BflyStdc['ORDINAL'] = BflyStdc.apply(lambda x: datetime.date(int(x['Year']),int(x['Month']),int(x['Day'])).toordinal(),axis=1)
BflyStdc['FRACYEAR'] = BflyStdc.apply(lambda x: x['Year'].astype(int)
                                            + (  datetime.date(x['Year'].astype(int),x['Month'].astype(int),x['Day'].astype(int)).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                            / (  datetime.date(x['Year'].astype(int)+1,1,1).toordinal()
                                               - datetime.date(x['Year'].astype(int),1,1).toordinal() )
                                  ,axis=1)

BflyGal

# %% [markdown]
# ## Pre-allocating Variables
#
# We create a variable to store each unique observer.  **NOTE THAT CURRENTLY EACH STATION HAS _ONLY_ ONE OBSERVER SO WE DO ALL OUR LOGICAL OPERATIONS USING THE 'STATION' FIELD**.
#
# We also remove the '0' station which indicates a day without observations.

# %%
UnObs = sp.unique(GN_Dat.STATION)
UnObs = UnObs[UnObs>0]
UnObs.shape

# %% [markdown]
# ### Day adjacency tolerance
#
# Since the amount of groups visible in the Sun don't change radically from day to day, we specify a window of tolerance in which two observations are considered to be _"simultaneous"_.  First priority is given to observations taken on the same calendar date, then to observations within one day are considered, then within two days, and so on until reaching the size of the tolerance window.  The process is ran simultaneously for prior and following days.

# %%
DayTol = 2

# %% [markdown]
# ### Minimum days of overlap
#
# In order to establish a link between two observers in the network, we require to have minimum amount of days of overlap (considering as overlapping days days within DayTol of each other).

# %%
MinOvr = 60

# %% [markdown]
# ## Creating the adjacency matrix
#
# Now we go through all observers **_(stations)_** executing the following algorithm:
#
# 1. Pick an observer _(station)_.
# 2. Find all observers _(stations)_ that have overlapping observations within DayTol.
# 3. Go through all overlaping observers _(stations)_ and define the adjacency matrices.

# %%
# Minimum number of different group numbers to consider an observer valid
minObVal = 4

#Number of days of overlap
LnkDays = sp.sparse.lil_matrix((UnObs.shape[0], UnObs.shape[0]))

pos = {} #Dictionary with positions for plotting the network
for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    Obs1 = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
    #Update position dictionary of for plotting
    pos.update({iObs1: [ sp.mean(Obs1.FRACYEAR), sp.random.random_sample()]})
    
    # Create an array of valid date of observations including +/- DayTol
    DatObs1 = Obs1.ORDINAL
    TmpList = DatObs1
    for iDat in range(DayTol):
        TmpList = pd.concat([TmpList,DatObs1 + iDat + 1])
        TmpList = pd.concat([TmpList,DatObs1 - iDat - 1])    
    DatObs1 = sp.unique(TmpList)

    
    #---------------------
    # 2. Find all observers that have overlapping observations within DayTol
    #---------------------
    
    ObsOvrlp = sp.in1d(GN_Dat.ORDINAL, DatObs1)
    ObsOvrlp = sp.unique(GN_Dat.STATION[ObsOvrlp])
    ObsOvrlp = ObsOvrlp[sp.logical_and(ObsOvrlp != 0, ObsOvrlp != UnObs[iObs1])]
    
    # Void observer if it doesn't have enough values
    if np.unique(Obs1['GROUPS']).shape[0] < minObVal:
        ObsOvrlp = np.array([]) 
    
    #---------------------
    # 3. Go through all overlaping observers and define the adjacency matrices.
    #---------------------
    
    for iObsOvrlp in range(ObsOvrlp.shape[0]):
        
        #Determine the overlapping observer
        iObs2 = (UnObs==ObsOvrlp[iObsOvrlp]).nonzero()
        Obs2 = GN_Dat[GN_Dat.STATION == ObsOvrlp[iObsOvrlp]]
        
        # Remove zeros to include only matches that could potentially signify a useful connection
        #Obs2 = Obs2[Obs2['GROUPS'] > 0]
        
        DatObs2 = Obs2.ORDINAL
        
        
        # Void observer if it doesn't have enough values
        if np.unique(Obs2['GROUPS']).shape[0] < minObVal:
            DatObs2 = np.array([])        
        
        #Find the dates in common (including days within DayTol)
        DatIntrsct = sp.intersect1d(DatObs1,DatObs2)
        
        #Fill the days of overlap adjacency matrix
        if DatIntrsct.shape[0] >= MinOvr:
            LnkDays[iObs1,iObs2] = DatIntrsct.shape[0]

# %% [markdown]
# ## Blending Group Number Observations

# %%
# Defining Time Grid
Y1 = 1600  # First year of the window of interest
Y2 = 2020  # Second year of the window of interest
dYr = 0.25 # Coarness of the year grid
YrBl = np.arange(Y1+0.5-dYr/2,Y2+dYr,dYr)

# Defining group Grid
dG = 0.25  # Coarness of the Group grid
Gmax = 17 # Maximum group number to consider
GrpsBl = np.arange(0,Gmax+dG,dG)

# Creating Blending Matrix
GrpsBM = np.zeros((YrBl.shape[0],GrpsBl.shape[0]))

# Interpolating to grid

## Svalgaard and Schatten
# Create array with missing dates
tmpyr = YrBl.copy()+dYr/2
tmpyr = tmpyr[np.logical_and(tmpyr>np.min(SS_Dat['Year']),tmpyr<np.max(SS_Dat['Year']))]
MisYr = np.setdiff1d(tmpyr,SS_Dat['Year'])

# Create temporary dataframe to interpolate and append
pdAppnd = np.zeros((MisYr.shape[0], SS_Dat.shape[1]))
pdAppnd.fill(np.nan)
pdAppnd[:,0] = MisYr
pdAppnd = pd.DataFrame(pdAppnd, columns=['Year','G','S'])

# Interpolate missing values
tmp_int = interpolate.splrep(SS_Dat['Year'].values, SS_Dat['G'].values, s=1)
pdAppnd['G'] = interpolate.splev(pdAppnd['Year'], tmp_int)
tmp_int = interpolate.splrep(SS_Dat['Year'].values, SS_Dat['S'].values, s=1)
pdAppnd['S'] = interpolate.splev(pdAppnd['Year'], tmp_int)

# Append
SS_DatInt = SS_Dat.append(pdAppnd)
SS_DatInt = SS_DatInt.sort_values('Year').reset_index(drop=True)



# Usoskin etal
#Create array with missing dates
tmpyr = YrBl.copy()+dYr/2
tmpyr = tmpyr[np.logical_and(tmpyr>np.min(U_Dat['Year']),tmpyr<np.max(U_Dat['Year']))]
MisYr = np.setdiff1d(tmpyr,U_Dat['Year'])

# Create temporary dataframe to interpolate and append
pdAppnd = np.zeros((MisYr.shape[0], U_Dat.shape[1]))
pdAppnd.fill(np.nan)
pdAppnd[:,0] = MisYr
# pdAppnd = pd.DataFrame(pdAppnd, columns=['Year','G','S1','S2','S'])
pdAppnd = pd.DataFrame(pdAppnd, columns=['Year','G','S'])

# Interpolate missing values
tmp_int = interpolate.splrep(U_Dat.loc[np.isfinite(U_Dat['G']),'Year'].values, U_Dat.loc[np.isfinite(U_Dat['G']),'G'].values, s=2)
pdAppnd['G'] = interpolate.splev(pdAppnd['Year'], tmp_int)
tmp_int = interpolate.splrep(U_Dat.loc[np.isfinite(U_Dat['G']),'Year'].values, U_Dat.loc[np.isfinite(U_Dat['G']),'S'].values, s=2)
pdAppnd['S'] = interpolate.splev(pdAppnd['Year'], tmp_int)

# Append
U_DatInt = U_Dat.append(pdAppnd)
U_DatInt = U_DatInt.sort_values('Year').reset_index(drop=True)

# Remove NaN intervals
for i in range(0, U_Dat.shape[0]):
    if np.isnan(U_Dat['G'].values[i]):
        U_DatInt.loc[np.logical_and(U_DatInt['Year']>U_Dat['Year'].values[i]-1, 
                      JU_DatInt['Year']<U_Dat['Year'].values[i]+1), ['G','S']] = np.nan


## Vaquero etal
# Create array with missing dates
tmpyr = YrBl.copy()+dYr/2
tmpyr = tmpyr[np.logical_and(tmpyr>np.min(JV_Dat['Year']),tmpyr<np.max(JV_Dat['Year']))]
MisYr = np.setdiff1d(tmpyr,JV_Dat['Year'])

# Create temporary dataframe to interpolate and append
pdAppnd = np.zeros((MisYr.shape[0], JV_Dat.shape[1]))
pdAppnd.fill(np.nan)
pdAppnd[:,0] = MisYr
pdAppnd = pd.DataFrame(pdAppnd, columns=['Year','G','S1','S2','S'])

# Interpolate missing values
tmp_int = interpolate.splrep(JV_Dat.loc[np.isfinite(JV_Dat['G']),'Year'].values, JV_Dat.loc[np.isfinite(JV_Dat['G']),'G'].values, s=1)
pdAppnd['G'] = interpolate.splev(pdAppnd['Year'], tmp_int)
tmp_int = interpolate.splrep(JV_Dat.loc[np.isfinite(JV_Dat['G']),'Year'].values, JV_Dat.loc[np.isfinite(JV_Dat['G']),'S'].values, s=1)
pdAppnd['S'] = interpolate.splev(pdAppnd['Year'], tmp_int)

# Append
JV_DatInt = JV_Dat.append(pdAppnd)
JV_DatInt = JV_DatInt.sort_values('Year').reset_index(drop=True)

# Remove NaN intervals
for i in range(0, JV_Dat.shape[0]):
    if np.isnan(JV_Dat['G'].values[i]):
        JV_DatInt.loc[np.logical_and(JV_DatInt['Year']>JV_Dat['Year'].values[i]-1, 
                      JV_DatInt['Year']<JV_Dat['Year'].values[i]+1), ['G','S']] = np.nan


## Chatzistergos etal
# Create array with missing dates
tmpyr = YrBl.copy()+dYr/2
tmpyr = tmpyr[np.logical_and(tmpyr>np.min(Ch_Dat['Year']),tmpyr<np.max(Ch_Dat['Year']))]
MisYr = np.setdiff1d(tmpyr,Ch_Dat['Year'])

# Create temporary dataframe to interpolate and append
pdAppnd = np.zeros((MisYr.shape[0], Ch_Dat.shape[1]))
pdAppnd.fill(np.nan)
pdAppnd[:,0] = MisYr
pdAppnd = pd.DataFrame(pdAppnd, columns=['Year','G','S1','S2','S'])

# Interpolate missing values
tmp_int = interpolate.splrep(Ch_Dat.loc[np.isfinite(Ch_Dat['G']),'Year'].values, Ch_Dat.loc[np.isfinite(Ch_Dat['G']),'G'].values, s=1)
pdAppnd['G'] = interpolate.splev(pdAppnd['Year'], tmp_int)
tmp_int = interpolate.splrep(Ch_Dat.loc[np.isfinite(Ch_Dat['G']),'Year'].values, Ch_Dat.loc[np.isfinite(Ch_Dat['G']),'S'].values, s=1)
pdAppnd['S'] = interpolate.splev(pdAppnd['Year'], tmp_int)

# Append
Ch_DatInt = Ch_Dat.append(pdAppnd)
Ch_DatInt = Ch_DatInt.sort_values('Year').reset_index(drop=True)

# Remove NaN intervals
for i in range(0, Ch_Dat.shape[0]):
    if np.isnan(Ch_Dat['G'].values[i]):
        Ch_DatInt.loc[np.logical_and(Ch_DatInt['Year']>Ch_Dat['Year'].values[i]-1, 
                      Ch_DatInt['Year']<Ch_Dat['Year'].values[i]+1), ['G','S']] = np.nan
        
        
        
## Ivanov Miletsky
# Create array with missing dates
tmpyr = YrBl.copy()+dYr/2
tmpyr = tmpyr[np.logical_and(tmpyr>np.min(IM_Dat['Year']),tmpyr<np.max(IM_Dat['Year']))]
MisYr = np.setdiff1d(tmpyr,IM_Dat['Year'])

# Create temporary dataframe to interpolate and append
pdAppnd = np.zeros((MisYr.shape[0], IM_Dat.shape[1]))
pdAppnd.fill(np.nan)
pdAppnd[:,0] = MisYr
pdAppnd = pd.DataFrame(pdAppnd, columns=['Year','G','S'])

# Interpolate missing values
tmp_int = interpolate.splrep(IM_Dat.loc[np.isfinite(IM_Dat['G']),'Year'].values, IM_Dat.loc[np.isfinite(IM_Dat['G']),'G'].values, s=1)
pdAppnd['G'] = interpolate.splev(pdAppnd['Year'], tmp_int)
tmp_int = interpolate.splrep(IM_Dat.loc[np.isfinite(IM_Dat['G']),'Year'].values, IM_Dat.loc[np.isfinite(IM_Dat['G']),'S'].values, s=1)
pdAppnd['S'] = interpolate.splev(pdAppnd['Year'], tmp_int)

# Append
IM_DatInt = IM_Dat.append(pdAppnd)
IM_DatInt = IM_DatInt.sort_values('Year').reset_index(drop=True)

# Remove NaN intervals
for i in range(0, IM_Dat.shape[0]):
    if np.isnan(IM_Dat['G'].values[i]):
        IM_DatInt.loc[np.logical_and(IM_DatInt['Year']>IM_Dat['Year'].values[i]-1, 
                      IM_DatInt['Year']<IM_Dat['Year'].values[i]+1), ['G','S']] = np.nan

# Filling the Blending Matrix
for i in range(0,YrBl.shape[0]):
    # Create temporary storage variable for blending
    tmpBl = GrpsBl*0
    
    # Add Svalgaard and Schatten
    if np.sum(np.logical_and(SS_DatInt['Year']>YrBl[i],SS_DatInt['Year']<YrBl[i]+1)) > 0:
        # Find groups and error
        tmpDat = SS_DatInt[np.logical_and(SS_DatInt['Year']>YrBl[i],SS_DatInt['Year']<YrBl[i]+1)]
        # Add gaussian to temporary blending variable
        Gauss =  sp.stats.norm(loc = tmpDat['G'].values[0], scale = tmpDat['S'].values[0])
        tmpBl = tmpBl + Gauss.cdf(GrpsBl+1) - Gauss.cdf(GrpsBl)
        
    # Add Usoskin et al.
    if np.sum(np.logical_and(U_DatInt['Year']>YrBl[i],U_DatInt['Year']<YrBl[i]+1)) > 0:
        # Find groups and error
        tmpDat = U_DatInt[np.logical_and(U_DatInt['Year']>YrBl[i],U_DatInt['Year']<YrBl[i]+1)]
        # Add gaussian to temporary blending variable
        Gauss =  sp.stats.norm(loc = tmpDat['G'].values[0], scale = tmpDat['S'].values[0])
        tmpBl = tmpBl + Gauss.cdf(GrpsBl+1) - Gauss.cdf(GrpsBl)
        
        
    # Add Vaquero et al.
    if np.sum(np.logical_and(JV_DatInt['Year']>YrBl[i],JV_DatInt['Year']<YrBl[i]+1)) > 0:
        # Find groups and error
        tmpDat = JV_DatInt[np.logical_and(JV_DatInt['Year']>YrBl[i],JV_DatInt['Year']<YrBl[i]+1)]
        
        if np.isfinite(tmpDat['G'].values[0]):
            # Add gaussian to temporary blending variable
            Gauss =  sp.stats.norm(loc = tmpDat['G'].values[0], scale = tmpDat['S'].values[0])
            tmpBl = tmpBl + Gauss.cdf(GrpsBl+1) - Gauss.cdf(GrpsBl)
            
            
    # Add Chatzistergos et al.
    if np.sum(np.logical_and(Ch_DatInt['Year']>YrBl[i],Ch_DatInt['Year']<YrBl[i]+1)) > 0:
        # Find groups and error
        tmpDat = Ch_DatInt[np.logical_and(Ch_DatInt['Year']>YrBl[i],Ch_DatInt['Year']<YrBl[i]+1)]
        
        if np.isfinite(tmpDat['G'].values[0]):
            # Add gaussian to temporary blending variable
            Gauss =  sp.stats.norm(loc = tmpDat['G'].values[0], scale = tmpDat['S'].values[0])
            tmpBl = tmpBl + Gauss.cdf(GrpsBl+1) - Gauss.cdf(GrpsBl)            

    # Add Ivanov & Miletsky.
    if np.sum(np.logical_and(IM_DatInt['Year']>YrBl[i],IM_DatInt['Year']<YrBl[i]+1)) > 0:
        # Find groups and error
        tmpDat = IM_DatInt[np.logical_and(IM_DatInt['Year']>YrBl[i],IM_DatInt['Year']<YrBl[i]+1)]
        
        if np.isfinite(tmpDat['G'].values[0]):
            # Add gaussian to temporary blending variable
            Gauss =  sp.stats.norm(loc = tmpDat['G'].values[0], scale = tmpDat['S'].values[0])
            tmpBl = tmpBl + Gauss.cdf(GrpsBl+1) - Gauss.cdf(GrpsBl)            
        
    GrpsBM[i,:] = tmpBl

# Cleaning matrix of small values for plotting
GrpsBM[GrpsBM==0] = np.nan    
tmp = np.squeeze(GrpsBM.reshape((1,-1)))
GrpsBM[GrpsBM<np.percentile(tmp[np.isfinite(tmp)],70)] = np.nan

# print(Ch_Dat, Ch_DatInt)

# %% [markdown]
# ## Creating and plotting graph network

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 4      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+3*ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)', 'Lockwood et al. (2014)', 'Cliver & Ling (2016)', 'Zolotova & Ponyavin (2015)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)

for i in range(0,4):
    t = leg.get_texts()[i]
    t.set_fontproperties(t.get_fontproperties())
    t.set_color('r')
    
t = leg.get_texts()[9]
t.set_fontproperties(t.get_fontproperties())
t.set_color('b')    

alphUn = 0.4

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(LO_Dat['Year'],LO_Dat['G']-LO_Dat['S'], LO_Dat['G']+LO_Dat['S'],color='c', edgecolor='none',alpha=alphUn)
ax1.fill_between(CL_Dat['Year'],CL_Dat['G']-CL_Dat['S'], CL_Dat['G']+CL_Dat['S'],color='m', edgecolor='none',alpha=alphUn)
ax1.fill_between(HS_Dat['Year'],HS_Dat['G']-HS_Dat['S'], HS_Dat['G']+HS_Dat['S'],color=cmap(7), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax1.fill_between(NZM_Dat['Date'],NZM_Dat['G']-NZM_Dat['S'], NZM_Dat['G']+NZM_Dat['S'],color=cmap(4), edgecolor='none',alpha=0.2)


ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')
# ax2.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')



ax4 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax4.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

ax4.set_xlim(left=Y1, right=Y2)
ax4.xaxis.tick_top()
ax4.set_xticklabels([])
ax4.set_xticks(np.arange(1600,2050,50))
# ax4.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')
ax4.set_ylabel('Current-Series Blended\nGroup Number')

ax4.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')



ax3 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyHvl['Year'],BflyHvl['Lat'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyHvl['Year'])+np.max(BflyHvl['Year']))/2, 57, 'Hevelius', va='top', ha='center',color=cmap(2), fontsize=FZ)


ax3.scatter(BflyMM1['Year'],BflyMM1['Lat'],MZ,alpha=0.5,color=cmap(4))
ax3.text( (np.min(BflyMM1['Year'])+np.max(BflyMM1['Year']))/2, -57, 'Malapert, Gassendi\n& Marcgraf', va='bottom', ha='center',color=cmap(4), fontsize=FZ)


ax3.scatter(BflySchn['FRACYEAR'],BflySchn['Lat'],MZ,alpha=0.5,color=cmap(1))
ax3.text( (np.min(BflySchn['FRACYEAR'])+np.max(BflySchn['FRACYEAR']))/2+8, 40, 'Scheiner', va='center', ha='center',color=cmap(1), fontsize=FZ)

ax3.scatter(BflyGal['FRACYEAR'],BflyGal['Latitude'],MZ,alpha=0.5,color=cmap(7))
ax3.text( (np.min(BflyGal['FRACYEAR'])+np.max(BflyGal['FRACYEAR']))/2, 57, 'Galileo', va='top', ha='center',color=cmap(7), fontsize=FZ)

ax3.scatter(BflyMM['Year'],BflyMM['Latitude'],MZ,alpha=0.8,color=cmap(6))
ax3.text( (np.min(BflyMM['Year'])+np.max(BflyMM['Year']))/2, -57, 'Spoerer and Observers\nfrom the\nParis Observatory', va='bottom', ha='center',color=cmap(6), fontsize=FZ)

ax3.scatter(BflyStdc['FRACYEAR'],BflyStdc['Lat'],MZ,alpha=0.5,color=cmap(7))
ax3.text( (np.min(BflyStdc['FRACYEAR'])+np.max(BflyStdc['FRACYEAR']))/2, 57, 'Staudacher', va='top', ha='center',color=cmap(7), fontsize=FZ)


ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(0))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'Schwabe', va='bottom', ha='center',color=cmap(0), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(4))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, 57, 'Spörer', va='top', ha='center',color=cmap(4), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'RGO', va='bottom', ha='center',color=cmap(2), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(5))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-4, 57, 'Debrecen', va='top', ha='center',color=cmap(5), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color='k')
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'KMAS', va='bottom', ha='center', fontsize=FZ)




ax3.annotate('(d)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')


fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI, bbox_inches='tight')

# fig.savefig('F2_Solar_Cycle_Obs_Data.pdf', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+1*ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)', 'Lockwood et al. (2014)', 'Cliver & Ling (2016)', 'Zolotova & Ponyavin (2015)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)

for i in range(0,4):
    t = leg.get_texts()[i]
    t.set_fontproperties(t.get_fontproperties())
    t.set_color('r')
    
t = leg.get_texts()[9]
t.set_fontproperties(t.get_fontproperties())
t.set_color('b')    

alphUn = 0.4

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(LO_Dat['Year'],LO_Dat['G']-LO_Dat['S'], LO_Dat['G']+LO_Dat['S'],color='c', edgecolor='none',alpha=alphUn)
ax1.fill_between(CL_Dat['Year'],CL_Dat['G']-CL_Dat['S'], CL_Dat['G']+CL_Dat['S'],color='m', edgecolor='none',alpha=alphUn)
ax1.fill_between(HS_Dat['Year'],HS_Dat['G']-HS_Dat['S'], HS_Dat['G']+HS_Dat['S'],color=cmap(7), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax1.fill_between(NZM_Dat['Date'],NZM_Dat['G']-NZM_Dat['S'], NZM_Dat['G']+NZM_Dat['S'],color=cmap(4), edgecolor='none',alpha=0.2)


ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Group Number')
ax1.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')
ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')



fig.savefig('F1_Solar_Cycle_Obs_Data_Red.png',dpi=figDPI)

# %% [markdown]
# # Creating Matrix containing all observations in a window of interest

# %%
## Maunder Minimum

Y1 = 1634  # First year of the window of interest
Y2 = 1727  # Second year of the window of interest

# Find all observers with at least one day in the window of interest
UnqObs = GN_Dat.loc[np.logical_and(np.logical_and(GN_Dat.YEAR>=Y1, GN_Dat.YEAR<=Y2),GN_Dat.STATION>0), 'STATION'].values
UnqObs = np.unique(UnqObs)

# Finding the ordinal days that contain these observers completely
OD1 = 9999999
OD2 = 0
for Obs in UnqObs:
    OD1 = np.min(np.append(GN_Dat.loc[GN_Dat.STATION == Obs, 'ORDINAL'].values, OD1))
    OD2 = np.max(np.append(GN_Dat.loc[GN_Dat.STATION == Obs, 'ORDINAL'].values, OD2))
    
# Create Matrix for observations
ObsMat = np.zeros([OD2-OD1+1, UnqObs.shape[0]])*np.nan

# Fill it with observations
for i in range(0,UnqObs.shape[0]):
    ObsData = GN_Dat.loc[GN_Dat.STATION == UnqObs[i], ['ORDINAL','GROUPS']].values
    ObsMat[ObsData[:,0].astype(int)-OD1,i] = ObsData[:,1]
    
    
## Modern period

Y12 = 1865  # First year of the window of interest
Y22 = Y12+Y2-Y1  # Second year of the window of interest

# Find all observers with at least one day in the window of interest
UnqObs2 = GN_Dat.loc[np.logical_and(np.logical_and(GN_Dat.YEAR>=Y12, GN_Dat.YEAR<=Y22),GN_Dat.STATION>0), 'STATION'].values
UnqObs2 = np.unique(UnqObs2)

# Finding the ordinal days that contain these observers completely
OD12 = 9999999
OD22 = 0
for Obs in UnqObs2:
    OD12 = np.min(np.append(GN_Dat.loc[GN_Dat.STATION == Obs, 'ORDINAL'].values, OD12))
    OD22 = np.max(np.append(GN_Dat.loc[GN_Dat.STATION == Obs, 'ORDINAL'].values, OD22))
    
# Create Matrix for observations
ObsMat2 = np.zeros([OD22-OD12+1, UnqObs2.shape[0]])*np.nan

# Fill it with observations
for i in range(0,UnqObs2.shape[0]):
    ObsData2 = GN_Dat.loc[GN_Dat.STATION == UnqObs2[i], ['ORDINAL','GROUPS']].values
    ObsMat2[ObsData2[:,0].astype(int)-OD12,i] = ObsData2[:,1]

# %% [markdown]
# # Calculate activity mask based on observation matrix and Yearly Coverage

# %%
## Maunder Minimum

# Add observations of all observers to identify active days
ADmsk = (np.nansum(ObsMat,axis=1)>0).astype(float) + 1

# Calculate the number of NaNs each day to identify missing days
MDmsk = np.sum(np.isnan(ObsMat),axis=1)==UnqObs.shape[0]

# Fill days with missing observations with NaN
ADmsk[MDmsk] = ADmsk[MDmsk]-1

# Creating fractional year variable

# Ordinal to fractional
def ordinal2fracy(Ord):
    YEAR = datetime.date.fromordinal(Ord).year     
    return YEAR + ( Ord - datetime.date(YEAR,1,1).toordinal() )/(datetime.date(YEAR+1,1,1).toordinal() - datetime.date(YEAR,1,1).toordinal() )

vordinal2fracy=np.vectorize(ordinal2fracy)
fracyr = vordinal2fracy(np.arange(OD1,OD2+1))



## Modern Period

# Add observations of all observers to identify active days
ADmsk2 = (np.nansum(ObsMat2,axis=1)>0).astype(float) + 1

# Calculate the number of NaNs each day to identify missing days
MDmsk2 = np.sum(np.isnan(ObsMat2),axis=1)==UnqObs2.shape[0]

# Fill days with missing observations with NaN
ADmsk2[MDmsk2] = ADmsk2[MDmsk2]-1

# Creating fractional year variable
fracyr2 = vordinal2fracy(np.arange(OD12,OD22+1))

# %% [markdown]
# ## Unify Butterfly diagram observations

# %%
# Modern data
BflyAll = BflyMod.copy()
BflyAll.loc[BflyAll['CORR_A']<ALim,'LATITUDE'] = np.NaN
BflyAll = BflyAll.loc[np.isfinite(BflyAll['CORR_A']),('FRACYEAR','ORDINAL','LATITUDE')].values

# Hevelius
tmp = BflyHvl.loc[:,('Year','Lat')]
tmp['ORDINAL'] = tmp.apply(lambda x: np.round(datetime.date(np.floor(x['Year']).astype(int),1,1).toordinal()
                           + (x['Year']-np.floor(x['Year']))*(datetime.date(np.floor(x['Year']).astype(int)+1,1,1).toordinal()
                                                             -datetime.date(np.floor(x['Year']).astype(int),1,1).toordinal())).astype(int)
                                  ,axis=1)
tmp = tmp.loc[:,('Year','ORDINAL','Lat')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Galileo
tmp = BflyGal.loc[:,('FRACYEAR','ORDINAL','Latitude')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Scheiner
tmp = BflySchn.loc[:,('FRACYEAR','ORDINAL','Lat')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Harriot
tmp = BflyHrt.loc[:,('FRACYEAR','ORDINAL','Lat')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Staudacher
tmp = BflyStdc.loc[:,('FRACYEAR','ORDINAL','Lat')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Sporer and Ribes 
tmp = BflyMM.loc[:,('Year','Latitude')]
tmp['ORDINAL'] = tmp.apply(lambda x: np.round(datetime.date(np.floor(x['Year']).astype(int),1,1).toordinal()
                           + (x['Year']-np.floor(x['Year']))*(datetime.date(np.floor(x['Year']).astype(int)+1,1,1).toordinal()
                                                             -datetime.date(np.floor(x['Year']).astype(int),1,1).toordinal())).astype(int)
                                  ,axis=1)
tmp = tmp.loc[:,('Year','ORDINAL','Latitude')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Malapert, and so on 
tmp = BflyMM1.loc[:,('Year','Lat')]


tmp['ORDINAL'] = tmp.apply(lambda x: np.round(datetime.date(np.floor(x['Year']).astype(int),1,1).toordinal()
                           + (x['Year']-np.floor(x['Year']))*(datetime.date(np.floor(x['Year']).astype(int)+1,1,1).toordinal()
                                                             -datetime.date(np.floor(x['Year']).astype(int),1,1).toordinal())).astype(int)
                                  ,axis=1)
tmp = tmp.loc[:,('Year','ORDINAL','Lat')].values
BflyAll = np.append(BflyAll,tmp,axis=0)

# Create pandas dataframe
BflyAll = pd.DataFrame(BflyAll, columns=['FRACYEAR','ORDINAL','LATITUDE'])
BflyAll['ORDINAL'] = BflyAll['ORDINAL'].astype(int)

# %% [markdown]
# ## Calculating coverage of sunspot group observations

# %%
# Define years to calculate butterfly diagram coverage
Y1 = 1600
Y2 = 1900

# Number of years in coverage window
YrCum = 2

# Removing days wihout observers
CvrObs = GN_Dat['ORDINAL'].copy()
CrvObs = CvrObs[GN_Dat['STATION']>0]

# Identifying each day with an observation
CrvObs = np.unique(CrvObs)

# Calculate the corresponding fractional year
CrvObsFy = vordinal2fracy(CrvObs)

# Set up repository variables
CrvObsYr = np.arange(Y1,Y2+1,YrCum)
CrvObsCv = CrvObsYr.copy().astype(float)*0

for i in np.arange(0,CrvObsYr.shape[0]):
    # Calculate number of days in year
    NdaysYr = datetime.date(CrvObsYr[i]+YrCum,1,1).toordinal()-datetime.date(CrvObsYr[i],1,1).toordinal()
    
    # Find number of observations in first year of the bin
    NdaysOb = np.sum(np.floor(CrvObsFy)==CrvObsYr[i])
    
    # Add other years to the bin
    for j in np.arange(1,YrCum):
        NdaysOb = NdaysOb + np.sum(np.floor(CrvObsFy)==CrvObsYr[i]+j)
    
    # Calculate coverage
    CrvObsCv[i] = NdaysOb/NdaysYr

# %% [markdown]
# ## Calculate observational coverage based on butterfly diagram

# %%
Y1 = 1600
Y2 = 1980

# Calculate a unique list of days represented in the butterfly diagram
ObsDsBfly = np.unique(BflyAll['ORDINAL'])
    
# Calculate the corresponding fractional year
BfObsFy = vordinal2fracy(ObsDsBfly)

# Set up repository variables
BfObsYr = np.arange(Y1,Y2+1,YrCum)
BfObsCv = BfObsYr.copy().astype(float)*0

for i in np.arange(0,BfObsYr.shape[0]):
    # Calculate number of days in year
    NdaysYr = datetime.date(BfObsYr[i]+YrCum,1,1).toordinal()-datetime.date(BfObsYr[i],1,1).toordinal()
    
    # Find number of observations in first year of the bin
    NdaysOb = np.sum(np.floor(BfObsFy)==BfObsYr[i])
    
    # Add other years to the bin
    for j in np.arange(1,YrCum):
        NdaysOb = NdaysOb + np.sum(np.floor(BfObsFy)==BfObsYr[i]+j)
    
    # Calculate coverage
    BfObsCv[i] = NdaysOb/NdaysYr    

# %% [markdown]
# ## Plot activity mask by reshaping into months

# %%
Y1 = 1634  # First year of the window of interest
Y2 = 1727  # Second year of the window of interest

# Number of days per "month"
MoLngt = 30

## Maunder Minimum

ADmskPlt = ADmsk[0:np.int(ADmsk.shape[0]/MoLngt)*MoLngt].copy()
fracyrPlt = fracyr[0:np.int(ADmsk.shape[0]/MoLngt)*MoLngt].copy()

# Reshaping
ADmskPlt = ADmskPlt.reshape((-1,MoLngt))
fracyrPlt = fracyrPlt.reshape((-1,MoLngt))
fracyrPlt = np.mean(fracyrPlt,axis=1)


# Observational coverage variable
ObsYr = np.arange(np.min(np.floor(fracyr)),np.max(np.floor(fracyr)))
ObsCv = ObsYr.copy()*0
                  
for i in np.arange(0,ObsCv.shape[0]):    
    ObsCv[i] = np.sum(ADmskPlt[np.floor(fracyrPlt)==ObsYr[i],:]>0)/( np.sum(np.floor(fracyrPlt)==ObsYr[i])*MoLngt )
  
xMesh = np.insert(fracyrPlt,0,fracyrPlt[0])
yMesh = np.arange(0,MoLngt+1)
    
## Modern Period

ADmskPlt2 = ADmsk2[0:np.int(ADmsk2.shape[0]/MoLngt)*MoLngt].copy()
fracyrPlt2 = fracyr2[0:np.int(ADmsk2.shape[0]/MoLngt)*MoLngt].copy()

# Reshaping
ADmskPlt2 = ADmskPlt2.reshape((-1,MoLngt))
fracyrPlt2 = fracyrPlt2.reshape((-1,MoLngt))
fracyrPlt2 = np.mean(fracyrPlt2,axis=1)


# Observational coverage variable
ObsYr2 = np.arange(np.min(np.floor(fracyr2)),np.max(np.floor(fracyr2)))
ObsCv2 = ObsYr2.copy()*0
                  
for i in np.arange(0,ObsCv.shape[0]):    
    ObsCv2[i] = np.sum(ADmskPlt2[np.floor(fracyrPlt2)==ObsYr2[i],:]>0)/( np.sum(np.floor(fracyrPlt2)==ObsYr2[i])*MoLngt )    
    
xMesh2 = np.insert(fracyrPlt2,0,fracyrPlt2[0])


plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 4000   # Horizontal size of each panel
pxy = 1500    # Vertical size of each panel

nph = 2      # Number of horizontal panels
npv = 4      # Number of vertical panels

# Padding
padv  = 350 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 520 #Horizontal padding in pixels at the edge of the figure
padh2 = 400 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

#Colormap
colors = [(1, 1, 1), Clr[4], Clr[2]]
cmap = clrs.LinearSegmentedColormap.from_list('cmap', colors, N=3)

ax1 = fig.add_axes([ppadh, ppadv+2*ppxy, ppxx, ppxy])
pch = ax1.pcolormesh(xMesh, yMesh, np.transpose(ADmskPlt), cmap = cmap, alpha = 0.45, linewidth = 2)
ax1.set_ylim(bottom = 0.1, top = MoLngt)
ax1.set_xlim(left=Y1, right=Y2)

# Axes properties
ax1.set_ylabel('Day Number\nin the Month')
ax1.minorticks_on()

ax1.annotate('(c)',xy=(0, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='left')


# Add Colorbar
axcb1  = fig.add_axes([ppadh+ppxx, ppadv+2*ppxy, ppadh2, ppxy])
axcb1.fill([0,0,1,1], [0, 1, 1, 0], color=(1, 1, 1), alpha = 0.6, edgecolor = 'none')
axcb1.fill([0,0,1,1], [1, 2, 2, 1], color=Clr[4], alpha = 0.6, edgecolor = 'none')
axcb1.fill([0,0,1,1], [2, 3, 3, 2], color=Clr[2], alpha = 0.6, edgecolor = 'none')

axcb1.annotate('Missing', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center', rotation='vertical', 
           fontsize=27,zorder=3, )
axcb1.annotate('Quiet', xy = (0.5, 1.5), horizontalalignment='center', verticalalignment='center', rotation='vertical', 
           fontsize=27,zorder=3, )
axcb1.annotate('Active', xy = (0.5, 2.5), horizontalalignment='center', verticalalignment='center', rotation='vertical', 
           fontsize=27,zorder=3, )

axcb1.set_ylim(bottom = 0, top = 3)
axcb1.set_xlim(left=0, right=1)    

axcb1.set_xticks([])
axcb1.set_yticks([])

y2 = 15

ax2 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax2.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax2.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax2.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax2.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax2.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)


ax2.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax2.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax2.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax2.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax2.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')


#Colormap for transparency mask
colorsMsk = [(0.65,0.5,0.35), (0.60,0.60,0.4), (0.42,0.5,0.56), (0.5,0.5,0.5)]
color0 = (0.61,0.38,0.38)

MaskMaxAl = 0.8

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax2.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

    
ax2.set_ylim(bottom = 0, top = y2)
ax2.set_xlim(left=Y1, right=Y2)

# Axes properties
ax2.set_ylabel('GN Masked Using\nObserved Days')
ax2.minorticks_on()

ax2.annotate('(e)',xy=(0, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='left')
ax2.set_yticks(np.arange(2,16,2))

# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])



## Modern Period

ax6 = fig.add_axes([ppadh+ppxx+ppadh2, ppadv+3*ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax6.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax6.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax6.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax6.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)


ax6.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax6.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax6.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax6.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')

    
ax6.set_ylim(bottom = 0, top = y2)
ax6.set_xlim(left=Y12, right=Y22)
ax6.yaxis.tick_right()
ax6.yaxis.set_label_position('right')
ax6.xaxis.tick_top()
ax6.xaxis.set_label_position('top')


# Axes properties
ax6.set_ylabel('Group Number')
ax6.minorticks_on()
ax6.set_yticks(np.arange(2,16,2))

ax6.text(0.5, 1.15,'Modern Period', horizontalalignment='center',
         fontsize=30,
         transform = ax6.transAxes);

ax6.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax3 = fig.add_axes([ppadh, ppadv+3*ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax3.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax3.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax3.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax3.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax3.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')

ax3.legend(('Svalgaard & Schatten (2016)', 'Willamo et al. (2017)','Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)'), 
           ncol = 4, loc = (0.1,0.74), fontsize=27, frameon=False, columnspacing=1)

alphUn = 0.4

ax3.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax3.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax3.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax3.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax3.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)


ax3.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax3.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax3.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax3.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax3.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')

    
ax3.set_ylim(bottom = 0, top = y2)
ax3.set_xlim(left=Y1, right=Y2)
ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')

# Axes properties
ax3.set_ylabel('Group Number')
ax3.minorticks_on()
ax3.set_yticks(np.arange(2,16,2))

ax3.text(0.5, 1.15,'Maunder Minimum', horizontalalignment='center',
         fontsize=30,
         transform = ax3.transAxes);

ax3.annotate('(a)',xy=(0, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='left')


#Colormap
colors = [(1, 1, 1), Clr[4], Clr[2]]
cmap = clrs.LinearSegmentedColormap.from_list('cmap', colors, N=3)

ax4 = fig.add_axes([ppadh+ppxx+ppadh2, ppadv+2*ppxy, ppxx, ppxy])
ax4.pcolormesh(xMesh2, yMesh, np.transpose(ADmskPlt2), cmap = cmap, alpha = 0.45, linewidth = 2)
ax4.set_ylim(bottom = 0.1, top = MoLngt)
ax4.set_xlim(left=Y12, right=Y22)

# Axes properties
ax4.set_ylabel('Day Number\nin the Month')
ax4.minorticks_on()
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position('right')

ax4.annotate('(d)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax5 = fig.add_axes([ppadh+ppxx+ppadh2, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax5.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax5.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax5.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax5.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)

ax5.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax5.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax5.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax5.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')

 

ax5.set_ylim(bottom = 0, top = y2)
ax5.set_xlim(left=Y12, right=Y22)

# Axes properties
ax5.set_ylabel('GN Masked Using\nObserved Days')
ax5.minorticks_on()
ax5.set_yticks(np.arange(2,16,2))
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position('right')

ax5.annotate('(f)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax7 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax7.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')


ax7.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.5,color='k')

ax7.set_ylim(top=60, bottom=-60)
ax7.set_xlim(left=Y1, right=Y2)

# Axes properties
ax7.set_ylabel('Latitude (o)');
ax7.minorticks_on()

ax7.annotate('(g)',xy=(0, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='left')

ax8 = fig.add_axes([ppadh+ppxx+ppadh2, ppadv, ppxx, ppxy])


# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax8.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')


ax8.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.5,color='k')



ax8.set_ylim(top=60, bottom=-60)
ax8.set_xlim(left=Y12, right=Y22)

# Axes properties
ax8.set_ylabel('Latitude (o)');
# ax8.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')
ax8.minorticks_on()
ax8.yaxis.tick_right()
ax8.yaxis.set_label_position('right')

ax8.annotate('(h)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')



fig.savefig('F2_MM_vs_Modern.png',dpi=figDPI)

# %%
colors = [(0.65,0.5,0.35), (0.60,0.60,0.4), (0.42,0.5,0.56), (0.5,0.5,0.5)]
color0 = (0.61,0.38,0.38)
cmapMsk = clrs.LinearSegmentedColormap.from_list('cmapMsk', colors, N=5)
norm = clrs.Normalize(vmin=0, vmax=0.15, clip=True)

colors[(int(np.min([3,np.floor(0.2/0.05)])))]

# %% [markdown]
# ## Long-term historic data plot

# %%
#Plotting Graph Network

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1500    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3     # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))


plt.rc('font', **font)


ax1 = fig.add_axes([ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax1)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax1)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax1)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax1, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax1)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax1, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax1)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax1, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax1)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax1, width=LW)


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_yticks([])
ax1.set_ylabel('Networks of\nOverlaping\nObservers')
# ax1.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')
ax1.xaxis.set_label_position('top')
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')



ax3 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax3.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax3.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax3.set_xlim(left=Y1, right=Y2)
ax3.xaxis.tick_top()
ax3.set_xticklabels([])
ax3.set_ylabel('GN Masked Using\nObserved Days')

# ax3.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

# ax2.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

fig.savefig('F3a_Bfly_GN_Coverage.png',dpi=figDPI)

# %%


# %%


# %%
#Plotting Graph Network

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1500    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3     # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))


plt.rc('font', **font)


ax1 = fig.add_axes([ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)

ax1.legend(('Svalgaard & Schatten (2015)', 'Willamo et al. (2017)'
            ,'Ivanov & Miletsky (2017)','Chatzistergos et al. (2017)','Vaquero et al. (2015)'), 
           ncol = 4, loc = 2, fontsize=29, frameon=False, columnspacing=1)

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)


ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)

ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=17, bottom=0)
ax1.set_xticklabels([])
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Group Number')
ax1.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')



ax3 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax3.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax3.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax3.set_xlim(left=Y1, right=Y2)
ax3.xaxis.tick_top()
ax3.set_xticklabels([])
ax3.set_ylabel('GN Masked Using\nObserved Days')

ax3.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

ax2.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

fig.savefig('Network_Bfly_GN_Coverage.png',dpi=figDPI)

# %%
# Recalculating day mask for alternative figure
Y1 = 1600  # First year of the window of interest
Y2 = 2020  # Second year of the window of interest

# Find all observers with at least one day in the window of interest
UnqObs = GN_Dat.loc[np.logical_and(np.logical_and(GN_Dat.YEAR>=Y1, GN_Dat.YEAR<=Y2),GN_Dat.STATION>0), 'STATION'].values
UnqObs = np.unique(UnqObs)

# Finding the ordinal days that contain these observers completely
OD1 = 9999999
OD2 = 0
for Obs in UnqObs:
    OD1 = np.min(np.append(GN_Dat.loc[GN_Dat.STATION == Obs, 'ORDINAL'].values, OD1))
    OD2 = np.max(np.append(GN_Dat.loc[GN_Dat.STATION == Obs, 'ORDINAL'].values, OD2))
    
# Create Matrix for observations
ObsMat = np.zeros([OD2-OD1+1, UnqObs.shape[0]])*np.nan

# Fill it with observations
for i in range(0,UnqObs.shape[0]):
    ObsData = GN_Dat.loc[GN_Dat.STATION == UnqObs[i], ['ORDINAL','GROUPS']].values
    ObsMat[ObsData[:,0].astype(int)-OD1,i] = ObsData[:,1]
    
    
    
# Add observations of all observers to identify active days
ADmsk = (np.nansum(ObsMat,axis=1)>0).astype(float) + 1

# Calculate the number of NaNs each day to identify missing days
MDmsk = np.sum(np.isnan(ObsMat),axis=1)==UnqObs.shape[0]

# Fill days with missing observations with NaN
ADmsk[MDmsk] = ADmsk[MDmsk]-1

# Creating fractional year variable

vordinal2fracy=np.vectorize(ordinal2fracy)
fracyr = vordinal2fracy(np.arange(OD1,OD2+1))    
    
    
    
ADmskPlt = ADmsk[0:np.int(ADmsk.shape[0]/MoLngt)*MoLngt].copy()
fracyrPlt = fracyr[0:np.int(ADmsk.shape[0]/MoLngt)*MoLngt].copy()

# Reshaping
ADmskPlt = ADmskPlt.reshape((-1,MoLngt))
fracyrPlt = fracyrPlt.reshape((-1,MoLngt))
fracyrPlt = np.mean(fracyrPlt,axis=1)


# Observational coverage variable
ObsYr = np.arange(np.min(np.floor(fracyr)),np.max(np.floor(fracyr)))
ObsCv = ObsYr.copy()*0
                  
for i in np.arange(0,ObsCv.shape[0]):    
    ObsCv[i] = np.sum(ADmskPlt[np.floor(fracyrPlt)==ObsYr[i],:]>0)/( np.sum(np.floor(fracyrPlt)==ObsYr[i])*MoLngt )
  
xMesh = np.insert(fracyrPlt,0,fracyrPlt[0])
yMesh = np.arange(0,MoLngt+1)

# %%
#Plotting Graph Network

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1500    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 4     # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))


plt.rc('font', **font)


ax1 = fig.add_axes([ppadh, ppadv+3*ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)

ax1.legend(('Svalgaard & Schatten (2015)', 'Willamo et al. (2017)'
            ,'Ivanov & Miletsky (2017)','Chatzistergos et al. (2017)','Vaquero et al. (2015)'), 
           ncol = 4, loc = 2, fontsize=29, frameon=False, columnspacing=1)

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)


ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)

ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=17, bottom=0)
ax1.set_xticks(np.arange(1600,2050,50))
ax1.set_xticklabels([])
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')


#Colormap
colors = [(1, 1, 1), Clr[4], Clr[2]]
cmap = clrs.LinearSegmentedColormap.from_list('cmap', colors, N=3)

ax4 = fig.add_axes([ppadh, ppadv+2*ppxy, ppxx, ppxy])
#ax4.pcolormesh(xMesh2, yMesh, np.transpose(ADmskPlt2), cmap = cmap, alpha = 0.45, linewidth = 2)
pch = ax4.pcolormesh(xMesh, yMesh, np.transpose(ADmskPlt), cmap = cmap, alpha = 0.45, linewidth = 2)
ax4.set_ylim(bottom = 0.1, top = MoLngt)
ax4.set_xlim(left=Y1, right=Y2)

# Axes properties
ax4.set_ylabel('Day Number\nin the Month')
ax4.set_xticks(np.arange(1600,2050,50))
ax4.minorticks_on()
ax4.set_xticklabels([])
# ax4.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

ax4.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


# Add Colorbar
axcb1  = fig.add_axes([ppadh+ppxx, ppadv+2*ppxy, ppadh2, ppxy])
axcb1.fill([0,0,1,1], [0, 1, 1, 0], color=(1, 1, 1), alpha = 0.6, edgecolor = 'none')
axcb1.fill([0,0,1,1], [1, 2, 2, 1], color=Clr[4], alpha = 0.6, edgecolor = 'none')
axcb1.fill([0,0,1,1], [2, 3, 3, 2], color=Clr[2], alpha = 0.6, edgecolor = 'none')

axcb1.annotate('Missing', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center', rotation='vertical', 
           fontsize=27,zorder=3, )
axcb1.annotate('Quiet', xy = (0.5, 1.5), horizontalalignment='center', verticalalignment='center', rotation='vertical', 
           fontsize=27,zorder=3, )
axcb1.annotate('Active', xy = (0.5, 2.5), horizontalalignment='center', verticalalignment='center', rotation='vertical', 
           fontsize=27,zorder=3, )

axcb1.set_ylim(bottom = 0, top = 3)
axcb1.set_xlim(left=0, right=1)    

axcb1.set_xticks([])
axcb1.set_yticks([])


ax3 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax3.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax3.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax3.set_xlim(left=Y1, right=Y2)
ax3.xaxis.tick_top()
ax3.set_xticks(np.arange(1600,2050,50))
ax3.set_xticklabels([])
ax3.set_ylabel('GN Masked Using\nObserved Days')
# ax3.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

ax3.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

ax2.annotate('(d)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)')
ax2.set_xticks(np.arange(1600,2050,50))
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F3_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('F3_Bfly_GN_Coverage.pdf', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 1810

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('F4a_Bfly_GN_Coverage.pdf', bbox_inches='tight')

# %%
plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1810
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax1.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

ax2.annotate('(d)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4b_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('F4b_Bfly_GN_Coverage.pdf', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 4      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 500  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 1810

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+3*ppxy+ppadv2, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv+2*ppxy+ppadv2, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv+2*ppxy+ppadv2, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+3*ppxy+ppadv2, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])



Y1 = 1810
Y2 = 2020
ax1b = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1b.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1b.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax1b.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1b.set_xlim(left=Y1, right=Y2)
ax1b.xaxis.tick_top()
ax1b.xaxis.set_label_position('top')
ax1b.set_ylabel('GN Masked Using\nObserved Days')

ax2b = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2b.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2b.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

ax2b.annotate('(d)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2b.set_xlim(left=Y1, right=Y2)
ax2b.set_ylim(top=59.5, bottom=-59.5)
ax2b.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3b  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3b.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3b.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3b.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3b.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3b.set_ylim(bottom = 0, top = 4)
axcb3b.set_xlim(left=0, right=1)    

axcb3b.set_xticks([])
axcb3b.set_yticks([])
axcb3b.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2b  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2b.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2b.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2b.set_ylim(bottom = 1, top = 6)
axcb2b.set_xlim(left=0, right=1)    

axcb2b.set_xticks([])
axcb2b.set_yticks([])



fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI, bbox_inches='tight')
# fig.savefig('F4_Bfly_GN_Coverage.pdf', bbox_inches='tight')

# %%
#Plotting Graph Network

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1500    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2     # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))


plt.rc('font', **font)


ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)

# ax1.legend(('Svalgaard & Schatten (2015)', 'Willamo et al. (2017)'
#             ,'Ivanov & Miletsky (2017)','Chatzistergos et al. (2017)','Vaquero et al. (2015)'), 
#            ncol = 4, loc = 2, fontsize=29, frameon=False, columnspacing=1)

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)


ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3), linewidth=LW,linestyle='--')
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=17, bottom=0)
# ax1.set_xticklabels([])
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Group Number')
ax1.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')



ax3 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax3.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax3.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

ax3.set_xlim(left=Y1, right=Y2)
# ax3.xaxis.tick_top()
# ax3.set_xticklabels([])
ax3.set_ylabel('GN Masked Using\nObserved Days')

# ax3.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


# ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# # Plot transparency mask
# for i in np.arange(0,BfObsYr.shape[0]):
#     if BfObsCv[i]==0:
#         clr = color0
#     else:
#         clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
#     ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
# ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

# # ax2.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')


# ax2.set_xlim(left=Y1, right=Y2)
# ax2.set_ylim(top=59.5, bottom=-59.5)
# ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

fig.savefig('F3_Bfly_GN_Coverage_redux.png',dpi=figDPI)

# %%
MG_Obs = pd.read_csv('magnetic_clean.csv', quotechar = '"', encoding = 'ansi')
MG_Obs

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)', 'Lockwood et al. (2014)', 'Cliver & Ling (2016)', 'Zolotova & Ponyavin (2015)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)

for i in range(0,4):
    t = leg.get_texts()[i]
    t.set_fontproperties(t.get_fontproperties())
    t.set_color('r')
    
t = leg.get_texts()[9]
t.set_fontproperties(t.get_fontproperties())
t.set_color('b')    

alphUn = 0.4

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(LO_Dat['Year'],LO_Dat['G']-LO_Dat['S'], LO_Dat['G']+LO_Dat['S'],color='c', edgecolor='none',alpha=alphUn)
ax1.fill_between(CL_Dat['Year'],CL_Dat['G']-CL_Dat['S'], CL_Dat['G']+CL_Dat['S'],color='m', edgecolor='none',alpha=alphUn)
ax1.fill_between(HS_Dat['Year'],HS_Dat['G']-HS_Dat['S'], HS_Dat['G']+HS_Dat['S'],color=cmap(7), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax1.fill_between(NZM_Dat['Date'],NZM_Dat['G']-NZM_Dat['S'], NZM_Dat['G']+NZM_Dat['S'],color=cmap(4), edgecolor='none',alpha=0.2)


ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyHvl['Year'],BflyHvl['Lat'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyHvl['Year'])+np.max(BflyHvl['Year']))/2, 57, 'Hevelius', va='top', ha='center',color=cmap(2), fontsize=FZ)


ax3.scatter(BflyMM1['Year'],BflyMM1['Lat'],MZ,alpha=0.5,color=cmap(4))
ax3.text( (np.min(BflyMM1['Year'])+np.max(BflyMM1['Year']))/2, -57, 'Malapert, Gassendi\n& Marcgraf', va='bottom', ha='center',color=cmap(4), fontsize=FZ)


ax3.scatter(BflySchn['FRACYEAR'],BflySchn['Lat'],MZ,alpha=0.5,color=cmap(1))
ax3.text( (np.min(BflySchn['FRACYEAR'])+np.max(BflySchn['FRACYEAR']))/2+8, 40, 'Scheiner', va='center', ha='center',color=cmap(1), fontsize=FZ)

ax3.scatter(BflyGal['FRACYEAR'],BflyGal['Latitude'],MZ,alpha=0.5,color=cmap(7))
ax3.text( (np.min(BflyGal['FRACYEAR'])+np.max(BflyGal['FRACYEAR']))/2, 57, 'Galileo', va='top', ha='center',color=cmap(7), fontsize=FZ)

ax3.scatter(BflyMM['Year'],BflyMM['Latitude'],MZ,alpha=0.8,color=cmap(6))
ax3.text( (np.min(BflyMM['Year'])+np.max(BflyMM['Year']))/2, -57, 'Spoerer and Observers\nfrom the\nParis Observatory', va='bottom', ha='center',color=cmap(6), fontsize=FZ)

ax3.scatter(BflyStdc['FRACYEAR'],BflyStdc['Lat'],MZ,alpha=0.5,color=cmap(7))
ax3.text( (np.min(BflyStdc['FRACYEAR'])+np.max(BflyStdc['FRACYEAR']))/2, 57, 'Staudacher', va='top', ha='center',color=cmap(7), fontsize=FZ)


ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(0))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'Schwabe', va='bottom', ha='center',color=cmap(0), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(4))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, 57, 'Spörer', va='top', ha='center',color=cmap(4), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'RGO', va='bottom', ha='center',color=cmap(2), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(5))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-4, 57, 'Debrecen', va='top', ha='center',color=cmap(5), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color='k')
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-1, -57, 'KMAS', va='bottom', ha='center', fontsize=FZ)

ax3.scatter(MG_Obs['FRACYR'],MG_Obs['LAT'],MZ,alpha=0.5,color='c')
ax3.text(1985, -57, 'Magnetic', va='bottom', ha='center', fontsize=FZ, color='c')


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_all.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)', 'Lockwood et al. (2014)', 'Cliver & Ling (2016)', 'Zolotova & Ponyavin (2015)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)

for i in range(0,4):
    t = leg.get_texts()[i]
    t.set_fontproperties(t.get_fontproperties())
    t.set_color('r')
    
t = leg.get_texts()[9]
t.set_fontproperties(t.get_fontproperties())
t.set_color('b')    

alphUn = 0.4

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(LO_Dat['Year'],LO_Dat['G']-LO_Dat['S'], LO_Dat['G']+LO_Dat['S'],color='c', edgecolor='none',alpha=alphUn)
ax1.fill_between(CL_Dat['Year'],CL_Dat['G']-CL_Dat['S'], CL_Dat['G']+CL_Dat['S'],color='m', edgecolor='none',alpha=alphUn)
ax1.fill_between(HS_Dat['Year'],HS_Dat['G']-HS_Dat['S'], HS_Dat['G']+HS_Dat['S'],color=cmap(7), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax1.fill_between(NZM_Dat['Date'],NZM_Dat['G']-NZM_Dat['S'], NZM_Dat['G']+NZM_Dat['S'],color=cmap(4), edgecolor='none',alpha=0.2)


ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


# ax3.scatter(BflyHvl['Year'],BflyHvl['Lat'],MZ,alpha=0.5,color=cmap(2))
# ax3.text( (np.min(BflyHvl['Year'])+np.max(BflyHvl['Year']))/2, 57, 'Hevelius', va='top', ha='center',color=cmap(2), fontsize=FZ)


# ax3.scatter(BflyMM1['Year'],BflyMM1['Lat'],MZ,alpha=0.5,color=cmap(4))
# ax3.text( (np.min(BflyMM1['Year'])+np.max(BflyMM1['Year']))/2, -57, 'Malapert, Gassendi\n& Marcgraf', va='bottom', ha='center',color=cmap(4), fontsize=FZ)


# ax3.scatter(BflySchn['FRACYEAR'],BflySchn['Lat'],MZ,alpha=0.5,color=cmap(1))
# ax3.text( (np.min(BflySchn['FRACYEAR'])+np.max(BflySchn['FRACYEAR']))/2+8, 40, 'Scheiner', va='center', ha='center',color=cmap(1), fontsize=FZ)

# ax3.scatter(BflyGal['FRACYEAR'],BflyGal['Latitude'],MZ,alpha=0.5,color=cmap(7))
# ax3.text( (np.min(BflyGal['FRACYEAR'])+np.max(BflyGal['FRACYEAR']))/2, 57, 'Galileo', va='top', ha='center',color=cmap(7), fontsize=FZ)

# ax3.scatter(BflyMM['Year'],BflyMM['Latitude'],MZ,alpha=0.8,color=cmap(6))
# ax3.text( (np.min(BflyMM['Year'])+np.max(BflyMM['Year']))/2, -57, 'Spoerer and Observers\nfrom the\nParis Observatory', va='bottom', ha='center',color=cmap(6), fontsize=FZ)

# ax3.scatter(BflyStdc['FRACYEAR'],BflyStdc['Lat'],MZ,alpha=0.5,color=cmap(7))
# ax3.text( (np.min(BflyStdc['FRACYEAR'])+np.max(BflyStdc['FRACYEAR']))/2, 57, 'Staudacher', va='top', ha='center',color=cmap(7), fontsize=FZ)


# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(0))
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'Schwabe', va='bottom', ha='center',color=cmap(0), fontsize=FZ)

# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(4))
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, 57, 'Spörer', va='top', ha='center',color=cmap(4), fontsize=FZ)

# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(2))
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'RGO', va='bottom', ha='center',color=cmap(2), fontsize=FZ)

# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(5))
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-4, 57, 'Debrecen', va='top', ha='center',color=cmap(5), fontsize=FZ)

# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color='k')
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'KMAS', va='bottom', ha='center', fontsize=FZ)

ax3.scatter(MG_Obs['FRACYR'],MG_Obs['LAT'],MZ,alpha=0.5,color='c')
ax3.text(1985, -57, 'Magnetic', va='bottom', ha='center', fontsize=FZ, color='c')


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_01.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)', 'Lockwood et al. (2014)', 'Cliver & Ling (2016)', 'Zolotova & Ponyavin (2015)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)

for i in range(0,4):
    t = leg.get_texts()[i]
    t.set_fontproperties(t.get_fontproperties())
    t.set_color('r')
    
t = leg.get_texts()[9]
t.set_fontproperties(t.get_fontproperties())
t.set_color('b')    

alphUn = 0.4

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(LO_Dat['Year'],LO_Dat['G']-LO_Dat['S'], LO_Dat['G']+LO_Dat['S'],color='c', edgecolor='none',alpha=alphUn)
ax1.fill_between(CL_Dat['Year'],CL_Dat['G']-CL_Dat['S'], CL_Dat['G']+CL_Dat['S'],color='m', edgecolor='none',alpha=alphUn)
ax1.fill_between(HS_Dat['Year'],HS_Dat['G']-HS_Dat['S'], HS_Dat['G']+HS_Dat['S'],color=cmap(7), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax1.fill_between(NZM_Dat['Date'],NZM_Dat['G']-NZM_Dat['S'], NZM_Dat['G']+NZM_Dat['S'],color=cmap(4), edgecolor='none',alpha=0.2)


ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


# ax3.scatter(BflyHvl['Year'],BflyHvl['Lat'],MZ,alpha=0.5,color=cmap(2))
# ax3.text( (np.min(BflyHvl['Year'])+np.max(BflyHvl['Year']))/2, 57, 'Hevelius', va='top', ha='center',color=cmap(2), fontsize=FZ)


# ax3.scatter(BflyMM1['Year'],BflyMM1['Lat'],MZ,alpha=0.5,color=cmap(4))
# ax3.text( (np.min(BflyMM1['Year'])+np.max(BflyMM1['Year']))/2, -57, 'Malapert, Gassendi\n& Marcgraf', va='bottom', ha='center',color=cmap(4), fontsize=FZ)


# ax3.scatter(BflySchn['FRACYEAR'],BflySchn['Lat'],MZ,alpha=0.5,color=cmap(1))
# ax3.text( (np.min(BflySchn['FRACYEAR'])+np.max(BflySchn['FRACYEAR']))/2+8, 40, 'Scheiner', va='center', ha='center',color=cmap(1), fontsize=FZ)

# ax3.scatter(BflyGal['FRACYEAR'],BflyGal['Latitude'],MZ,alpha=0.5,color=cmap(7))
# ax3.text( (np.min(BflyGal['FRACYEAR'])+np.max(BflyGal['FRACYEAR']))/2, 57, 'Galileo', va='top', ha='center',color=cmap(7), fontsize=FZ)

# ax3.scatter(BflyMM['Year'],BflyMM['Latitude'],MZ,alpha=0.8,color=cmap(6))
# ax3.text( (np.min(BflyMM['Year'])+np.max(BflyMM['Year']))/2, -57, 'Spoerer and Observers\nfrom the\nParis Observatory', va='bottom', ha='center',color=cmap(6), fontsize=FZ)

# ax3.scatter(BflyStdc['FRACYEAR'],BflyStdc['Lat'],MZ,alpha=0.5,color=cmap(7))
# ax3.text( (np.min(BflyStdc['FRACYEAR'])+np.max(BflyStdc['FRACYEAR']))/2, 57, 'Staudacher', va='top', ha='center',color=cmap(7), fontsize=FZ)


# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(0))
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'Schwabe', va='bottom', ha='center',color=cmap(0), fontsize=FZ)

# ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(4))
# ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, 57, 'Spörer', va='top', ha='center',color=cmap(4), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'RGO', va='bottom', ha='center',color=cmap(2), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(5))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-4, 57, 'Debrecen', va='top', ha='center',color=cmap(5), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color='k')
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-1, -57, 'KMAS', va='bottom', ha='center', fontsize=FZ)

ax3.scatter(MG_Obs['FRACYR'],MG_Obs['LAT'],MZ,alpha=0.5,color='c')
ax3.text(1985, -57, 'Magnetic', va='bottom', ha='center', fontsize=FZ, color='c')


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_02.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    ObsDat = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
    # Plot observer
#     ax1.scatter(Obs1['FRACYEAR'], Obs1['GROUPS']+np.random.rand(1)[0]-0.5,MZ,alpha=0.5,color=0.9*np.random.random(3)+0.1)

    # Finding missing days
    ObsInt  = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL']+1))
    MisDays =  np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL'] ))

    # Creating dataframe with NaNs for days without observations
    year  = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
    month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
    day   = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

    station  = day*0+UnObs[iObs1]
    observer = day*0+1
    groups   = day*np.nan

    fractyear = np.array(list(map(lambda year, month, day : year + (  datetime.date(year, month , day).toordinal()
                                                - datetime.date(year,1,1).toordinal() )
                                                / (  datetime.date(year+1,1,1).toordinal()
                                                - datetime.date(year,1,1).toordinal() ), year, month, day)))

    NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)), columns = ObsDat.columns.values)

    # Append dataframe with missing days
    ObsDat = ObsDat.append(NoObs,ignore_index=True)

    # Recast using original data types
    origType = GN_Dat.dtypes.to_dict()
    ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

    # Sorting according to date
    ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)
    # Smoothing for plotting
    Gss_1D_ker = conv.Gaussian1DKernel(180)
    ObsDat['AVGROUPS']  = conv.convolve(ObsDat['GROUPS'].values,Gss_1D_ker, boundary = 'extend', preserve_nan=True)
    
    ax1.plot(ObsDat['FRACYEAR'], ObsDat['AVGROUPS'], zorder=1, color=0.9*np.random.random(3)+0.1, linewidth=LW)
    

ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyHvl['Year'],BflyHvl['Lat'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyHvl['Year'])+np.max(BflyHvl['Year']))/2, 57, 'Hevelius', va='top', ha='center',color=cmap(2), fontsize=FZ)


ax3.scatter(BflyMM1['Year'],BflyMM1['Lat'],MZ,alpha=0.5,color=cmap(4))
ax3.text( (np.min(BflyMM1['Year'])+np.max(BflyMM1['Year']))/2, -57, 'Malapert, Gassendi\n& Marcgraf', va='bottom', ha='center',color=cmap(4), fontsize=FZ)


ax3.scatter(BflySchn['FRACYEAR'],BflySchn['Lat'],MZ,alpha=0.5,color=cmap(1))
ax3.text( (np.min(BflySchn['FRACYEAR'])+np.max(BflySchn['FRACYEAR']))/2+8, 40, 'Scheiner', va='center', ha='center',color=cmap(1), fontsize=FZ)

ax3.scatter(BflyGal['FRACYEAR'],BflyGal['Latitude'],MZ,alpha=0.5,color=cmap(7))
ax3.text( (np.min(BflyGal['FRACYEAR'])+np.max(BflyGal['FRACYEAR']))/2, 57, 'Galileo', va='top', ha='center',color=cmap(7), fontsize=FZ)

ax3.scatter(BflyMM['Year'],BflyMM['Latitude'],MZ,alpha=0.8,color=cmap(6))
ax3.text( (np.min(BflyMM['Year'])+np.max(BflyMM['Year']))/2, -57, 'Spoerer and Observers\nfrom the\nParis Observatory', va='bottom', ha='center',color=cmap(6), fontsize=FZ)

ax3.scatter(BflyStdc['FRACYEAR'],BflyStdc['Lat'],MZ,alpha=0.5,color=cmap(7))
ax3.text( (np.min(BflyStdc['FRACYEAR'])+np.max(BflyStdc['FRACYEAR']))/2, 57, 'Staudacher', va='top', ha='center',color=cmap(7), fontsize=FZ)


ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(0))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'Schwabe', va='bottom', ha='center',color=cmap(0), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(4))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==2,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, 57, 'Spörer', va='top', ha='center',color=cmap(4), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(2))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1004,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2, -57, 'RGO', va='bottom', ha='center',color=cmap(2), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color=cmap(5))
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1003,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-4, 57, 'Debrecen', va='top', ha='center',color=cmap(5), fontsize=FZ)

ax3.scatter(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'],BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'LATITUDE'],MZ,alpha=0.5,color='k')
ax3.text( (np.min(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR'])+np.max(BflyMod.loc[np.logical_and(BflyMod['SURVEY']==1002,BflyMod['CORR_A']>=ALim),'FRACYEAR']))/2-1, -57, 'KMAS', va='bottom', ha='center', fontsize=FZ)

ax3.scatter(MG_Obs['FRACYR'],MG_Obs['LAT'],MZ,alpha=0.5,color='c')
ax3.text(1985, -57, 'Magnetic', va='bottom', ha='center', fontsize=FZ, color='c')


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_03.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5


ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)
# ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)


leg = ax1.legend(('Wolf Sunspot Number/20\nClette & Lefèvre (2016)', 'Hoyt & Schatten (1998)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)


for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    ObsDat = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
    # Plot observer
#     ax1.scatter(Obs1['FRACYEAR'], Obs1['GROUPS']+np.random.rand(1)[0]-0.5,MZ,alpha=0.5,color=0.9*np.random.random(3)+0.1)

    # Finding missing days
    ObsInt  = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL']+1))
    MisDays =  np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL'] ))

    # Creating dataframe with NaNs for days without observations
    year  = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
    month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
    day   = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

    station  = day*0+UnObs[iObs1]
    observer = day*0+1
    groups   = day*np.nan

    fractyear = np.array(list(map(lambda year, month, day : year + (  datetime.date(year, month , day).toordinal()
                                                - datetime.date(year,1,1).toordinal() )
                                                / (  datetime.date(year+1,1,1).toordinal()
                                                - datetime.date(year,1,1).toordinal() ), year, month, day)))

    NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)), columns = ObsDat.columns.values)

    # Append dataframe with missing days
    ObsDat = ObsDat.append(NoObs,ignore_index=True)

    # Recast using original data types
    origType = GN_Dat.dtypes.to_dict()
    ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

    # Sorting according to date
    ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)
    # Smoothing for plotting
    Gss_1D_ker = conv.Gaussian1DKernel(180)
    ObsDat['AVGROUPS']  = conv.convolve(ObsDat['GROUPS'].values,Gss_1D_ker, boundary = 'extend', preserve_nan=True)
    
    ax1.plot(ObsDat['FRACYEAR'], ObsDat['AVGROUPS'], zorder=1, color=0.9*np.random.random(3)+0.1, linewidth=LW)
    

ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_SN.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)
ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)


leg = ax1.legend(('Wolf Sunspot Number/20\nClette & Lefèvre (2016)', 'Hoyt & Schatten (1998)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)


for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    ObsDat = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
    # Plot observer
#     ax1.scatter(Obs1['FRACYEAR'], Obs1['GROUPS']+np.random.rand(1)[0]-0.5,MZ,alpha=0.5,color=0.9*np.random.random(3)+0.1)

    # Finding missing days
    ObsInt  = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL']+1))
    MisDays =  np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL'] ))

    # Creating dataframe with NaNs for days without observations
    year  = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
    month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
    day   = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

    station  = day*0+UnObs[iObs1]
    observer = day*0+1
    groups   = day*np.nan

    fractyear = np.array(list(map(lambda year, month, day : year + (  datetime.date(year, month , day).toordinal()
                                                - datetime.date(year,1,1).toordinal() )
                                                / (  datetime.date(year+1,1,1).toordinal()
                                                - datetime.date(year,1,1).toordinal() ), year, month, day)))

    NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)), columns = ObsDat.columns.values)

    # Append dataframe with missing days
    ObsDat = ObsDat.append(NoObs,ignore_index=True)

    # Recast using original data types
    origType = GN_Dat.dtypes.to_dict()
    ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

    # Sorting according to date
    ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)
    # Smoothing for plotting
    Gss_1D_ker = conv.Gaussian1DKernel(180)
    ObsDat['AVGROUPS']  = conv.convolve(ObsDat['GROUPS'].values,Gss_1D_ker, boundary = 'extend', preserve_nan=True)
    
    ax1.plot(ObsDat['FRACYEAR'], ObsDat['AVGROUPS'], zorder=1, color=[0.8,0.8,0.8],alpha=1, linewidth=LW)
    

ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)    
ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
# ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
# ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
# ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
# ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
# ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
# ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
# ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')



    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_SN01.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)
# ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)


leg = ax1.legend(('Wolf Sunspot Number/20\nClette & Lefèvre (2016)', 'Hoyt & Schatten (1998)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)


# for iObs1 in range(UnObs.shape[0]):

#     #---------------------
#     # 1. Pick an observer    
#     #---------------------
    
#     ObsDat = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
#     # Plot observer
# #     ax1.scatter(Obs1['FRACYEAR'], Obs1['GROUPS']+np.random.rand(1)[0]-0.5,MZ,alpha=0.5,color=0.9*np.random.random(3)+0.1)

#     # Finding missing days
#     ObsInt  = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL']+1))
#     MisDays =  np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL'] ))

#     # Creating dataframe with NaNs for days without observations
#     year  = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
#     month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
#     day   = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

#     station  = day*0+UnObs[iObs1]
#     observer = day*0+1
#     groups   = day*np.nan

#     fractyear = np.array(list(map(lambda year, month, day : year + (  datetime.date(year, month , day).toordinal()
#                                                 - datetime.date(year,1,1).toordinal() )
#                                                 / (  datetime.date(year+1,1,1).toordinal()
#                                                 - datetime.date(year,1,1).toordinal() ), year, month, day)))

#     NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)), columns = ObsDat.columns.values)

#     # Append dataframe with missing days
#     ObsDat = ObsDat.append(NoObs,ignore_index=True)

#     # Recast using original data types
#     origType = GN_Dat.dtypes.to_dict()
#     ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

#     # Sorting according to date
#     ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)
#     # Smoothing for plotting
#     Gss_1D_ker = conv.Gaussian1DKernel(180)
#     ObsDat['AVGROUPS']  = conv.convolve(ObsDat['GROUPS'].values,Gss_1D_ker, boundary = 'extend', preserve_nan=True)
    
#     ax1.plot(ObsDat['FRACYEAR'], ObsDat['AVGROUPS'], zorder=1, color=[0.8,0.8,0.8],alpha=1, linewidth=LW)
    

    
# ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
# ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
# ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
# ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
# ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
# ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
# ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
# ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)


    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_WSN01.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)    
ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)


leg = ax1.legend(('Wolf Sunspot Number/20\nClette & Lefèvre (2016)', 'Hoyt & Schatten (1998)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)



for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    ObsDat = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
    # Plot observer
#     ax1.scatter(Obs1['FRACYEAR'], Obs1['GROUPS']+np.random.rand(1)[0]-0.5,MZ,alpha=0.5,color=0.9*np.random.random(3)+0.1)

    # Finding missing days
    ObsInt  = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL']+1))
    MisDays =  np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL'] ))

    # Creating dataframe with NaNs for days without observations
    year  = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
    month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
    day   = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

    station  = day*0+UnObs[iObs1]
    observer = day*0+1
    groups   = day*np.nan

    fractyear = np.array(list(map(lambda year, month, day : year + (  datetime.date(year, month , day).toordinal()
                                                - datetime.date(year,1,1).toordinal() )
                                                / (  datetime.date(year+1,1,1).toordinal()
                                                - datetime.date(year,1,1).toordinal() ), year, month, day)))

    NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)), columns = ObsDat.columns.values)

    # Append dataframe with missing days
    ObsDat = ObsDat.append(NoObs,ignore_index=True)

    # Recast using original data types
    origType = GN_Dat.dtypes.to_dict()
    ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

    # Sorting according to date
    ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)
    # Smoothing for plotting
    Gss_1D_ker = conv.Gaussian1DKernel(180)
    ObsDat['AVGROUPS']  = conv.convolve(ObsDat['GROUPS'].values,Gss_1D_ker, boundary = 'extend', preserve_nan=True)
    
    ax1.plot(ObsDat['FRACYEAR'], ObsDat['AVGROUPS'], zorder=1, color=[0.8,0.8,0.8],alpha=1, linewidth=LW)
    

ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)    
ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
# ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
# ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
# ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
# ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
# ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
# ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')



    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_SN02.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)


for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    ObsDat = GN_Dat[GN_Dat['STATION'] == UnObs[iObs1]]
    
    # Plot observer
#     ax1.scatter(Obs1['FRACYEAR'], Obs1['GROUPS']+np.random.rand(1)[0]-0.5,MZ,alpha=0.5,color=0.9*np.random.random(3)+0.1)

    # Finding missing days
    ObsInt  = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL']+1))
    MisDays =  np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL'] ))

    # Creating dataframe with NaNs for days without observations
    year  = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
    month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
    day   = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

    station  = day*0+UnObs[iObs1]
    observer = day*0+1
    groups   = day*np.nan

    fractyear = np.array(list(map(lambda year, month, day : year + (  datetime.date(year, month , day).toordinal()
                                                - datetime.date(year,1,1).toordinal() )
                                                / (  datetime.date(year+1,1,1).toordinal()
                                                - datetime.date(year,1,1).toordinal() ), year, month, day)))

    NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)), columns = ObsDat.columns.values)

    # Append dataframe with missing days
    ObsDat = ObsDat.append(NoObs,ignore_index=True)

    # Recast using original data types
    origType = GN_Dat.dtypes.to_dict()
    ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

    # Sorting according to date
    ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)
    # Smoothing for plotting
    Gss_1D_ker = conv.Gaussian1DKernel(180)
    ObsDat['AVGROUPS']  = conv.convolve(ObsDat['GROUPS'].values,Gss_1D_ker, boundary = 'extend', preserve_nan=True)
    
    ax1.plot(ObsDat['FRACYEAR'], ObsDat['AVGROUPS'], zorder=1, color=[0.8,0.8,0.8],alpha=1, linewidth=LW)
    

    
ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
# ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
# ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
# ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
# ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
# ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
# ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
# ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)


    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_SN03.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(LO_Dat['Year'],LO_Dat['G'], zorder=1, color='c',linestyle='-', linewidth=LW)
ax1.plot(CL_Dat['Year'],CL_Dat['G'], zorder=1, color='m',linestyle='-', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

leg = ax1.legend(('Hoyt & Schatten (1998)', 'Lockwood et al. (2014)', 'Cliver & Ling (2016)', 'Zolotova & Ponyavin (2015)','Svalgaard & Schatten (2016)', 'Willamo et al. (2017)'
            , 'Ivanov & Miletsky (2017)','Vaquero et al. (2015)', 'Chatzistergos et al. (2017)','Wolf Sunspot Number/20\nClette & Lefèvre (2016)'), 
           ncol = 4, loc = 2, fontsize=28, frameon=False, columnspacing=1)

for i in range(0,4):
    t = leg.get_texts()[i]
    t.set_fontproperties(t.get_fontproperties())
    t.set_color('r')
    
t = leg.get_texts()[9]
t.set_fontproperties(t.get_fontproperties())
t.set_color('b')    

alphUn = 0.4

ax1.fill_between(U_Dat['Year'],U_Dat['G']-U_Dat['S'], U_Dat['G']+U_Dat['S'],color=cmap(6), edgecolor='none',alpha=alphUn)
ax1.fill_between(SS_Dat['Year'],SS_Dat['G']-SS_Dat['S'], SS_Dat['G']+SS_Dat['S'],color=cmap(1), edgecolor='none',alpha=alphUn)
ax1.fill_between(LO_Dat['Year'],LO_Dat['G']-LO_Dat['S'], LO_Dat['G']+LO_Dat['S'],color='c', edgecolor='none',alpha=alphUn)
ax1.fill_between(CL_Dat['Year'],CL_Dat['G']-CL_Dat['S'], CL_Dat['G']+CL_Dat['S'],color='m', edgecolor='none',alpha=alphUn)
ax1.fill_between(HS_Dat['Year'],HS_Dat['G']-HS_Dat['S'], HS_Dat['G']+HS_Dat['S'],color=cmap(7), edgecolor='none',alpha=alphUn)
ax1.fill_between(JV_Dat['Year'],JV_Dat['S1'], JV_Dat['S2'],color=cmap(0), edgecolor='none',alpha=alphUn)
ax1.fill_between(Ch_Dat['Year'],Ch_Dat['G']-Ch_Dat['S2'], Ch_Dat['G']+Ch_Dat['S1'],color=cmap(2), edgecolor='none',alpha=alphUn)
ax1.fill_between(IM_Dat['Year'],IM_Dat['G']-IM_Dat['S'], IM_Dat['G']+IM_Dat['S'],color=cmap(3), edgecolor='none',alpha=alphUn)
ax1.fill_between(NZM_Dat['Date'],NZM_Dat['G']-NZM_Dat['S'], NZM_Dat['G']+NZM_Dat['S'],color=cmap(4), edgecolor='none',alpha=0.2)


ax1.plot(HS_Dat['Year'],HS_Dat['G'], zorder=1, color=cmap(7), linewidth=LW)
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color=cmap(1),linestyle=':',linewidth=LW)
ax1.plot(U_Dat['Year'],U_Dat['G'], zorder=1, color=cmap(6),linestyle='--', linewidth=LW)
ax1.plot(NZM_Dat['Date'],NZM_Dat['G'], zorder=1, color=cmap(4),linestyle='-.', linewidth=LW)
ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)

ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_SN04.png', bbox_inches='tight')

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 1700    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 3      # Number of vertical panels

# Padding
padv  = 200 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 300 #Horizontal padding in pixels at the edge of the figure
padh2 = 150 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2020

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([1.6*ppadh, ppadv+ppxy, ppxx, ppxy])

#Colors
cmap = plt.cm.get_cmap('Set1')

#LineWidth
LW = 5

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))


ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=19, bottom=0)
ax1.set_xticklabels([])
ax1.set_ylabel('Group Number')
# ax1.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax2 = fig.add_axes([1.6*ppadh, ppadv, ppxx, ppxy])

#Node Size
NZ =50

#LineWidth
LW = 2

nx.draw_networkx_nodes(ObsNtwrk,pos,node_size=15, node_color='k',alpha=0.3, ax = ax2)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ, node_color='k',alpha=0.2, ax = ax2)
nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[0]),pos,node_size=NZ,alpha=0.1, node_color='g', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[0]),pos,alpha=0.2, edge_color='g', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[1]),pos,node_size=NZ, node_color='b', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[1]),pos,alpha=0.2, edge_color='b', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[2]),pos,node_size=NZ, node_color='r', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[2]),pos,alpha=0.2, edge_color='r', ax = ax2, width=LW)

nx.draw_networkx_nodes(ObsNtwrk.subgraph(Clst[3]),pos,node_size=NZ, node_color='c', ax = ax2)
nx.draw_networkx_edges(ObsNtwrk.subgraph(Clst[3]),pos,alpha=0.2, edge_color='c', ax = ax2, width=LW)


ax2.set_xlim(left=Y1, right=Y2)
ax2.set_yticks([])
# ax2.set_xticklabels([])
ax2.set_ylabel('Networks of\nOverlaping\nObservers')


ax3 = fig.add_axes([1.6*ppadh, ppadv+2*ppxy, ppxx, ppxy])

#Area limit for modern data
ALim = 10

#Marker Size
MZ = 10

#Font size
FZ = 30


cmap = plt.cm.get_cmap('Accent')


ax3.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color=[0.3,0.3,0.3])


ax3.set_xlim(left=Y1, right=Y2)
ax3.set_ylim(top=60, bottom=-60)
ax3.set_ylabel('Latitude (o)');
# ax3.grid(color=(0.2,0.2,0.2), linestyle='--', linewidth=1, axis='x', which='both')

ax3.xaxis.tick_top()
ax3.xaxis.set_label_position('top')


# fig.savefig('F2_Solar_Cycle_Obs_Data.png',dpi=figDPI)

fig.savefig('Pres_Solar_Cycle_Obs_Data_SN05.png', bbox_inches='tight')

# %%


# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 4000   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units

MZ = 50
## Start Figure
Y1 = 1670
Y2 = 1725

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# # Plot transparency mask
# for i in np.arange(0,CrvObsYr.shape[0]):
#     if CrvObsCv[i]==0:
#         clr = color0
#     else:
#         clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
#     ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
# for i in np.arange(0,BfObsYr.shape[0]):
#     if BfObsCv[i]==0:
#         clr = color0
#     else:
#         clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
#     ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.9,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage.png', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 4000   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units

MZ = 50
## Start Figure
Y1 = 1950
Y2 = 2005

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# # Plot transparency mask
# for i in np.arange(0,CrvObsYr.shape[0]):
#     if CrvObsCv[i]==0:
#         clr = color0
#     else:
#         clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
#     ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
# for i in np.arange(0,BfObsYr.shape[0]):
#     if BfObsCv[i]==0:
#         clr = color0
#     else:
#         clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
#     ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.9,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage_02.png', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 4000   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units

MZ = 50
## Start Figure
Y1 = 1670
Y2 = 1725

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# # Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.9,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage_03.png', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 4000   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units

MZ = 50
## Start Figure
Y1 = 1950
Y2 = 2005

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# # Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.9,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage_04.png', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2010

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage_05.png', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 2      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 0  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + (npv-1)*padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 2010

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.cm.get_cmap('Accent')
ax1.plot(SS_Dat['Year'],SS_Dat['G'], zorder=1, color='k',linestyle='-',linewidth=6)
# ax1.plot(IM_Dat['Year'],IM_Dat['G'], zorder=1, color=cmap(3),linestyle='--', linewidth=LW)
# ax1.plot(JV_Dat['Year'],JV_Dat['G'], zorder=1, color=cmap(0), linewidth=LW)
# ax1.plot(Ch_Dat['Year'],Ch_Dat['G'], zorder=1, color=cmap(2), linewidth=LW,linestyle='--')
# ax1.plot(WSN_Dat['Year'],WSN_Dat['G'], zorder=1, color='k',linestyle=':', linewidth=LW)


    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.set_ylim(top=16.49, bottom=0)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('Group\nNumber')

ax2 = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# # Plot transparency mask
# for i in np.arange(0,BfObsYr.shape[0]):
#     if BfObsCv[i]==0:
#         clr = color0
#     else:
#         clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
#     ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])

# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage_05a.png', bbox_inches='tight')

# %%

plt.rc('font', **font)

# Size definitions
dpi = 300
pxx = 8600   # Horizontal size of each panel
pxy = 2000    # Vertical size of each panel

nph = 1      # Number of horizontal panels
npv = 4      # Number of vertical panels

# Padding
padv  = 300 #Vertical padding in pixels
padv2 = 500  #Vertical padding in pixels between panels
padh  = 600 #Horizontal padding in pixels at the edge of the figure
padh2 = 350 #Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv*pxy + 2*padv + padv2 )      #Vertical size of figure in pixels
fszh = (nph*pxx + 2*padh + (nph-1)*padh2 )      #Horizontal size of figure in pixels

# Conversion to relative unites
ppxx   = pxx/fszh
ppxy   = pxy/fszv
ppadv  = padv/fszv     #Vertical padding in relative units
ppadv2 = padv2/fszv    #Vertical padding in relative units
ppadh  = padh/fszh     #Horizontal padding the edge of the figure in relative units
ppadh2 = padh2/fszh    #Horizontal padding between panels in relative units


## Start Figure
Y1 = 1600
Y2 = 1810

fig = plt.figure(figsize=(fszh/dpi,fszv/dpi))

ax1 = fig.add_axes([ppadh, ppadv+3*ppxy+ppadv2, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1.annotate('(a)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1.set_xlim(left=Y1, right=Y2)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylabel('GN Masked Using\nObserved Days')

ax2 = fig.add_axes([ppadh, ppadv+2*ppxy+ppadv2, ppxx, ppxy])

# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

# ax2.annotate('(b)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2.set_xlim(left=Y1, right=Y2)
ax2.set_ylim(top=59.5, bottom=-59.5)
ax2.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3  = fig.add_axes([ppadh+ppxx, ppadv+2*ppxy+ppadv2, ppadh2, ppxy])

axcb3.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3.set_ylim(bottom = 0, top = 4)
axcb3.set_xlim(left=0, right=1)    

axcb3.set_xticks([])
axcb3.set_yticks([])
axcb3.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2  = fig.add_axes([ppadh+ppxx, ppadv+3*ppxy+ppadv2, ppadh2, ppxy])
for i in range(1,6):
    axcb2.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2.set_ylim(bottom = 1, top = 6)
axcb2.set_xlim(left=0, right=1)    

axcb2.set_xticks([])
axcb2.set_yticks([])



Y1 = 1810
Y2 = 2020
ax1b = fig.add_axes([ppadh, ppadv+ppxy, ppxx, ppxy])

cmap = plt.get_cmap('Purples')
pch = ax1b.pcolormesh(YrBl, GrpsBl, np.transpose(GrpsBM), alpha =1, cmap=cmap,edgecolors='None',vmin = -0.5, vmax = np.nanmax(GrpsBM))

# Plot transparency mask
for i in np.arange(0,CrvObsYr.shape[0]):
    if CrvObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(CrvObsCv[i]/0.05)])))]
    ax1b.fill([CrvObsYr[i], CrvObsYr[i], CrvObsYr[i]+YrCum, CrvObsYr[i]+YrCum], [0, 17, 17, 0], color=clr, alpha = (1-CrvObsCv[i])*MaskMaxAl, zorder=2, edgecolor = 'none')

# ax1b.annotate('(c)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')    
    
ax1b.set_xlim(left=Y1, right=Y2)
ax1b.xaxis.tick_top()
ax1b.xaxis.set_label_position('top')
ax1b.set_ylabel('GN Masked Using\nObserved Days')

ax2b = fig.add_axes([ppadh, ppadv, ppxx, ppxy])

# Plot transparency mask
# Plot transparency mask
for i in np.arange(0,BfObsYr.shape[0]):
    if BfObsCv[i]==0:
        clr = color0
    else:
        clr = colorsMsk[(int(np.min([3,np.floor(BfObsCv[i]/0.05)])))]
    ax2b.fill([BfObsYr[i], BfObsYr[i], BfObsYr[i]+YrCum, BfObsYr[i]+YrCum], [-60, 60, 60, -60], color=clr, alpha = (1-BfObsCv[i])*MaskMaxAl, zorder=0, edgecolor = 'none')
        
ax2b.scatter(BflyAll['FRACYEAR'],BflyAll['LATITUDE'],MZ,alpha=0.3,color='k')

# ax2b.annotate('(d)',xy=(1, 0.99),xycoords='axes fraction',fontsize=60,va='top', ha='right')

ax2b.set_xlim(left=Y1, right=Y2)
ax2b.set_ylim(top=59.5, bottom=-59.5)
ax2b.set_ylabel('Latitude (o)');
# ax2.grid(color=(0.5,0.5,0.5), linestyle='--', linewidth=1, axis='x', which='both')

# Add Colorbar
axcb3b  = fig.add_axes([ppadh+ppxx, ppadv, ppadh2, ppxy])

axcb3b.fill([0,0,1,1], [0, 1, 1, 0], color=color0, alpha = MaskMaxAl, edgecolor = 'none')
axcb3b.annotate('0%', xy = (0.5, 0.5), horizontalalignment='center', verticalalignment='center',
               fontsize=27,zorder=3, )
for i in range(1,4):
    clr = colorsMsk[i-1]
    axcb3b.fill([0,0,1,1], [i, i+1, i+1, i], color=clr, alpha = (1-i/20)*MaskMaxAl, edgecolor = 'none')
    axcb3b.annotate('<'+str(5*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb3b.set_ylim(bottom = 0, top = 4)
axcb3b.set_xlim(left=0, right=1)    

axcb3b.set_xticks([])
axcb3b.set_yticks([])
axcb3b.set_xlabel('Obsv.\nCoverage')


# Add Colorbar
axcb2b  = fig.add_axes([ppadh+ppxx, ppadv+ppxy, ppadh2, ppxy])
for i in range(1,6):
    axcb2b.fill([0,0,1,1], [i, i+1, i+1, i], color=(0.5,0.5,0.5), alpha = (1-i/5)*MaskMaxAl, edgecolor = 'none')
    axcb2b.annotate(str(20*i)+'%', xy = (0.5, i+0.5), horizontalalignment='center', verticalalignment='center',
                   fontsize=27,zorder=3, )

axcb2b.set_ylim(bottom = 1, top = 6)
axcb2b.set_xlim(left=0, right=1)    

axcb2b.set_xticks([])
axcb2b.set_yticks([])



# fig.savefig('F4a_Bfly_GN_Coverage.png',dpi=figDPI)
fig.savefig('Pres_Solar_Cycle_Bfly_GN_Coverage_06.png', bbox_inches='tight')

# %%


