# ---------------------------------------------------------------------------
# LEGACY NOTEBOOK EXPORT (code + markdown only; outputs/figures stripped)
# Source:   SN_01.ipynb
# Status:   tracked, modified (uncommitted)
# SHA-256 of original .ipynb: 5d5d26406200df3c2f96aaad991c0373bdd9335db9f179e84ba452e3a92583a4
# Cells:    25 (code 15, markdown 10, raw 0)
# Exported: 2026-10-28 by legacy/tools/export_notebook_text.ps1
# Format:   py:percent ('# %%' = code cell, '# %% [markdown]' = markdown cell)
# Data paths inside refer to the ORIGINAL repo layout; see legacy/README.md.
# ---------------------------------------------------------------------------

# %% [markdown]
# ## Setup

# %%
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd 
import time
import datetime
import calendar
import networkx as nx

# %% [markdown]
# ## Reading Observations and Observers
#
# The files used were downloaded from [SILSO's Group Number page](http://www.sidc.be/silso/groupnumberv3).  The current version used is JV_V1-12, but turned into a .csv file.  The original file contains leap years that should not be there, which were removed by hand (1700, 1800, 1900).
#
# ### Reading Observations

# %%
GN_Dat = pd.read_csv('GNObservations_JV_V1-12.csv', quotechar = '"', encoding = 'ansi',header = 15)

# %% [markdown]
# We add two more columns to the data:
#     1. ORDINAL:  number of days after (0001,01,01).
#     2. FRACYEAR: Fractional year for plotting.

# %%
GN_Dat['ORDINAL'] = GN_Dat.apply(lambda x: datetime.date(x['YEAR'],x['MONTH'],x['DAY']).toordinal(),axis=1)
GN_Dat['FRACYEAR'] = GN_Dat.apply(lambda x: x['YEAR']
                                            + (  datetime.date(x['YEAR'],x['MONTH'],x['DAY']).toordinal()
                                               - datetime.date(x['YEAR'],1,1).toordinal() )
                                            / (  datetime.date(x['YEAR']+1,1,1).toordinal()
                                               - datetime.date(x['YEAR'],1,1).toordinal() )
                                  ,axis=1)
print(GN_Dat)

# %% [markdown]
# ### Reading Observers

# %%
GN_Obs = pd.read_csv('GNObservers_JV_V1-12.csv', quotechar = '"', encoding = 'ansi')
print(GN_Obs)

# %% [markdown]
# ## Pre-allocating Variables
#
# We create a variable to store each unique observer.  **NOTE THAT CURRENTLY EACH STATION HAS _ONLY_ ONE OBSERVER SO WE DO ALL OUR LOGICAL OPERATIONS USING THE 'STATION' FIELD**.
#
# We also remove the '0' station which indicates a day without observations.

# %%
UnObs = sp.unique(GN_Dat.STATION)
UnObs = UnObs[UnObs>0]

# %% [markdown]
# And create empty sparse connectivity matrices that will be used to define the properties of the observation network

# %%
#Number of days of overlap
LnkDays = sp.sparse.lil_matrix((UnObs.shape[0], UnObs.shape[0]))

# %% [markdown]
# ### Day adjacency tolerance
#
# Since the amount of groups visible in the Sun don't change radically from day to day, we specify a window of tolerance in which two observations are considered to be _"simultaneous"_.  First priority is given to observations taken on the same calendar date, then to observations within one day are considered, then within two days, and so on until reaching the size of the tolerance window.  The process is ran simultaneously for prior and following days.

# %%
DayTol = 10

# %% [markdown]
# ### Minimum days of overlap
#
# In order to establish a link between two observers in the network, we require to have minimum amount of days of overlap (considering as overlapping days days within DayTol of each other).

# %%
MinOvr = 10

# %% [markdown]
# ## Creating the adjacency matrix
#
# Now we go through all observers **_(stations)_** executing the following algorithm:
#
# 1. Pick an observer _(station)_.
# 2. Find all observers _(stations)_ that have overlapping observations within DayTol.
# 3. Go through all overlaping observers _(stations)_ and define the adjacency matrices.

# %%
pos = {} #Dictionary with positions for plotting the network
for iObs1 in range(UnObs.shape[0]):

    #---------------------
    # 1. Pick an observer    
    #---------------------
    
    Obs1 = GN_Dat[GN_Dat.STATION == UnObs[iObs1]]
    
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
    
    
    #---------------------
    # 3. Go through all overlaping observers and define the adjacency matrices.
    #---------------------
    
    for iObsOvrlp in range(ObsOvrlp.shape[0]):
        
        #Determine the overlapping observer
        iObs2 = (UnObs==ObsOvrlp[iObsOvrlp]).nonzero()
        Obs2 = GN_Dat[GN_Dat.STATION == ObsOvrlp[iObsOvrlp]]
        DatObs2 = Obs2.ORDINAL
        
        #Find the dates in common (including days within DayTol)
        DatIntrsct = sp.intersect1d(DatObs1,DatObs2)
        
        #Fill the days of overlap adjacency matrix
        if DatIntrsct.shape[0] >= MinOvr:
            LnkDays[iObs1,iObs2] = DatIntrsct.shape[0]

# %% [markdown]
# ## Creating and plotting graph network

# %%
#Creating Graph Network
ObsNtwrk = nx.from_scipy_sparse_matrix(LnkDays,create_using=nx.Graph())

#Clusters existing in the network
Clst = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)

#Plotting Graph Network
plt.figure(figsize=(17,4))

#SubNtwrk = ObsNtwrk.subgraph(Clst[1])
SubNtwrk = ObsNtwrk

nx.draw_networkx_nodes(SubNtwrk,pos,node_size=10)
nx.draw_networkx_edges(SubNtwrk,pos,alpha=0.1)
plt.show()

# %%
t = sorted(nx.connected_components(ObsNtwrk), key = len, reverse=True)
len(t[1])

# %%
subNtwrk = ObsNtwrk.subgraph(Clst[1])

# %%
a = 3

# %%
a

# %%


