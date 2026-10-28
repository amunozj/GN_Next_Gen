# SN_Daisy (legacy, 2017)

`legacy/SN_Daisy/notebooks_text/SN_01.py`: reads the GN database V1-12, defines a day-adjacency tolerance (`DayTol`)
under which observations on nearby days count as simultaneous (same day first, then ±1, ±2 …), requires a minimum number
of overlap days, and fills sparse connectivity matrices describing the observer network. It is the earliest version of
the network logic later used in `AMJ_JMV_MM_Perspective` (Python) and `GN_Series_matlab` (MATLAB). Nothing here needs
porting beyond what [gn_series_matlab.md](gn_series_matlab.md) and [reuse_map.md](reuse_map.md) already cover.
