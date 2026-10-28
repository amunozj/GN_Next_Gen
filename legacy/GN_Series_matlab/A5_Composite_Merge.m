clear
close all

load GN_181_composites.mat

Mrg = nansum(CalCmp,1)./nansum(isfinite(CalCmp));
Unc = nanstd(CalCmp,1)


plot(date2,Mrg,'.')


