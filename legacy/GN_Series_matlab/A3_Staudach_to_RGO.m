close all

clear

% load GN_Links_Mat_Fits.mat;
% load GN_Links_Mat_Fits_Non_Robust.mat;

% load GN_Links_Mat_Fits_Bisquare_All.mat;
% load GN_Links_Mat_Fits_LAR_with_Catch.mat
% load GN_Links_Mat_Fits_Robust;
load GN_Links_Mat_Fits_Bisquare_3.mat;


load GNobservations_JV_V1-12.mat


Clr100 = [255,208,171]/255;
Clr90 = [114,178,229]/255;
Clr70 = [81,178,76]/255;
Clr50 = [111,40,124]/255;
Clr30 = [80,60,0]/255;

ClrD = [1 0 0];
ClrN = [[1 1 0];[1 0 1];[0 1 1]];


Clr = [[1 1 1]*0.6;ClrD;Clr70;Clr90;Clr50;Clr30];



date = datenum(GN_Dat(:,1),GN_Dat(:,2),GN_Dat(:,3));
date2 = GN_Dat(:,1) + (date-datenum(GN_Dat(:,1),1,1))./(datenum(GN_Dat(:,1)+1,1,1)-datenum(GN_Dat(:,1),1,1));

GN_Dat = [date2 date GN_Dat];

date = min(date):max(date);
[Y,M,D] = datevec(date);
date2 = Y + (date-datenum(Y,1,1))./(datenum(Y+1,1,1)-datenum(Y,1,1));

%Removing days without observations

GN_Dat = GN_Dat(GN_Dat(:,7)==1,:);

%Removing RGO counts before 1915
GN_Dat((GN_Dat(:,6)==332)&(GN_Dat(:,3)<1915),:)=NaN;
% GN_Dat((GN_Dat(:,6)==332),:)=NaN;

%Choosing a specific year
% GN_Dat(GN_Dat(:,3)>1950,:)=NaN;

GN_Dat = GN_Dat(isfinite(GN_Dat(:,3)),:);

%% Creating link matrix

Unob = unique(GN_Dat(:,6));
Unob = Unob(Unob>0);


dtol = 2;
MinCon = 4;
MinRho = 0.6;
Rtol = 0.9;

LnkMO((LnkMPt<MinCon)|(LnkMR<MinRho)|isnan(LnkMR)) = 0;

%% Finding shortest distances with different metrics

%DM1 = max(LnkMO(:))-LnkMO+1; %Size of overlap
DM1 = (LnkUNP./LnkVlP).^2;

%Finding spots where we prefer a linear fit
LinIn = (LnkUNPR<Rtol*LnkUNLR)&( abs(LnkVlP-LnkVlLa)>LnkUNLa )&( abs(LnkVlLb)>LnkUNLb );

DM1(LinIn) = (LnkUNLa(LinIn)./LnkVlLa(LinIn)).^2;

% DM1 = DM1./LnkMO;

DM1(LnkUNP==0) = 0;

%% Starting Node
%Staudach
stnd  = 445;

%% Ending Node
%RGO
endnd = 311;

DM1(DM1==0) = Inf;

[shortestPaths, totalCosts] = kShortestPath(DM1, stnd, endnd, 10000);

save('GN_Links_Mat_Fits_Bisquare_paths_10000.mat', 'dpos','ypos', 'shortestPaths', 'totalCosts' ,'LnkMO','LnkMPt', 'LnkMR', 'LnkVlP', 'LnkUNP', 'LnkUNPR', 'LnkVlLa', 'LnkUNLa', 'LnkVlLb', 'LnkUNLb', 'LnkUNLR')
