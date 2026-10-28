clear

close all



load GNobservations_JV_V1-12.mat;

date = datenum(GN_Dat(:,1),GN_Dat(:,2),GN_Dat(:,3));
date2 = GN_Dat(:,1) + (date-datenum(GN_Dat(:,1),1,1))./(datenum(GN_Dat(:,1)+1,1,1)-datenum(GN_Dat(:,1),1,1));

GN_Dat = [date2 date GN_Dat];

date = min(date):max(date);
[Y,M,D] = datevec(date);
date2 = Y + (date-datenum(Y,1,1))./(datenum(Y+1,1,1)-datenum(Y,1,1));

%Removing days without observations

GN_Dat = GN_Dat(GN_Dat(:,7)==1,:);
minSrs = [date2' date' date'*NaN date'*0];

for i=min(date):max(date)
    
    [i max(date)]
    
    tmpin = find(GN_Dat(:,2)==i);
    
    if ~isempty(tmpin)
    
        minSrs(i-min(date)+1,3) = min(GN_Dat(tmpin,8));
        minSrs(i-min(date)+1,4) = length(tmpin);
        
    end
    
end



tmp = minSrs(:,3);
tmp(minSrs(:,4)<2) = NaN;

figure(66)
h = gcf;
set(h,'Position',[50 50 1500 500], 'Color','w');
h1 = axes('FontSize',17);

plot(minSrs(:,1),tmp)

title('Minimum Sunspot Group seen by at least 2 obervers')
ylabel('Group Number')

ylim([0 20])

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = 0.075;
p1(3) = 0.9;
h1.Position = p1;

export_fig 'Min_GN_2_Observers.png' -nocrop -m2



tmp = minSrs(:,3);
tmp(minSrs(:,4)<3) = NaN;

figure(67)
h = gcf;
set(h,'Position',[50 50 1500 500], 'Color','w');
h1 = axes('FontSize',17);

plot(minSrs(:,1),tmp)

title('Minimum Sunspot Group seen by at least 3 obervers')
ylabel('Group Number')

ylim([0 20])

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = 0.075;
p1(3) = 0.9;
h1.Position = p1;

export_fig 'Min_GN_3_Observers.png' -nocrop -m2


tmp = minSrs(:,3);
tmp(minSrs(:,4)<4) = NaN;

figure(68)
h = gcf;
set(h,'Position',[50 50 1500 500], 'Color','w');
h1 = axes('FontSize',17);

plot(minSrs(:,1),tmp)

title('Minimum Sunspot Group seen by at least 4 obervers')
ylabel('Group Number')

ylim([0 20])

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = 0.075;
p1(3) = 0.9;
h1.Position = p1;

export_fig 'Min_GN_4_Observers.png' -nocrop -m2
