close all

clear

tx = 0.97;
ty = 0.94;
txsz = 28;

% load GN_Links_Mat_Fits.mat;
% load GN_Links_Mat_Fits_Non_Robust.mat;

% load GN_Links_Mat_Fits_Bisquare_All.mat;
% load GN_Links_Mat_Fits_LAR_with_Catch.mat
% load GN_Links_Mat_Fits_Robust;
load GN_Links_Mat_Fits_Bisquare_paths_1000.mat;
load GN_Links_Mat_Fits_Bisquare_3.mat;

load SSN_Vers.mat


load GNobservations_JV_V1-12.mat
load Zolotova.mat
load Vaquero.mat

alp = 0.75;

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
MinCon = 10;
MinRho = 0.0;
Rtol = 1;

LnkMO((LnkMPt<MinCon)|(LnkMR<MinRho)|isnan(LnkMR)) = 0;

%% Finding shortest distances with different metrics

%DM1 = max(LnkMO(:))-LnkMO+1; %Size of overlap
DM1 = (LnkUNP./LnkVlP).^2;

%Finding spots where we prefer a linear fit
LinIn = (LnkUNPR<Rtol*LnkUNLR)&( abs(LnkVlP-LnkVlLa)>LnkUNLa )&( abs(LnkVlLb)>LnkUNLb );

DM1(LinIn) = (LnkUNLa(LinIn)./LnkVlLa(LinIn)).^2;

DM1 = DM1./LnkMO;

DM1(LnkUNP==0) = 0;

%% Starting Node

%% Ending Node


CalCmp = [];
UcnCmp = [];


G  = graph((DM1>0)|(DM1'>0));
G1 = digraph(DM1);

%Joint segments
bins = conncomp(G,'OutputForm','cell');
bins = bins(cellfun(@length,bins)>1);

%Node Colors
Clrbns = zeros(length(Unob),1);

%Number of connections
Deg   = degree(G);
szDeg = 1.25*sqrt(Deg+1)+1;


%Source nodes for edges
[sOut,tOut] = findedge(G);
Clredg = sOut*0;

Lndate = NaN(length(date),length(bins));

Ends = [zeros(length(bins),1) ones(length(bins),1)*(max(date)+1000) zeros(length(bins),4)];

for i = 1:length(bins)
    
    tmp = bins{i};
    
    tmpdat = isnan(GN_Dat(:,1));
    for j = 1:length(tmp)
        
        %Defining node colors
        Clrbns(tmp(j)) = 3;%mod(i,4)+2;
        
        %Defining edge colors
        Clredg(sOut==tmp(j)) = 3;%mod(i,4)+2;
        
        %Finding dates covered by the linked observer
        tmpdat = tmpdat|(GN_Dat(:,6)==Unob(tmp(j)));
        
        %Finding node extrema
        tmpdat2 = GN_Dat(GN_Dat(:,6)==Unob(tmp(j)),2);
        
        %Checking if it extends farther in time
        if (min(tmpdat2)<Ends(i,2))||((min(tmpdat2)==Ends(i,2))&&(Deg(tmp(j))>Ends(i,3)))
            Ends(i,1) = tmp(j);
            Ends(i,2) = min(tmpdat2);
            Ends(i,3) = Deg(tmp(j));
        end
        
        if (max(tmpdat2)>Ends(i,5))||((max(tmpdat2)==Ends(i,5))&&(Deg(tmp(j))>Ends(i,6))&&(max(tmpdat2)~=max(GN_Dat(:,2))))
            Ends(i,4) = tmp(j);
            Ends(i,5) = max(tmpdat2);
            Ends(i,6) = Deg(tmp(j));
        end
        
        %Adding Locarno
        if Ends(i,5)==max(GN_Dat(:,2))
            Ends(i,4) = 590;
        end
        
    end
    
%     if Ends(i,1)~=Ends(i,4)
%         Clrbns(Ends(i,1)) = 1;
%         Clrbns(Ends(i,4)) = 1;
%     end
    
    Lnkdates = GN_Dat(tmpdat,2);
    [C,ia,ib] = intersect(date,Lnkdates);
    Lndate(ia,i) = 1;
    
    
end




Mndate = NaN(length(date),length(bins));

% ypos = rand(length(dpos),1)-0.5;



figure(1)
colormap(Clr)
h = gcf;
set(h,'Position',[0 100 1925 700], 'Color','w');
h2 = subplot(2,1,1);

plot(0,0,'o','Color', Clr90,'MarkerFaceColor', Clr90 ,'MarkerSize',7)
hold on
plot(0,0,'o','Color', [1 1 1]*0.6,'MarkerFaceColor', [1 1 1]*0.6 ,'MarkerSize',7)

% h = plot(G,'XData', dpos,'YData', ypos,'EdgeAlpha',0.2,'MarkerSize',szDeg,'NodeCData',Clrbns,'EdgeCData',Clredg,'LineWidth',1);
h = plot(G,'XData', dpos,'YData', ypos,'EdgeAlpha',0.2,'MarkerSize',5,'NodeCData',Clrbns,'EdgeCData',Clredg,'LineWidth',2);

xlim([1600 2020])
caxis([-0.5 5.5])
ylim([-1 1]*0.8+0.1)

legend('Observers with > 10 days in common with another observer','Observers with < 10 days in common with another observer','Location','NorthWest', 'orientation','horizontal')
legend boxoff

text(tx,ty,'(b)','Units','normalized','FontSize',txsz)


h1 = subplot(2,1,2);

jbfill([0 1],[1 1],[0 0],[1 1],Clr50,Clr50,1,alp)
hold on
jbfill([0 1],[1 1],[0 0],[1 1],Clr70,Clr70,1,alp)
hold on
jbfill([0 1],[1 1],[0 0],[1 1],Clr100,Clr100,1,alp)
hold on
plot(Zolotova(:,1), Zolotova(:,2),'k--','LineWidth',2)
plot(Vaquero(:,1), Vaquero(:,2),'r','LineWidth',3)



jbfill(GNbb2y(:,1),GNbb2y(:,2),GNbb2y(:,2)*0,ones(size(GNbb2y,1),1),Clr70,Clr70,1,alp)
hold on
jbfill(GNhsy(:,1),GNhsy(:,2),GNhsy(:,2)*0,ones(size(GNhsy,1),1),Clr50,Clr50,1,alp)
hold on
jbfill(GNiuy2(:,1),GNiuy2(:,2),GNiuy2(:,2)*0,ones(size(GNiuy2,1),1),Clr100,Clr100,1,alp)
hold on
plot(GNbb2y(:,1),GNbb2y(:,2),'--','Color',Clr70,'LineWidth',2)
plot(GNhsy(:,1),GNhsy(:,2),'-','Color',Clr50,'LineWidth',2)

plot(Zolotova(:,1), Zolotova(:,2),'k--','LineWidth',2)
plot(Vaquero(:,1), Vaquero(:,2),'r','LineWidth',3)

box on

text(tx,ty,'(a)','Units','normalized','FontSize',txsz)

hold off

legend('Hoyt & Schatten (1998)','Svalgaard & Schatten (2016)','Usoskin et al. (2016)', 'Zolotova & Ponyavin (2015)','Vaquero et al. (2015)','Location','NorthWest', 'orientation','horizontal')
legend boxoff
ylabel('Group Number')



xlim([1600 2020])
ylim([0 15])


%
% p2(1) = 0.055;
% p2(2) = 0.05;
% p2(3) = 0.43;
% p2(4) = 0.45;
%
% p4(1) = p2(1)+p2(3)+0.025;
% p4(2) = p2(2);
% p4(3) = p2(3);
% p4(4) = p2(4);

h2.FontSize = 20;
p3 = h2.Position;
p3(1) = 0.04;
p3(2) = 0.06;
p3(3) = 0.95;
p3(4) = 0.44;
h2.Position = p3;
% set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
set(h2,'YTick',[])
set(h2,'XMinorTick','on')
set(h2,'layer','top')

h1.FontSize = 20;
p1 = h1.Position;
p1(1) = p3(1);
p1(2) = p3(2)+p3(4);
p1(3) = p3(3);
p1(4) = p3(4);
h1.Position = p1;
% set(h1, 'YTick', [-90:30:90])
set(h1,'XMinorTick','on')
set(h1,'layer','top')
set(h1,'XAxisLocation','top')


export_fig 'D:\Mis Documentos\AAResearch\Proposals & Applications\Proposals\2018\Early Career\3 page\Latex\F1_Long_Term.pdf' -nocrop 
