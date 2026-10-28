clear

close all

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
MinCon = 10;
MinRho = 0.6;

LnkMO = zeros(length(Unob));  %Days of overalp
LnkMR = LnkMO;  %Correlation between observers

LnkVlP = LnkMO;  %Fit value proportional
LnkUNP = LnkMO;  %Fit uncertainty proportional
LnkUNPR = LnkMO;  %Fit r squared

LnkVlLa = LnkMO;  %Fit value Linear a
LnkUNLa = LnkMO;  %Fit uncertainty Linear a
LnkVlLb = LnkMO;  %Fit value Linear b
LnkUNLb = LnkMO;  %Fit uncertainty Linear b
LnkUNLR = LnkMO;  %Fit r squared

IntrD = cell(length(Unob));

for i = 1:length(Unob)
    i
    
    Ob1 = GN_Dat(GN_Dat(:,6)==Unob(i),2);
    %Adding adjacent days
    for k = 1:dtol
        Ob1 = [Ob1;GN_Dat(GN_Dat(:,6)==Unob(i),2)-k;GN_Dat(GN_Dat(:,6)==Unob(i),2)+k];
    end
    Ob1 = unique(Ob1);
    
    Ovrlp = ismember(GN_Dat(:,2),Ob1)&(GN_Dat(:,6)~=0)&(GN_Dat(:,6)~=Unob(i));
    MObs  = unique(GN_Dat(Ovrlp,6));
    
    Ob1 = GN_Dat(GN_Dat(:,6)==Unob(i),:);
    %Adding adjacent days
    for k = 1:dtol
        
        tmp2 = Ob1;
        tmp2(:,2) = tmp2(:,2)+k;
        [C,ia] = setdiff(tmp2(:,2),Ob1(:,2));
        Ob1 = [Ob1;tmp2(ia,:)];
        
        tmp2 = Ob1;
        tmp2(:,2) = tmp2(:,2)-k;
        [C,ia] = setdiff(tmp2(:,2),Ob1(:,2));
        Ob1 = [Ob1;tmp2(ia,:)];
    end
    
    
    for j=1:length(MObs)
        
        i2 = find(Unob==MObs(j));
        
        %         Making sure it was not visited already
        %         if LnkMO(i,i2)==0
        
        Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
        
        [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
        RHO = corr(Ob2(ib,8),Ob1(ia,8),'type','Spearman');
        
        LnkMO(i,i2) = length(C);
        LnkMO(i2,i) = length(C);
        LnkMR(i2,i) = RHO;
        LnkMR(i,i2) = RHO;        
        
        if (length(C) >= MinCon)&&(RHO>=MinRho)
            
            %% Proportional
%             fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
%                 'StartPoint',[1],'MaxFunEvals',1000,'MaxIter',1000);
%             fo = fitoptions('Method','LinearLeastSquares', 'Robust', 'LAR',...
%                 'StartPoint',[1],'MaxFunEvals',1000,'MaxIter',1000);
            ft = fittype('a*x');

            %Ob2 to Ob1
            [PropFit1,Propgof1] = fit(Ob2(ib,8),Ob1(ia,8),ft,'Robust', 'LAR', 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);            
            if ~isempty(lastwarn)
               [PropFit1,Propgof1] = fit(Ob2(ib,8),Ob1(ia,8),ft, 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
               lastwarn('')
            end
            
            PrpVa1 = coeffvalues(PropFit1);
            PrpUn1 = confint(PropFit1);
            PrpUn1 = abs(PrpUn1(1)-PrpVa1);
            PrpR21 = Propgof1.adjrsquare;
            
            LnkVlP(i2,i) = PrpVa1;
            LnkUNP(i2,i) = PrpUn1;
                        
            %Ob1 to Ob2
            [PropFit2,Propgof2] = fit(Ob1(ia,8),Ob2(ib,8),ft,'Robust', 'LAR', 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
            if ~isempty(lastwarn)
                [PropFit2,Propgof2] = fit(Ob1(ia,8),Ob2(ib,8),ft, 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
                lastwarn('')
            end
            
            PrpVa2 = coeffvalues(PropFit2);
            PrpUn2 = confint(PropFit2);
            PrpUn2 = abs(PrpUn2(1)-PrpVa2);
            PrpR22 = Propgof2.adjrsquare;
            
            LnkVlP(i,i2) = PrpVa2;
            LnkUNP(i,i2) = PrpUn2;
            LnkUNPR(i,i2) = min([PrpR21 PrpR22]);
            LnkUNPR(i2,i) = min([PrpR21 PrpR22]);            
            
            %% Linear

            %Ob2 to Ob1
            [LinFit1,Lingof1] = fit(Ob2(ib,8),Ob1(ia,8),'poly1','Robust', 'LAR');
            if ~isempty(lastwarn)
                [LinFit1,Lingof1] = fit(Ob2(ib,8),Ob1(ia,8),'poly1');
                lastwarn('')
            end
            
            LinVa1 = coeffvalues(LinFit1);
            LinUn1 = confint(LinFit1);
            LinUn1 = [abs(LinVa1(1)-LinUn1(1,1)) abs(LinVa1(2)-LinUn1(1,2))];
            LinR21 = Lingof1.adjrsquare;
            
            LnkVlLa(i2,i) = LinVa1(1);
            LnkUNLa(i2,i) = LinUn1(1);
            LnkVlLb(i2,i) = LinVa1(2);
            LnkUNLb(i2,i) = LinUn1(2);
            
            %Ob1 to Ob2
            [LinFit2,Lingof2] = fit(Ob1(ia,8),Ob2(ib,8),'poly1','Robust', 'LAR');
            if ~isempty(lastwarn)
                [LinFit2,Lingof2] = fit(Ob1(ia,8),Ob2(ib,8),'poly1');
                lastwarn('')
            end
            
            LinVa2 = coeffvalues(LinFit2);
            LinUn2 = confint(LinFit2);
            LinUn2 = [abs(LinVa2(1)-LinUn2(1,1)) abs(LinVa2(2)-LinUn2(1,2))];
            LinR22 = Lingof2.adjrsquare;
            
            LnkVlLa(i,i2) = LinVa2(1);
            LnkUNLa(i,i2) = LinUn2(1);
            LnkVlLb(i,i2) = LinVa2(2);
            LnkUNLb(i,i2) = LinUn2(2);
            
            LnkUNLR(i,i2) = min([LinR21 LinR22]);
            LnkUNLR(i2,i) = min([LinR21 LinR22]);            
            
        end
              
        %         end
        
    end
    
end


dpos = Unob*0;
for i = 1:length(Unob)
    
    dpos(i) = mean(GN_Dat(GN_Dat(:,6)==Unob(i),1));
    
end

% load GN_Links_Mat_2d.mat;

LnkMO((LnkMO<MinCon)|(LnkMR<MinRho)|isnan(LnkMR)) = 0;

G = graph(LnkMO);

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
        Clrbns(tmp(j)) = mod(i,4)+2;
        
        %Defining edge colors
        Clredg(sOut==tmp(j)) = mod(i,4)+2;
        
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
    
    if Ends(i,1)~=Ends(i,4)
        Clrbns(Ends(i,1)) = 1;
        Clrbns(Ends(i,4)) = 1;
    end
    
    Lnkdates = GN_Dat(tmpdat,2);
    [C,ia,ib] = intersect(date,Lnkdates);
    Lndate(ia,i) = 1;
    
    
end


%% Finding shortest distances with different metrics

%Size of overlap
% DM1           = max(LnkMO(:))-LnkMO+1;
DM1           = LnkMO.*LnkMR;
DM1           = max(DM1(:))-DM1+1;
% DM1             = LnkMO*0+1;
% DM1           = max(LnkMY(:))-LnkMY+1;
DM1(LnkMO==0) = 0;
G1            = graph(DM1);

% %Size of time ending difference
% DM2           = max(LnkMY(:))-LnkMY+1;
% DM2(LnkMY==0) = 0;
% G2            = graph(DM2);

Mndate = NaN(length(date),length(bins));

ypos = rand(length(dpos),1)-0.5;

for i = 1:length(bins)
    
    tmp = shortestpath(G1,Ends(i,1),Ends(i,4));
    path{1,i} = tmp;
    if length(tmp)>1
        
        for j = 1:length(tmp)
            Clrbns(tmp(j)) = 1;
            
            
            
            Lnkdates = GN_Dat((GN_Dat(:,6)==Unob(tmp(j))),2);
            [C,ia,ib] = intersect(date,Lnkdates);
            Mndate(ia,i) = 1;
            
        end
        
    end
    
end



figure(1)
colormap(Clr)
h = gcf;
set(h,'Position',[0 100 1925 700], 'Color','w');
h1 = subplot(2,1,1);



h = plot(G,'XData', dpos,'YData', ypos,'EdgeAlpha',0.2,'MarkerSize',szDeg,'NodeCData',Clrbns,'EdgeCData',Clredg,'LineWidth',1);
hold on

ObsNm = [GN_Obs{:,1}];
ObsLb = GN_Obs(:,5);
for i = 1:length(bins)
    
    tmp = path{1,i};
    if length(tmp)>1
        plot(dpos(tmp),ypos(tmp),'--','Color',ClrD,'LineWidth',2)
        
        for j = 1:length(tmp)
            
            fin = find(ObsNm==Unob(tmp(j)));
            txt = ObsLb{fin};
            
            k = strfind(txt,',');
            
            if ~isempty(k)
                txt = txt(1:k(1)-1);
            end
            
            text(dpos(tmp(j)),ypos(tmp(j))-0.04,txt,'HorizontalAlignment','center')
            
        end
        
    end
    
end


hold off

xlim([1600 2020])
caxis([-0.5 5.5])

h2 = subplot(2,1,2);

for i = 1:length(bins)
    
    plot(date2,Lndate(:,i)-i,'.','Color', Clr((mod(i,4)+3),:),'LineWidth',3,'MarkerSize',3)
    hold on
    plot(date2,Mndate(:,i)-i-0.25,'.','Color', ClrD,'LineWidth',3,'MarkerSize',3)
    
end
hold off

xlim([1600 2020])
ylim([-length(bins) 1])

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

h2.FontSize = 17;
p2 = h2.Position;
p2(1) = 0.02;
p2(2) = 0.07;
p2(3) = 0.97;
p2(4) = 0.25;
h2.Position = p2;
% set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
set(h2,'YTick',[])
set(h2,'XMinorTick','on')
set(h2,'layer','top')

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = p2(1);
p1(2) = p2(2)+p2(4);
p1(3) = p2(3);
p1(4) = 0.6;
h1.Position = p1;
% set(h1, 'YTick', [-90:30:90])
set(h1,'YTick',[])
set(h1,'XMinorTick','on')
set(h1,'layer','top')
set(h1,'XAxisLocation','top')


figure(2)
colormap(Clr)
h = gcf;
set(h,'Position',[0 100 1925 700], 'Color','w');
h1 = subplot(2,1,1);

h = plot(G,'XData', dpos,'YData', ypos,'EdgeAlpha',0.2,'MarkerSize',szDeg,'NodeCData',Clrbns,'EdgeCData',Clredg,'LineWidth',1);
hold on

%% Identifying Networks

for i = 1:length(bins)
    
    tmp = path{1,i};
    if length(tmp)>1
        
        for j = 2:length(tmp)
            
            %Identifying network
            Strt1 = (sOut==tmp(j));
            End1  = (tOut==tmp(j));
            Cnnct1 = unique([tOut(Strt1);sOut(End1)]);
            
            Strt2 = (sOut==tmp(j-1));
            End2  = (tOut==tmp(j-1));
            Cnnct2 = unique([tOut(Strt2);sOut(End2)]);
            
            Cnnct = intersect(Cnnct1,Cnnct2);
            
            for k = 1:length(Cnnct)
                
                tmpClr = ClrN(mod(j,3)+1,:);
                tmpLn = ':';
                if mod(j,2)==0
                    tmpLn = '--';
                end
                
                plot(dpos([tmp(j) Cnnct(k)]),ypos([tmp(j) Cnnct(k)]),tmpLn,'Color',tmpClr,'LineWidth',2)
                plot(dpos([tmp(j-1) Cnnct(k)]),ypos([tmp(j-1) Cnnct(k)]),tmpLn,'Color',tmpClr,'LineWidth',2)
                
            end
            
        end
        
    end
    
end


ObsNm = [GN_Obs{:,1}];
ObsLb = GN_Obs(:,5);
for i = 1:length(bins)
    
    tmp = path{1,i};
    if length(tmp)>1
        plot(dpos(tmp),ypos(tmp),'--','Color',ClrD,'LineWidth',2)
        
        for j = 1:length(tmp)
            
            fin = find(ObsNm==Unob(tmp(j)));
            txt = ObsLb{fin};
            
            k = strfind(txt,',');
            
            if ~isempty(k)
                txt = txt(1:k(1)-1);
            end
            
            text(dpos(tmp(j)),ypos(tmp(j))-0.04,txt,'HorizontalAlignment','center')
            
        end
        
    end
    
end


hold off

xlim([1600 2020])
caxis([-0.5 5.5])

h2 = subplot(2,1,2);

for i = 1:length(bins)
    
    plot(date2,Lndate(:,i)-i,'.','Color', Clr((mod(i,4)+3),:),'LineWidth',3,'MarkerSize',3)
    hold on
    plot(date2,Mndate(:,i)-i-0.25,'.','Color', ClrD,'LineWidth',3,'MarkerSize',3)
    
end
hold off

xlim([1600 2020])
ylim([-length(bins) 1])

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

h2.FontSize = 17;
p2 = h2.Position;
p2(1) = 0.02;
p2(2) = 0.07;
p2(3) = 0.97;
p2(4) = 0.25;
h2.Position = p2;
% set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
set(h2,'YTick',[])
set(h2,'XMinorTick','on')
set(h2,'layer','top')

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = p2(1);
p1(2) = p2(2)+p2(4);
p1(3) = p2(3);
p1(4) = 0.6;
h1.Position = p1;
% set(h1, 'YTick', [-90:30:90])
set(h1,'YTick',[])
set(h1,'XMinorTick','on')
set(h1,'layer','top')
set(h1,'XAxisLocation','top')




figure(3)
colormap(Clr)
h = gcf;
set(h,'Position',[0 100 1925 700], 'Color','w');
h1 = subplot(2,1,1);

h = plot(G,'XData', dpos,'YData', ypos,'EdgeAlpha',0.2,'MarkerSize',szDeg,'NodeCData',Clrbns,'EdgeCData',Clredg,'LineWidth',1);
hold on

%% Identifying Networks

for i = 1:length(bins)
    
    tmp = path{1,i};
    if length(tmp)>2
        
        for j = 3:length(tmp)
            
            %Identifying network
            Strt1 = (sOut==tmp(j));
            End1  = (tOut==tmp(j));
            Cnnct1 = unique([tOut(Strt1);sOut(End1)]);
            
            Strt2 = (sOut==tmp(j-1));
            End2  = (tOut==tmp(j-1));
            Cnnct2 = unique([tOut(Strt2);sOut(End2)]);
            
            Strt3 = (sOut==tmp(j-2));
            End3  = (tOut==tmp(j-2));
            Cnnct3 = unique([tOut(Strt3);sOut(End3)]);
            
            Cnnct = intersect(Cnnct1,Cnnct2);
            Cnnct = intersect(Cnnct,Cnnct3);
            
            for k = 1:length(Cnnct)
                
                tmpClr = ClrN(mod(j,3)+1,:);
                tmpLn = ':';
                if mod(j,2)==0
                    tmpLn = '--';
                end
                
                plot(dpos([tmp(j) Cnnct(k)]),ypos([tmp(j) Cnnct(k)]),tmpLn,'Color',tmpClr,'LineWidth',2)
                plot(dpos([tmp(j-1) Cnnct(k)]),ypos([tmp(j-1) Cnnct(k)]),tmpLn,'Color',tmpClr,'LineWidth',2)
                plot(dpos([tmp(j-2) Cnnct(k)]),ypos([tmp(j-2) Cnnct(k)]),tmpLn,'Color',tmpClr,'LineWidth',2)
                
            end
            
        end
        
    end
    
end


ObsNm = [GN_Obs{:,1}];
ObsLb = GN_Obs(:,5);
for i = 1:length(bins)
    
    tmp = path{1,i};
    if length(tmp)>1
        plot(dpos(tmp),ypos(tmp),'--','Color',ClrD,'LineWidth',2)
        
        for j = 1:length(tmp)
            
            fin = find(ObsNm==Unob(tmp(j)));
            txt = ObsLb{fin};
            
            k = strfind(txt,',');
            
            if ~isempty(k)
                txt = txt(1:k(1)-1);
            end
            
            text(dpos(tmp(j)),ypos(tmp(j))-0.04,txt,'HorizontalAlignment','center')
            
        end
        
    end
    
end


hold off

xlim([1600 2020])
caxis([-0.5 5.5])

h2 = subplot(2,1,2);

for i = 1:length(bins)
    
    plot(date2,Lndate(:,i)-i,'.','Color', Clr((mod(i,4)+3),:),'LineWidth',3,'MarkerSize',3)
    hold on
    plot(date2,Mndate(:,i)-i-0.25,'.','Color', ClrD,'LineWidth',3,'MarkerSize',3)
    
end
hold off

xlim([1600 2020])
ylim([-length(bins) 1])

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

h2.FontSize = 17;
p2 = h2.Position;
p2(1) = 0.02;
p2(2) = 0.07;
p2(3) = 0.97;
p2(4) = 0.25;
h2.Position = p2;
% set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
set(h2,'YTick',[])
set(h2,'XMinorTick','on')
set(h2,'layer','top')

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = p2(1);
p1(2) = p2(2)+p2(4);
p1(3) = p2(3);
p1(4) = 0.6;
h1.Position = p1;
% set(h1, 'YTick', [-90:30:90])
set(h1,'YTick',[])
set(h1,'XMinorTick','on')
set(h1,'layer','top')
set(h1,'XAxisLocation','top')




%% Making scatterplot of 2 way networks

outFld = '2way';
outFld = [pwd '\' outFld];
mkdir(outFld)

for i = 1:length(bins)
    
    tmp = path{1,i};
    if length(tmp)>1
        
        
        for j = 2:length(tmp)
            
            mkdir([outFld '\' num2str(i) '_' num2str(j)])
            
            %Identifying network
            Strt1 = (sOut==tmp(j));
            End1  = (tOut==tmp(j));
            Cnnct1 = unique([tOut(Strt1);sOut(End1)]);
            
            Strt2 = (sOut==tmp(j-1));
            End2  = (tOut==tmp(j-1));
            Cnnct2 = unique([tOut(Strt2);sOut(End2)]);
            
            Cnnct = intersect(Cnnct1,Cnnct2);
            
            fin = find(ObsNm==Unob(tmp(j)));
            txt1 = ObsLb{fin};
            k = strfind(txt1,',');
            if ~isempty(k)
                txt1 = txt1(1:k(1)-1);
            end
            
            Ob1 = GN_Dat(GN_Dat(:,6)==Unob(tmp(j)),:);
            %Adding adjacent days
            for k = 1:dtol
                
                tmp2 = Ob1;
                tmp2(:,2) = tmp2(:,2)+k;
                [C,ia] = setdiff(tmp2(:,2),Ob1(:,2));
                Ob1 = [Ob1;tmp2(ia,:)];
                
                tmp2 = Ob1;
                tmp2(:,2) = tmp2(:,2)-k;
                [C,ia] = setdiff(tmp2(:,2),Ob1(:,2));
                Ob1 = [Ob1;tmp2(ia,:)];
            end
            
            fin = find(ObsNm==Unob(tmp(j-1)));
            txt2 = ObsLb{fin};
            k = strfind(txt2,',');
            if ~isempty(k)
                txt2 = txt2(1:k(1)-1);
            end
            
            Ob2 = GN_Dat(GN_Dat(:,6)==Unob(tmp(j-1)),:);
            %Adding adjacent days
            for k = 1:dtol
                tmp2 = Ob2;
                tmp2(:,2) = tmp2(:,2)+k;
                [C,ia] = setdiff(tmp2(:,2),Ob2(:,2));
                Ob2 = [Ob2;tmp2(ia,:)];
                
                tmp2 = Ob2;
                tmp2(:,2) = tmp2(:,2)-k;
                [C,ia] = setdiff(tmp2(:,2),Ob2(:,2));
                Ob2 = [Ob2;tmp2(ia,:)];
            end
            
            [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
            RHO = corr(Ob2(ib,8),Ob1(ia,8),'type','Spearman');
            
            fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                'StartPoint',[1]);
            ft = fittype('a*x','options',fo);
            [PropFit,Propgof] = fit(Ob2(ib,8),Ob1(ia,8),ft);
            
            PrpVa0 = coeffvalues(PropFit);
            PrpUn0 = confint(PropFit);
            PrpUn0 = abs(PrpUn0(1)-PrpVa0);
            PrpCf = [tmp(j-1) tmp(j) PrpVa0 PrpUn0 0 0 0 0 PrpVa0 PrpUn0];
            
            fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                'StartPoint',[1 0]);
            ft = fittype('a*x+b','options',fo);
            [LinFit,Lingof] = fit(Ob2(ib,8),Ob1(ia,8),ft);
            
            LinVa0 = coeffvalues(LinFit);
            LinUn0 = confint(LinFit);
            LinUn0 = [abs(LinVa0(1)-LinUn0(1,1)) abs(LinVa0(2)-LinUn0(1,2))];
            LinCf = [tmp(j-1) tmp(j) LinVa0 LinUn0 0 0 0 0 0 0 LinVa0 LinUn0];
            
            for k = 1:length(Cnnct)
                
                fin = find(ObsNm==Unob(Cnnct(k)));
                txt3 = ObsLb{fin};
                kt = strfind(txt3,',');
                if ~isempty(kt)
                    txt3 = txt3(1:kt(1)-1);
                end
                
                Ob3 = GN_Dat(GN_Dat(:,6)==Unob(Cnnct(k)),:);
                
                name = [txt1 '_' txt2 '_' txt3 '.png'];
                name = [outFld '\' num2str(i) '_' num2str(j) '\' num2str(k) '_' name];
                
                figure(4)
                h = gcf;
                set(h,'Position',[100 100 1300 600], 'Color','w');
                h1 = subplot(1,2,2);
                
                [C,ia,ib] = intersect(Ob1(:,2),Ob3(:,2));
                RHO = corr(Ob3(ib,8),Ob1(ia,8),'type','Spearman');
                
                fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                    'StartPoint',[1]);
                ft = fittype('a*x','options',fo);
                [PropFit,Propgof] = fit(Ob3(ib,8),Ob1(ia,8),ft);
                
                PrpVa1 = coeffvalues(PropFit);
                PrpUn1 = confint(PropFit);
                PrpUn1 = abs(PrpUn1(1)-PrpVa1);
                
                
                fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                    'StartPoint',[1 0]);
                ft = fittype('a*x+b','options',fo);
                [LinFit,Lingof] = fit(Ob3(ib,8),Ob1(ia,8),ft);
                
                LinVa1 = coeffvalues(LinFit);
                LinUn1 = confint(LinFit);
                LinUn1 = [abs(LinVa1(1)-LinUn1(1,1)) abs(LinVa1(2)-LinUn1(1,2))];
                
                
                fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                    'StartPoint',[1 0 0],'Lower',[0 0 0]);
                ft = fittype('a*x.^b+c','options',fo);
                [PowFit,Powgof] = fit(Ob3(ib,8),Ob1(ia,8),ft);
                
                x = [0:0.1:max(Ob3(ib,8))+1];
                
                %                 plot(x,PropFit(x),'--',x,LinFit(x),'-.',x,PowFit(x),'-','LineWidth',2)
                plot(x,PropFit(x),'--',x,LinFit(x),'-.','LineWidth',2)
                hold on
                scatter(Ob3(ib,8),Ob1(ia,8),'o','filled','MarkerFaceAlpha',0.3)
                text(0.05,0.95,['\rho = ' num2str(round(RHO,2))],'Units','normalized','Fontsize',15)
                hold off
                
                xlabel(txt3)
                ylabel(txt1)
                
                xlim([0 max(Ob3(ib,8))+1])
                ylim([0 max(Ob1(ia,8))+1])
                
                %                 legend(['Prp adR^2 = ' num2str(round(Propgof.adjrsquare,2))], ['Lin adR^2 = ' num2str(round(Lingof.adjrsquare,2))], ['Pow adR^2 = ' num2str(round(Powgof.adjrsquare,2))],'Location','SouthEast')
                legend(['Prp adR^2 = ' num2str(round(Propgof.adjrsquare,2))], ['Lin adR^2 = ' num2str(round(Lingof.adjrsquare,2))],'Location','SouthEast')
                legend boxoff
                
                
                h2 = subplot(1,2,1);
                
                [C,ia,ib] = intersect(Ob2(:,2),Ob3(:,2));
                RHO = corr(Ob2(ia,8),Ob3(ib,8),'type','Spearman');
                
                fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                    'StartPoint',[1]);
                ft = fittype('a*x','options',fo);
                [PropFit,Propgof] = fit(Ob2(ia,8),Ob3(ib,8),ft);
                
                PrpVa2 = coeffvalues(PropFit);
                PrpUn2 = confint(PropFit);
                PrpUn2 = abs(PrpUn2(1)-PrpVa2);
                
                
                fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                    'StartPoint',[1 0]);
                ft = fittype('a*x+b','options',fo);
                [LinFit,Lingof] = fit(Ob2(ia,8),Ob3(ib,8),ft);
                
                LinVa2 = coeffvalues(LinFit);
                LinUn2 = confint(LinFit);
                LinUn2 = [abs(LinVa2(1)-LinUn2(1,1)) abs(LinVa2(2)-LinUn2(1,2))];
                
                
                fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
                    'StartPoint',[1 0 0],'Lower',[0 0 0]);
                ft = fittype('a*x.^b+c','options',fo);
                [PowFit,Powgof] = fit(Ob2(ia,8),Ob3(ib,8),ft);
                
                x = [0:0.1:max(Ob2(ia,8))+1];
                
                %                 plot(x,PropFit(x),'--',x,LinFit(x),'-.',x,PowFit(x),'-','LineWidth',2)
                plot(x,PropFit(x),'--',x,LinFit(x),'-.','LineWidth',2)
                hold on
                scatter(Ob2(ia,8),Ob3(ib,8),'o','filled','MarkerFaceAlpha',0.3)
                text(0.05,0.95,['\rho = ' num2str(round(RHO,2))],'Units','normalized','Fontsize',15)
                hold off
                
                ylabel(txt3)
                xlabel(txt2)
                
                ylim([0 max(Ob3(ib,8))+1])
                xlim([0 max(Ob2(ia,8))+1])
                
                %                 legend(['Prp adR^2 = ' num2str(round(Propgof.adjrsquare,2))], ['Lin adR^2 = ' num2str(round(Lingof.adjrsquare,2))], ['Pow adR^2 = ' num2str(round(Powgof.adjrsquare,2))],'Location','SouthEast')
                legend(['Prp adR^2 = ' num2str(round(Propgof.adjrsquare,2))], ['Lin adR^2 = ' num2str(round(Lingof.adjrsquare,2))],'Location','SouthEast')
                legend boxoff
                h1.FontSize = 17;
                h2.FontSize = 17;
                
                export_fig(name, '-nocrop')
                
                PrpVa3 = PrpVa2*PrpVa1;
                PrpUn3 = PrpVa3*sqrt( (PrpUn1/PrpVa1)^2 + (PrpUn2/PrpVa2)^2 );
                PrpCf = [PrpCf;[tmp(j-1) Cnnct(k) PrpVa2 PrpUn2 Cnnct(k) tmp(j) PrpVa1 PrpUn1 PrpVa3 PrpUn3]];
                
                
                LinVa3 = LinVa1(1)*LinVa2(2) + LinVa1(2);
                LinUn3 = sqrt( (LinVa1(1)*LinVa2(2))^2*( (LinUn1(1)/LinVa1(1))^2 + (LinUn2(2)/LinVa2(2))^2 ) + LinUn1(2)^2);
                
                LinVa3 = [LinVa2(1)*LinVa1(1) LinVa3];
                LinUn3 = [LinVa2(1)*LinVa1(1)*sqrt( (LinUn1(1)/LinVa1(1))^2 + (LinUn2(1)/LinVa2(1))^2 ) LinUn3];
                
                LinCf = [LinCf;[tmp(j-1) Cnnct(k) LinVa2 LinUn2 Cnnct(k) tmp(j) LinVa1 LinUn1 LinVa3 LinUn3]];
                
                close(h)
                
            end
            
            PrpCf(length(Cnnct)+2,9) = mean(PrpCf(2:length(Cnnct)+1,9));
            PrpCf(length(Cnnct)+2,10) = sqrt(sum(PrpCf(2:length(Cnnct)+1,10).^2))/length(Cnnct);
            
            LinCf(length(Cnnct)+2,13) = mean(LinCf(2:length(Cnnct)+1,13));
            LinCf(length(Cnnct)+2,14) = mean(LinCf(2:length(Cnnct)+1,14));
            LinCf(length(Cnnct)+2,15) = sqrt(sum(LinCf(2:length(Cnnct)+1,15).^2))/length(Cnnct);
            LinCf(length(Cnnct)+2,16) = sqrt(sum(LinCf(2:length(Cnnct)+1,16).^2))/length(Cnnct);
            
            name = [txt1 '_' txt2 '.png'];
            name = [outFld '\' num2str(i) '_' num2str(j) '\0_' name];
            
            figure(5)
            h = gcf;
            set(h,'Position',[100 100 1300 600], 'Color','w');
            h1 = subplot(1,2,1);
            
            [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
            
            scatter(Ob2(ib,8),Ob1(ia,8),'o','filled','MarkerFaceAlpha',0.3)
            hold on
            
            x = [0:0.1:max(Ob2(ib,8))+1];
            for k = 2:length(Cnnct)+1
                
                plot(x,x*PrpCf(k,9),'Color',[1 1 1]*0.75)
                
            end
            
            plot(x,x*PrpCf(1,9),'Color',[1 1 1]*0,'LineWidth',3)
            plot(x,x*PrpCf(length(Cnnct)+2,9),'Color',[0 0 1],'LineWidth',3)
            
            hold off
            
            xlabel(txt2)
            ylabel(txt1)
            
            xlim([0 max(Ob2(ib,8))+1])
            ylim([0 max(Ob1(ia,8))+1])
            
            
            h2 = subplot(1,2,2);
            
            scatter(Ob2(ib,8),Ob1(ia,8),'o','filled','MarkerFaceAlpha',0.3)
            hold on
            
            for k = 2:length(Cnnct)+1
                
                plot(x,x*LinCf(k,13)+LinCf(k,14),'Color',[1 1 1]*0.75)
                
            end
            
            plot(x,x*LinCf(1,13)+LinCf(1,14),'Color',[1 1 1]*0,'LineWidth',3)
            plot(x,x*LinCf(length(Cnnct)+2,13)+LinCf(length(Cnnct)+2,14),'Color',[0 0 1],'LineWidth',3)
            
            hold off
            
            xlabel(txt2)
            ylabel(txt1)
            
            xlim([0 max(Ob2(ib,8))+1])
            ylim([0 max(Ob1(ia,8))+1])
            
            
            h1.FontSize = 17;
            h2.FontSize = 17;
            
            export_fig(name, '-nocrop')
            
            close(h)
            
        end
        
    end
    
end





