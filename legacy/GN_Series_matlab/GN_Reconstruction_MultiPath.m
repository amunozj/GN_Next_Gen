close all

clear

% load GN_Links_Mat_Fits.mat;
% load GN_Links_Mat_Fits_Non_Robust.mat;

% load GN_Links_Mat_Fits_Bisquare_All.mat;
% load GN_Links_Mat_Fits_LAR_with_Catch.mat
% load GN_Links_Mat_Fits_Robust;
load GN_Links_Mat_Fits_Bisquare_with_Catch_weighted_4.mat;


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
MinCon = 5;
MinRho = 0.6;

LnkMO((LnkMO<MinCon)|(LnkMR<MinRho)|isnan(LnkMR)) = 0;

%% Finding shortest distances with different metrics

%DM1 = max(LnkMO(:))-LnkMO+1; %Size of overlap
DM1 = abs(LnkUNP./LnkVlP);
% DM1 = abs(LnkUNLa./LnkVlLa);
DM1(LnkUNLR>LnkUNPR) = abs(LnkUNLa(LnkUNLR>LnkUNPR)./LnkVlLa(LnkUNLR>LnkUNPR));

% DM1 = LnkUNLa;
DM1(LnkMO==0) = 0;

%% Starting Node

%% Ending Node


CalCmp = [];
UcnCmp = [];

for mp = 1:10
    
    if mp==4
        mp
    end
    
    G = graph(LnkMO);
    G1            = digraph(DM1);
    
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
    
    
    
    
    Mndate = NaN(length(date),length(bins));
    
    ypos = rand(length(dpos),1)-0.5;
    
    for i = 1:length(bins)
        
        [tmp,d] = shortestpath(G1,Ends(i,1),Ends(i,4))
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
    p3 = h2.Position;
    p3(1) = 0.02;
    p3(2) = 0.07;
    p3(3) = 0.97;
    p3(4) = 0.25;
    h2.Position = p3;
    % set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
    set(h2,'YTick',[])
    set(h2,'XMinorTick','on')
    set(h2,'layer','top')
    
    h1.FontSize = 17;
    p1 = h1.Position;
    p1(1) = p3(1);
    p1(2) = p3(2)+p3(4);
    p1(3) = p3(3);
    p1(4) = 0.6;
    h1.Position = p1;
    % set(h1, 'YTick', [-90:30:90])
    set(h1,'YTick',[])
    set(h1,'XMinorTick','on')
    set(h1,'layer','top')
    set(h1,'XAxisLocation','top')
    
    
    
    %% Trimming the path using link between tree observers
    
    %Resetting colors
    for i = 1:length(bins)
        
        tmp = bins{i};
        for j = 1:length(tmp)
            
            %Defining node colors
            Clrbns(tmp(j)) = mod(i,4)+2;
            
        end
    end
    
    for n = 1:length(bins)
        
        tmp = path{1,n};
        if length(tmp)>2
            
            sw_dn = true;
            
            while sw_dn
                
                for j = 3:length(tmp)
                    
                    %Looking for match
                    Strt1 = find((sOut==tmp(j-2))&(tOut==tmp(j))|(sOut==tmp(j))&(tOut==tmp(j-2)));
                    
                    if ~isempty(Strt1)
                        
                        fin = find(ObsNm==Unob(tmp(j-2)));
                        txt1 = ObsLb{fin};
                        k = strfind(txt1,',');
                        if ~isempty(k)
                            txt1 = txt1(1:k(1)-1);
                        end
                        
                        fin = find(ObsNm==Unob(tmp(j-1)));
                        txt2 = ObsLb{fin};
                        k = strfind(txt2,',');
                        if ~isempty(k)
                            txt2 = txt2(1:k(1)-1);
                        end
                        
                        fin = find(ObsNm==Unob(tmp(j)));
                        txt3 = ObsLb{fin};
                        k = strfind(txt3,',');
                        if ~isempty(k)
                            txt3 = txt3(1:k(1)-1);
                        end
                        
                        
                        figure(5)
                        h = gcf;
                        set(h,'Position',[0 100 1920 600], 'Color','w');
                        
                        h1 = subplot(1,3,1);
                        
                        i = tmp(j-2);
                        i2 = tmp(j-1);
                        
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
                        
                        Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
                        
                        [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
                        scatter(Ob1(ia,8),Ob2(ib,8),'ob','MarkerFaceAlpha',0.3)
                        hold on
                        scatter(Ob1(ia,8),Ob2(ib,8),'ob','filled','MarkerFaceAlpha',0.1)
                        
                        x = [0:0.1:max(Ob1(ia,8))+1];
                        
                        plot(x,x*LnkVlP(i,i2),'-.','Color',[1 1 0],'LineWidth',3)
                        plot(x,x*LnkVlLa(i,i2) + LnkVlLb(i,i2),':','Color',[0 1 1],'LineWidth',3)
                        
                        text(0.05,0.95,{['a = ' num2str(LnkVlP(i,i2)) ' +/- ' num2str(LnkUNP(i,i2))],['Prp R2 = ' num2str(LnkUNPR(i,i2))],['Lin R2 = ' num2str(LnkUNLR(i,i2))],['No days = ' num2str(length(C))]},'Units','normalized','Fontsize',15)
                        
                        hold off
                        
                        ylabel(txt2)
                        xlabel(txt1)
                        
                        ylim([0 max(Ob2(ib,8))+1])
                        xlim([0 max(Ob1(ia,8))+1])
                        
                        h2 = subplot(1,3,2);
                        
                        i = tmp(j-1);
                        i2 = tmp(j);
                        
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
                        
                        Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
                        
                        [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
                        scatter(Ob1(ia,8),Ob2(ib,8),'ob','MarkerFaceAlpha',0.3)
                        hold on
                        scatter(Ob1(ia,8),Ob2(ib,8),'ob','filled','MarkerFaceAlpha',0.1)
                        
                        x = [0:0.1:max(Ob1(ia,8))+1];
                        
                        plot(x,x*LnkVlP(i,i2),'-.','Color',[1 1 0],'LineWidth',3)
                        plot(x,x*LnkVlLa(i,i2) + LnkVlLb(i,i2),':','Color',[0 1 1],'LineWidth',3)
                        
                        text(0.05,0.95,{['a = ' num2str(LnkVlP(i,i2)) ' +/- ' num2str(LnkUNP(i,i2))],['Prp R2 = ' num2str(LnkUNPR(i,i2))],['Lin R2 = ' num2str(LnkUNLR(i,i2))],['No days = ' num2str(length(C))]},'Units','normalized','Fontsize',15)
                        
                        hold off
                        
                        ylabel(txt3)
                        xlabel(txt2)
                        
                        ylim([0 max(Ob2(ib,8))+1])
                        xlim([0 max(Ob1(ia,8))+1])
                        
                        
                        h3 = subplot(1,3,3);
                        
                        %Combined uncertainty
                        cumun = LnkVlP(tmp(j-2),tmp(j-1))*LnkVlP(tmp(j-1),tmp(j))*sqrt((LnkUNP(tmp(j-2),tmp(j-1))/LnkVlP(tmp(j-2),tmp(j-1)))^2 + (LnkUNP(tmp(j-1),tmp(j))/LnkVlP(tmp(j-1),tmp(j)))^2);
                        Ca = LnkVlP(tmp(j-2),tmp(j-1))*LnkVlP(tmp(j-1),tmp(j));
                        
                        i = tmp(j-2);
                        i2 = tmp(j);
                        
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
                        
                        Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
                        
                        [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
                        scatter(Ob1(ia,8),Ob2(ib,8),'ob','MarkerFaceAlpha',0.3)
                        hold on
                        scatter(Ob1(ia,8),Ob2(ib,8),'ob','filled','MarkerFaceAlpha',0.01)
                        
                        x = [0:0.1:max(Ob1(ia,8))+1];
                        
                        plot(x,x*LnkVlP(i,i2),'-.','Color',[1 1 0],'LineWidth',3)
                        plot(x,x*LnkVlLa(i,i2) + LnkVlLb(i,i2),':','Color',[0 1 1],'LineWidth',3)
                        
                        text(0.05,0.95,{['a = ' num2str(LnkVlP(i,i2)) ' +/- ' num2str(LnkUNP(i,i2))],['Prp R2 = ' num2str(LnkUNPR(i,i2))],['Lin R2 = ' num2str(LnkUNLR(i,i2))],['No days = ' num2str(length(C))],['Ca = ' num2str(Ca) ' +/- ' num2str(cumun)]},'Units','normalized','Fontsize',15)
                        
                        hold off
                        
                        ylabel(txt3)
                        xlabel(txt1)
                        
                        ylim([0 max(Ob2(ib,8))+1])
                        xlim([0 max(Ob1(ia,8))+1])
                        
                        h1.FontSize = 17;
                        h2.FontSize = 17;
                        h3.FontSize = 17;
                        
                        close(h)
                        
                        %% Removing intermediate link
                        %                     tmp(j-1) = NaN;
                        %                     tmp = tmp(isfinite(tmp));
                        if length(tmp)<3
                            sw_dn = false;
                        end
                        %                     break
                        
                        
                    end
                    
                end
                
                if j==length(tmp)
                    sw_dn = false;
                end
                
            end
            
            path{1,n} = tmp;
            
        end
        
        %Adding colors and calculating calibration factors and uncertainties
        if length(tmp)>1
            
            clFct = tmp*0+1;
            clUnc = tmp*0;
            
            for j = 1:length(tmp)
                Clrbns(tmp(j)) = 1;
                
                %Start finding calibration factors starting with the second
                %link
                if j>1
                    
                    i = tmp(j-1);
                    i2 = tmp(j);
                    
                    for nc = 2:j
                        
                        clFct(nc-1) = clFct(nc-1)*LnkVlP(i,i2);
                        clUnc(nc-1) = clUnc(nc-1) + ( LnkUNP(i,i2)/LnkVlP(i,i2) )^2;
                        
                    end
                    
                end
                
            end
            
            clUnc = sqrt(clUnc).*clFct;
            
            Cali{1,n} = clFct;
            Cali{2,n} = clUnc;
            
        end
        
    end   
    
    %% Plotting relationships along optimal path
    for n = 2:length(tmp)
        
        i  = tmp(n-1);
        i2 = tmp(n);
        
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
        
        fin = find(ObsNm==Unob(i));
        txt1 = ObsLb{fin};
        k = strfind(txt1,',');
        if ~isempty(k)
            txt1 = txt1(1:k(1)-1);
        end
        
        Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
        
        fin = find(ObsNm==Unob(i2));
        txt2 = ObsLb{fin};
        k = strfind(txt2,',');
        if ~isempty(k)
            txt2 = txt2(1:k(1)-1);
        end
        
        [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
        RHO = corr(Ob1(ia,8),Ob2(ib,8),'type','Spearman');
        
        ft = fittype('a*x');
        %Ob1 to Ob2
        [PropFit2,Propgof2] = fit(Ob1(ia,8),Ob2(ib,8),ft,'Robust', 'on', 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
        if ~isempty(lastwarn)
            [PropFit2,Propgof2] = fit(Ob1(ia,8),Ob2(ib,8),ft, 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
            lastwarn('')
        end
        
        PrpVa2 = coeffvalues(PropFit2);
        PrpUn2 = confint(PropFit2);
        PrpUn2 = abs(PrpUn2(1)-PrpVa2);
        PrpR22 = Propgof2.adjrsquare;
        
        %Ob1 to Ob2
        [LinFit2,Lingof2] = fit(Ob1(ia,8),Ob2(ib,8),'poly1');
        if ~isempty(lastwarn)
            [LinFit2,Lingof2] = fit(Ob1(ia,8),Ob2(ib,8),'poly1','Robust', 'on');
            lastwarn('')
        end
        
        LinVa2 = coeffvalues(LinFit2);
        LinUn2 = confint(LinFit2);
        LinUn2 = [abs(LinVa2(1)-LinUn2(1,1)) abs(LinVa2(2)-LinUn2(1,2))];
        LinR22 = Lingof2.adjrsquare;
        
        
        figure(5)
        h = gcf;
        set(h,'Position',[100 100 650 600], 'Color','w');

        scatter(Ob1(ia,8),Ob2(ib,8),'ob','MarkerFaceAlpha',0.3)
        hold on
        scatter(Ob1(ia,8),Ob2(ib,8),'ob','filled','MarkerFaceAlpha',0.1)
        
        x = [0:0.1:max(Ob1(ia,8))+1];
        
        plot(x,x*LnkVlP(i,i2),'-.','Color',[1 1 0],'LineWidth',3)
        plot(x,x*LnkVlLa(i,i2) + LnkVlLb(i,i2),':','Color',[0 1 1],'LineWidth',3)
        
        text(0.05,0.95,{['a = ' num2str(LnkVlP(i,i2)) ' +/- ' num2str(LnkUNP(i,i2))],['Prp R2 = ' num2str(LnkUNPR(i,i2))],['Lin R2 = ' num2str(LnkUNLR(i,i2))]},'Units','normalized','Fontsize',15)
        
        hold off
        
        ylabel(txt2)
        xlabel(txt1)
        
        ylim([0 max(Ob2(ib,8))+1])
        xlim([0 max(Ob1(ia,8))+1])
        
        h1 = gca;
        h1.FontSize = 17;
        
        if LnkUNLR(i,i2)>LnkUNPR(i,i2)
           LnkUNPR(i,i2) 
        end
        
        close(h)
    end
    
    %% Creating calibrated composite and removing nodes
    
    CpmObs = [];
    CpmUnc = [];
    ii = size(Cali,2);
    Cal = Cali{1,ii};
    Unc = Cali{2,ii};
    Pat = path{1,ii};
    
    for j = 1:length(Pat)-1
        
        tmpCmp = date*NaN;
        tmpUnc = date*NaN;
        Ob1 = GN_Dat(GN_Dat(:,6)==Unob(Pat(j)),:);
        [C,ia,ib] = intersect(Ob1(:,2),date);
        
        tmpCmp(ib) = Ob1(ia,8);
        tmpUnc(ib) = Ob1(ia,8)*0+1;
        
        CpmObs = [CpmObs;tmpCmp];
        CpmUnc = [CpmUnc;tmpUnc];
 
        i  = Pat(j);
        i2 = Pat(j+1);        
       
        if (LnkUNPR(i,i2)>=LnkUNLR(i,i2))||isnan(LnkUNLR(i,i2))
           
            CpmObs = CpmObs*LnkVlP(i,i2);
            CpmUnc = CpmObs.*LnkVlP(i,i2).*sqrt( (LnkUNP(i,i2)/LnkVlP(i,i2))^2 + (CpmUnc./CpmObs).^2 );
            
        else        

            CpmObs = CpmObs*LnkVlLa(i,i2) + LnkVlLb(i,i2);
            CpmUnc = sqrt( (CpmObs.*LnkVlLa(i,i2)).^2.*( (LnkUNLa(i,i2)/LnkVlLa(i,i2))^2 + (CpmUnc./CpmObs).^2 ) + LnkUNLb(i,i2)^2);

        end
        
        
        if Pat(j)~=590
            DM1(Pat(j),:) = 0;
            LnkMO(Pat(j),:) = 0;
            DM1(:,Pat(j)) = 0;
            LnkMO(:,Pat(j)) = 0;
        end
        
    end
    
%     CpmObs = round(CpmObs);
%     CpmUnc = ceil(CpmUnc);
    
    Wght = 1./(CpmUnc.^2);
    
    
    
    figure(2)
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
    
    plot(date2,CpmObs,'.')
    
    xlim([1745 2011])
    ylim([0 30])
    
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
    p3 = h2.Position;
    p3(1) = 0.02;
    p3(2) = 0.07;
    p3(3) = 0.97;
    p3(4) = 0.25;
    h2.Position = p3;
    % set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
    set(h2,'XMinorTick','on')
    set(h2,'layer','top')
    
    h1.FontSize = 17;
    p1 = h1.Position;
    p1(1) = p3(1);
    p1(2) = p3(2)+p3(4);
    p1(3) = p3(3);
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
    set(h,'Position',[0 5 1925 1080], 'Color','w');
    h1 = subplot(3,1,1);
    
    
    
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
    
    xlim([1745 2011])
    caxis([-0.5 5.5])
    
    h2 = subplot(3,1,2);
    
    plot(date2,CpmObs,'.')
    
    xlim([1745 2011])
    ylim([0 30])
    

    h3 = subplot(3,1,3);
    
    plot(date2,CpmUnc,'.')
    
    xlim([1745 2011])
    ylim([0 4])
    
    
    h3.FontSize = 17;
    p3 = h3.Position;
    p3(1) = 0.02;
    p3(2) = 0.07;
    p3(3) = 0.97;
    p3(4) = 0.20;
    h3.Position = p3;
    % set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
    set(h3,'XMinorTick','on')
    set(h3,'layer','top')
    
    h2.FontSize = 17;
    p2 = h2.Position;
    p2(1) = p3(1);
    p2(2) = p3(2)+p3(4);
    p2(3) = p3(3);
    p2(4) = p3(4);
    h2.Position = p2;
    % set(h2, 'YTick', [10.^0 10.^1 10.^2 10.^3 10.^4 10.^5 10.^6 10.^7 10.^8 10.^9 10.^10])
    set(h2,'XTick',[])
    set(h2,'XMinorTick','on')
    set(h2,'layer','top')
    
    h1.FontSize = 17;
    p1 = h1.Position;
    p1(1) = p3(1);
    p1(2) = p3(2)+2*p3(4);
    p1(3) = p3(3);
    p1(4) = 0.47;
    h1.Position = p1;
    % set(h1, 'YTick', [-90:30:90])
    set(h1,'YTick',[])
    set(h1,'XMinorTick','on')
    set(h1,'layer','top')
    set(h1,'XAxisLocation','top')
    
    CalCmp = [CalCmp; CpmObs];
    UcnCmp = [UcnCmp; CpmUnc];
    
    
end


