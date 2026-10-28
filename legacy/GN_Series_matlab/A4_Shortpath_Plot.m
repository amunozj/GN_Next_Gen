close all

clear

% load GN_Links_Mat_Fits.mat;
% load GN_Links_Mat_Fits_Non_Robust.mat;

% load GN_Links_Mat_Fits_Bisquare_All.mat;
% load GN_Links_Mat_Fits_LAR_with_Catch.mat
% load GN_Links_Mat_Fits_Robust;
load GN_Links_Mat_Fits_Bisquare_paths_10000.mat;
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

DM1 = DM1./LnkMO;

DM1(LnkUNP==0) = 0;

%% Starting Node
%Staudach
stnd  = 445;

%% Ending Node
%RGO
endnd = 311;

PathLn = cellfun(@length,shortestPaths);
shortestPaths = shortestPaths(PathLn<=10);
totalCosts = totalCosts(PathLn<=10);

CalCmp = [];
UcnCmp = [];

for mp = 1:100:length(totalCosts)
    mp
    
    G = graph((DM1>0)|(DM1'>0));
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
    
    %     ypos = rand(length(dpos),1)-0.5;
    
    [tmp,d] = shortestpath(G1,stnd,endnd)
    path{1,1} = tmp;
    if length(tmp)>1
        
        for j = 1:length(tmp)
            Clrbns(tmp(j)) = 1;
            
            Lnkdates = GN_Dat((GN_Dat(:,6)==Unob(tmp(j))),2);
            [C,ia,ib] = intersect(date,Lnkdates);
            Mndate(ia,i) = 1;
            
        end
        
    end
    
    %% Creating calibrated composite and removing nodes
    
    %Resetting colors
    for i = 1:length(bins)
        
        tmp = bins{i};
        for j = 1:length(tmp)
            
            %Defining node colors
            Clrbns(tmp(j)) = mod(i,4)+2;
            
        end
    end
    
    CpmObs = [];
    CpmUnc = [];
    Pat = shortestPaths{1,mp};
    
    Clrbns(Pat(1)) = 1;
    
    for j = 1:length(Pat)-1
        
        Clrbns(Pat(j+1)) = 1;
        
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
        
        
        if (LnkUNPR(i,i2)<Rtol*LnkUNLR(i,i2))&&( abs(LnkVlP(i,i2)-LnkVlLa(i,i2))>LnkUNLa(i,i2) )&&( abs(LnkVlLb(i,i2))>LnkUNLb(i,i2) )
            
            CpmObs = CpmObs*LnkVlP(i,i2);
            CpmUnc = CpmObs.*LnkVlP(i,i2).*sqrt( (LnkUNP(i,i2)/LnkVlP(i,i2))^2 + (CpmUnc./CpmObs).^2 );
            
        else
            
            CpmObs = CpmObs*LnkVlLa(i,i2) + LnkVlLb(i,i2);
            CpmUnc = sqrt( (CpmObs.*LnkVlLa(i,i2)).^2.*( (LnkUNLa(i,i2)/LnkVlLa(i,i2))^2 + (CpmUnc./CpmObs).^2 ) + LnkUNLb(i,i2)^2 );
            
        end
        
    end
    
    tmpCmp = date*NaN;
    Ob1 = GN_Dat(GN_Dat(:,6)==Unob(endnd),:);
    [C,ia,ib] = intersect(Ob1(:,2),date);
    
    tmpCmp(ib) = Ob1(ia,8);
    
    CpmObs = [CpmObs;tmpCmp];
    CpmUnc = [CpmUnc;tmpCmp*0];
    
    figure(3)
    colormap(Clr)
    hf = gcf;
    set(hf,'Position',[0 5 1925 1080], 'Color','w');
    h1 = subplot(3,1,1);
    
    
    
    h = plot(G,'XData', dpos,'YData', ypos,'EdgeAlpha',0.2,'MarkerSize',szDeg,'NodeCData',Clrbns,'EdgeCData',Clredg,'LineWidth',1);
    hold on
    
    ObsNm = [GN_Obs{:,1}];
    ObsLb = GN_Obs(:,5);
    
    tmp = shortestPaths{1,mp};
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
    
    
    
    hold off
    
    xlim([1745 2020])
    caxis([-0.5 5.5])
    
    h2 = subplot(3,1,2);
    
    plot(date2,CpmObs,'.')
    hold on
    
    text(0.99,0.9,'Calibrated Counts','units','normalized','FontSize',17,'HorizontalAlignment','right')
    
    hold off
    
    xlim([1745 2020])
    ylim([0 30])
    
    
    h3 = subplot(3,1,3);
    
    plot(date2,CpmUnc,'.')
    hold on
    text(0.99,0.9,'Uncertainty','units','normalized','FontSize',17,'HorizontalAlignment','right')
    hold off
    
    xlim([1745 2020])
    ylim([0 30])
    
    
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
    
    outFld = 'Scw_iter';
    outFld = [pwd '\' outFld];
    mkdir(outFld)
    
    name = [num2str(mp) 'Sch_Lcrn' '.png'];
    name = [outFld '\' name];
    
    export_fig(name, '-nocrop')
    
    
    
%     %% Plotting relationships along optimal path
%     tmp = shortestPaths{1,mp};
%     for n = 2:length(tmp)
%         
%         i  = tmp(n-1);
%         i2 = tmp(n);
%         
%         fin = find(ObsNm==Unob(i));
%         txt1 = ObsLb{fin};
%         k = strfind(txt1,',');
%         if ~isempty(k)
%             txt1 = txt1(1:k(1)-1);
%         end
%         
%         Ob1 = GN_Dat(GN_Dat(:,6)==Unob(i),2);
%         %Adding adjacent days
%         for k = 1:dtol
%             Ob1 = [Ob1;GN_Dat(GN_Dat(:,6)==Unob(i),2)-k;GN_Dat(GN_Dat(:,6)==Unob(i),2)+k];
%         end
%         Ob1 = unique(Ob1);
%         
%         Ovrlp = ismember(GN_Dat(:,2),Ob1)&(GN_Dat(:,6)~=0)&(GN_Dat(:,6)~=Unob(i));
%         MObs  = unique(GN_Dat(Ovrlp,6));
%         
%         Ob1 = GN_Dat(GN_Dat(:,6)==Unob(i),:);
%         %Adding adjacent days
%         for k = 1:dtol
%             
%             tmp2 = Ob1;
%             tmp2(:,2) = tmp2(:,2)+k;
%             [C,ia] = setdiff(tmp2(:,2),Ob1(:,2));
%             Ob1 = [Ob1;tmp2(ia,:)];
%             
%             tmp2 = Ob1;
%             tmp2(:,2) = tmp2(:,2)-k;
%             [C,ia] = setdiff(tmp2(:,2),Ob1(:,2));
%             Ob1 = [Ob1;tmp2(ia,:)];
%         end
%         
%         
%         fin = find(ObsNm==Unob(i2));
%         txt2 = ObsLb{fin};
%         k = strfind(txt2,',');
%         if ~isempty(k)
%             txt2 = txt2(1:k(1)-1);
%         end
%         
%         Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
%         
%         [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
%         RHO = corr(Ob1(ia,8),Ob2(ib,8),'type','Spearman');
%         
%         %Unraveling pairs to find number of incidence of each pair
%         
%         linIn = sub2ind([max(Ob2(ib,8)) max(Ob1(ia,8))]+1, Ob2(ib,8)+1, Ob1(ia,8)+1);
%         UnIn = unique(linIn);
%         Wg   = histc(linIn,UnIn);
%         [Ob2R,Ob1R] = ind2sub([max(Ob2(ib,8)) max(Ob1(ia,8))]+1,UnIn);
%         Ob2R = Ob2R - 1;
%         Ob1R = Ob1R - 1;
%         
%         
%         figure(4)
%         h = gcf;
%         set(h,'Position',[100 100 600 600], 'Color','w');
%         h1 = gca;
%         
%         x = [0:0.1:max(Ob1(ia,8))+1];
%         plot(x,x*LnkVlP(i,i2),'-.','Color',[0.5 0 0],'LineWidth',3)
%         hold on
%         plot(x,x*LnkVlLa(i,i2) + LnkVlLb(i,i2),':','Color',[0 1 1]*.75,'LineWidth',3)
%         
%         scatter(Ob1R,Ob2R,'b','filled','SizeData', 10*sqrt(Wg)+1)
%         
%         text(0.7,0.1,{['\rho = ' num2str(roundsd(RHO,2))],['Prp AR2 = ' num2str(roundsd(LnkUNPR(i,i2),2))],['Lin AR2 = ' num2str(roundsd(LnkUNLR(i,i2),2))]},'Units','normalized','Fontsize',15)
%         
%         hold off
%         
%         legend(['y = (' num2str(roundsd(LnkVlP(i,i2),2)) ' +/- ' num2str(roundsd(LnkUNP(i,i2),2)) ')x'], ['y = (' num2str(roundsd(LnkVlLa(i,i2),2)) ' +/- ' num2str(roundsd(LnkUNLa(i,i2),2)) ')x + (' num2str(roundsd(LnkVlLb(i,i2),2)) ' +/- ' num2str(roundsd(LnkUNLb(i,i2),2)) ')'], 'Location', 'NorthWest')
%         legend boxoff
%         
%         xlabel(txt1)
%         ylabel(txt2)
%         
%         xlim([0 max(Ob1R)+round(max(Ob1R)*0.2)])
%         ylim([0 max(Ob2R)+round(max(Ob2R)*0.2)])
%         
%         h1.FontSize = 17;
%         
%         name = [txt1 '_' txt2 '.png'];
%         name = [outFld '\' name];
%         
% %         export_fig(name, '-nocrop')
%         
%         close(h)
%         
%     end
    
    close(hf)

    %% Creating composite
    
    tmpCmp = date*NaN;
    tmpUnc = date*NaN;
    for kk = size(CpmObs,1):-1:1
       
        tmpUnc(isnan(tmpCmp)&isfinite(CpmObs(kk,:))) = CpmUnc(kk,isnan(tmpCmp)&isfinite(CpmObs(kk,:)));
        tmpCmp(isnan(tmpCmp)&isfinite(CpmObs(kk,:))) = CpmObs(kk,isnan(tmpCmp)&isfinite(CpmObs(kk,:)));
        
    end
    
    tmpCmp = round(tmpCmp);
    tmpUnc = ceil(tmpUnc);
    
    
    CalCmp = [CalCmp; tmpCmp];
    UcnCmp = [UcnCmp; tmpUnc];
    
    
end



