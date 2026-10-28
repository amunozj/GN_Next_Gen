close all

clear

load GN_Links_Mat_Fits.mat;


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



%% Finding non-zero matches


[row,col] = find(LnkVlP~=0);

ObsNm = [GN_Obs{:,1}];
ObsLb = GN_Obs(:,5);
for n = 1:100
    
    ind = floor(rand(1,1)*length(row)) + 1;
    i  = col(ind);
    i2 = row(ind);
    
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
    RHO = corr(Ob2(ib,8),Ob1(ia,8),'type','Spearman');

    
    %% Proportional
    %             fo = fitoptions('Method','NonlinearLeastSquares', 'Robust', 'LAR',...
    %                 'StartPoint',[1],'MaxFunEvals',1000,'MaxIter',1000);
    %             fo = fitoptions('Method','LinearLeastSquares', 'Robust', 'LAR',...
    %                 'StartPoint',[1],'MaxFunEvals',1000,'MaxIter',1000);
    ft = fittype('a*x');
    
    %Ob2 to Ob1
    [PropFit1,Propgof1] = fit(Ob2(ib,8),Ob1(ia,8),ft,'Robust', 'on', 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
    if ~isempty(lastwarn)
        [PropFit1,Propgof1] = fit(Ob2(ib,8),Ob1(ia,8),ft, 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000);
        lastwarn('')
    end
    
    PrpVa1 = coeffvalues(PropFit1);
    PrpUn1 = confint(PropFit1);
    PrpUn1 = abs(PrpUn1(1)-PrpVa1);
    PrpR21 = Propgof1.adjrsquare;
    
    
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
    
    

    
    
    %% Linear
    
    %Ob2 to Ob1
    [LinFit1,Lingof1] = fit(Ob2(ib,8),Ob1(ia,8),'poly1','Robust', 'on');
    if ~isempty(lastwarn)
        [LinFit1,Lingof1] = fit(Ob2(ib,8),Ob1(ia,8),'poly1');
        lastwarn('')
    end
    
    LinVa1 = coeffvalues(LinFit1);
    LinUn1 = confint(LinFit1);
    LinUn1 = [abs(LinVa1(1)-LinUn1(1,1)) abs(LinVa1(2)-LinUn1(1,2))];
    LinR21 = Lingof1.adjrsquare;
    
    %Ob1 to Ob2
    [LinFit2,Lingof2] = fit(Ob1(ia,8),Ob2(ib,8),'poly1','Robust', 'on');
    if ~isempty(lastwarn)
        [LinFit2,Lingof2] = fit(Ob1(ia,8),Ob2(ib,8),'poly1');
        lastwarn('')
    end
    
    LinVa2 = coeffvalues(LinFit2);
    LinUn2 = confint(LinFit2);
    LinUn2 = [abs(LinVa2(1)-LinUn2(1,1)) abs(LinVa2(2)-LinUn2(1,2))];
    LinR22 = Lingof2.adjrsquare;
    
    
    
    figure(5)
    h = gcf;
    set(h,'Position',[100 100 1300 600], 'Color','w');
    h1 = subplot(1,2,1);

    scatter(Ob2(ib,8),Ob1(ia,8),'o','filled','MarkerFaceAlpha',0.3)
    hold on
    
    x = [0:0.1:max(Ob2(ib,8))+1];
    
    plot(x,x*PrpVa1,'Color',[1 0 0]*0,'LineWidth',3)
    plot(x,x*LinVa1(1) + LinVa1(2),'--','Color',[0 0 1],'LineWidth',3)

    plot(x,x*LnkVlP(i2,i),'-.','Color',[1 1 0],'LineWidth',3)
    plot(x,x*LnkVlLa(i2,i) + LnkVlLb(i2,i),':','Color',[0 1 1],'LineWidth',3)
    
    hold off
    
    xlabel(txt2)
    ylabel(txt1)
    
    xlim([0 max(Ob2(ib,8))+1])
    ylim([0 max(Ob1(ia,8))+1])
    
    
    h2 = subplot(1,2,2);
    
    scatter(Ob1(ia,8),Ob2(ib,8),'o','filled','MarkerFaceAlpha',0.3)
    hold on
    
    x = [0:0.1:max(Ob1(ia,8))+1];
    
    plot(x,x*PrpVa2,'Color',[1 0 0]*0,'LineWidth',3)
    plot(x,x*LinVa2(1) + LinVa2(2),'Color',[0 0 1],'LineWidth',3)

    plot(x,x*LnkVlP(i,i2),'-.','Color',[1 1 0],'LineWidth',3)
    plot(x,x*LnkVlLa(i,i2) + LnkVlLb(i,i2),':','Color',[0 1 1],'LineWidth',3)

    
    hold off
    
    ylabel(txt2)
    xlabel(txt1)
    
    ylim([0 max(Ob2(ib,8))+1])
    xlim([0 max(Ob1(ia,8))+1])
    
    
    h1.FontSize = 17;
    h2.FontSize = 17;
    
    close(h)
    
    
end







