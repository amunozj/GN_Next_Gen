clear

close all

load GNobservations_JV_V1-12.mat

lastwarn('')

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
MinCon = 3;
MinRho = 0.6;

LnkMO = zeros(length(Unob));  %Days of overalp
LnkMPt = LnkMO;  %Number of unique pairs
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
        
        Ob2 = GN_Dat(GN_Dat(:,6)==Unob(i2),:);
        
        [C,ia,ib] = intersect(Ob1(:,2),Ob2(:,2));
        RHO = corr(Ob1(ia,8),Ob2(ib,8),'type','Spearman');
        
        %Unraveling pairs to find number of incidence of each pair
        
        linIn = sub2ind([max(Ob2(ib,8)) max(Ob1(ia,8))]+1, Ob2(ib,8)+1, Ob1(ia,8)+1);
        UnIn = unique(linIn);
        Wg   = histc(linIn,UnIn);
        [Ob2R,Ob1R] = ind2sub([max(Ob2(ib,8)) max(Ob1(ia,8))]+1,UnIn);
        Ob2R = Ob2R - 1;
        Ob1R = Ob1R - 1;
        
        LnkMO(i,i2) = length(C);
        LnkMO(i2,i) = length(C);
        
        %Only add the entry if the link moves the correlation forward in
        %time
        if ((max(Ob1(:,2))<max(Ob2(:,2)))&&(mean(Ob1(:,2))<mean(Ob2(:,2))))
            
            LnkMPt(i,i2) = length(UnIn);
            LnkMR(i,i2) = RHO;
            
            if (length(UnIn) >= MinCon)&&(RHO>=MinRho)
                
                %% Proportional
                ft = fittype('a*x');
                
                %Ob1 to Ob2
                [PropFit2,Propgof2] = fit(Ob1R,Ob2R,ft,'Robust', 'Bisquare', 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000,'Weights',Wg);
                if ~isempty(lastwarn)
                    [PropFit2,Propgof2] = fit(Ob1R,Ob2R,ft, 'Startpoint', [1],'MaxFunEvals',1000,'MaxIter',1000,'Weights',Wg);
                    lastwarn('')
                end
                
                PrpVa2 = coeffvalues(PropFit2);
                PrpUn2 = confint(PropFit2);
                PrpUn2 = abs(PrpUn2(1)-PrpVa2);
                PrpR22 = Propgof2.adjrsquare;
                
                LnkVlP(i,i2) = PrpVa2;
                LnkUNP(i,i2) = PrpUn2;
                LnkUNPR(i,i2) = PrpR22;
                
                %% Linear
                
                %Ob1 to Ob2
                [LinFit2,Lingof2] = fit(Ob1R,Ob2R,'poly1','Robust', 'Bisquare','Weights',Wg);
                if ~isempty(lastwarn)
                    [LinFit2,Lingof2] = fit(Ob1R,Ob2R,'poly1','Weights',Wg);
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
                
                LnkUNLR(i,i2) = LinR22;
                
                
            end
            
        end
        
    end
    
end


dpos = Unob*0;
for i = 1:length(Unob)
    
    dpos(i) = mean(GN_Dat(GN_Dat(:,6)==Unob(i),1));
    
end

save(['GN_Links_Mat_Fits_Bisquare_' num2str(MinCon) '.mat'], 'dpos','LnkMO','LnkMPt', 'LnkMR', 'LnkVlP', 'LnkUNP', 'LnkUNPR', 'LnkVlLa', 'LnkUNLa', 'LnkVlLb', 'LnkUNLb', 'LnkUNLR')
