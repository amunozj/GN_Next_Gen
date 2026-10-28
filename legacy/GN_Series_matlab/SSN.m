clear 

close all

load SSN_Vers.mat

Clr100 = [255,208,171]/255;
Clr90 = [114,178,229]/255;
Clr70 = [81,178,76]/255;
Clr50 = [111,40,124]/255;
Clr30 = [80,60,0]/255;
alp = 0.75;

figure(1)
h = gcf;
set(h,'Position',[50 50 1500 500], 'Color','w');
h1 = axes('FontSize',17);

% plot(GNhsy(:,1),GNhsy(:,2))
jbfill(GNhsy(:,1),GNhsy(:,2),GNhsy(:,2)*0,ones(size(GNhsy,1),1),Clr50,Clr50,1,alp)

box on

legend('Hoyt & Schatten (1998)','Location','NorthWest')
legend boxoff
ylabel('Group Number')


xlim([1610 2020])
ylim([0 15])

h1.FontSize = 17;
p1 = h1.Position;
p1(1) = 0.075;
p1(3) = 0.9;
h1.Position = p1;

export_fig 'SSNx1.png' -nocrop -m2



figure(2)
h = gcf;
set(h,'Position',[50 50 1500 500], 'Color','w');
h1 = axes('FontSize',17);

jbfill([0 1],[1 1],[0 0],[1 1],Clr50,Clr50,1,alp)
hold on
jbfill(GNbb2y(:,1),GNbb2y(:,2),GNbb2y(:,2)*0,ones(size(GNbb2y,1),1),Clr70,Clr70,1,alp)
hold on
jbfill(GNhsy(:,1),GNhsy(:,2),GNhsy(:,2)*0,ones(size(GNhsy,1),1),Clr50,Clr50,1,alp)
hold on
plot(GNbb2y(:,1),GNbb2y(:,2),'--','Color',Clr70,'LineWidth',2)
hold off

box on

legend('Hoyt & Schatten (1998)','Svalgaard & Schatten (2015)','Location','NorthWest')
legend boxoff
ylabel('Group Number')


xlim([1610 2020])
ylim([0 15])


h1.FontSize = 17;
p1 = h1.Position;
p1(1) = 0.075;
p1(3) = 0.9;
h1.Position = p1;

export_fig 'SSNx2.png' -nocrop -m2


figure(3)
h = gcf;
set(h,'Position',[50 50 1500 500], 'Color','w');
h1 = axes('FontSize',17);


jbfill([0 1],[1 1],[0 0],[1 1],Clr50,Clr50,1,alp)
hold on
jbfill([0 1],[1 1],[0 0],[1 1],Clr70,Clr70,1,alp)
hold on
jbfill([0 1],[1 1],[0 0],[1 1],Clr100,Clr100,1,alp)
hold on
jbfill(GNbb2y(:,1),GNbb2y(:,2),GNbb2y(:,2)*0,ones(size(GNbb2y,1),1),Clr70,Clr70,1,alp)
hold on
jbfill(GNhsy(:,1),GNhsy(:,2),GNhsy(:,2)*0,ones(size(GNhsy,1),1),Clr50,Clr50,1,alp)
hold on
jbfill(GNiuy2(:,1),GNiuy2(:,2),GNiuy2(:,2)*0,ones(size(GNiuy2,1),1),Clr100,Clr100,1,alp)
hold on
plot(GNbb2y(:,1),GNbb2y(:,2),'--','Color',Clr70,'LineWidth',2)
plot(GNhsy(:,1),GNhsy(:,2),'-','Color',Clr50,'LineWidth',2)

box on

hold off

legend('Hoyt & Schatten (1998)','Svalgaard & Schatten (2015)','Usoskin et al. (2016)','Location','NorthWest')
legend boxoff
ylabel('Group Number')



xlim([1610 2020])
ylim([0 15])


h1.FontSize = 17;
p1 = h1.Position;
p1(1) = 0.075;
p1(3) = 0.9;
h1.Position = p1;

export_fig 'SSNx3.png' -nocrop -m2

