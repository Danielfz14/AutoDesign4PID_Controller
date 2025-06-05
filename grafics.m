

close all
load('Datos_completos_DE.mat')
load('Datos_completos_GA.mat')
load('Datos_completos_MHnew.mat')
load('Datos_completos_PSO.mat')
N=54;
%% Fitness all
ffit=Graphics('Fig1');
boxplot(([fitMHnew(1:N),fitpso(1:N),fitGA(1:N),fit_DE(1:N)]),'Labels',{'MH$^*$','PSO','GA','DE'})
xlim([0 5])
hold on
c = [0 .5 .5]	;
s=swarmchart([0.5+ones(N,1),0.5+2*ones(N,1),0.5+3*ones(N,1)...
    0.5+4*ones(N,1)],[fitMHnew(1:N),fitpso(1:N),fitGA(1:N),fit_DE(1:N)],[],c,'.', 'XJitter','randn','XJitterWidth',0.1);
ylabel("Fitness")
xlabel("Metaheuristics")
ylabel("Fitness")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

%% Fitness MHnew and PSO
N=54;
ffit=Graphics('Fig2');
boxplot([fitMHnew(1:N),fitpso(1:N),],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("Fitness")
xlabel("Metaheuristics")

 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
%% KP all
ffit=Graphics('Fig3');
boxplot(([MH_new_kp_ki_kd(1:N,1),MH_PSO_kp_ki_kd(1:N,1),MH_GA_kp_ki_kd(1:N,1),MH_DE_kp_ki_kd(1:N,1)]),'Labels',{'MH$^*$','PSO','GA','DE'},'Symbol','rx')
ylabel("K$_p$")
xlabel("Metaheuristics")
%set(gca,'YScale','log');
% ylim([10^2 1000])

 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
%% Ki all
ffit=Graphics('Fig4');
boxplot([MH_new_kp_ki_kd(1:N,2),MH_PSO_kp_ki_kd(1:N,2),MH_GA_kp_ki_kd(1:N,2),MH_DE_kp_ki_kd(1:N,2)],'Labels',{'MH$^*$','PSO','GA','DE'},'Symbol','rx')
ylabel("K$_i$")
xlabel("Metaheuristics")
set(gca,'YScale','log');
ylim([10^-2 100000])
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
%% kd all
ffit=Graphics('Fig5');
boxplot([MH_new_kp_ki_kd(1:N,3),MH_PSO_kp_ki_kd(1:N,3),MH_GA_kp_ki_kd(1:N,3),MH_DE_kp_ki_kd(1:N,3)],'Labels',{'MH$^*$','PSO','GA','DE'},'Symbol','rx')
ylabel("K$_d$")
xlabel("Metaheuristics")
set(gca,'YScale','log');
ylim([10^-2 100])
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
% %% kd MHnew and PSO
% ffit=Graphics('Fig6');
% boxplot([MH_new_kp_ki_kd(1:N,3),MH_PSO_kp_ki_kd(1:N,3)],'Notch','on','Labels',{'MH$^*$','PSO'})
% ylabel("K$_d$")
% xlabel("Metaheuristics")
%  setup(ffit);
%  setsize(ffit,2,[4,2]);
%  setfont(ffit,12)
%% MP all
ffit=Graphics('Fig7');
boxplot([L_MH_new',L_PSO',L_GA',L_DE'],'Labels',{'MH$^*$','PSO','GA','DE'},'Symbol','rx')
ylabel("M$_p$")
xlabel("Metaheuristics")
set(gca,'YScale','log');
 ylim([10^-5 1000])
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

 ffit=Graphics('Fig8');
boxplot([L_MH_new',L_PSO'],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("M$_p$")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

 %% Ts

ffit=Graphics('Fig9');
boxplot([Ts_MH_new',Ts_PSO',Ts_GA',Ts_DE'],'Labels',{'MH$^*$','PSO','GA','DE'},'Symbol','rx')
ylabel("T$_s$")
xlabel("Metaheuristics")
set(gca,'YScale','log');
 % ylim([10^-3 0.05])
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
%%
ffit=Graphics('Fig10');
boxplot([Ts_MH_new',Ts_PSO'],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("T$_s$")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
 ylim([3e-4 3.8e-4])
 %%
 ffit=Graphics('Fig11');
boxplot([IAE1_mhnew',IAE1_pso',IAE1_ga'],'Notch','on','Labels',{'MH$^*$','PSO','GA'})
ylabel("IAE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
 %%
  ffit=Graphics('Fig11');
boxplot([IAE1_mhnew',IAE1_pso'],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("IAE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
%%
 ffit=Graphics('Fig12');
boxplot([ISE1_mhnew',ISE1_pso',ISE1_ga'],'Notch','on','Labels',{'MH$^*$','PSO','GA'})
ylabel("ISE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

 ffit=Graphics('Fig12');
boxplot([ISE1_mhnew',ISE1_pso'],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("ISE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

%%

 ffit=Graphics('Fig13');
boxplot([ITAE1_mhnew',ITAE1_pso',ITSE1_ga'],'Notch','on','Labels',{'MH$^*$','PSO','GA'})
ylabel("ITAE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

 ffit=Graphics('Fig13');
boxplot([ITAE1_mhnew',ITAE1_pso'],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("ITAE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
%%

 ffit=Graphics('Fig14');
boxplot([ITSE1_mhnew',ITSE1_pso',ITSE1_ga'],'Notch','on','Labels',{'MH$^*$','PSO','GA'})
ylabel("ITSE")
xlabel("Metaheuristics")
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)

 ffit=Graphics('Fig13');
boxplot([ITSE1_mhnew',ITSE1_pso'],'Notch','on','Labels',{'MH$^*$','PSO'})
ylabel("ITSE")
xlabel("Metaheuristics")
ylim([5.8e-8 8e-8])
 setup(ffit);
 setsize(ffit,2,[4,2]);
 setfont(ffit,12)
