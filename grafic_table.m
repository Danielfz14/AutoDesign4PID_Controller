
close all
load('Datos_completos_DE.mat')
load('Datos_completos_GA.mat')
load('Datos_completos_MHnew.mat')
load('Datos_completos_PSO.mat')

% Fitness all
N=54;
k=4;
esc=15;

Fit_all=[fitMHnew(1:N),fitpso(1:N),fitGA(1:N),fit_DE(1:N)];
label=["MH$^*$","PSO","GA","DE"];
ffit=Graphics('Fig1');
boxplot(Fit_all(:,k),'Notch','on','Labels',{[char(label(k)) ...
    '']})
set(gca,'YScale','log');
hold on
 c = [0 .5 .5]	;
s=swarmchart(0.25+ones(length(Fit_all(:,k))),Fit_all(:,k),[],c,'filled','o', 'XJitter','randn','XJitterWidth',0.1);
ylabel("Fitness")
xlabel("Metaheuristics")
max_a=max(Fit_all(:,k));
min_a=min(Fit_all(:,k));
limite=[min_a-min_a/esc max_a+max_a/esc];
ylim(limite)
 setup(ffit);
 setsize(ffit,2,[4,5]);
 setfont(ffit,12)
 ffit=Graphics('Fig1');
 histfit(fitMHnew,2,'beta')
 D_hist_log=log10(Fit_all(:,k));
  h= histogram(Fit_all(:,k),10,'FaceColor',[0 .5 .5]);
  xlim(limite)
 set(gca,'YDir','reverse','XAxisLocation','top','YAxisLocation','right','XScale','log');
 camroll(90)
 setup(ffit);
 setsize(ffit,2,[4,5]);
 setfont(ffit,12)

% C_V fitness

C_v_mhnew_fit=std(fitMHnew)/mean(fitMHnew)*100;
C_v_pso_fit=std(fitpso)/mean(fitpso)*100;
C_v_GA_fit=std(fitGA)/mean(fitGA)*100;
C_v_DE_fit=std(fit_DE)/mean(fit_DE)*100;

Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};
Best_fitness = [min(fitMHnew);min(fitpso);min(fitGA);min(fit_DE)];
Average      = [mean(fitMHnew);mean(fitpso);mean(fitGA);mean(fit_DE)] ;
STD          = [std(fitMHnew);std(fitpso);std(fitGA);std(fit_DE)] ;
C_v          = [C_v_mhnew_fit;C_v_pso_fit;C_v_GA_fit;C_v_DE_fit];
IQR          = [iqr(fitMHnew);iqr(fitpso);iqr(fitGA);iqr(fit_DE)];
T1 = table(Metaheuristics,Best_fitness,Average,STD,C_v,IQR)
% constantes

C_v_mhnew_k=std(MH_new_kp_ki_kd)./mean(MH_new_kp_ki_kd)*100;
C_v_pso_k=std(MH_PSO_kp_ki_kd)./mean(MH_PSO_kp_ki_kd)*100;
C_v_GA_k=std(MH_GA_kp_ki_kd)./mean(MH_GA_kp_ki_kd)*100;
C_v_DE_k=std(MH_DE_kp_ki_kd)./mean(MH_DE_kp_ki_kd)*100;

Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};

Average_kp_ki_kd  = [mean(MH_new_kp_ki_kd);mean(MH_PSO_kp_ki_kd);mean(MH_GA_kp_ki_kd);mean(MH_DE_kp_ki_kd)];
STD_kp_ki_kd      = [std(MH_new_kp_ki_kd);std(MH_PSO_kp_ki_kd);std(MH_GA_kp_ki_kd);std(MH_DE_kp_ki_kd)] ;
C_v_kp_ki_kd      = [C_v_mhnew_k;C_v_pso_k;C_v_GA_k;C_v_DE_k];
T2 = table(Metaheuristics,Average_kp_ki_kd,STD_kp_ki_kd,C_v_kp_ki_kd)

% TS

C_v_mhnew_TS =std(Ts_MH_new)./mean(Ts_MH_new)*100;
C_v_pso_TS   =std(Ts_PSO)./mean(Ts_PSO)*100;
C_v_GA_TS    =std(Ts_GA)./mean(Ts_GA)*100;
C_v_DE_TS    =std(Ts_DE)./mean(Ts_DE)*100;

Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};

Average_Ts  = [mean(Ts_MH_new);mean(Ts_PSO);mean(Ts_GA);mean(Ts_DE)];
STD_TS      = [std(Ts_MH_new);std(Ts_PSO);std(Ts_GA);std(Ts_DE)] ;
C_v_TS      = [C_v_mhnew_TS;C_v_pso_TS;C_v_GA_TS;C_v_DE_TS];
T3 = table(Metaheuristics,Average_Ts,STD_TS,C_v_TS)

%

C_v_mhnew_L =std(L_MH_new)./mean(L_MH_new)*100;
C_v_pso_L   =std(L_PSO)./mean(L_PSO)*100;
C_v_GA_L    =std(L_GA)./mean(L_GA)*100;
C_v_DE_L    =std(L_DE)./mean(L_DE)*100;

Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};

Average_L  = [mean(L_MH_new);mean(L_PSO);mean(L_GA);mean(L_DE)];
STD_L      = [std(L_MH_new);std(L_PSO);std(L_GA);std(L_DE)] ;
C_v_L      = [C_v_mhnew_L;C_v_pso_L;C_v_GA_L;C_v_DE_L];
T4 = table(Metaheuristics,Average_L,STD_L,C_v_L)
% performance IAE

C_v_mhnew_IAE =std(IAE1_mhnew)./mean(IAE1_mhnew)*100;
C_v_pso_IAE   =std(IAE1_pso)./mean(IAE1_pso)*100;
C_v_GA_IAE   =std(IAE1_ga)./mean(IAE1_ga)*100;
C_v_DE_IAE    =std(IAE1_de)./mean(IAE1_de)*100;
Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};
Average_IAE  = [mean(IAE1_mhnew);mean(IAE1_pso);mean(IAE1_ga);mean(IAE1_de)];
STD_IAE     = [std(IAE1_mhnew);std(IAE1_pso);std(IAE1_ga);std(IAE1_de)] ;
C_v_IAE      = [C_v_mhnew_IAE;C_v_pso_IAE;C_v_GA_IAE;C_v_DE_IAE];
T5 = table(Metaheuristics,Average_IAE,STD_IAE,C_v_IAE)

% performance ISE
C_v_mhnew_ISE =std(ISE1_mhnew)./mean(ISE1_mhnew)*100;
C_v_pso_ISE   =std(ISE1_pso)./mean(ISE1_pso)*100;
C_v_GA_ISE   =std(ISE1_ga)./mean(ISE1_ga)*100;
C_v_DE_ISE    =std(ISE1_de)./mean(ISE1_de)*100;
Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};
Average_ISE  = [mean(ISE1_mhnew);mean(ISE1_pso);mean(ISE1_ga);mean(ISE1_de)];
STD_ISE     = [std(ISE1_mhnew);std(ISE1_pso);std(ISE1_ga);std(ISE1_de)] ;
C_v_ISE      = [C_v_mhnew_ISE;C_v_pso_ISE;C_v_GA_ISE;C_v_DE_ISE];
T6 = table(Metaheuristics,Average_ISE,STD_ISE,C_v_ISE)
% performance ITAE
C_v_mhnew_ITAE =std(ITAE1_mhnew)./mean(ITAE1_mhnew)*100;
C_v_pso_ITAE  =std(ITAE1_pso)./mean(ITAE1_pso)*100;
C_v_GA_ITAE   =std(ITAE1_ga)./mean(ITAE1_ga)*100;
C_v_DE_ITAE   =std(ITAE1_de)./mean(ITAE1_de)*100;
Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};
Average_ITAE = [mean(ITAE1_mhnew);mean(ITAE1_pso);mean(ITAE1_ga);mean(ITAE1_de)];
STD_ITAE    = [std(ITAE1_mhnew);std(ITAE1_pso);std(ITAE1_ga);std(ITAE1_de)] ;
C_v_ITAE      = [C_v_mhnew_ITAE;C_v_pso_ITAE;C_v_GA_ITAE;C_v_DE_ITAE];
T7 = table(Metaheuristics,Average_ITAE,STD_ITAE,C_v_ITAE)
% performance ITSE
C_v_mhnew_ITSE=std(ITSE1_mhnew)./mean(ITSE1_mhnew)*100;
C_v_pso_ITSE  =std(ITAE1_pso)./mean(ITAE1_pso)*100;
C_v_GA_ITSE   =std(ITSE1_ga)./mean(ITSE1_ga)*100;
C_v_DE_ITSE   =std(ITSE1_de)./mean(ITSE1_de)*100;
Metaheuristics = {'MH$^*$';'PSO';'GA';'DE'};
Average_ITSE = [mean(ITSE1_mhnew);mean(ITSE1_pso);mean(ITSE1_ga);mean(ITSE1_de)];
STD_ITSE    = [std(ITSE1_mhnew);std(ITSE1_pso);std(ITSE1_ga);std(ITSE1_de)] ;
C_v_ITSE      = [C_v_mhnew_ITSE;C_v_pso_ITSE;C_v_GA_ITSE;C_v_DE_ITSE];
T8 = table(Metaheuristics,Average_ITSE,STD_ITSE,C_v_ITSE)


