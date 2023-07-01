function fcost=ripp_buck2(x1,x2,x3)
kp=x1; %0.05
ki=x2; % 8.072
kd=x3; %0.05
% lambda=x4;
% miu=x5;
F=40e3;%1e3;
T0=0.00448;
L0=41.841;
L1=1e-3*(1.0);
C1=100e-6*(1.0);
Vg=36;
D=0.103;
R1=6;
Ts=1e-7;
Ref=3;
Tsin=0.01;
options = simset('SrcWorkspace','current');
simout=sim('buck_N',[],options);
 Vout=Vout.signals.values;
 re=referencia.signals.values;
 rui_=ruido.signals.values;
% ITAE=simout.ITAE;
t=simout;
H=stepinfo(Vout,t,3,'SettlingTimeThreshold',0.05)
Ts=H.SettlingTime;
L = H.Overshoot;
 ffit=Graphics('Fig1');
 hold on
plot(t,re,'b',LineWidth=1.2)
plot(t,Vout,'r',LineWidth=1.2)
grid on
%xlim([0 4e-3])
 ylim([0 9.5])
 %¿yline(3,'-.k',LineWidth=1);
 %xline(3.1480e-04,'-.g',LineWidth=1)

 ylabel("Output voltage change (V)")
xlabel("Time (s) ")
 setup(ffit);
 setsize(ffit,2,[2,1]);
 setfont(ffit,12)

 ffit=Graphics('Fig1');
 hold on
 yyaxis left
plot(t,Vout,LineWidth=1.2)
 ylabel("Output")

ylim([0 3.5])
yyaxis right
plot(t,rui_,LineWidth=0.8)
grid on
ylim([0 1.0])
% %xlim([0 4e-3])
  
xlabel("Time (s) ")
legend('Output voltage','Disturbance')
setup(ffit);
 setsize(ffit,2,[2,1]);
 setfont(ffit,12)

  if isnan(Ts),  Ts=100;  end
  if isnan(L),  L=100;  end
% title(['L=',num2str(L),' TS=',num2str(Ts)])

%fcost=0.3*abs(Ts)/T0  + 0.7*abs(L)/L+ 100*ITAE(end)0;
fcost=+0.3*abs(Ts)/T0  + 0.7*abs(L)/L0+15*abs(Vout(end)-3);
%5.13045631e+01, 2.57148710e+01 ,9.22672071e-03
end