function fcost=HHbuck(kp,ki,Kd)
format long 
alfa=0.5;
L0= 10;
T0= 0.0012;
r=6;
l=1e-3;
d=1/12;
c=100e-6;
vg=36;
Gp=tf([vg*d*r/l],[r*c 1 r/l])
[y,tOut] = step(Gp,1);
 C = pid(kp,ki,Kd);
 sys = feedback(Gp*C,1);
[y,tOut] = step(sys,3);
 setup(ffit);
 setsize(ffit,2,[4,2]); 
 setfont(ffit,10)
H= stepinfo(sys);
L = H.Overshoot;
Ts = H.SettlingTime;
ye=y(end-0.2*length(y):end);
E=abs(1-sum(ye)/length(ye));
fcost= alfa*abs(L-L0)/L0 + (1-alfa)*abs(Ts-T0)/T0 +E;
end

