% aula 12

% filtrando usando a funcao eegfilt

clear
clc
clf

srate = 1000;
dt = 1/srate;
t = dt:dt:5;

LFP = sin(2*pi*10*t)+sin(2*pi*30*t)+sin(2*pi*50*t);

% funcao eegfilt chama filtfilt
% que aplica o filtro duas vezes (uma reversa
% pra corrigir distorcao de fase)

% ordem default do filtro eh dada por:

% default 3*fix(srate/locutoff)

% Low-pass filter (filtro passa baixa)
% [filtrado,K] = eegfilt(LFP,srate,0,40);

% se quiser determinar a ordem do filtro
% ordem = 50
% [filtrado,K] = eegfilt(LFP,srate,0,40,0,ordem);

% High-pass filter (filtro passa alta)
% [filtrado,K] = eegfilt(LFP,srate,40,0);
% ordem = 200
% [filtrado,K] = eegfilt(LFP,srate,2,0,0,ordem);


% Band-pass filter (filtro passa banda)
[filtrado,K]  = eegfilt(LFP,srate,20,40);
% ordem = 500
% [filtrado,K] = eegfilt(LFP,srate,40,60,0,ordem);

% mudando a ordem do filtro
% ordem = 300
% filtrado = eegfilt(LFP,srate,20,40,0,ordem);

% Notch filter (filtro rejeita banda)
% [filtrado,K]  = eegfilt(LFP,srate,20,40,0,300,1);

[Pxx F] = pwelch(LFP,length(LFP),[],2^16,srate);
[PxxFilt F] = pwelch(filtrado,length(LFP),[],2^16,srate);

[PxxKernell F2] = pwelch(K,length(K),[],2^16,srate);


figure(2)

subplot(311)
plot(t,LFP)
hold on
plot(t,filtrado,'r-')
hold off
xlim([0.8 1.2])

subplot(312)
plot(F,Pxx)
hold on
plot(F,PxxFilt,'r-')
hold off
xlim([0 70])

% %

subplot(313)
% plot(F,PxxFilt,'r-')
% xlim([0 70])

subplot(313)
plot(F2,PxxKernell,'r-')
xlim([0 70])

%% added 2024:

minfac         = 3;    % this many (lo)cutoff-freq cycles in filter
min_filtorder  = 15;   % minimum filter length
trans          = 0.01; % fractional width of transition zones

%  f=[MINFREQ locutoff*(1-trans)/nyq locutoff/nyq 1];
%  fprintf('eegfilt() - highpass transition band width is %1.1g Hz.\n',(f(3)-f(2))*srate/2);
 m=[   0             0                   1      1];
 
 nyq = 1000/2
 locutoff = 15
 
f = [0 locutoff*(1-trans)/nyq locutoff/nyq 1]
 m= [0             0                   1      1];
 m= [1             1                   0      0];
 
 
%  f = [0 98/nyq 100/nyq  150/nyq  152/nyq 1]
%  m= [0    0       1         1      0  0];

  f = [0 19/nyq 20/nyq  40/nyq  42/nyq 1]
 m= [0    0       1         1      0  0];
 
% f = [0 18/nyq 20/nyq   80/nyq 82/nyq 1]
%  m= [0  0        1         1    0   0];
 
 
% filtorder = 3*fix(srate/locutoff)
filtorder = 500
filtwts = firls(filtorder,f,m)


% %

clf
subplot(211)
[Pxx F]=pwelch(filtwts,length(filtwts),[],10*srate,srate)

plot(F,Pxx)
xlim([0 100])

subplot(212)

plot(filtwts)

%%

% filtrado = filtfilt(filtwts,1,LFP);
filtrado = filtfilt(filtwts,1,LFP);

% hold on
plot(LFP)
hold on
plot(filtrado)
hold off

%%

% Python

% % def eegfilt(data,srate,flow,fhigh):
% %    
% %     # fir LS
% %     trans = 0.15
% %     nyq = srate*0.5
% %     f=[0, (1-trans)*flow/nyq, flow/nyq, fhigh/nyq, (1+trans)*fhigh/nyq, 1]
% %     m=[0,0,1,1,0,0]
% %     filt_order = 3*np.fix(srate/flow)
% %     if filt_order % 2 == 0:
% %         filt_order = filt_order + 1
% %      
% %     filtwts = signal.firls(filt_order,f,np.double(m))
% %     data_filt = signal.filtfilt(filtwts,1, data)
% %    
% %     return(data_filt)



%% Nenhum filtro ? perfeito

% clf

figure(1)
% clf

LowFreqCutoff=20
HighFreqCutoff=40

whitenoise = randn(1,1000000);
 
% filtered = eegfilt(whitenoise,srate,...
%     LowFreqCutoff,HighFreqCutoff);

ordem = 500
filtered = eegfilt(whitenoise,srate,...
    LowFreqCutoff,HighFreqCutoff,0,ordem);

% ordem = 100
% filtered = eegfilt(whitenoise,srate,...
%     LowFreqCutoff,HighFreqCutoff,0,ordem,1);

[PxxW F] = pwelch(whitenoise,srate,[],2^16,srate);
[PxxF F] = pwelch(filtered,srate,[],2^16,srate);

% %

subplot(211)
plot(F,PxxW,'k-')
hold on
plot(F,PxxF)
plot([HighFreqCutoff HighFreqCutoff],[0 max(PxxW)*1.2],'k-')
plot([LowFreqCutoff LowFreqCutoff],[0 max(PxxW)*1.2],'k-')

hold off

% xlim([0 70])

xlim([0 100])

%% entendendo o 5o input da funcao eegfilt

% epochframes = frames per epoch 
%(filter each epoch separately {def/0: data is 1 epoch}

t = dt:dt:1;

LFP1 = sin(2*pi*10*t);
LFP2 = sin(2*pi*10*t+pi);
LFP3 = sin(2*pi*10*t+pi/2);
LFP4 = sin(2*pi*10*t+pi/3);

LFP = [LFP1 LFP2 LFP3 LFP4];

filtradoTrials = eegfilt(LFP,srate,0,15,1000);
filtradoSingle = eegfilt(LFP,srate,0,15,0);

clf

plot(LFP,'k-','linew',2)
hold on
plot(filtradoSingle,'b-')
plot(filtradoTrials,'r-')
hold off

%% Efeito da ordem novamente (no leakage temporal)

srate = 1000
dt = 1/srate
t = dt:dt:4;

lfp = sin(2*pi*10*t);
lfp(1:2000)=0;
lfp(3000:end)=0;

ordem = 50
filtrado = eegfilt(lfp,srate,0,15,0,ordem);

plot(t,lfp)
hold on
plot(t,filtrado,'r-','linew',2)
hold off


xlim([1.5 3.5])

%% Nenhum filtro eh perfeito 2


srate = 1000;
dt = 1/srate;
t = dt:dt:4;

lfp = sin(2*pi*10*t);
lfp = lfp + 0.0*randn(size(t));

filtrado = eegfilt(lfp,srate,15,0);

[Pxx F] = pwelch(lfp,4*srate,[],2^16,srate);
[PxxF F] = pwelch(filtrado,4*srate,[],2^16,srate);


subplot(221)
plot(t,lfp)
hold on
plot(t,filtrado)
hold off
xlim([1 1.5])

%%
subplot(222)
plot(F,Pxx)
hold on
plot(F,PxxF)
hold off
xlim([0 20])

subplot(223)
plot(t,filtrado,'r-')
xlim([1 1.5])

subplot(224)
plot(F,PxxF,'r-')
xlim([0 20])

%% O eegfilt aceita matrizes como input

% neste caso ele filtra por linha

% Channel vs timepoints (ou seja, cada linha ? um canal)

LFPAll(1,:) = sin(2*pi*5*t);
LFPAll(2,:) = sin(2*pi*15*t);
LFPAll(3,:) = sin(2*pi*30*t);

filtered = eegfilt(LFPAll,srate,10,25,0,[],1);


h1 = subplot(311);
plot(t,LFPAll(1,:))
hold on
plot(t,filtered(1,:))
hold off
ylim([-3 3])

h2 = subplot(312);
plot(t,LFPAll(2,:))
hold on
plot(t,filtered(2,:))
hold off
ylim([-3 3])

h3 = subplot(313);
plot(t,LFPAll(3,:))
hold on
plot(t,filtered(3,:))
hold off
ylim([-3 3])

linkaxes([h1 h2 h3])

% linkaxes([h1 h2 h3],'x')

%%


















