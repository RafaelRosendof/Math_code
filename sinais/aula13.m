% aula 13

% Transformada de Hilbert

clear, clc, clf

srate = 1000;
dt = 1/srate;
t = dt:dt:3;

% LFP = sin(2*pi*8*t);
% 
% LFP = sin(2*pi*8*t)+...
%     sin(2*pi*30*t)+...
%     0*sin(2*pi*1*t);

% LFP = randn(size(t));

ruidobranco = randn(size(t));
LFP = eegfilt(ruidobranco,srate,8,10);
% LFP = LFP+1;
% LFP = exp(t);
% LFP = eegfilt(ruidobranco,srate,4,20);

% LFP = ruidobranco;
% A funcao hilbert, built-in do Matlab, tem
% como output a Representacao Analitica do Sinal
% definida por RA = LFP + iH(LFP), onde 
% H(LFP) eh a transformada de Hilbert do sinal

RA = hilbert(LFP);

sinal_original = real(RA);
HT = imag(RA);


% %

subplot(1,1,1)
plot(t,LFP)
hold on
% plot(t,sinal_original)
plot(t,HT,'r-')
hold off


%%

% % framework geral
% filtrado = eegfilt(LFP,srate,lowfreq,highfreq);
% Amp = abs(hilbert(filtrado));


%%

Amp = abs(hilbert(LFP));

for n = 1:3:3000; % amostra
subplot(311)
plot(t,LFP)
hold on
plot(t,HT,'r-')
plot(t,Amp,'k-','linewidth',2)

plot(t(n),LFP(n),'bo')
plot(t(n),HT(n),'ro')
plot(t(n),Amp(n),'ko')

hold off


xlim([0 3])

subplot(3,1,[2 3])

plot([0 LFP(n)],[0 HT(n)],'k-')
hold on
plot(LFP(n),HT(n),'ko')

plot(LFP(n),0,'bo')
plot(0,HT(n),'ro')

title(['Amplitude instant?nea = ' ...
    num2str(Amp(n))])

axis square

% xlim([-.2 .2])
xlim([-1 1]*0.2)
ylim(xlim())


hold off
pause(0.001)
end


%%

LFP = sin(2*pi*8*t).*sin(2*pi*0.5*t);

HT = imag(hilbert(LFP));

EnvAmp = abs(hilbert(LFP));

subplot(311)
plot(t,LFP)
hold on
plot(t,HT,'r-')
plot(t,EnvAmp,'k-','linew',3)
hold off

%%

subplot(3,1,[2 3])
% v = 1:10;
% plot3(v,v.^2,v) % plot3(x,y,z)

% %
plot3(t,LFP,HT)
hold on
plot3(t,zeros(size(t)),zeros(size(t)),...
    'r-','linew',1)
hold off

xlabel('time (s)')
ylabel('Real')
zlabel('Imag')

%% observacao: 

% a transformada de hilbert de um sinal
% constante eh zero

X = -ones(1,1000);
HT = imag(hilbert(X));

AmplitudeEnvelope = abs(hilbert(X));

plot(X)
hold on
plot(HT)
plot(AmplitudeEnvelope,'k-','linew',2)
hold off

ylim([-2 2])

%%
LFP = 3*sin(2*pi*20*t)
ampenv = abs(hilbert(LFP));

plot(LFP)
hold on
plot(ampenv)
hold off

%% Amplitude Spectrum

clear, clc, clf

srate = 1000;
dt = 1/srate;
t = dt:dt:10;

LFP = 4*sin(2*pi*20*t) + 3*sin(2*pi*70*t);

order = 300 % ordem do filtro

f = 20;

kernel = sin(2*pi*f*t(1:order));
kernel = kernel/sum(kernel.^2);

Filtrado = conv(LFP,kernel,'same');
Filtrado = conv(Filtrado,kernel(end:-1:1),'same');

HT = imag(hilbert(Filtrado));
Amp = abs(hilbert(Filtrado));

% computando a amplitude media
MeanAmp = mean(Amp);

% computando o Root Mean Square
RMS = sqrt(mean(Filtrado.^2));

subplot(111)
plot(t,LFP)
hold on
plot(t,Filtrado,'b-')
plot(t,HT,'r-')
plot(t,Amp,'k-','linew',2)
xlim([1 1.2])
hold off

title(['Freq = ' int2str(f) ...
    ' Hz;  Amp = ' num2str(MeanAmp) ...
    ';  RMS = ' num2str(RMS)])

%% implementando um loop nas frequencias

clear, clc, clf

srate = 1000;
dt = 1/srate;
t = dt:dt:10;

% LFP = 4*sin(2*pi*20*t) + 2*sin(2*pi*70*t);

LFP = sin(2*pi*20*t) + sin(2*pi*40*t)+ sin(2*pi*70*t);

% LFP = 3*sin(2*pi*20*t) + sin(2*pi*40*t)+ sin(2*pi*70*t);

order = 300 % ordem do filtro
freqvector = 1:1:100;

for f = freqvector;

kernel = sin(2*pi*f*t(1:order));
kernel = kernel/sum(kernel.^2);

Filtrado = conv(LFP,kernel,'same');
Filtrado = conv(Filtrado,kernel(end:-1:1),'same');

HT = imag(hilbert(Filtrado));
Amp = abs(hilbert(Filtrado));

% computando a amplitude media
MeanAmp = mean(Amp);
AmpSpectrum(f) = MeanAmp;

% computando o Root Mean Square
RMS = sqrt(mean(Filtrado.^2));
RMSSpectrum(f) = RMS;


subplot(311)
plot(t,LFP)
hold on
plot(t,Filtrado,'b-')
plot(t,HT,'r-')
plot(t,Amp,'k-','linew',2)
xlim([1 1.2])
hold off

title(['Freq = ' int2str(f) ...
    ' Hz;  Amp = ' num2str(MeanAmp) ...
    ';  RMS = ' num2str(RMS)])

pause(0.05)

end

%%

subplot(3,2,[3 5])
plot(freqvector,AmpSpectrum)
xlabel('Frequency (Hz)')
ylabel('Amplitude (mV)')

subplot(3,2,[4 6])
plot(freqvector,RMSSpectrum)
xlabel('Frequency (Hz)')
ylabel('RMS (mV)')


%% Computando o espectro de amplitude via eegfilt

srate = 1000;
dt = 1/srate;
t = dt:dt:10;

LFP = sin(2*pi*20*t) + sin(2*pi*40*t)+ sin(2*pi*70*t);

freqvector = 1:1:100
bandwidth = 6
clear AmpSpectrum
for flow = freqvector
flow
fhigh = flow+bandwidth;
% filtrado = eegfilt(LFP,srate,flow,fhigh,0,300);
filtrado = eegfilt(LFP,srate,flow,fhigh);
AmpSpectrum(flow) = mean(abs(hilbert(filtrado)));

end


%%
% subplot(3,2,[4 6])
% subplot(111)
subplot(211)
plot(freqvector+bandwidth/2,AmpSpectrum)
xlabel('Frequency (Hz)')
ylabel('Amplitude (mV)')

%%
subplot(212)
[Pxx F] = pwelch(LFP,2*srate,[],freqvector,srate)
hold on
plot(F,Pxx)
xlim([0 120])

%% added 2024

flow = 70
fhigh = flow+bandwidth;
[filtrado,K] = eegfilt(LFP,srate,flow,fhigh) %,0,300);

subplot(212)
[Pxx F] = pwelch(K,length(K),[],freqvector,srate)
% [Pxx F] = pwelch(K,length(K),[],2^16,srate)

hold on
plot(F,Pxx)
% xlim([0 120])
%%
%% TFD "continua" - time frequency decomposition

LFP = sin(2*pi*10*t) + 1*sin(2*pi*40*t);
LFP(1:5000) = 0;

LFPfiltered  = eegfilt(LFP,srate,5,15);
AmpEnv = abs(hilbert(LFPfiltered));
% AmpEnv = abs(hilbert(LFP));

clf
plot(t,LFP)
hold on
plot(t,LFPfiltered,'r-')
plot(t,AmpEnv,'k-','linew',2)
hold off

% xlim([4.5 5.5])

%%

clear TFD

freqvector = 0:1:100;

tic
count = 0;
for f = freqvector
    count = count + 1;
    
% LFPfiltered = eegfilt(LFP,srate,f,f+4,0,200);
LFPfiltered = eegfilt(LFP,srate,f,f+4);

%    LFPfiltered = eegfilt(LFP,srate,f,f+4,0,300);
   AmpEnv = abs(hilbert(LFPfiltered));
   TFD(count,:) = AmpEnv;
   
  plot(t,LFP)
hold on
plot(t,LFPfiltered,'r-')
plot(t,AmpEnv,'k-','linew',2)
hold off

title(['Frequency (Hz) = ' num2str(f+2) ' Hz'])
xlim([4.5 5.5])

pause(0.1)
   
end
toc


%%

subplot(212)
% subplot(111)
imagesc(t,freqvector+2,TFD)
% imagesc(t,freqvector+5,TFD)

axis xy
xlabel('Time (s)')
ylabel('Frequency (Hz)')
colorbar

%%

subplot(211)
tic
[S,F,T,TFD2]=spectrogram(LFP,0.2*srate,[],freqvector+2,srate);
toc
imagesc(t,F,TFD2)
axis xy
xlabel('Time (s)')
ylabel('Frequency (Hz)')
colorbar








