% Aula 10 - 25/5/23

% aula final coerencia

% implementando um sinal com freq variavel

srate = 1000;
dt = 1/srate;
t = dt:dt:10;
% t2 = cumsum(dt*ones(1,10000));

% f = 8;

% FM = frequency modulation
f = 20 + 10*sin(2*pi*1*t);


subplot(211)
plot(t,f)
ylabel('inst freq (hz)')
% ylim([0 15])
xlim([1 3])

% %

% forma errada de programar pois nao considera
% os valores passados
% LFP1 = sin(2*pi*f.*t);
% subplot(212)
% plot(t,LFP1)
% ylabel('mV')
% xlim([1 3])


% forma correta eh usando a funcao
% cumulative sum

LFP1 = sin(cumsum(2*pi*f.*dt));
subplot(212)
plot(t,LFP1)
ylabel('mV')
xlim([1 3])

%%

srate = 1000;
dt = 1/srate;
t = dt:dt:200;

% f1 = 8 + 4*randn(size(t));
% f2 = 8 + 4*randn(size(t));

% % programando um ruido comum:
ruido = 1*randn(size(t));
f1 = 8 + 4*ruido;
f2 = 8 + 4*ruido;


LFP1 = sin(cumsum(2*pi*f1.*dt))+0*sin(2*pi*16*t);
LFP2 = sin(cumsum(2*pi*f2.*dt))+0*sin(2*pi*16*t+pi);

LFP1 = LFP1 + 1*randn(size(LFP1));
LFP2 = LFP2 + 1*randn(size(LFP2));

% %
subplot(211)
plot(t,LFP1)
hold on
plot(t,LFP2)
hold off
xlim([0. 0.5]+0.5)

%%

[Pxx1 F] = pwelch(LFP1,1*srate,[],10*srate,srate);
[Pxx2 F] = pwelch(LFP2,1*srate,[],10*srate,srate);

[Cxy F] = mscohere(LFP1,LFP2,1*srate,[],10*srate,srate);


subplot(2,3,4)
plot(F,Pxx1)
xlim([0 20])
ylabel('power')
xlabel('freq hz')

subplot(2,3,5)
plot(F,Pxx2)
xlim([0 20])
ylabel('power')
xlabel('freq hz')
% %

subplot(2,3,6)
plot(F,Cxy)
xlim([0 20])
ylabel('coherence')
xlabel('freq hz')

ylim([0 1])

%%

clear CxySurrAll
for Nsurr = 1:20

i = 30000+randi(50000);
Xsurr = circshift(LFP1,i);

[CxySurr F] = mscohere(Xsurr,LFP2,1*srate,[],10*srate,srate);
   
plot(F,CxySurr)
xlim([0 20])
ylim([0 1.1])
pause(0.001)
  CxySurrAll(Nsurr,:) =   CxySurr;
    
end

% %

clf
plot(F,Cxy)
hold on
plot(F,mean(CxySurrAll),'k-','linew',2)
hold  on
plot(F,mean(CxySurrAll)-3*std(CxySurrAll),'k--')
plot(F,mean(CxySurrAll)+3*std(CxySurrAll),'k--')
hold  off

xlabel('Freq (Hz)')
ylabel('Coherence')
xlim([0 20])

ylim([0 1])


%% Coherograma

srate = 1000;
dt = 1/srate;
t = dt:dt:180;

f1 = 8 + 4*randn(size(t));
f2 = 8 + 4*randn(size(t));

LFP1 = sin(cumsum(2*pi*f1.*dt));
LFP2 = sin(cumsum(2*pi*f2.*dt));

LFP2(90*srate:end) = LFP1(90*srate:end);

LFP1 = LFP1 + 0.5*randn(size(LFP1));
LFP2 = LFP2 + 0.5*randn(size(LFP2));

% parametros do "janelao", ou seja,
% a janela para computar cada Cxy
win = 20*srate;
step = 0.1*win;
Nwin = (length(t)-win)/step + 1;

% parametros da Cxy
cohwin = 2*srate;
overlap = 0.5*cohwin;
nfft = 10*srate;

clear Coherogram T

for nwin = 1:Nwin
    nwin
    Idx = (1:win) + (nwin-1)*step;
    [Cxy F] = mscohere(LFP1(Idx),LFP2(Idx),cohwin,overlap,nfft,srate);
Coherogram(nwin,:) = Cxy;
% T(nwin) = t(Idx(win/2))
T(nwin) = mean(t(Idx));

end

%%

subplot(111)
imagesc(T,F,Coherogram')

axis xy
ylim([0 20])
xlabel('Frequency (hz)')
ylabel('Coherence')
colorbar
caxis([0 1])




%% Adendo 2024
% surrogate coherogram


clear CoherogramSurr T

for nsurr = 1:10

i = 40000+randi(20000);
LFP1surr = circshift(LFP1,i);

for nwin = 1:Nwin
    nwin
    Idx = (1:win) + (nwin-1)*step;
    [Cxy F] = mscohere(LFP1surr(Idx),LFP2(Idx),cohwin,overlap,nfft,srate);
CoherogramSurr(nsurr,nwin,:) = Cxy;
% T(nwin) = t(Idx(win/2))
T(nwin) = mean(t(Idx));

end
end

%%

nsurr = 10
subplot(212)
imagesc(T,F,squeeze(CoherogramSurr(nsurr,:,:))')
imagesc(T,F,squeeze(mean(CoherogramSurr))')


axis xy
ylim([0 20])
xlabel('Frequency (hz)')
ylabel('Coherence')
colorbar
caxis([0 1])

%%
figure(2)



subplot(211)
imagesc(T,F,Coherogram')

axis xy
ylim([0 20])
xlabel('Frequency (hz)')
ylabel('Coherence')
colorbar
caxis([0 1])

subplot(212)
Msurr = squeeze(mean(CoherogramSurr))';
% SDsurr = squeeze(std(CoherogramSurr))';

% imagesc(T,F,Coherogram'-(Msurr+2*SDsurr))
imagesc(T,F,Coherogram'-Msurr)

axis xy
ylim([0 20])
xlabel('Frequency (hz)')
ylabel('Coherence')
colorbar
caxis([0 0.8])


