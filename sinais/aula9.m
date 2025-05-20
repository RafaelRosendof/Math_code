% aula 9 - ASC - graduacao 25/1

% continuacao de coerencia

% usando a funcao built in

% mscohere -> ms significa magnitude squared
% ou seja, a funcao eleva a coerencia ao quadrado

clear, clc, clf
%%
srate = 1000;
dt = 1/srate;
t = dt:dt:10;

ruidoX =  1*randn(size(t));
ruidoY =  1*randn(size(t));

phi = 0*pi
X = 3*sin(2*pi*8*t) + ruidoX;
Y = 3*sin(2*pi*8*t+phi) + ruidoY;

subplot(311)
plot(t,X)
hold on
plot(t,Y)
hold off

xlim([1 2])

% %

% clf

% ver incerteza vs finite sample bias
win = 10*srate;
overlap = 0*win;
% freqvect = 0:0.1:20;

nfft = 10*srate; % neste caso, a freq resolution eh
% dada por srate/nfft

% % %%%%%%%%%%%%%

% [Cxy F] = mscohere(X,Y,win,overlap,freqvect,srate);
[Cxy F] = mscohere(X,Y,win,overlap,nfft,srate);

%%%%%%%%%%%%%%%%

subplot(3,1,[2 3])
hold on
plot(F,Cxy,'b-')
% hold on
% plot(F,sqrt(Cxy),'r-')
% % 
% hold off

xlabel('Freq (Hz)')
ylabel('Coherence')

ylim([0 1.1])
xlim([0 20])

%% fazendo varios exps
% computando a coerencia media

clear, clc, clf

srate = 1000;
dt = 1/srate;
t = dt:dt:30;

clear CxyAll

for nexp = 1:10
phi = pi;
X = 3*sin(2*pi*8*t) + 0.3*randn(size(t));
Y = 0.5*sin(2*pi*8*t+phi) + 0.3*randn(size(t));    
   
win = 2*srate;
overlap = 0.25*win;
nfft = 10*srate; % neste caso, a freq resolution eh
[Cxy F] = mscohere(X,Y,win,overlap,nfft,srate);
   
CxyAll(nexp,:)=Cxy;

subplot(1,1,1)
plot(F,Cxy,'k-')
xlabel('Freq (Hz)')
ylabel('Coherence')
xlim([0 20])
pause(0.01)

end

%%
plot(F,mean(CxyAll),'k-','linew',2)
hold  on
plot(F,mean(CxyAll)-std(CxyAll)/sqrt(nexp),'k--')
plot(F,mean(CxyAll)+std(CxyAll)/sqrt(nexp),'k--')
hold  off

xlabel('Freq (Hz)')
ylabel('Coherence')
xlim([0 20])
ylim([0 1.1])

%% trabalhando com surrogados
% para ver a significancia estatistica
% da coerencia

phi = 1*pi;

ruido = 0.8*randn(size(t));
X = 3*sin(2*pi*8*t) + 1*randn(size(t));
Y = 4*sin(2*pi*8*t+phi) + 1*randn(size(t));    
  
% X = 3*sin(2*pi*8*t) + ruido;
% Y = 3*sin(2*pi*8*t+0.64*phi) + ruido;    
 

win = 1*srate;
overlap = 0.5*win;
nfft = 10*srate; % neste caso, a freq resolution eh
[Cxy F] = mscohere(X,Y,win,overlap,nfft,srate);
 
subplot(311)
plot(t,X)
hold on
plot(t,Y)
hold off
xlim([0 1])

subplot(3,1,[2 3])
plot(F,Cxy)

xlabel('Freq (Hz)')
ylabel('Coherence')
xlim([0 20])
ylim([0 1.1])


%%

I = find(F>7 & F<9);
CohTheta= mean(Cxy(I))

I = find(F>0.5 & F<5);
CohDelta= mean(Cxy(I))



%% surrogado do tipo shuffling

IrandX = randperm(length(X));
IrandY = randperm(length(Y));

Xsurr = X(IrandX);
Ysurr = Y(IrandY);

[CxySurr F] = mscohere(Xsurr,Ysurr,win,overlap,nfft,srate);

hold on
plot(F,CxySurr,'k-')
hold off

%%
clear CxySurrAll
for Nsurr = 1:50
IrandX = randperm(length(X));
IrandY = randperm(length(Y));

Xsurr = X(IrandX);
Ysurr = Y(IrandY);

[CxySurr F] = mscohere(Xsurr,Ysurr,win,overlap,nfft,srate);
   
  CxySurrAll(Nsurr,:) =   CxySurr;
    
end

%%

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

%%

I = find(F>7 & F<9);
CohThetaSurr= mean(CxySurrAll(:,I),2)

I = find(F>0.5 & F<5);
CohDeltaSurr= mean(CxySurrAll(:,I),2)

bar([1 2],[CohDelta CohTheta],0.25)
hold on
bar([1.3 2.3],[mean(CohDeltaSurr) mean(CohThetaSurr)],0.25)

errorbar([1.3 2.3],[mean(CohDeltaSurr) mean(CohThetaSurr)],...
    3*[std(CohDeltaSurr) std(CohThetaSurr)],'k.')

hold off
ylim([0 1.1])

%%

plot(t,Xsurr)
hold  on
plot(t,Ysurr)
hold off
xlim([0 0.1])

%%

sound(sin(2*pi*30*t))
sound(Xsurr)


%% Adendo 2024
% programando circular shift

plot(t,X)
xlim([1 2])

%% sortear um numero randomico entre
% 1 e 3 segundos

i = 1000+randi(2000);
Xsurr = [X(i:end) X(1:i-1)];
Xsurr = circshift(X,i);

hold on
plot(t,Xsurr)
hold off

%%
clear CxySurrAll
for Nsurr = 1:100

i = 1000+randi(2000);
Xsurr = circshift(X,i);

[CxySurr F] = mscohere(Xsurr,Y,win,overlap,nfft,srate);
   
plot(F,CxySurr)
xlim([0 20])
ylim([0 1.1])
pause(0.001)
  CxySurrAll(Nsurr,:) =   CxySurr;
    
end

%% usando cumulative sum

DT = dt*ones(size(X))

t2 = cumsum(DT)

%%


f = 7+4*rand(size(X))
% f = 7 + 3*sin(2*pi*3*t)

LFP = sin(2*pi*f.*t);
subplot(211)
plot(t,LFP)
xlim([4 5])

% %

LFP1 = sin(cumsum(2*pi*f.*dt));
subplot(212)
% hold on
plot(t,LFP1)
hold off
xlim([4 5])


%%

[Pxx F] = pwelch(LFP,2*srate,[],10*srate,srate);
subplot(211)
plot(F,Pxx)
xlim([0 40])
% ylim([0 1])

[Pxx F] = pwelch(LFP1,2*srate,[],10*srate,srate);
subplot(212)
% hold on
plot(F,Pxx)
xlim([0 40])


%%

% aula final coerencia

% implementando um sinal com freq variavel

srate = 1000;
dt = 1/srate;
t = dt:dt:10;
% t2 = cumsum(dt*ones(1,10000));

% f = 8;

% FM = frequency modulation
% f = 20 + 10*sin(2*pi*1*t);
f = 20 + randn(size(t));


subplot(211)
plot(t,f)
ylabel('inst freq (hz)')
% ylim([0 15])
xlim([1 3])

%%

% forma errada de programar pois nao considera
% os valores passados
LFP1 = sin(2*pi*f.*t);
subplot(212)
plot(t,LFP1)
ylabel('mV')
xlim([0 10])

%% forma correta eh usando a funcao
% cumulative sum

LFP1 = sin(cumsum(2*pi*f.*dt));
subplot(212)
plot(t,LFP1)
ylabel('mV')
xlim([1 3])

%%

[Pxx F] = pwelch(LFP1,10*srate,[],10*srate,srate);
subplot(211)
plot(F,Pxx)
xlim([0 40])


%%

y = chirp(t,2,9,10)

plot(t,y)
















