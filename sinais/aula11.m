% Aula 11 - 2025

X = [0 2 1 1 0]
%  2 2 1 
K = [1 2 2] %                 



%%                     
ConvXK = conv(X,K)


%%
K =        [1 2 2] %                 
%        [0 1 1 2 0]
% X = [0 2 1 1 0]

% ConvXK = conv(K,X)


%%
ConvXK = conv(X,K,'same')

% ConvXK = conv(K,X,'same')


%%

X = [0 2 1 1 -1 7 -2 -3 8 1 -1 2 -3 4]
K = [1 1 -2]

ConvXK = conv(X,K)
ConvXKB = conv(X,K,'same')
clf

subplot(311)
plot(2:15,X,'bo-')
xlim([0 17])

subplot(312)
% plot(K,'ro-')
% hold on
j=10
plot([0 1 2]+j,K(end:-1:1),'ro-')
hold off
xlim([0 17])

subplot(313)
plot(ConvXK,'ko-')
hold on
plot(2:15,ConvXKB,'yo-')
hold off
xlim([0 17])

%% Exemplo de aplicacao

clf

srate = 1000;
dt = 1/srate;
t = dt:dt:2;

LFP = randn(size(t));
subplot(211)
plot(t,LFP)


%%

ordem = 100

% moving average 
K = ones(1,ordem)/ordem;

% %

% Convola = conv(LFP,K);
Convol = conv(LFP,K,'same');
subplot(211)
plot(t,LFP)
hold on
plot(t,Convol,'k-','linew',3)
hold off

% %

subplot(212)
[PxxConv F] = pwelch(Convol,length(Convol),[],2^16,srate);
[PxxLFP F] = pwelch(LFP,length(Convol),[],2^16,srate);

plot(F,PxxLFP)
hold on
plot(F,PxxConv)
hold off
xlim([0 50])
xlabel('freq (hz)')


%%


subplot(311)
[Pxx F] = pwelch(LFP,length(K),[],2^16,srate);
hold off
plot(F,Pxx*length(LFP))
% xlim([0 100])
xlabel('freq (hz)')
ylabel('Pxx')

subplot(312)

K = ones(1,20)

[Pkk F] = pwelch(K,length(K),[],2^16,srate);
hold off
plot(F,Pkk*length(K))
% xlim([0 100])
xlabel('freq (hz)')
ylabel('Pkk')


Convol = conv(LFP,K);
subplot(313)
[Pcc F] = pwelch(Convol,length(K),[],2^16,srate);
hold off
plot(F,Pcc*2)
hold on

hold off
% xlim([0 100])
xlabel('freq (hz)')
ylabel('P convol')

%% Novo 2025

srate = 1000;
dt = 1/srate;
t = dt:dt:1;

LFP = randn(size(t));

clf
nfft = length(LFP)
subplot(311)
[Pxx F] = pwelch(LFP,rectwin(length(LFP)),0,nfft,srate);
hold off
plot(F,Pxx*length(LFP))
% xlim([0 100])
xlabel('freq (hz)')
ylabel('Pxx')

subplot(312)

K = ones(1,20)/sqrt(20)

% K_padded = zeros(1, length(LFP));
% center = floor(length(LFP)/2) - floor(length(K)/2);
% K_padded(center + (1:length(K))) = K;

[Pkk F] = pwelch(K,rectwin(length(K)),0,nfft,srate);
hold off
plot(F,Pkk*length(K))
% xlim([0 100])
xlabel('freq (hz)')
ylabel('Pkk')

% Convol = conv(LFP,K,'same');
Convol = conv(LFP,K);

subplot(313)
[Pcc F] = pwelch(Convol,rectwin(length(Convol)),0,nfft,srate);
hold off
plot(F,Pcc*length(Convol))
hold on
plot(F,Pkk*length(K).*Pxx*length(LFP))
hold off
xlabel('freq (hz)')
ylabel('P convol')




%%
subplot(121)
plot(F,Pkk*length(K).*Pxx*length(LFP))

xlim([0 50])

subplot(122)
plot(F,Pcc*length(Convol)*dt*2)

xlim([0 50])



%%

% clf

srate = 1000;
dt = 1/srate;
t = dt:dt:2;

LFP = randn(size(t));
LFP(500:1500) = LFP(500:1500) + 30*exp(-t(500:1500));

ordem = 40;
K = ones(1,ordem)/ordem;

Convol = conv(LFP,K,'same')

subplot(111)
plot(t,LFP)
hold on
plot(t,Convol,'r-','linew',3)
plot([0.2 0.2],[-5 20],'r--','linew',2)
hold off

xlabel('time (s)')

%%

% K = [ K zeros(size(LFP))];
% K = K(1:length(LFP));

% K = [ zeros(size(LFP)) K];
% K = K(end-length(LFP)+1:end);

delta = length(LFP)-length(K)
K = [zeros(1,delta/2) K zeros(1,delta/2)]

figure(2)
hold on
plot(t,ifft(fft(LFP).*fft(K)),'g-')


%% De novo

clf
clear
clc

% %
srate = 1000;
dt = 1/srate;
t = dt:dt:1;

% LFP= sin(2*pi*10*t)+1*sin(2*pi*0.5*t)+1*sin(2*pi*40*t);
LFP= sin(2*pi*10*t)+1*sin(2*pi*13*t)+1*sin(2*pi*40*t);

% LFP= sin(2*pi*10*t);
LFP = LFP + 0*randn(size(t));

order = 100;
K = sin(2*pi*10*t(1:order));
norm = sum(K.^2); % to avoid adding or removing energy
% norm = 1
% %

figure(2)

subplot(311)
[Pxx F] = pwelch(LFP,length(LFP),[],2^16,srate);
hold off
plot(F,Pxx)
xlim([0 50])
xlabel('freq (hz)')
ylabel('Pxx')


% %
subplot(312)
[Pkk F] = pwelch(K/norm,length(K),[],2^16,srate);
hold off
plot(F,Pkk)
xlim([0 50])
xlabel('freq (hz)')
ylabel('Pkk')

% %
subplot(313)
plot(F,Pkk.*Pxx)
xlim([0 50])
xlabel('freq (hz)')
ylabel('Power convol')

% %

Convol = conv(LFP,K/norm,'same')
[Pcc F] = pwelch(Convol,length(Convol),[],2^16,srate);
hold off
plot(F,Pcc)
xlim([0 50])
xlabel('freq (hz)')
ylabel('Pcc')
% ylim([0 0.4])

%%

srate = 1000;
dt = 1/srate;
t = dt:dt:1;

% LFP= sin(2*pi*10*t)+1*sin(2*pi*0.5*t)+1*sin(2*pi*40*t);
% LFP= sin(2*pi*10*t)+1*sin(2*pi*11*t)+1*sin(2*pi*40*t);

LFP= sin(2*pi*10*t) + 0*sin(2*pi*20*t);
LFP = LFP + 0*randn(size(t));

order = 100;
K = sin(2*pi*10*t(1:order));
norm = sum(K.^2); % to avoid adding or removing energy
% norm = 1
% %


figure(1)
% padding with zeros
LFP = [zeros(1,order) LFP zeros(1,order)];
t = (1:length(LFP))*dt;

for j = 0:2:length(LFP)-order
%     j = 200    
subplot(211)
plot(t,LFP)
hold on
plot(t((1:order)+j),K(order:-1:1),'r-','linew',3)
hold off
ylim([-4 4])

subplot(212)

plot(t(order/2+j),sum(K(order:-1:1).*LFP((1:order)+j))/norm,'k.')
hold on
xlim([0 t(end)])
ylim([- 4 4])
pause(0.001)
end

%%
clf

plot(K)


%%
figure(1)
Convol = conv(LFP,K/norm,'same')

% to avoid phase shifts, convolve the 
% convolution with the inverted kernell
Convol2 = conv(Convol,K(end:-1:1)/norm,'same')
% Convol2 = conv(Convol(end:-1:1),K/norm,'same')

subplot(212)
hold on
plot(t,Convol,'r-')
plot(t,Convol2,'g-','linew',3)
hold off

%%
subplot(211)
hold on
plot(t,Convol,'r-')
plot(t,Convol2,'g-','linew',3)
hold off



%%
figure(2)
subplot(313)

[Pkk F] = pwelch(Convol2,length(Convol),[],2^16,srate);
hold on
plot(F,Pkk)
xlim([0 50])
xlabel('freq (hz)')
ylabel('Pkk')






































